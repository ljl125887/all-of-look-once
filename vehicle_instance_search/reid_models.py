"""车辆/人员实例特征模型。

FastReID 网络结构根据 JDAI-CV/fast-reid（Apache-2.0）精简实现，仅保留
VeRi 推理所需的 ResNet50-IBN、Non-local 与 GeM/BN neck。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

from person_osnet import load_osnet_x1_0
from face_models import FaceFeatureExtractor


PERSON_MODEL_FASTREID = "FastReID基线"
PERSON_MODEL_OSNET = "OSNet人员专用"
PERSON_MODEL_FUSION = "FastReID+OSNet融合（推荐）"
PERSON_MODEL_OPTIONS = (PERSON_MODEL_FUSION, PERSON_MODEL_OSNET, PERSON_MODEL_FASTREID)


class IBN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2
        self.half = half
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(channels - half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = torch.split(x, [self.half, x.shape[1] - self.half], dim=1)
        return torch.cat((self.IN(left.contiguous()), self.BN(right.contiguous())), dim=1)


class NonLocal(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # FastReID v1.0 VeRi 权重使用 1 个中间通道。
        inner = 1
        self.in_channels = channels
        self.inter_channels = inner
        self.g = nn.Conv2d(channels, inner, 1)
        self.W = nn.Sequential(nn.Conv2d(inner, channels, 1), nn.BatchNorm2d(channels))
        self.theta = nn.Conv2d(channels, inner, 1)
        self.phi = nn.Conv2d(channels, inner, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        g_x = self.g(x).view(batch, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch, self.inter_channels, -1)
        affinity = torch.matmul(theta_x, phi_x) / max(1, phi_x.shape[-1])
        y = torch.matmul(affinity, g_x).permute(0, 2, 1).contiguous()
        y = y.view(batch, self.inter_channels, *x.shape[2:])
        return self.W(y) + x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        with_ibn: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if with_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNetIBNNonLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(64, 3, stride=1, with_ibn=True)
        self.layer2 = self._make_layer(128, 4, stride=2, with_ibn=True)
        self.layer3 = self._make_layer(256, 6, stride=2, with_ibn=True)
        self.layer4 = self._make_layer(512, 3, stride=1, with_ibn=False)
        self.NL_1 = nn.ModuleList()
        self.NL_2 = nn.ModuleList([NonLocal(512) for _ in range(2)])
        self.NL_3 = nn.ModuleList([NonLocal(1024) for _ in range(3)])
        self.NL_4 = nn.ModuleList()
        self.NL_1_idx: list[int] = []
        self.NL_2_idx = [2, 3]
        self.NL_3_idx = [3, 4, 5]
        self.NL_4_idx: list[int] = []

    def _make_layer(self, planes: int, blocks: int, stride: int, with_ibn: bool) -> nn.Sequential:
        outplanes = planes * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, with_ibn)]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, with_ibn=with_ibn))
        return nn.Sequential(*layers)

    @staticmethod
    def _run_stage(x: torch.Tensor, stage: nn.Sequential, modules: nn.ModuleList, indices: list[int]) -> torch.Tensor:
        nl_index = 0
        for index, block in enumerate(stage):
            x = block(x)
            if nl_index < len(indices) and index == indices[nl_index]:
                x = modules[nl_index](x)
                nl_index += 1
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self._run_stage(x, self.layer1, self.NL_1, self.NL_1_idx)
        x = self._run_stage(x, self.layer2, self.NL_2, self.NL_2_idx)
        x = self._run_stage(x, self.layer3, self.NL_3, self.NL_3_idx)
        return self._run_stage(x, self.layer4, self.NL_4, self.NL_4_idx)


class GeM(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=1e-6).pow(self.p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)


class ReIDHead(nn.Module):
    def __init__(self, num_classes: int, flat_bnneck: bool = False):
        super().__init__()
        self.pool_layer = GeM()
        self.flat_bnneck = flat_bnneck
        if flat_bnneck:
            self.bnneck = nn.BatchNorm1d(2048)
        else:
            self.bottleneck = nn.Sequential(nn.BatchNorm2d(2048))
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_layer(x)
        if self.flat_bnneck:
            return self.bnneck(pooled.flatten(1))
        return self.bottleneck(pooled).flatten(1)


class FastReIDNet(nn.Module):
    def __init__(self, num_classes: int, flat_bnneck: bool = False):
        super().__init__()
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1))
        self.backbone = ResNetIBNNonLocal()
        self.heads = ReIDHead(num_classes, flat_bnneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        return F.normalize(self.heads(self.backbone(x)), dim=1)


@dataclass(slots=True)
class FeatureBundle:
    vehicle: np.ndarray
    person_osnet: np.ndarray | None
    visual: np.ndarray
    color: np.ndarray
    aspect: float
    modality: str
    face: np.ndarray | None = None


@dataclass(slots=True)
class Similarity:
    score: float
    vehicle: float
    visual: float
    color: float
    shape: float
    reference_index: int
    face: float = -1.0


def _safe_crop(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    return crop


def infer_modality(crop: np.ndarray) -> str:
    """粗分可见光/低照度/热红外，只在同模态参考之间做实例匹配。"""
    crop = _safe_crop(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[..., 2].mean())
    saturation = float(hsv[..., 1].mean())
    channel_spread = float(np.mean(np.max(crop, axis=2) - np.min(crop, axis=2)))
    if brightness < 38.0:
        return "低照度"
    if saturation < 9.0 and channel_spread < 8.0:
        return "热红外"
    return "可见光"


def _color_descriptor(crop: np.ndarray) -> np.ndarray:
    crop = _safe_crop(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 降低暗背景影响，保留车身主体的色相/饱和度分布。
    mask = cv2.inRange(hsv, np.array([0, 18, 28], np.uint8), np.array([179, 255, 255], np.uint8))
    hist = cv2.calcHist([hsv], [0, 1], mask, [24, 8], [0, 180, 0, 256]).reshape(-1)
    if float(hist.sum()) < 1.0:
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256]).reshape(-1)
    norm = float(np.linalg.norm(hist))
    return (hist / max(norm, 1e-8)).astype(np.float32)


def _resize_rgb(crop: np.ndarray, size: int, preserve_aspect: bool) -> np.ndarray:
    rgb = cv2.cvtColor(_safe_crop(crop), cv2.COLOR_BGR2RGB)
    if not preserve_aspect:
        return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    h, w = rgb.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    x, y = (size - nw) // 2, (size - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def _resize_rgb_hw(crop: np.ndarray, height: int, width: int) -> np.ndarray:
    rgb = cv2.cvtColor(_safe_crop(crop), cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)


class VehicleFeatureExtractor:
    """共享 DINOv2，并为车辆、人员分别使用专用 FastReID 权重。"""

    def __init__(
        self,
        fastreid_weights: Path,
        dino_weights: Path,
        person_weights: Path | None = None,
        person_osnet_weights: Path | None = None,
        face_detector_weights: Path | None = None,
        face_recognizer_weights: Path | None = None,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.models: dict[str, FastReIDNet] = {}
        self.models["车辆"] = self._load_reid_model(fastreid_weights)
        self.person_weights = person_weights
        if person_weights is not None and person_weights.is_file():
            self.models["人员"] = self._load_reid_model(person_weights)
        self.person_osnet_weights = person_osnet_weights
        self.person_osnet = None
        if person_osnet_weights is not None and person_osnet_weights.is_file():
            self.person_osnet = load_osnet_x1_0(person_osnet_weights, self.device)
        self.face_extractor = None
        if (
            face_detector_weights is not None
            and face_recognizer_weights is not None
            and face_detector_weights.is_file()
            and face_recognizer_weights.is_file()
        ):
            self.face_extractor = FaceFeatureExtractor(face_detector_weights, face_recognizer_weights)

        self.visual_model = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0)
        dino_state = torch.load(dino_weights, map_location="cpu", weights_only=True)
        self.visual_model.load_state_dict(dino_state, strict=False)
        self.visual_model.eval().to(self.device)

    def _load_reid_model(self, path: Path) -> FastReIDNet:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        classifier = state.get("heads.classifier.weight")
        num_classes = int(classifier.shape[0]) if classifier is not None else 1
        flat_bnneck = "heads.bnneck.weight" in state and "heads.bottleneck.0.weight" not in state
        model = FastReIDNet(num_classes, flat_bnneck)
        incompatible = model.load_state_dict(state, strict=False)
        bad_missing = [key for key in incompatible.missing_keys if not key.startswith("heads.bnneck")]
        bad_unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("heads.bnneck")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"FastReID 权重不兼容: missing={bad_missing[:8]}, unexpected={bad_unexpected[:8]}")
        return model.eval().to(self.device)

    def _reid_tensor(self, crops: Iterable[np.ndarray], category: str) -> torch.Tensor:
        if category == "人员":
            arrays = [_resize_rgb_hw(crop, 384, 128) for crop in crops]
        else:
            arrays = [_resize_rgb_hw(crop, 256, 256) for crop in crops]
        return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float()

    def _visual_tensor(self, crops: Iterable[np.ndarray]) -> torch.Tensor:
        arrays = [_resize_rgb(crop, 518, preserve_aspect=True) for crop in crops]
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def _osnet_tensor(self, crops: Iterable[np.ndarray]) -> torch.Tensor:
        arrays = [_resize_rgb_hw(crop, 256, 128) for crop in crops]
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def extract(self, crops: list[np.ndarray], category: str = "车辆") -> list[FeatureBundle]:
        if not crops:
            return []
        if category not in self.models:
            if category == "人员" and self.person_weights is not None:
                raise FileNotFoundError(f"人员 ReID 权重尚未下载完整：{self.person_weights}")
            raise ValueError(f"暂不支持类别：{category}")
        use_amp = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            vehicle = self.models[category](self._reid_tensor(crops, category).to(self.device)).float().cpu().numpy()
            visual = self.visual_model(self._visual_tensor(crops).to(self.device)).float()
            visual = F.normalize(visual, dim=1).cpu().numpy()
            person_osnet = None
            if category == "人员":
                if self.person_osnet is None:
                    raise FileNotFoundError(f"人员OSNet权重尚未下载完整：{self.person_osnet_weights}")
                person_osnet = self.person_osnet(self._osnet_tensor(crops).to(self.device)).float().cpu().numpy()
        bundles: list[FeatureBundle] = []
        faces = (
            self.face_extractor.extract(crops)
            if category == "人员" and self.face_extractor is not None
            else [None] * len(crops)
        )
        for index, crop in enumerate(crops):
            h, w = crop.shape[:2]
            bundles.append(
                FeatureBundle(
                    vehicle=vehicle[index].astype(np.float32),
                    person_osnet=None if person_osnet is None else person_osnet[index].astype(np.float32),
                    visual=visual[index].astype(np.float32),
                    color=_color_descriptor(crop),
                    aspect=float(w / max(1, h)),
                    modality=infer_modality(crop),
                    face=faces[index],
                )
            )
        return bundles

    @staticmethod
    def compare(
        candidate: FeatureBundle,
        references: list[FeatureBundle],
        category: str = "车辆",
        person_model: str = PERSON_MODEL_FUSION,
    ) -> Similarity:
        similarities: list[Similarity] = []
        for index, reference in enumerate(references):
            if candidate.modality != reference.modality:
                continue
            vehicle = float(np.dot(candidate.vehicle, reference.vehicle))
            osnet = (
                float(np.dot(candidate.person_osnet, reference.person_osnet))
                if candidate.person_osnet is not None and reference.person_osnet is not None
                else -1.0
            )
            face = (
                float(np.dot(candidate.face, reference.face))
                if candidate.face is not None and reference.face is not None
                else -1.0
            )
            visual = float(np.dot(candidate.visual, reference.visual))
            color = float(np.dot(candidate.color, reference.color))
            shape = float(np.exp(-abs(np.log(max(candidate.aspect, 1e-4) / max(reference.aspect, 1e-4)))))
            if category == "人员":
                if person_model == PERSON_MODEL_FASTREID:
                    reid = vehicle
                    score = 0.56 * vehicle + 0.31 * visual + 0.10 * color + 0.03 * shape
                elif person_model == PERSON_MODEL_OSNET:
                    reid = osnet
                    score = 0.68 * osnet + 0.22 * visual + 0.07 * color + 0.03 * shape
                else:
                    reid = 0.50 * vehicle + 0.50 * osnet
                    score = 0.34 * vehicle + 0.34 * osnet + 0.24 * visual + 0.06 * color + 0.02 * shape
            else:
                # 专用车辆 ReID 与 DINOv2 为主；颜色和轮廓只做弱约束。
                reid = vehicle
                score = 0.48 * vehicle + 0.37 * visual + 0.10 * color + 0.05 * shape
            current = Similarity(score, reid, visual, color, shape, index, face)
            similarities.append(current)
        if not similarities:
            return Similarity(-1.0, -1.0, -1.0, -1.0, -1.0, -1)
        similarities.sort(key=lambda item: item.score, reverse=True)
        best = similarities[0]
        if category == "人员" and len(similarities) >= 2:
            # 人员多参考图要求至少两张证据一致，避免某一张偶然与同制服人员相似。
            top = similarities[:2]
            return Similarity(
                score=float(np.mean([item.score for item in top])),
                vehicle=float(np.mean([item.vehicle for item in top])),
                visual=float(np.mean([item.visual for item in top])),
                color=float(np.mean([item.color for item in top])),
                shape=float(np.mean([item.shape for item in top])),
                reference_index=best.reference_index,
                face=max((item.face for item in similarities), default=-1.0),
            )
        return best


# 保留原有导入名，避免旧脚本失效。
FastReIDVehicleNet = FastReIDNet
