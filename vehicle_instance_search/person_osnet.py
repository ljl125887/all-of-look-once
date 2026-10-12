"""OSNet x1.0 人员 ReID 推理网络。

网络结构精简自 Kaiyang Zhou 的 deep-person-reid（MIT License），只保留
OSNet x1.0 特征提取所需代码。原项目：https://github.com/KaiyangZhou/deep-person-reid
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(ConvLayer):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, 1)


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * weights


class OSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(
            LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid)
        )
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.gate(self.conv2a(x)) + self.gate(self.conv2b(x)) + self.gate(self.conv2c(x)) + self.gate(self.conv2d(x))
        return F.relu(self.conv3(x) + identity)


class OSNetX10(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        channels = [64, 256, 384, 512]
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(channels[0], channels[1], reduce=True)
        self.conv3 = self._make_layer(channels[1], channels[2], reduce=True)
        self.conv4 = self._make_layer(channels[2], channels[3], reduce=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, reduce: bool) -> nn.Sequential:
        layers: list[nn.Module] = [OSBlock(in_channels, out_channels), OSBlock(out_channels, out_channels)]
        if reduce:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        x = self.global_avgpool(x).flatten(1)
        return F.normalize(self.fc(x), dim=1)


def load_osnet_x1_0(weights_path: Path, device: torch.device) -> OSNetX10:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    classifier = state.get("classifier.weight")
    if classifier is None:
        raise RuntimeError(f"OSNet权重缺少classifier.weight：{weights_path}")
    model = OSNetX10(int(classifier.shape[0]))
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)
