"""车辆/人员实例查找核心流程。"""

from __future__ import annotations

import csv
import json
import math
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

from reid_models import (
    PERSON_MODEL_FASTREID,
    PERSON_MODEL_FUSION,
    PERSON_MODEL_OSNET,
    FeatureBundle,
    Similarity,
    VehicleFeatureExtractor,
)


VEHICLE_CLASSES = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck
CATEGORY_CLASSES = {"车辆": VEHICLE_CLASSES, "人员": (0,)}
SENSITIVITY_THRESHOLDS = {
    "车辆": {"严格": 0.70, "标准": 0.60, "宽松": 0.52},
}
PERSON_THRESHOLDS = {
    PERSON_MODEL_FASTREID: {"严格": 0.90, "标准": 0.86, "宽松": 0.80},
    PERSON_MODEL_OSNET: {"严格": 0.88, "标准": 0.84, "宽松": 0.78},
    PERSON_MODEL_FUSION: {"严格": 0.88, "标准": 0.84, "宽松": 0.78},
}


def threshold_for(category: str, sensitivity: str, person_model: str = PERSON_MODEL_FUSION) -> float:
    if category == "人员":
        table = PERSON_THRESHOLDS.get(person_model, PERSON_THRESHOLDS[PERSON_MODEL_FUSION])
        return table.get(sensitivity, table["标准"])
    # 车辆阈值独立且冻结，人员模型切换不会进入此分支。
    return SENSITIVITY_THRESHOLDS["车辆"].get(sensitivity, 0.60)


def candidate_threshold_for(category: str, sensitivity: str, person_model: str = PERSON_MODEL_FUSION) -> float:
    threshold = threshold_for(category, sensitivity, person_model)
    if category == "人员":
        # 单张远距离人员图可能只有0.63~0.70；仅列为人工候选，不自动确认。
        return max(0.62, threshold - 0.20)
    return threshold - 0.08


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path: str | Path, image: np.ndarray, quality: int = 94) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def safe_name(value: str) -> str:
    value = re.sub(r'[\\/:*?"<>|\s]+', "_", value.strip())
    return value[:80] or "未命名目标"


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{sec:05.2f}"


@dataclass(slots=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


@dataclass(slots=True)
class ScoredDetection:
    detection: Detection
    similarity: Similarity
    status: str


@dataclass(slots=True)
class TopShot:
    score: float
    timestamp: float
    frame: np.ndarray
    crop: np.ndarray
    box: tuple[int, int, int, int]
    components: dict[str, float]


@dataclass(slots=True)
class AppearanceEvent:
    index: int
    start_s: float
    end_s: float
    hit_count: int = 0
    best_score: float = 0.0
    top_shots: list[TopShot] = field(default_factory=list)

    def add(self, timestamp: float, frame: np.ndarray, crop: np.ndarray, scored: ScoredDetection) -> None:
        self.end_s = timestamp
        self.hit_count += 1
        self.best_score = max(self.best_score, scored.similarity.score)
        sim = scored.similarity
        shot = TopShot(
            score=sim.score,
            timestamp=timestamp,
            frame=frame.copy(),
            crop=crop.copy(),
            box=scored.detection.box,
            components={
                "vehicle": sim.vehicle,
                "visual": sim.visual,
                "color": sim.color,
                "shape": sim.shape,
            },
        )
        self.top_shots.append(shot)
        self.top_shots.sort(key=lambda item: item.score, reverse=True)
        del self.top_shots[3:]


class EventAggregator:
    def __init__(self, gap_seconds: float = 5.0):
        self.gap_seconds = gap_seconds
        self.events: list[AppearanceEvent] = []
        self.current: AppearanceEvent | None = None

    def add(self, timestamp: float, frame: np.ndarray, crop: np.ndarray, scored: ScoredDetection) -> None:
        if self.current is None or timestamp - self.current.end_s > self.gap_seconds:
            if self.current is not None:
                self.events.append(self.current)
            self.current = AppearanceEvent(len(self.events) + 1, timestamp, timestamp)
        self.current.add(timestamp, frame, crop, scored)

    def finish(self) -> list[AppearanceEvent]:
        if self.current is not None:
            self.events.append(self.current)
            self.current = None
        if not self.events:
            return []
        # 轨迹短暂漏检后可能重新确认，并把新段开始时间回溯到轨迹起点；统一合并重叠/短断裂段。
        merged: list[AppearanceEvent] = []
        for event in sorted(self.events, key=lambda item: item.start_s):
            if merged and event.start_s <= merged[-1].end_s + self.gap_seconds:
                previous = merged[-1]
                previous.start_s = min(previous.start_s, event.start_s)
                previous.end_s = max(previous.end_s, event.end_s)
                previous.hit_count += event.hit_count
                previous.best_score = max(previous.best_score, event.best_score)
                previous.top_shots = sorted(
                    [*previous.top_shots, *event.top_shots], key=lambda item: item.score, reverse=True
                )[:5]
            else:
                event.index = len(merged) + 1
                merged.append(event)
        self.events = merged
        return self.events

    @property
    def visible_count(self) -> int:
        return len(self.events) + (1 if self.current is not None else 0)


@dataclass(slots=True)
class _PersonTrack:
    track_id: int
    box: tuple[int, int, int, int]
    start_s: float
    last_s: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    anchor_history: deque[bool] = field(default_factory=lambda: deque(maxlen=5))
    weak_history: deque[bool] = field(default_factory=lambda: deque(maxlen=12))
    confirmed: bool = False


class PersonIdentityTracker:
    """人员专用的短时轨迹维持；ReID确认身份后，不再要求每帧重复过阈值。"""

    def __init__(self, max_missing_s: float = 2.5):
        self.max_missing_s = max_missing_s
        self.tracks: dict[int, _PersonTrack] = {}
        self.next_id = 1

    @staticmethod
    def _center(box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    def _associate(
        self,
        timestamp: float,
        detections: list[Detection],
        similarities: list[Similarity] | None = None,
    ) -> dict[int, _PersonTrack]:
        self.tracks = {
            key: track for key, track in self.tracks.items() if timestamp - track.last_s <= self.max_missing_s
        }
        pairs: list[tuple[float, int, int]] = []
        for det_index, detection in enumerate(detections):
            center = self._center(detection.box)
            for track_id, track in self.tracks.items():
                dt = max(1e-3, timestamp - track.last_s)
                predicted = self._center(track.box) + track.velocity * dt
                height = max(1, track.box[3] - track.box[1], detection.box[3] - detection.box[1])
                distance = float(np.linalg.norm(center - predicted))
                allowed = max(70.0, 2.2 * height + 45.0 * dt)
                if distance <= allowed:
                    # 已确认身份的轨迹在人员靠近/交叉时加入当前外观证据，降低纯中心点
                    # 最近邻造成的换人概率；无外观证据时仍退回运动连续性。
                    appearance_bonus = 0.0
                    if track.confirmed and similarities is not None and det_index < len(similarities):
                        item = similarities[det_index]
                        appearance_bonus = 0.28 * max(0.0, item.score) + 0.16 * max(0.0, item.face)
                    pairs.append((distance / allowed - appearance_bonus, det_index, track_id))
        assigned_detections: set[int] = set()
        assigned_tracks: set[int] = set()
        result: dict[int, _PersonTrack] = {}
        for _, det_index, track_id in sorted(pairs):
            if det_index in assigned_detections or track_id in assigned_tracks:
                continue
            track = self.tracks[track_id]
            detection = detections[det_index]
            dt = max(1e-3, timestamp - track.last_s)
            measured_velocity = (self._center(detection.box) - self._center(track.box)) / dt
            track.velocity = 0.55 * track.velocity + 0.45 * measured_velocity
            track.box = detection.box
            track.last_s = timestamp
            result[det_index] = track
            assigned_detections.add(det_index)
            assigned_tracks.add(track_id)
        for det_index, detection in enumerate(detections):
            if det_index in assigned_detections:
                continue
            track = _PersonTrack(self.next_id, detection.box, timestamp, timestamp)
            self.next_id += 1
            self.tracks[track.track_id] = track
            result[det_index] = track
        return result

    def update(
        self,
        timestamp: float,
        detections: list[Detection],
        similarities: list[Similarity],
        threshold: float,
        candidate_threshold: float,
        anchor_allowed: list[bool] | None = None,
    ) -> list[tuple[str, int, float]]:
        assignments = self._associate(timestamp, detections, similarities)
        scores = [item.score for item in similarities]
        order = np.argsort(scores)[::-1].tolist() if scores else []
        best_index = order[0] if order else -1
        second_score = scores[order[1]] if len(order) >= 2 else -1.0
        statuses: list[tuple[str, int, float]] = []
        for index, similarity in enumerate(similarities):
            track = assignments[index]
            # 多目标实时监控会先做跨标签互斥分配。未获得当前检测框归属权的
            # 标签不能建立身份锚点，避免同一个人同时确认成“男生”和“女生”。
            allowed = anchor_allowed[index] if anchor_allowed is not None else True
            # 人脸证据也必须通过跨标签归属门控；阈值需按部署样本独立校准。
            face_anchor = allowed and similarity.face >= 0.65
            absolute_anchor = allowed and (similarity.score >= threshold or face_anchor)
            relative_anchor = (
                allowed
                and
                len(order) >= 2
                and index == best_index
                and similarity.score >= candidate_threshold
                and similarity.score - second_score >= 0.12
            )
            weak_relative_evidence = (
                allowed
                and
                len(order) >= 2
                and index == best_index
                and similarity.score >= 0.52
                and similarity.score - second_score >= 0.14
            )
            track.anchor_history.append(absolute_anchor or relative_anchor)
            track.weak_history.append(weak_relative_evidence)
            # 强人脸仍需通过外层2/3连续抽帧才报警，因此可在本轨迹立即锁定；
            # 只有身体证据时继续保持原来的多帧确认要求。
            if not track.confirmed and (
                face_anchor or sum(track.anchor_history) >= 3 or sum(track.weak_history) >= 8
            ):
                track.confirmed = True
            if track.confirmed and (similarity.score >= 0.25 or similarity.face >= 0.30):
                status = "match"
            elif absolute_anchor or relative_anchor or similarity.score >= candidate_threshold:
                status = "candidate"
            else:
                status = "reject"
            statuses.append((status, track.track_id, track.start_s))
        return statuses


class InstanceDetector:
    def __init__(self, weights: Path, device: str = "0"):
        self.model = YOLO(str(weights))
        self.device = device
        self.names = self.model.names

    @staticmethod
    def _tiles(frame: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
        h, w = frame.shape[:2]
        tile_w = min(w, math.ceil(w * 0.58))
        tile_h = min(h, math.ceil(h * 0.58))
        xs = sorted(set((0, max(0, w - tile_w))))
        ys = sorted(set((0, max(0, h - tile_h))))
        return [(frame[y : y + tile_h, x : x + tile_w], x, y) for y in ys for x in xs]

    def detect(
        self,
        frame: np.ndarray,
        high_accuracy: bool = True,
        category: str | list[str] | tuple[str, ...] = "车辆",
    ) -> list[Detection]:
        h, w = frame.shape[:2]
        tiled = high_accuracy and max(h, w) >= 3000
        entries = self._tiles(frame) if tiled else [(frame, 0, 0)]
        sources = [entry[0] for entry in entries]
        categories = (category,) if isinstance(category, str) else tuple(category)
        class_ids = sorted({item for name in categories for item in CATEGORY_CLASSES.get(name, VEHICLE_CLASSES)})
        results = self.model.predict(
            source=sources,
            device=self.device,
            imgsz=1280,
            conf=0.08,
            iou=0.55,
            classes=class_ids,
            verbose=False,
        )
        boxes: list[list[float]] = []
        confidences: list[float] = []
        classes: list[int] = []
        for result, (_, offset_x, offset_y) in zip(results, entries):
            for item in result.boxes:
                x1, y1, x2, y2 = item.xyxy[0].tolist()
                boxes.append([x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y])
                confidences.append(float(item.conf.item()))
                classes.append(int(item.cls.item()))
        if not boxes:
            return []
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
        score_tensor = torch.tensor(confidences, dtype=torch.float32)
        keep = nms(box_tensor, score_tensor, 0.55).tolist()
        detections: list[Detection] = []
        for index in keep:
            x1, y1, x2, y2 = boxes[index]
            x1, y1 = max(0, round(x1)), max(0, round(y1))
            x2, y2 = min(w, round(x2)), min(h, round(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = classes[index]
            detections.append(
                Detection((x1, y1, x2, y2), confidences[index], class_id, str(self.names[class_id]))
            )
        return detections


def padded_crop(frame: np.ndarray, box: tuple[int, int, int, int], ratio: float = 0.06) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = round((x2 - x1) * ratio), round((y2 - y1) * ratio)
    return frame[max(0, y1 - pad_y) : min(h, y2 + pad_y), max(0, x1 - pad_x) : min(w, x2 + pad_x)]


@dataclass(slots=True)
class SearchConfig:
    target_name: str
    video_path: Path
    output_root: Path
    category: str = "车辆"
    person_model: str = PERSON_MODEL_FUSION
    sensitivity: str = "标准"
    sample_fps: float = 4.0
    high_accuracy: bool = True
    event_gap_seconds: float = 5.0
    min_width: int = 40


ProgressCallback = Callable[[dict], None]
LogCallback = Callable[[str], None]


class InstanceSearchPipeline:
    def __init__(self, project_root: Path, log: LogCallback | None = None):
        self.project_root = project_root
        self.log = log or (lambda message: None)
        model_dir = project_root / "vehicle_instance_search" / "models"
        required = [
            project_root / "yolov8l.pt",
            model_dir / "veri_sbs_R50-ibn.pth",
            model_dir / "dinov2_vits14_pretrain.pth",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "请先按 docs/MODELS.md 准备模型文件：" + ", ".join(missing)
            )
        self.log("正在加载 YOLOv8l 车辆/人员检测模型……")
        self.detector = InstanceDetector(project_root / "yolov8l.pt", "0" if torch.cuda.is_available() else "cpu")
        self.log("正在加载冻结的车辆FastReID、人员FastReID/OSNet与DINOv2特征模型……")
        self.extractor = VehicleFeatureExtractor(
            model_dir / "veri_sbs_R50-ibn.pth",
            model_dir / "dinov2_vits14_pretrain.pth",
            model_dir / "market_sbs_R50-ibn.pth",
            model_dir / "osnet_x1_0_msmt17.pth",
            model_dir / "face_detection_yunet_2026may.onnx",
            model_dir / "face_recognition_sface_2021dec.onnx",
        )
        self.log("模型加载完成。")

    def _draw_preview(self, frame: np.ndarray, scored: list[ScoredDetection], timestamp: float) -> np.ndarray:
        view = frame.copy()
        for item in scored:
            x1, y1, x2, y2 = item.detection.box
            if item.status == "match":
                color, thickness = (40, 220, 40), 5
            elif item.status == "candidate":
                color, thickness = (0, 190, 255), 3
            elif item.status == "low_resolution":
                color, thickness = (180, 180, 180), 2
            else:
                color, thickness = (80, 110, 255), 2
            cv2.rectangle(view, (x1, y1), (x2, y2), color, thickness)
            label = f"{item.status} {item.similarity.score:.3f} det={item.detection.confidence:.2f}"
            text_y = max(28, y1 - 7)
            cv2.rectangle(view, (x1, text_y - 26), (min(view.shape[1], x1 + 390), text_y + 3), (10, 10, 10), -1)
            cv2.putText(view, label, (x1 + 5, text_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        cv2.putText(view, format_time(timestamp), (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        return view

    def _save_results(
        self,
        config: SearchConfig,
        references: list[np.ndarray],
        events: list[AppearanceEvent],
        metadata: dict,
        audit_rows: list[dict] | None = None,
        candidate_events: list[AppearanceEvent] | None = None,
    ) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = config.output_root / safe_name(config.target_name) / f"{stamp}_{safe_name(config.video_path.stem)}"
        reference_dir = session / "references"
        event_root = session / "events"
        candidate_root = session / "candidate_review"
        reference_dir.mkdir(parents=True, exist_ok=True)
        event_root.mkdir(parents=True, exist_ok=True)
        candidate_root.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(references, start=1):
            write_image(reference_dir / f"reference_{index:02d}.jpg", image)

        rows: list[dict] = []
        for event in events:
            event_dir = event_root / f"event_{event.index:03d}_{event.start_s:.2f}s"
            event_dir.mkdir(parents=True, exist_ok=True)
            for rank, shot in enumerate(event.top_shots, start=1):
                annotated = shot.frame.copy()
                x1, y1, x2, y2 = shot.box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 40), 6)
                cv2.putText(
                    annotated,
                    f"MATCH {shot.score:.3f}  {format_time(shot.timestamp)}",
                    (max(10, x1), max(40, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (40, 220, 40),
                    3,
                    cv2.LINE_AA,
                )
                base = f"top_{rank:02d}_{shot.timestamp:.2f}s_score_{shot.score:.3f}"
                write_image(event_dir / f"{base}_frame.jpg", annotated)
                write_image(event_dir / f"{base}_crop.jpg", shot.crop)
                (event_dir / f"{base}_details.json").write_text(
                    json.dumps(shot.components, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            rows.append(
                {
                    "event_id": event.index,
                    "start_seconds": round(event.start_s, 3),
                    "end_seconds": round(event.end_s, 3),
                    "start_time": format_time(event.start_s),
                    "end_time": format_time(event.end_s),
                    "hit_count": event.hit_count,
                    "best_score": round(event.best_score, 5),
                    "folder": str(event_dir),
                }
            )

        with (session / "events.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
                "event_id", "start_seconds", "end_seconds", "start_time", "end_time", "hit_count", "best_score", "folder"
            ])
            writer.writeheader()
            writer.writerows(rows)
        candidate_rows: list[dict] = []
        for event in candidate_events or []:
            event_dir = candidate_root / f"candidate_{event.index:03d}_{event.start_s:.2f}s"
            event_dir.mkdir(parents=True, exist_ok=True)
            for rank, shot in enumerate(event.top_shots, 1):
                annotated = shot.frame.copy()
                x1, y1, x2, y2 = shot.box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 190, 255), 5)
                cv2.putText(annotated, f"REVIEW {shot.score:.3f}  {format_time(shot.timestamp)}", (max(10, x1), max(40, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 190, 255), 3, cv2.LINE_AA)
                base = f"top_{rank:02d}_{shot.timestamp:.2f}s_score_{shot.score:.3f}"
                write_image(event_dir / f"{base}_frame.jpg", annotated)
                write_image(event_dir / f"{base}_crop.jpg", shot.crop)
            candidate_rows.append(
                {
                    "candidate_event_id": event.index,
                    "start_seconds": round(event.start_s, 3),
                    "end_seconds": round(event.end_s, 3),
                    "start_time": format_time(event.start_s),
                    "end_time": format_time(event.end_s),
                    "hit_count": event.hit_count,
                    "best_score": round(event.best_score, 5),
                    "folder": str(event_dir),
                }
            )
        candidate_fields = ["candidate_event_id", "start_seconds", "end_seconds", "start_time", "end_time", "hit_count", "best_score", "folder"]
        with (session / "candidate_events.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=candidate_fields)
            writer.writeheader()
            writer.writerows(candidate_rows)
        audit_rows = audit_rows or []
        audit_fields = [
            "timestamp_seconds", "time", "detection_index", "class_name", "detection_confidence",
            "x1", "y1", "x2", "y2", "width", "height", "score", "reid", "visual", "color",
            "shape", "face", "reference_index", "status", "temporal_confirmed",
            "selected_for_event", "track_id", "track_start_seconds",
        ]
        with (session / "candidate_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=audit_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(audit_rows)
        summary = {
            "target_name": config.target_name,
            "video": str(config.video_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sensitivity": config.sensitivity,
            "category": config.category,
            "person_model": config.person_model if config.category == "人员" else "车辆固定模型",
            "threshold": threshold_for(config.category, config.sensitivity, config.person_model),
            "sample_fps": config.sample_fps,
            "event_gap_seconds": config.event_gap_seconds,
            "event_count": len(events),
            "candidate_review_event_count": len(candidate_rows),
            "unique_target_count": 1 if events else 0,
            "video_metadata": metadata,
            "events": rows,
        }
        (session / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return session

    def search_video(
        self,
        config: SearchConfig,
        references: list[np.ndarray],
        stop_event: threading.Event,
        progress: ProgressCallback | None = None,
    ) -> dict:
        progress = progress or (lambda payload: None)
        reference_features = self.extractor.extract(references, config.category)
        threshold = threshold_for(config.category, config.sensitivity, config.person_model)
        candidate_threshold = candidate_threshold_for(config.category, config.sensitivity, config.person_model)
        cap = cv2.VideoCapture(str(config.video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频：{config.video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if frame_count > 0 else 0.0
        frame_step = max(1, round(fps / max(0.1, config.sample_fps)))
        metadata = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "frame_step": frame_step,
        }
        effective_event_gap = max(config.event_gap_seconds, 8.0) if config.category == "人员" else config.event_gap_seconds
        aggregator = EventAggregator(effective_event_gap)
        candidate_aggregator = EventAggregator(effective_event_gap)
        # 离线验收同样使用2/3抽帧确认，单帧偶然高相似不直接形成事件。
        temporal_window: deque[tuple[float, np.ndarray, np.ndarray, ScoredDetection] | None] = deque(maxlen=3)
        candidate_window: deque[tuple[float, np.ndarray, np.ndarray, ScoredDetection] | None] = deque(maxlen=3)
        emitted_timestamps: set[float] = set()
        emitted_candidate_timestamps: set[float] = set()
        confirmed_timestamps: set[float] = set()
        track_start_by_timestamp: dict[float, float] = {}
        person_tracker = PersonIdentityTracker() if config.category == "人员" else None
        audit_rows: list[dict] = []
        sampled = 0
        model_calls = 0
        started = time.perf_counter()
        frame_index = -1
        effective_min_width = min(config.min_width, 20) if config.category == "人员" else config.min_width
        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % frame_step != 0:
                    continue
                sampled += 1
                timestamp = frame_index / fps
                detections = self.detector.detect(frame, config.high_accuracy, config.category)
                model_calls += 1
                crops = [padded_crop(frame, detection.box) for detection in detections]
                features = self.extractor.extract(crops, config.category) if crops else []
                similarities = [
                    self.extractor.compare(feature, reference_features, config.category, config.person_model)
                    for feature in features
                ]
                person_statuses = (
                    person_tracker.update(timestamp, detections, similarities, threshold, candidate_threshold)
                    if person_tracker is not None
                    else []
                )
                scored_items: list[ScoredDetection] = []
                best_match: tuple[ScoredDetection, np.ndarray] | None = None
                best_candidate: tuple[ScoredDetection, np.ndarray] | None = None
                sample_audit_start = len(audit_rows)
                for detection_index, (detection, crop, similarity) in enumerate(zip(detections, crops, similarities)):
                    box_width = detection.box[2] - detection.box[0]
                    box_height = detection.box[3] - detection.box[1]
                    min_height = 36 if config.category == "人员" else max(20, effective_min_width // 2)
                    if box_width < effective_min_width or box_height < min_height:
                        status = "low_resolution"
                        track_id, track_start = -1, timestamp
                    elif person_tracker is not None:
                        status, track_id, track_start = person_statuses[detection_index]
                    elif similarity.score >= threshold:
                        status = "match"
                        track_id, track_start = -1, timestamp
                    elif similarity.score >= candidate_threshold:
                        status = "candidate"
                        track_id, track_start = -1, timestamp
                    else:
                        status = "reject"
                        track_id, track_start = -1, timestamp
                    scored = ScoredDetection(detection, similarity, status)
                    scored_items.append(scored)
                    x1, y1, x2, y2 = detection.box
                    audit_rows.append(
                        {
                            "timestamp_seconds": round(timestamp, 6),
                            "time": format_time(timestamp),
                            "detection_index": len(scored_items),
                            "class_name": detection.class_name,
                            "detection_confidence": round(detection.confidence, 6),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "score": round(similarity.score, 6),
                            "reid": round(similarity.vehicle, 6),
                            "visual": round(similarity.visual, 6),
                            "color": round(similarity.color, 6),
                            "shape": round(similarity.shape, 6),
                            "face": round(similarity.face, 6),
                            "reference_index": similarity.reference_index,
                            "status": status,
                            "temporal_confirmed": False,
                            "selected_for_event": False,
                            "track_id": track_id,
                            "track_start_seconds": round(track_start, 6),
                        }
                    )
                    if status == "match" and (best_match is None or similarity.score > best_match[0].similarity.score):
                        best_match = (scored, crop)
                        track_start_by_timestamp[round(timestamp, 6)] = track_start
                    elif status == "candidate" and (best_candidate is None or similarity.score > best_candidate[0].similarity.score):
                        best_candidate = (scored, crop)
                if best_match is not None:
                    best_box = best_match[0].detection.box
                    best_score = best_match[0].similarity.score
                    for row in audit_rows[sample_audit_start:]:
                        row_box = (row["x1"], row["y1"], row["x2"], row["y2"])
                        if row_box == best_box and abs(float(row["score"]) - best_score) < 1e-5:
                            row["selected_for_event"] = True
                            break
                temporal_window.append(
                    (timestamp, frame.copy(), best_match[1].copy(), best_match[0]) if best_match is not None else None
                )
                candidate_window.append(
                    (timestamp, frame.copy(), best_candidate[1].copy(), best_candidate[0])
                    if best_match is None and best_candidate is not None else None
                )
                if sum(item is not None for item in temporal_window) >= 2:
                    for item in temporal_window:
                        if item is None or item[0] in emitted_timestamps:
                            continue
                        item_timestamp, item_frame, item_crop, item_scored = item
                        aggregator.add(item_timestamp, item_frame, item_crop, item_scored)
                        if config.category == "人员" and aggregator.current is not None:
                            track_start = track_start_by_timestamp.get(round(item_timestamp, 6), item_timestamp)
                            aggregator.current.start_s = min(aggregator.current.start_s, track_start)
                        emitted_timestamps.add(item_timestamp)
                        confirmed_timestamps.add(round(item_timestamp, 6))
                if sum(item is not None for item in candidate_window) >= 2:
                    for item in candidate_window:
                        if item is None or item[0] in emitted_candidate_timestamps:
                            continue
                        item_timestamp, item_frame, item_crop, item_scored = item
                        candidate_aggregator.add(item_timestamp, item_frame, item_crop, item_scored)
                        emitted_candidate_timestamps.add(item_timestamp)
                preview = self._draw_preview(frame, scored_items, timestamp)
                elapsed = time.perf_counter() - started
                progress(
                    {
                        "type": "progress",
                        "frame": preview,
                        "percent": (frame_index + 1) / max(1, frame_count) * 100.0 if frame_count else 0.0,
                        "timestamp": timestamp,
                        "duration": duration,
                        "sampled": sampled,
                        "detections": len(detections),
                        "matches": sum(item.status == "match" for item in scored_items),
                        "candidates": sum(item.status == "candidate" for item in scored_items),
                        "events": aggregator.visible_count,
                        "elapsed": elapsed,
                    }
                )
        finally:
            cap.release()
        events = aggregator.finish()
        candidate_events = candidate_aggregator.finish()
        for row in audit_rows:
            row["temporal_confirmed"] = (
                bool(row["selected_for_event"])
                and row["status"] == "match"
                and float(row["timestamp_seconds"]) in confirmed_timestamps
            )
        output_dir = self._save_results(config, references, events, metadata, audit_rows, candidate_events)
        result = {
            "cancelled": stop_event.is_set(),
            "events": events,
            "candidate_events": candidate_events,
            "output_dir": output_dir,
            "sampled": sampled,
            "model_calls": model_calls,
            "elapsed": time.perf_counter() - started,
            "metadata": metadata,
        }
        progress({"type": "done", **result})
        return result


# 兼容现有车辆验收脚本使用的类名。
VehicleDetector = InstanceDetector
VehicleSearchPipeline = InstanceSearchPipeline
