"""四路实时视频流采集、实例匹配、报警去重及前后录像。"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from core import (
    Detection,
    InstanceSearchPipeline,
    PersonIdentityTracker,
    candidate_threshold_for,
    padded_crop,
    safe_name,
    threshold_for,
    write_image,
)
from profiles import TargetLibrary, TargetProfile


MessageCallback = Callable[[dict], None]


@dataclass(slots=True)
class StreamConfig:
    index: int
    name: str
    source: str
    enabled: bool = True


@dataclass(slots=True)
class MatchState:
    history: deque[bool] = field(default_factory=lambda: deque(maxlen=3))
    active: bool = False
    last_match_at: float = 0.0
    last_alert_at: float = 0.0


@dataclass(slots=True)
class GlobalIdentityState:
    global_id: str
    last_stream: int | None = None
    last_alert_stream: int | None = None
    last_seen_at: float = 0.0
    last_score: float = 0.0
    appearances: int = 0


@dataclass(slots=True)
class PendingRecording:
    path: Path
    end_at: float
    frames: list[bytes]


def _source_value(source: str):
    stripped = source.strip()
    return int(stripped) if stripped.isdigit() else stripped


class StreamCapture:
    def __init__(
        self,
        config: StreamConfig,
        callback: MessageCallback,
        pre_seconds: float = 10.0,
        post_seconds: float = 20.0,
        record_fps: float = 5.0,
    ):
        self.config = config
        self.callback = callback
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.record_fps = record_fps
        self.stop_event = threading.Event()
        self.shutdown_requested = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_seq = 0
        self.latest_wall_time = 0.0
        self.prebuffer: deque[tuple[float, bytes]] = deque(maxlen=max(1, round(pre_seconds * record_fps)))
        self.pending: list[PendingRecording] = []
        self.connected = False

    def start(self) -> None:
        # 非守护线程：关闭GUI时仍会完成已触发报警的后20秒采集和视频封装，避免损坏mp4。
        self.thread = threading.Thread(target=self._run, daemon=False, name=f"stream-{self.config.index}")
        self.thread.start()

    def stop(self) -> None:
        self.shutdown_requested.set()
        with self.lock:
            if not self.pending:
                self.stop_event.set()

    def get_latest(self) -> tuple[int, np.ndarray | None, float]:
        with self.lock:
            return self.latest_seq, None if self.latest_frame is None else self.latest_frame.copy(), self.latest_wall_time

    def start_recording(self, path: Path) -> None:
        with self.lock:
            frames = [data for _, data in self.prebuffer]
            self.pending.append(PendingRecording(path, time.monotonic() + self.post_seconds, frames))

    def _publish_status(self, status: str, detail: str = "") -> None:
        self.callback({"type": "live_status", "stream": self.config.index, "status": status, "detail": detail})

    def _open(self) -> cv2.VideoCapture:
        source = _source_value(self.config.source)
        cap = cv2.VideoCapture(source)
        if isinstance(source, str) and source.lower().startswith("rtsp://"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def _write_recording(self, pending: PendingRecording, size: tuple[int, int]) -> None:
        def work() -> None:
            pending.path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(pending.path), cv2.VideoWriter_fourcc(*"mp4v"), self.record_fps, size
            )
            written = 0
            try:
                for data in pending.frames:
                    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    if (frame.shape[1], frame.shape[0]) != size:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                    written += 1
            finally:
                writer.release()
            self.callback(
                {
                    "type": "recording_done",
                    "stream": self.config.index,
                    "path": str(pending.path),
                    "frames": written,
                }
            )

        threading.Thread(target=work, daemon=False, name=f"record-{self.config.index}").start()

    def _run(self) -> None:
        source_path = Path(self.config.source)
        is_file = source_path.is_file()
        while not self.stop_event.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                self._publish_status("重连中", "无法打开视频源")
                cap.release()
                self.stop_event.wait(2.0)
                continue
            self.connected = True
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self._publish_status("在线", f"输入 {fps:.1f} FPS")
            last_record = 0.0
            file_next = time.monotonic()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        file_next = time.monotonic()
                        continue
                    break
                if is_file:
                    file_next += 1.0 / max(1.0, fps)
                    delay = file_next - time.monotonic()
                    if delay > 0:
                        self.stop_event.wait(min(delay, 0.1))
                now = time.monotonic()
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_seq += 1
                    self.latest_wall_time = time.time()
                if now - last_record < 1.0 / self.record_fps:
                    continue
                last_record = now
                ok_encode, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
                if not ok_encode:
                    continue
                data = encoded.tobytes()
                finished: list[PendingRecording] = []
                with self.lock:
                    self.prebuffer.append((now, data))
                    for item in self.pending:
                        item.frames.append(data)
                        if now >= item.end_at:
                            finished.append(item)
                    for item in finished:
                        self.pending.remove(item)
                for item in finished:
                    self._write_recording(item, (frame.shape[1], frame.shape[0]))
                if self.shutdown_requested.is_set():
                    with self.lock:
                        has_pending = bool(self.pending)
                    if not has_pending:
                        self.stop_event.set()
            cap.release()
            self.connected = False
            if not self.stop_event.is_set():
                self._publish_status("重连中", "视频源中断")
                self.stop_event.wait(1.0)
        self._publish_status("已停止")


class LiveMonitorManager:
    """单GPU轮询四路最新帧；准确率阈值固定，仅自适应抽帧频率。"""

    def __init__(
        self,
        pipeline: InstanceSearchPipeline,
        library: TargetLibrary,
        profiles: list[TargetProfile],
        streams: list[StreamConfig],
        output_root: Path,
        callback: MessageCallback,
    ):
        self.pipeline = pipeline
        self.library = library
        self.profiles = [item for item in profiles if item.enabled]
        self.streams = [item for item in streams if item.enabled and item.source.strip()]
        self.output_root = output_root
        self.callback = callback
        self.stop_event = threading.Event()
        self.captures: dict[int, StreamCapture] = {}
        self.thread: threading.Thread | None = None
        self.reference_features: dict[str, list] = {}
        self.states: dict[tuple[int, str], MatchState] = {}
        self.last_seq: dict[int, int] = {}
        self.last_infer: dict[int, float] = {}
        self.person_trackers: dict[tuple[int, str], PersonIdentityTracker] = {}
        self.global_identities: dict[str, GlobalIdentityState] = {
            profile.profile_id: GlobalIdentityState(f"PID-{profile.profile_id[-8:].upper()}")
            for profile in self.profiles
        }

    def start(self) -> None:
        if not self.profiles:
            raise ValueError("目标库中没有启用的监控目标")
        if not self.streams:
            raise ValueError("请至少启用并填写一路视频源")
        for profile in self.profiles:
            references = self.library.load_references(profile)
            self.reference_features[profile.profile_id] = self.pipeline.extractor.extract(references, profile.category)
        for config in self.streams:
            capture = StreamCapture(config, self.callback)
            self.captures[config.index] = capture
            capture.start()
        self.thread = threading.Thread(target=self._run, daemon=True, name="live-inference")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        for capture in self.captures.values():
            capture.stop()

    @staticmethod
    def _category_for_detection(detection: Detection) -> str:
        return "人员" if detection.class_id == 0 else "车辆"

    def _alert(
        self,
        stream: StreamConfig,
        profile: TargetProfile,
        frame: np.ndarray,
        crop: np.ndarray,
        detection: Detection,
        score: float,
    ) -> None:
        now = datetime.now()
        identity = self.global_identities[profile.profile_id]
        previous_stream = identity.last_alert_stream
        stream_names = {item.index: item.name for item in self.streams}
        if previous_stream is None:
            transition = "首次出现"
        elif previous_stream == stream.index:
            transition = f"{stream.name}重新进入"
        else:
            transition = f"{stream_names.get(previous_stream, f'视频流{previous_stream + 1}')} → {stream.name}"
        identity.last_alert_stream = stream.index
        identity.last_stream = stream.index
        identity.last_seen_at = time.monotonic()
        identity.last_score = score
        identity.appearances += 1
        event_dir = (
            self.output_root
            / safe_name(profile.name)
            / "live"
            / now.strftime("%Y%m%d")
            / safe_name(stream.name)
            / now.strftime("%H%M%S_%f")
        )
        event_dir.mkdir(parents=True, exist_ok=True)
        annotated = frame.copy()
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (30, 230, 70), 5)
        cv2.putText(
            annotated,
            f"MATCH {score:.3f}",
            (max(5, x1), max(35, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (30, 230, 70),
            3,
            cv2.LINE_AA,
        )
        full_path = event_dir / "alert_frame.jpg"
        crop_path = event_dir / "target_crop.jpg"
        write_image(full_path, annotated)
        write_image(crop_path, crop)
        metadata = {
            "time": now.isoformat(timespec="milliseconds"),
            "stream_index": stream.index,
            "stream_name": stream.name,
            "target_id": profile.profile_id,
            "target_name": profile.name,
            "global_id": identity.global_id,
            "transition": transition,
            "previous_stream_index": previous_stream,
            "category": profile.category,
            "sensitivity": profile.sensitivity,
            "person_model": profile.person_model if profile.category == "人员" else "车辆固定模型",
            "score": round(score, 6),
            "box": list(detection.box),
            "frame": str(full_path),
            "crop": str(crop_path),
            "recording": str(event_dir / "alert_pre10_post20.mp4"),
        }
        (event_dir / "event.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            self.library.add_positive(profile.profile_id, crop)
        except Exception as exc:
            self.callback({"type": "log", "message": f"正样本写入失败：{exc}"})
        self.captures[stream.index].start_recording(event_dir / "alert_pre10_post20.mp4")
        self.callback(
            {
                "type": "alert",
                "stream": stream.index,
                "stream_name": stream.name,
                "profile": profile,
                "score": score,
                "frame": annotated,
                "crop": crop,
                "output_dir": str(event_dir),
                "global_id": identity.global_id,
                "transition": transition,
            }
        )

    def _publish_global_identity(
        self,
        stream: StreamConfig,
        profile: TargetProfile,
        score: float,
    ) -> None:
        identity = self.global_identities[profile.profile_id]
        identity.last_stream = stream.index
        identity.last_seen_at = time.monotonic()
        identity.last_score = score
        self.callback(
            {
                "type": "global_identity",
                "global_id": identity.global_id,
                "profile_id": profile.profile_id,
                "target": profile.name,
                "stream": stream.index,
                "stream_name": stream.name,
                "score": score,
                "appearances": identity.appearances,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
        )

    def _process_stream(self, stream: StreamConfig, frame: np.ndarray) -> None:
        now = time.monotonic()
        categories = sorted({profile.category for profile in self.profiles})
        detections = self.pipeline.detector.detect(frame, high_accuracy=True, category=categories)
        grouped: dict[str, list[tuple[Detection, np.ndarray]]] = {category: [] for category in categories}
        for detection in detections:
            category = self._category_for_detection(detection)
            if category in grouped:
                grouped[category].append((detection, padded_crop(frame, detection.box)))

        best: dict[str, tuple[float, Detection, np.ndarray]] = {}
        for category, items in grouped.items():
            crops = [crop for _, crop in items]
            features = self.pipeline.extractor.extract(crops, category) if crops else []
            category_profiles = [item for item in self.profiles if item.category == category]
            similarity_by_profile = {}
            for profile in category_profiles:
                references = self.reference_features[profile.profile_id]
                similarity_by_profile[profile.profile_id] = [
                    self.pipeline.extractor.compare(feature, references, category, profile.person_model)
                    for feature in features
                ]

            # 每个检测框只能归属于一个人员标签。清晰人脸优先；看不到脸时，
            # 用人体ReID跨标签分差决定归属。分差不足则不建立新身份锚点。
            anchor_allowed: dict[str, list[bool]] = {
                profile.profile_id: [False] * len(items) for profile in category_profiles
            }
            if category == "人员":
                for detection_index in range(len(items)):
                    detection = items[detection_index][0]
                    x1, y1, x2, y2 = detection.box
                    box_width = x2 - x1
                    box_height = y2 - y1
                    frame_height, frame_width = frame.shape[:2]
                    complete_person = (
                        box_height >= 100
                        and box_height >= 1.05 * box_width
                        and x1 > 3
                        and y1 > 3
                        and x2 < frame_width - 3
                        and y2 < frame_height - 3
                    )
                    if not complete_person:
                        continue
                    ranked_body = sorted(
                        (
                            (similarity_by_profile[profile.profile_id][detection_index].score, profile)
                            for profile in category_profiles
                        ),
                        key=lambda entry: entry[0],
                        reverse=True,
                    )
                    ranked_face = sorted(
                        (
                            (similarity_by_profile[profile.profile_id][detection_index].face, profile)
                            for profile in category_profiles
                        ),
                        key=lambda entry: entry[0],
                        reverse=True,
                    )
                    chosen = None
                    face_second = ranked_face[1][0] if len(ranked_face) > 1 else -1.0
                    if ranked_face and ranked_face[0][0] >= 0.48 and ranked_face[0][0] - face_second >= 0.08:
                        chosen = ranked_face[0][1]
                    else:
                        body_second = ranked_body[1][0] if len(ranked_body) > 1 else -1.0
                        if ranked_body and (len(ranked_body) == 1 or ranked_body[0][0] - body_second >= 0.055):
                            chosen = ranked_body[0][1]
                    if chosen is not None:
                        anchor_allowed[chosen.profile_id][detection_index] = True

            for profile in category_profiles:
                threshold = threshold_for(profile.category, profile.sensitivity, profile.person_model)
                similarities = similarity_by_profile[profile.profile_id]
                person_statuses = None
                if category == "人员" and similarities:
                    tracker = self.person_trackers.setdefault(
                        (stream.index, profile.profile_id), PersonIdentityTracker(max_missing_s=5.0)
                    )
                    person_statuses = tracker.update(
                        now,
                        [detection for detection, _ in items],
                        similarities,
                        threshold,
                        candidate_threshold_for(category, profile.sensitivity, profile.person_model),
                        anchor_allowed[profile.profile_id],
                    )
                for item_index, ((detection, crop), similarity) in enumerate(zip(items, similarities)):
                    min_width = 18 if category == "人员" else 40
                    min_height = 34 if category == "人员" else 20
                    if detection.box[2] - detection.box[0] < min_width or detection.box[3] - detection.box[1] < min_height:
                        continue
                    is_match = (
                        person_statuses[item_index][0] == "match"
                        if person_statuses is not None
                        else similarity.score >= threshold
                    )
                    if is_match:
                        current = best.get(profile.profile_id)
                        if current is None or similarity.score > current[0]:
                            best[profile.profile_id] = (similarity.score, detection, crop)

        preview = frame.copy()
        preview_items: list[dict] = []
        for profile in self.profiles:
            state = self.states.setdefault((stream.index, profile.profile_id), MatchState())
            match = best.get(profile.profile_id)
            state.history.append(match is not None)
            if match is not None:
                state.last_match_at = now
                score, detection, crop = match
                x1, y1, x2, y2 = detection.box
                cv2.rectangle(preview, (x1, y1), (x2, y2), (30, 230, 70), 4)
                cv2.putText(preview, f"{profile.name} {score:.3f}", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 230, 70), 2, cv2.LINE_AA)
                preview_items.append({"target": profile.name, "score": score, "box": detection.box})
                self._publish_global_identity(stream, profile, score)
            # 2/3 抽帧确认；同路持续出现保持 active，不重复报警。
            if not state.active and sum(state.history) >= 2 and match is not None:
                state.active = True
                state.last_alert_at = now
                self._alert(stream, profile, frame, match[2], match[1], match[0])
            # 连续5秒未命中视为离开；再次进入立即重新报警，不受180秒影响。
            if state.active and now - state.last_match_at >= 5.0:
                state.active = False
                state.history.clear()

        self.callback(
            {
                "type": "live_preview",
                "stream": stream.index,
                "frame": preview,
                "detections": len(detections),
                "matches": preview_items,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
        )

    def _run(self) -> None:
        self.callback({"type": "log", "message": f"实时监控已启动：{len(self.streams)} 路，{len(self.profiles)} 个目标。"})
        try:
            while not self.stop_event.is_set():
                did_work = False
                for stream in self.streams:
                    if self.stop_event.is_set():
                        break
                    sequence, frame, _ = self.captures[stream.index].get_latest()
                    if frame is None or sequence == self.last_seq.get(stream.index):
                        continue
                    now = time.monotonic()
                    h, w = frame.shape[:2]
                    interval = 0.85 if max(h, w) >= 3000 else 0.50
                    if now - self.last_infer.get(stream.index, 0.0) < interval:
                        continue
                    self.last_seq[stream.index] = sequence
                    self.last_infer[stream.index] = now
                    self._process_stream(stream, frame)
                    did_work = True
                if not did_work:
                    self.stop_event.wait(0.03)
        except Exception as exc:
            import traceback

            self.callback({"type": "live_error", "message": str(exc), "traceback": traceback.format_exc()})
        finally:
            for capture in self.captures.values():
                capture.stop()
            self.callback({"type": "live_stopped"})
