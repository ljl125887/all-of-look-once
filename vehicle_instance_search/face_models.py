"""OpenCV Zoo YuNet + SFace本地人脸辅助特征。

只在人脸清晰可见时提供额外身份证据；没有脸、侧脸失败或背面时返回None，
由身体ReID和连续轨迹继续工作。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np


class FaceFeatureExtractor:
    def __init__(self, detector_path: Path, recognizer_path: Path):
        detector_path = self._opencv_path(detector_path)
        recognizer_path = self._opencv_path(recognizer_path)
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), score_threshold=0.60, nms_threshold=0.3, top_k=50
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    @staticmethod
    def _opencv_path(source: Path) -> Path:
        """OpenCV 5的Windows ONNX加载器不支持非ASCII路径，复制到英文缓存目录。"""
        source = source.resolve()
        try:
            str(source).encode("ascii")
            return source
        except UnicodeEncodeError:
            local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            cache = local_app_data / "vehicle_instance_search" / "models"
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / source.name
            if not target.is_file() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
            return target

    def extract_one(self, image: np.ndarray) -> np.ndarray | None:
        if image is None or image.size == 0:
            return None
        height, width = image.shape[:2]
        if min(height, width) < 32:
            return None
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or not len(faces):
            return None
        # 人体切片中取最大的人脸，仅让至少28像素的人脸参与自动身份锁定。
        face = max(faces, key=lambda item: float(item[2] * item[3]))
        if min(float(face[2]), float(face[3])) < 28.0:
            return None
        aligned = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm < 1e-8:
            return None
        return feature / norm

    def extract(self, images: list[np.ndarray]) -> list[np.ndarray | None]:
        return [self.extract_one(image) for image in images]
