"""持久化监控目标库：目标元数据、参考图、正样本与负样本。"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from core import read_image, safe_name, write_image
from reid_models import PERSON_MODEL_FASTREID, PERSON_MODEL_FUSION


@dataclass(slots=True)
class TargetProfile:
    profile_id: str
    name: str
    category: str
    sensitivity: str
    # 旧目标库没有该字段，按原FastReID基线读取，避免静默改变既有人员结果。
    person_model: str = PERSON_MODEL_FASTREID
    enabled: bool = True
    reference_paths: list[str] = field(default_factory=list)
    positive_paths: list[str] = field(default_factory=list)
    negative_paths: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "category": self.category,
            "sensitivity": self.sensitivity,
            "person_model": self.person_model,
            "enabled": self.enabled,
            "reference_paths": self.reference_paths,
            "positive_paths": self.positive_paths,
            "negative_paths": self.negative_paths,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TargetLibrary:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _folder(self, profile_id: str) -> Path:
        return self.root / profile_id

    def list_profiles(self) -> list[TargetProfile]:
        profiles: list[TargetProfile] = []
        for metadata in self.root.glob("*/metadata.json"):
            try:
                data = json.loads(metadata.read_text(encoding="utf-8"))
                profiles.append(TargetProfile(**data))
            except (OSError, ValueError, TypeError):
                continue
        return sorted(profiles, key=lambda item: item.updated_at, reverse=True)

    def get(self, profile_id: str) -> TargetProfile:
        data = json.loads((self._folder(profile_id) / "metadata.json").read_text(encoding="utf-8"))
        return TargetProfile(**data)

    def save(
        self,
        name: str,
        category: str,
        sensitivity: str,
        references: list[np.ndarray],
        profile_id: str | None = None,
        person_model: str = PERSON_MODEL_FUSION,
    ) -> TargetProfile:
        now = datetime.now().isoformat(timespec="seconds")
        existing = self.get(profile_id) if profile_id and (self._folder(profile_id) / "metadata.json").is_file() else None
        profile_id = profile_id or f"{safe_name(name)}_{uuid.uuid4().hex[:8]}"
        folder = self._folder(profile_id)
        refs = folder / "references"
        refs.mkdir(parents=True, exist_ok=True)
        for old in refs.glob("*.jpg"):
            old.unlink()
        paths: list[str] = []
        for index, image in enumerate(references, 1):
            path = refs / f"reference_{index:02d}.jpg"
            write_image(path, image)
            paths.append(str(path.relative_to(folder)))
        profile = TargetProfile(
            profile_id=profile_id,
            name=name,
            category=category,
            sensitivity=sensitivity,
            person_model=person_model if category == "人员" else "车辆固定模型",
            enabled=True if existing is None else existing.enabled,
            reference_paths=paths,
            positive_paths=[] if existing is None else existing.positive_paths,
            negative_paths=[] if existing is None else existing.negative_paths,
            created_at=now if existing is None else existing.created_at,
            updated_at=now,
        )
        (folder / "metadata.json").write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return profile

    def load_references(self, profile: TargetProfile) -> list[np.ndarray]:
        folder = self._folder(profile.profile_id)
        return [read_image(folder / path) for path in profile.reference_paths]

    def set_enabled(self, profile_id: str, enabled: bool) -> TargetProfile:
        profile = self.get(profile_id)
        profile.enabled = enabled
        profile.updated_at = datetime.now().isoformat(timespec="seconds")
        (self._folder(profile_id) / "metadata.json").write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return profile

    def add_positive(self, profile_id: str, image: np.ndarray) -> Path:
        profile = self.get(profile_id)
        folder = self._folder(profile_id)
        positive_dir = folder / "positive_samples"
        positive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = positive_dir / f"positive_{stamp}.jpg"
        write_image(path, image)
        profile.positive_paths.append(str(path.relative_to(folder)))
        profile.updated_at = datetime.now().isoformat(timespec="seconds")
        (folder / "metadata.json").write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def delete(self, profile_id: str) -> None:
        folder = self._folder(profile_id).resolve()
        root = self.root.resolve()
        if folder.parent != root:
            raise ValueError("目标目录越界")
        if folder.is_dir():
            shutil.rmtree(folder)
