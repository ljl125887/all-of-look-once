"""Load only tracker definitions so CI needs no GPU or model downloads."""
import ast
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "vehicle_instance_search" / "core.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"Detection", "_PersonTrack", "PersonIdentityTracker"}]
namespace = dict(np=np, deque=deque, dataclass=dataclass, field=field, Similarity=SimpleNamespace)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "tracker", "exec"), namespace)
Tracker = namespace["PersonIdentityTracker"]
Detection = namespace["Detection"]


class IdentityGateTests(unittest.TestCase):
    def test_disallowed_face_cannot_confirm_identity(self):
        tracker = Tracker()
        box = Detection((10, 10, 80, 200), .9, 0, "person")
        score = SimpleNamespace(score=.95, face=.99)
        for timestamp in range(8):
            result = tracker.update(float(timestamp), [box], [score], .84, .64, [False])
            self.assertNotEqual(result[0][0], "match")

    def test_low_similarity_does_not_confirm(self):
        tracker = Tracker()
        box = Detection((10, 10, 80, 200), .9, 0, "person")
        for timestamp in range(8):
            result = tracker.update(float(timestamp), [box], [SimpleNamespace(score=.25, face=.2)], .84, .64, [True])
            self.assertEqual(result[0][0], "reject")

    def test_strong_identity_confirms_and_expires_after_gap(self):
        tracker = Tracker()
        box = Detection((10, 10, 80, 200), .9, 0, "person")
        for timestamp in range(3):
            result = tracker.update(float(timestamp), [box], [SimpleNamespace(score=.9, face=-1)], .84, .64, [True])
        self.assertEqual(result[0][0], "match")
        result = tracker.update(10., [box], [SimpleNamespace(score=.25, face=-1)], .84, .64, [True])
        self.assertEqual(result[0][0], "reject")


if __name__ == "__main__":
    unittest.main()
