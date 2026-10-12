from pathlib import Path
import importlib.util
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_scan", ROOT / "tools" / "check_public_release.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


class ReleaseTests(unittest.TestCase):
    def test_public_tree_has_no_private_files(self):
        self.assertEqual(scan.scan(), [])

    def test_credentials_rejected(self):
        sample = "rtsp" + chr(58) + "//example" + chr(58) + "secret@camera.invalid/stream"
        self.assertIn("credential URL", scan.violations(Path("example.md"), sample))

    def test_location_and_runtime_files_rejected(self):
        self.assertIn("unexpected file type", scan.violations(Path("track.gpx"), ""))
        self.assertIn("runtime/private data", scan.violations(Path("config.json"), "{}"))


if __name__ == "__main__":
    unittest.main()
