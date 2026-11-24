"""Regression checks for the observed Windows status-publication failure."""

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from atomic_io import replace_with_retry, write_json

WEEK = Path(__file__).resolve().parents[1]


class AtomicPublicationTests(unittest.TestCase):
    def setUp(self):
        root = WEEK / ".cache" / "atomic_tests"
        root.mkdir(parents=True, exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(dir=root)
        self.path = Path(self.folder.name).resolve()
        if not self.path.is_relative_to(WEEK.resolve()):
            raise RuntimeError("Temporary test directory escaped the workspace")
        self.addCleanup(self.folder.cleanup)

    def test_json_replacement_keeps_a_complete_document(self):
        target = self.path / "status.json"
        write_json(target, {"iteration": 1})
        write_json(target, {"iteration": 2, "state": "completed"})
        self.assertEqual(json.loads(target.read_text())["iteration"], 2)
        self.assertFalse(target.with_suffix(".tmp").exists())

    def test_persistent_denial_is_bounded_and_preserves_both_files(self):
        source, target = self.path / "next", self.path / "old"
        source.write_text("new")
        target.write_text("old")
        with patch("atomic_io.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                replace_with_retry(source, target, attempts=2, delay_s=0)
        self.assertEqual(source.read_text(), "new")
        self.assertEqual(target.read_text(), "old")

    @unittest.skipUnless(os.name == "nt", "Windows file-sharing regression")
    def test_actual_windows_reader_lock_can_clear_during_retry(self):
        target = self.path / "status.json"
        write_json(target, {"iteration": 1})
        reader = target.open("rb")
        probe = self.path / "probe"
        probe.write_text("probe")
        try:
            with self.assertRaises(PermissionError):
                os.replace(probe, target)
        except BaseException:
            reader.close()
            raise
        release = threading.Timer(0.15, reader.close)
        release.start()
        try:
            write_json(target, {"iteration": 2})
        finally:
            release.join()
            reader.close()
        self.assertEqual(json.loads(target.read_text()), {"iteration": 2})


if __name__ == "__main__":
    unittest.main()
