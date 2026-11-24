"""Verify persisted progress and stale-report behavior without opening a GUI."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training_monitor import load_report


class MonitorTests(unittest.TestCase):
    def test_dead_process_is_not_shown_as_healthy_training(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "run.json").write_text(json.dumps({"iterations_requested": 1200}))
            (run / "status.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "iterations_completed": 120,
                        "elapsed_s": 600,
                        "pid": 42,
                    }
                )
            )
            with patch("training_monitor.process_alive", return_value=False):
                report = load_report(run)
            self.assertEqual(report["percent"], 10)
            self.assertEqual(report["eta"], 5400)
            self.assertTrue(report["stale"])

    def test_completed_run_needs_no_live_process(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "run.json").write_text(json.dumps({"iterations_requested": 800}))
            (run / "status.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "iterations_completed": 800,
                        "elapsed_s": 3500,
                    }
                )
            )
            report = load_report(run)
            self.assertEqual(report["percent"], 100)
            self.assertFalse(report["stale"])
            self.assertIsNone(report["eta"])

    def test_partial_or_missing_report_does_not_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "status.json").write_text('{"state":')
            report = load_report(run)
            self.assertEqual(report["completed"], 0)
            self.assertEqual(report["percent"], 0)


if __name__ == "__main__":
    unittest.main()
