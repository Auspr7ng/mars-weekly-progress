"""Headless scripted integration diagnostic; never presented as real Xbox input.

Load the production checkpoint/model/player, replacing only SDL input and the
web UI with an explicit scripted fixture. No robot hardware is opened.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

import run_demo  # establishes the same frozen source/dependency paths
import mujoco
import numpy as np

ROWS = []


class ScriptedReader:
    backend = "scripted fixture; no physical Xbox input"

    def __init__(self, control):
        self.control = control

    def start(self):
        pass

    def close(self):
        self.control.stop("Scripted diagnostic ended")


class ScriptedArrow:
    def __init__(self, control):
        self.control = control
        self.last_command = np.zeros(3)
        self.performance = {}
        self.seq = 0
        self.episode = 0
        self.last_time = 0.0
        self.forced_fall = False

    def stop(self):
        self.control.stop("Scripted diagnostic stop")

    def command(self, data):
        self.seq += 1
        t = data.time
        if t < self.last_time - 1e-8:
            self.episode += 1
        self.last_time = t
        p = dict(
            client="scripted-check",
            seq=self.seq,
            id="SCRIPTED (no physical controller)",
            mapping="standard",
            connected=True,
            focused=True,
            x=0.0,
            y=0.0,
            rx=0.0,
            ry=0.0,
            a=False,
            y_button=False,
            home=False,
            lb=False,
            b=False,
            menu=False,
        )
        if self.episode:
            self.control.submit(p)
            self.last_command = np.asarray(self.control.sample()["command"])
            return self.last_command.tolist()
        if 1 <= t < 1.2 or 7.5 <= t < 7.7:
            p["ry"] = -1.0
        elif 8.5 <= t < 8.8:
            p["y"] = -1.0
        elif 5.5 <= t < 5.6:
            p["home"] = True
        elif 2.5 <= t < 2.6:
            p["a"] = True
        elif 4 <= t < 4.1:
            p["y_button"] = True
        elif 1.2 <= t < 1.4:
            p["rx"] = 1.0
        elif t >= 9.3 and not self.forced_fall:
            # Exercise the same episode-end and reset path as a real fall.
            angle = np.radians(85.0)
            data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]
            data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, data)
            self.forced_fall = True
        self.control.submit(p)
        self.last_command = np.asarray(self.control.sample()["command"])
        return self.last_command.tolist()


class HeadlessViewer:
    headless = True

    def __init__(self, model, data, control, reader, port, upper_body):
        self.model, self.upper_body = model, upper_body
        self.reset_requested = self.stopped = False
        self.paused = SimpleNamespace(value=False)
        self.arrow = ScriptedArrow(control)
        self.arrow.model = model
        self.reset_count = 0

    def update(self, data, hands, feet, label):
        root = self.model.body("robot/base_link").id
        cos = float(data.xmat[root].reshape(3, 3)[2, 2])
        ROWS.append(
            dict(
                episode=self.arrow.episode,
                simulation_time_s=float(data.time),
                label=label,
                command=self.arrow.last_command.tolist(),
                tilt_deg=float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
                upper_body=self.upper_body.status(),
                measured={n: float(data.qpos[adr]) for n, adr in self.upper_body.qadr.items()},
            )
        )
        if "END:" in label:
            self.reset_count += 1
            self.reset_requested = True
        if self.arrow.episode and data.time >= 1.0:
            self.stopped = True

    def close(self):
        self.arrow.stop()


def main():
    ROWS.clear()
    import native_input
    import viewer

    with (
        patch.object(native_input, "NativeInput", ScriptedReader),
        patch.object(viewer, "Viewer", HeadlessViewer),
        patch.object(sys, "argv", ["run_demo.py", "--no-browser", "--duration", "45"]),
    ):
        run_demo.main()
    initial = [r for r in ROWS if r["episode"] == 0]
    reset_rows = [r for r in ROWS if r["episode"] == 1]
    posed_walk = [r for r in initial if 8.55 < r["simulation_time_s"] < 8.75]
    checks = dict(
        completed=bool(reset_rows and reset_rows[-1]["simulation_time_s"] >= 1.0),
        raised=any(r["upper_body"]["lift_rad"] > 0.05 for r in ROWS),
        spread=any(r["upper_body"]["spread_rad"] > 0.05 for r in ROWS),
        opened=any(r["upper_body"]["slider_target_m"] > 0.025 for r in ROWS),
        closed=any(r["upper_body"]["slider_target_m"] < 0.005 for r in ROWS),
        walking_at_nonhome_pose=bool(posed_walk)
        and all(r["command"][0] > 0 and r["upper_body"]["lift_rad"] > 0.05 for r in posed_walk),
        held_pose_while_walking=bool(posed_walk)
        and max(r["upper_body"]["lift_rad"] for r in posed_walk)
        - min(r["upper_body"]["lift_rad"] for r in posed_walk)
        < 1e-8,
        optional_home_return=any(
            r["upper_body"]["lift_rad"] == 0.0
            and r["upper_body"]["spread_rad"] == 0.0
            and not r["upper_body"]["manipulation_active"]
            for r in initial
            if 6.5 < r["simulation_time_s"] < 7.5
        ),
        episode_end_detected=any("END:" in r["label"] for r in initial),
        reset_recovered=bool(
            reset_rows
            and reset_rows[0]["tilt_deg"] < 5.0
            and reset_rows[0]["upper_body"]["lift_rad"] == 0.0
            and reset_rows[0]["upper_body"]["spread_rad"] == 0.0
            and not reset_rows[0]["upper_body"]["manipulation_active"]
        ),
        continued_after_reset=bool(
            reset_rows
            and reset_rows[-1]["simulation_time_s"] >= 1.0
            and all("END:" not in r["label"] for r in reset_rows)
        ),
    )
    result = dict(
        input_source="scripted fixture; no physical Xbox verification",
        scope="one short empty-hand scenario, not acceptance of the full pose envelope",
        checks=checks,
        passed=all(checks.values()),
        samples=len(ROWS),
        simulated_seconds=sum(
            max(r["simulation_time_s"] for r in ROWS if r["episode"] == e)
            for e in {r["episode"] for r in ROWS}
        ),
        max_tilt_deg=max((r["tilt_deg"] for r in ROWS), default=None),
        episode_end_labels=[r["label"] for r in ROWS if "END:" in r["label"]],
        final_upper_body=ROWS[-1]["upper_body"] if ROWS else None,
    )
    folder = Path(__file__).resolve().parents[1] / "results"
    (folder / "arm_simulation_check.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (folder / "arm_simulation_trace.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8"
    )
    print("ARM_SIMULATION_CHECK " + json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
