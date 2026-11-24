"""Read-only local training dashboard. Uses only the Python standard library."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk
import webbrowser

WEEK = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def process_alive(pid):
    """Query a Windows process without sending it a signal."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False if ctypes.get_last_error() == 87 else None
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def duration(seconds):
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def load_report(run):
    spec = read_json(run / "run.json")
    status = read_json(run / "status.json")
    total = status.get("iterations_requested", spec.get("iterations_requested", 0))
    completed = status.get("iterations_completed", 0)
    # Older resumed runs reported absolute iteration; use their spec as authority.
    elapsed = status.get("elapsed_s", 0)
    age = None
    try:
        updated = datetime.fromisoformat(status["updated"])
        age = max(0, time.time() - updated.timestamp())
    except (KeyError, ValueError):
        try:
            age = max(0, time.time() - (run / "status.json").stat().st_mtime)
        except OSError:
            pass
    state = status.get("state", "No status yet")
    live = (
        process_alive(status.get("pid"))
        if state in ("running", "initializing")
        else None
    )
    stale = state in ("running", "initializing") and (live is False or (age or 0) > 120)
    eta = (
        elapsed / completed * max(0, total - completed)
        if completed and state == "running"
        else None
    )
    return dict(
        spec=spec,
        status=status,
        total=total,
        completed=completed,
        percent=min(100, 100 * completed / total) if total else 0,
        age=age,
        live=live,
        stale=stale,
        eta=eta,
    )


class Dashboard:
    def __init__(self, root, initial_run=None):
        self.root = root
        self.run = initial_run
        self.metric_cache = {}
        root.title("Week7 - Training Monitor")
        root.geometry("1080x780")
        root.minsize(900, 720)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        main = ttk.Frame(root, padding=20)
        main.pack(fill="both", expand=True)
        ttk.Label(
            main, text="Sunday A1 | Week7 Training Monitor", style="Title.TLabel"
        ).pack(anchor="w")
        ttk.Label(main, text="Local simulation runs - refreshes every 2 seconds").pack(
            anchor="w", pady=(4, 15)
        )
        row = ttk.Frame(main)
        row.pack(fill="x")
        self.choice = ttk.Combobox(row, state="readonly", width=72)
        self.choice.pack(side="left", fill="x", expand=True)
        self.choice.bind("<<ComboboxSelected>>", self.select)
        ttk.Button(row, text="Active run", command=self.active).pack(
            side="left", padx=8
        )
        ttk.Button(row, text="Open run folder", command=self.open_folder).pack(
            side="left"
        )
        self.name = ttk.Label(
            main, text="Waiting for a run", font=("Segoe UI", 14, "bold")
        )
        self.name.pack(anchor="w", pady=(18, 4))
        self.state = ttk.Label(main)
        self.state.pack(anchor="w")
        self.bar = ttk.Progressbar(main, maximum=100)
        self.bar.pack(fill="x", pady=10)
        self.progress = ttk.Label(main)
        self.progress.pack(anchor="w")
        self.details = tk.Text(
            main,
            height=11,
            wrap="word",
            font=("Consolas", 10),
            relief="flat",
            bg="#f1f4f7",
        )
        self.details.pack(fill="x", pady=12)
        charts = ttk.Frame(main)
        charts.pack(fill="both", expand=True)
        self.charts = []
        for title in ("Mean episode reward", "Mean episode length (seconds)"):
            box = ttk.LabelFrame(charts, text=title, padding=5)
            box.pack(side="left", fill="both", expand=True, padx=4)
            canvas = tk.Canvas(box, bg="white", height=150, highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            self.charts.append(canvas)
        self.note = ttk.Label(
            main,
            text="Reward and episode length show learning progress, not gait acceptance.",
        )
        self.note.pack(anchor="w", pady=(10, 0))
        self.footer = ttk.Label(main, wraplength=1010)
        self.footer.pack(anchor="w", pady=(8, 0))
        self.refresh()

    def active(self):
        active = read_json(WEEK / "config/active_run.json")
        if active.get("run_directory"):
            self.run = WEEK / active["run_directory"]
        self.refresh(schedule=False)

    def select(self, event=None):
        self.run = WEEK / "runs" / self.choice.get()
        self.refresh(schedule=False)

    def open_folder(self):
        if self.run and self.run.is_dir():
            webbrowser.open(self.run.resolve().as_uri())

    def metrics(self):
        path = self.run / "metrics.jsonl"
        try:
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns)
            if self.metric_cache.get("key") != key:
                records = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        pass  # A writer may be in the middle of its final line.
                self.metric_cache = {"key": key, "records": records}
            return self.metric_cache["records"]
        except OSError:
            return []

    def plot(self, canvas, records, key, scale=1):
        canvas.delete("all")
        values = [
            (r.get("iterations_completed", i + 1), r.get(key))
            for i, r in enumerate(records)
        ]
        values = [(x, y * scale) for x, y in values if isinstance(y, (float, int))]
        if len(values) < 2:
            canvas.create_text(
                140, 65, text="Waiting for metric samples", fill="#687789"
            )
            return
        width, height = max(350, canvas.winfo_width()), max(120, canvas.winfo_height())
        low, high = min(y for _, y in values), max(y for _, y in values)
        span = max(high - low, 1e-6)
        points = []
        for x, y in values:
            points.extend(
                (
                    48
                    + (width - 65)
                    * (x - values[0][0])
                    / max(1, values[-1][0] - values[0][0]),
                    15 + (height - 45) * (high - y) / span,
                )
            )
        canvas.create_line(*points, fill="#1979af", width=2)
        canvas.create_text(4, 13, text=f"{high:.1f}", anchor="w", fill="#55616d")
        canvas.create_text(
            4, height - 30, text=f"{low:.1f}", anchor="w", fill="#55616d"
        )
        canvas.create_text(
            width - 10,
            height - 10,
            text=f"Update {values[-1][0]} | latest {values[-1][1]:.2f}",
            anchor="e",
        )

    def refresh(self, schedule=True):
        try:
            names = sorted(
                (p.parent.name for p in (WEEK / "runs").glob("*/run.json")),
                reverse=True,
            )
            self.choice["values"] = names
            if self.run is None:
                manifest = read_json(WEEK / "config/active_run.json")
                self.run = (
                    WEEK / manifest["run_directory"]
                    if manifest.get("run_directory")
                    else (WEEK / "runs" / names[0] if names else None)
                )
            if self.run:
                self.choice.set(self.run.name)
                report = load_report(self.run)
                s, spec = report["status"], report["spec"]
                self.name["text"] = s.get(
                    "run_name", spec.get("run_name", self.run.name)
                )
                label = s.get("state", "Waiting for status").upper()
                if report["stale"]:
                    label += " | STALE REPORT - check the log; process may have exited"
                elif report["live"]:
                    label += " | Process present"
                self.state.configure(
                    text=label,
                    foreground=(
                        "#b44b16"
                        if report["stale"] or label.startswith("FAILED")
                        else "#155b84"
                    ),
                )
                self.bar["value"] = report["percent"]
                self.progress["text"] = (
                    f"{report['completed']:,} / {report['total']:,} updates ({report['percent']:.1f}%)    Elapsed {duration(s.get('elapsed_s'))}    Estimated remaining {duration(report['eta'])}"
                )
                c = s.get("curriculum", spec.get("curriculum_initial", {}))
                reward, length = s.get("mean_reward"), s.get(
                    "mean_episode_length_steps"
                )
                losses = s.get("losses", {})
                age = f"{report['age']:.0f}s" if report["age"] is not None else "--"
                details = [
                    f"Round: {spec.get('training_round', 1)}   Environments: {spec.get('num_envs', '--')}   Seed: {spec.get('seed', '--')}   PID: {s.get('pid', '--')}",
                    f"Transitions this run: {s.get('transitions', 0):,}   Last update age: {age}",
                    (
                        f"Mean reward: {reward:.2f}"
                        if isinstance(reward, (int, float))
                        else "Mean reward: --"
                    ),
                    (
                        f"Mean episode length: {length * .02:.2f}s / 30s"
                        if isinstance(length, (int, float))
                        else "Mean episode length: --"
                    ),
                    f"Curriculum targets: forward {c.get('forward_cap_m_s', 0):.3f} m/s | whole-sole swing {1000*c.get('swing_target_m', 0):.1f} mm | arm fraction {c.get('arm_fraction', 0):.2f}",
                    "Losses: " + " | ".join(f"{k}={v:.4g}" for k, v in losses.items()),
                    "Latest checkpoint: "
                    + str(s.get("checkpoint") or "Waiting for first checkpoint"),
                    "Source: "
                    + str(
                        spec.get("init_from")
                        or spec.get("resume")
                        or spec.get("source_checkpoint", "--")
                    ),
                ]
                if s.get("error"):
                    details.append("ERROR: " + s["error"])
                self.details.configure(state="normal")
                self.details.delete("1.0", "end")
                self.details.insert("1.0", "\n".join(details))
                self.details.configure(state="disabled")
                records = self.metrics()
                self.plot(self.charts[0], records, "mean_reward")
                self.plot(self.charts[1], records, "mean_episode_length_steps", 0.02)
                self.footer["text"] = str(self.run.resolve())
        except (OSError, ValueError, KeyError, tk.TclError) as exc:
            self.footer["text"] = f"Waiting for readable report: {exc}"
        finally:
            if schedule:
                self.root.after(2000, self.refresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="Print a report without opening a window"
    )
    args = parser.parse_args()
    if args.json:
        active = read_json(WEEK / "config/active_run.json")
        run = args.run_dir or WEEK / active["run_directory"]
        print(json.dumps(load_report(run), indent=2))
        return
    root = tk.Tk()
    Dashboard(root, args.run_dir.resolve() if args.run_dir else None)
    root.mainloop()


if __name__ == "__main__":
    main()
