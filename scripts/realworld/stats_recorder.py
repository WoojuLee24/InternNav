"""
stats_recorder.py — structured experiment stats collection for InternNav.

Usage as module (in client/server scripts):
    from stats_recorder import StatsRecorder
    rec = StatsRecorder("bag073623_kv-on_temp0.8", control_vars={
        "bag_id": "073623", "kv_cache": True, "temperature": 0.8,
        "plan_step_gap": 12, "mode": "async"
    })
    rec.record(joint_latency_ms=8.3, response_type="trajectory",
               waypoints=[[0.1, 0.0, 0.05]], s2_latency_ms=315.0)
    rec.finalize()   # writes summary.json

Usage as CLI (parse existing client.log):
    python stats_recorder.py parse test_data/async_check_*/client.log --tag my_exp
    python stats_recorder.py summary stats/my_exp/
    python stats_recorder.py compare stats/exp_a/ stats/exp_b/
"""

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATS_DIR = _REPO_ROOT / "stats"


class StatsRecorder:
    """Thread-safe per-experiment metrics recorder.

    One instance per experiment run. All records go to a JSONL file so
    the plotter can stream them without loading everything into memory.
    """

    def __init__(self, tag: str, control_vars: Optional[Dict[str, Any]] = None):
        self.tag = tag
        self.control_vars = control_vars or {}
        self._t0 = time.time()
        self._n = 0
        self._traj_count = 0
        self._discrete_count = 0
        self._waiting_count = 0
        self._latency_window: deque = deque(maxlen=50)  # rolling 50-sample window
        self._s2_latency_window: deque = deque(maxlen=20)

        self._out_dir = _STATS_DIR / tag
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._out_dir / "metrics.jsonl"
        self._summary_path = self._out_dir / "summary.json"

        # Write experiment header
        with open(self._out_dir / "config.json", "w") as f:
            json.dump({"tag": tag, "control_vars": control_vars,
                       "started_at": self._t0}, f, indent=2)

    def record(
        self,
        joint_latency_ms: float = 0.0,
        response_type: str = "waiting",          # "trajectory" | "discrete" | "waiting"
        waypoints: Optional[List[List[float]]] = None,
        subgoal_px: Optional[List[float]] = None,
        s2_latency_ms: float = 0.0,
        s2_hz: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._n += 1
        t_rel = time.time() - self._t0

        if response_type == "trajectory":
            self._traj_count += 1
        elif response_type == "discrete":
            self._discrete_count += 1
        else:
            self._waiting_count += 1

        if joint_latency_ms > 0:
            self._latency_window.append(joint_latency_ms)
        if s2_latency_ms > 0:
            self._s2_latency_window.append(s2_latency_ms)

        record: Dict[str, Any] = {
            "ts": time.time(),
            "t_rel": round(t_rel, 3),
            "n": self._n,
            "joint_latency_ms": round(joint_latency_ms, 2),
            "response_type": response_type,
            "s2_latency_ms": round(s2_latency_ms, 2),
            "s2_hz": round(s2_hz, 3),
            "waypoints": waypoints,
            "subgoal_px": subgoal_px,
            "control_vars": self.control_vars,
        }
        if extra:
            record.update(extra)

        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def finalize(self) -> Dict[str, Any]:
        duration = time.time() - self._t0
        total = max(self._n, 1)
        latencies = list(self._latency_window) or [0]
        s2_latencies = list(self._s2_latency_window) or [0]

        summary = {
            "tag": self.tag,
            "control_vars": self.control_vars,
            "duration_s": round(duration, 1),
            "total_requests": total,
            "joint_req_hz": round(total / max(duration, 1), 3),
            "trajectory_ratio": round(self._traj_count / total, 4),
            "discrete_ratio": round(self._discrete_count / total, 4),
            "waiting_ratio": round(self._waiting_count / total, 4),
            "joint_latency_ms_mean": round(sum(latencies) / len(latencies), 2),
            "joint_latency_ms_p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "s2_latency_ms_mean": round(sum(s2_latencies) / len(s2_latencies), 2),
            "traj_count": self._traj_count,
            "discrete_count": self._discrete_count,
        }
        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    @property
    def trajectory_ratio(self) -> float:
        total = max(self._n, 1)
        return self._traj_count / total

    @property
    def rolling_latency_ms(self) -> float:
        if not self._latency_window:
            return 0.0
        return sum(self._latency_window) / len(self._latency_window)


# ── CLI: parse existing client.log ────────────────────────────────────────────

# Regex patterns matching http_internvla_client_debug.py log format
_LATENCY_RE  = re.compile(r"\[HTTP\].*?(\d+\.?\d*)\s*ms", re.IGNORECASE)
_TRAJ_RE     = re.compile(r"Received Trajectory", re.IGNORECASE)
_DISCRETE_RE = re.compile(r"Received Discrete", re.IGNORECASE)
_WAITING_RE  = re.compile(r"(No trajectory|status.*waiting)", re.IGNORECASE)
_S2_LAT_RE   = re.compile(r"s2[_\s]latency[:\s]+(\d+\.?\d*)\s*ms", re.IGNORECASE)
_WAYPTS_RE   = re.compile(r"waypoints?[:\s]+\[([^\]]+)\]", re.IGNORECASE)


def _parse_client_log(log_path: str, tag: str, control_vars: Dict) -> Dict:
    rec = StatsRecorder(tag, control_vars)
    with open(log_path) as f:
        for line in f:
            lat_m   = _LATENCY_RE.search(line)
            s2_m    = _S2_LAT_RE.search(line)
            is_traj = bool(_TRAJ_RE.search(line))
            is_disc = bool(_DISCRETE_RE.search(line))
            is_wait = bool(_WAITING_RE.search(line))

            if not any([lat_m, is_traj, is_disc, is_wait]):
                continue

            rtype = ("trajectory" if is_traj else
                     "discrete"   if is_disc else
                     "waiting")
            rec.record(
                joint_latency_ms=float(lat_m.group(1)) if lat_m else 0.0,
                s2_latency_ms=float(s2_m.group(1)) if s2_m else 0.0,
                response_type=rtype,
            )

    summary = rec.finalize()
    print(json.dumps(summary, indent=2))
    return summary


def _print_summary(stats_dir: str) -> None:
    p = Path(stats_dir) / "summary.json"
    if not p.exists():
        print(f"No summary.json in {stats_dir}")
        return
    with open(p) as f:
        d = json.load(f)
    print(f"\n{'─'*50}")
    print(f"  Experiment: {d['tag']}")
    print(f"{'─'*50}")
    print(f"  Control vars:     {d['control_vars']}")
    print(f"  Duration:         {d['duration_s']} s")
    print(f"  Total requests:   {d['total_requests']}")
    print(f"  joint_req_hz:     {d['joint_req_hz']} Hz")
    print(f"  trajectory_ratio: {d['trajectory_ratio']*100:.1f}%")
    print(f"  joint_latency:    {d['joint_latency_ms_mean']} ms (p95: {d['joint_latency_ms_p95']} ms)")
    print(f"  s2_latency:       {d['s2_latency_ms_mean']} ms")
    print(f"{'─'*50}")


def _compare(*dirs: str) -> None:
    rows = []
    for d in dirs:
        p = Path(d) / "summary.json"
        if p.exists():
            with open(p) as f:
                rows.append(json.load(f))

    if not rows:
        print("No summaries found."); return

    header = ["tag", "joint_req_hz", "trajectory_ratio", "joint_latency_ms_mean", "s2_latency_ms_mean"]
    widths  = [max(len(h), max(len(str(r.get(h, ""))) for r in rows)) + 2 for h in header]
    sep = "─" * sum(widths)
    print(sep)
    print("".join(h.ljust(w) for h, w in zip(header, widths)))
    print(sep)
    for r in rows:
        print("".join(str(r.get(h, "─")).ljust(w) for h, w in zip(header, widths)))
    print(sep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="InternNav experiment stats recorder / parser")
    sub = ap.add_subparsers(dest="cmd")

    p_parse = sub.add_parser("parse", help="Parse a client.log into stats JSONL")
    p_parse.add_argument("log", help="Path to client.log")
    p_parse.add_argument("--tag", default="", help="Experiment tag (default: log filename)")
    p_parse.add_argument("--bag-id", default="unknown")
    p_parse.add_argument("--kv-cache", action="store_true")
    p_parse.add_argument("--temperature", type=float, default=1.0)
    p_parse.add_argument("--mode", default="async")

    p_sum = sub.add_parser("summary", help="Print experiment summary")
    p_sum.add_argument("stats_dir")

    p_cmp = sub.add_parser("compare", help="Compare multiple experiment summaries")
    p_cmp.add_argument("dirs", nargs="+")

    args = ap.parse_args()

    if args.cmd == "parse":
        tag = args.tag or Path(args.log).parent.name
        cvars = {
            "bag_id": args.bag_id,
            "kv_cache": args.kv_cache,
            "temperature": args.temperature,
            "mode": args.mode,
        }
        _parse_client_log(args.log, tag, cvars)

    elif args.cmd == "summary":
        _print_summary(args.stats_dir)

    elif args.cmd == "compare":
        _compare(*args.dirs)

    else:
        ap.print_help()
