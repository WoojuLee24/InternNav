#!/usr/bin/env python3
"""Collect sync-run quality/speed metrics from client logs.

Outputs markdown/json for quick accept/reject decisions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any


LAT_RE = re.compile(r"\[([0-9]+\.[0-9]+)\].*\[HTTP\]\s+idx:\s*(\d+)\s*\|\s*Latency:\s*([0-9.]+)s")
TRAJ_RE = re.compile(r"\[Plan\]\s+Received Trajectory\.")
NO_TRAJ_RE = re.compile(r"\[Plan\]\s+No trajectory in response\.")
DISCRETE_RE = re.compile(r"\[Plan\]\s+Received Discrete Actions:")
FAIL_RE = re.compile(r"\[HTTP\]\s+(Request Failed|Bad status|Invalid JSON response)")


def parse_run(run_dir: pathlib.Path) -> dict[str, Any]:
    client_path = run_dir / "client.log"
    text = client_path.read_text(encoding="utf-8", errors="replace")

    lat_entries = [(float(m.group(1)), int(m.group(2)), float(m.group(3))) for m in LAT_RE.finditer(text)]
    latencies = [v for _, _, v in lat_entries]
    req_count = len(latencies)
    avg_latency = sum(latencies) / req_count if req_count > 0 else 0.0

    req_hz = 0.0
    if req_count >= 2:
        t0 = lat_entries[0][0]
        t1 = lat_entries[-1][0]
        elapsed = max(t1 - t0, 1e-6)
        req_hz = req_count / elapsed

    return {
        "run": run_dir.name,
        "req_count": req_count,
        "traj_count": len(TRAJ_RE.findall(text)),
        "no_traj_count": len(NO_TRAJ_RE.findall(text)),
        "discrete_count": len(DISCRETE_RE.findall(text)),
        "req_hz": req_hz,
        "http_avg_latency_s": avg_latency,
        "http_failures": len(FAIL_RE.findall(text)),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("# Sync Metrics Summary")
    lines.append("")
    lines.append("| Run | req_count | traj_count | no_traj_count | discrete_count | req_hz | http_avg_latency_s | http_failures |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {run} | {req_count} | {traj_count} | {no_traj_count} | {discrete_count} | {req_hz:.3f} | {http_avg_latency_s:.3f} | {http_failures} |".format(
                **r
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sync run metrics from client logs.")
    parser.add_argument("runs", nargs="+", help="Run directories under test_data")
    parser.add_argument("--json-out", default="", help="Optional output JSON path")
    parser.add_argument("--markdown-out", default="", help="Optional output markdown path")
    args = parser.parse_args()

    rows = [parse_run(pathlib.Path(r)) for r in args.runs]

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.markdown_out:
        out = pathlib.Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown(rows), encoding="utf-8")

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
