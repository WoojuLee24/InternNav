#!/usr/bin/env python3
"""Analyze Track-1 latency tail against server timing phases.

This script aligns client HTTP latency samples with server request records
by request order and reports correlation between latency tails and S2 events.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from typing import Any


CLIENT_RE = re.compile(r"\[HTTP\]\s+idx:\s*(\d+)\s*\|\s*Latency:\s*([0-9.]+)s")
S1_RE = re.compile(r"\[System 1\]\s+inference time:\s*([0-9.]+)s")
S2_RE = re.compile(r"\[System 2\]\s+inference time:\s*([0-9.]+)s")
DUAL_RE = re.compile(r"dual sys step time:\s*([0-9.]+)")
POST_RE = re.compile(r'"POST /eval_dual HTTP/1\.1" 200 -')


def parse_client(path: pathlib.Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for m in CLIENT_RE.finditer(text):
        rows.append({"idx": int(m.group(1)), "latency": float(m.group(2))})
    rows.sort(key=lambda x: x["idx"])
    return rows


def parse_server(path: pathlib.Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    records: list[dict[str, Any]] = []

    pending_s1: float | None = None
    pending_s2: list[float] = []
    pending_dual: float | None = None

    for line in lines:
        m = S1_RE.search(line)
        if m:
            pending_s1 = float(m.group(1))
            continue
        m = S2_RE.search(line)
        if m:
            pending_s2.append(float(m.group(1)))
            continue
        m = DUAL_RE.search(line)
        if m:
            pending_dual = float(m.group(1))
            continue
        if POST_RE.search(line):
            s2_total = sum(pending_s2) if pending_s2 else 0.0
            records.append(
                {
                    "s1": pending_s1 if pending_s1 is not None else math.nan,
                    "s2_total": s2_total,
                    "has_s2": bool(pending_s2),
                    "dual": pending_dual if pending_dual is not None else math.nan,
                }
            )
            pending_s1 = None
            pending_s2 = []
            pending_dual = None

    return records


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def run_analysis(run_dir: pathlib.Path) -> dict[str, Any]:
    client = parse_client(run_dir / "client.log")
    server = parse_server(run_dir / "server_stdout.log")
    n = min(len(client), len(server))
    pairs: list[dict[str, Any]] = []
    for i in range(n):
        pairs.append(
            {
                "idx": client[i]["idx"],
                "latency": client[i]["latency"],
                "s1": server[i]["s1"],
                "s2_total": server[i]["s2_total"],
                "has_s2": server[i]["has_s2"],
                "dual": server[i]["dual"],
            }
        )

    lats = [p["latency"] for p in pairs]
    p90 = percentile(lats, 0.90)
    tail = [p for p in pairs if p["latency"] >= p90]
    body = [p for p in pairs if p["latency"] < p90]

    tail_has_s2 = sum(1 for p in tail if p["has_s2"])
    body_has_s2 = sum(1 for p in body if p["has_s2"])

    def _vals(ps: list[dict[str, Any]], key: str) -> list[float]:
        return [float(p[key]) for p in ps if not math.isnan(float(p[key]))]

    result = {
        "run": run_dir.name,
        "aligned_count": n,
        "client_count": len(client),
        "server_count": len(server),
        "lat_mean": mean(lats),
        "lat_p90": p90,
        "tail_count": len(tail),
        "tail_has_s2_ratio": (tail_has_s2 / len(tail)) if tail else math.nan,
        "body_has_s2_ratio": (body_has_s2 / len(body)) if body else math.nan,
        "tail_lat_mean": mean(_vals(tail, "latency")),
        "tail_dual_mean": mean(_vals(tail, "dual")),
        "tail_s2_mean": mean(_vals(tail, "s2_total")),
        "body_lat_mean": mean(_vals(body, "latency")),
        "body_dual_mean": mean(_vals(body, "dual")),
        "body_s2_mean": mean(_vals(body, "s2_total")),
    }
    return result


def fmt(v: float, digits: int = 3) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    return f"{v:.{digits}f}"


def to_markdown(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Track-1 Tail Correlation Analysis")
    lines.append("")
    lines.append("Latency tail is defined as `latency >= p90` per run.")
    lines.append("")
    lines.append("| Run | aligned | lat_mean_s | lat_p90_s | tail_count | tail_has_s2_ratio | body_has_s2_ratio | tail_dual_mean_s | body_dual_mean_s | tail_s2_mean_s | body_s2_mean_s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            "| {run} | {aligned} | {lat_mean} | {lat_p90} | {tail_count} | {tail_s2_ratio} | {body_s2_ratio} | {tail_dual} | {body_dual} | {tail_s2} | {body_s2} |".format(
                run=r["run"],
                aligned=r["aligned_count"],
                lat_mean=fmt(r["lat_mean"]),
                lat_p90=fmt(r["lat_p90"]),
                tail_count=r["tail_count"],
                tail_s2_ratio=fmt(r["tail_has_s2_ratio"]),
                body_s2_ratio=fmt(r["body_has_s2_ratio"]),
                tail_dual=fmt(r["tail_dual_mean"]),
                body_dual=fmt(r["body_dual_mean"]),
                tail_s2=fmt(r["tail_s2_mean"]),
                body_s2=fmt(r["body_s2_mean"]),
            )
        )
    lines.append("")
    lines.append("## Interpretation hints")
    lines.append("")
    lines.append("- If `tail_has_s2_ratio` >> `body_has_s2_ratio`, latency tails align with S2-refresh windows.")
    lines.append("- If `tail_dual_mean_s` >> `body_dual_mean_s`, long latency requests are dominated by long dual-step execution.")
    lines.append("- Use these signals to guide adaptive cadence/token experiments in Track-2.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latency tail correlation for track-1 runs.")
    parser.add_argument("runs", nargs="+", help="Run directories containing client.log and server_stdout.log")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    parser.add_argument("--markdown-out", default="", help="Optional markdown output path")
    args = parser.parse_args()

    results = [run_analysis(pathlib.Path(run_dir)) for run_dir in args.runs]

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.markdown_out:
        out = pathlib.Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_markdown(results), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
