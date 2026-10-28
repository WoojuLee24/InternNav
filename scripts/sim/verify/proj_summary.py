"""Summarise proj_check.py output (③): err_z by camera / projection method / range bin, lidar coverage.

Usage: python3 scripts/sim/verify/proj_summary.py <out dir with proj.csv>
"""
import csv
import sys

import numpy as np

rows = list(csv.DictReader(open(f"{sys.argv[1]}/proj.csv")))
print(f"samples {len(rows)}")
print("| camera | proj | range | n | no_lidar or few | |err_z| p50 / p90 [m] | err_z median [m] | <=0.10 m | reproj p90 [px] |")
print("|---|---|---|---|---|---|---|---|---|")
for cam in sorted(set(r["pixel_frame"] for r in rows)):
    for proj in sorted(set(r["proj"] for r in rows)):
        for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 1e9)):
            sel = [r for r in rows if r["pixel_frame"] == cam and r["proj"] == proj
                   and lo <= float(r["range_m"]) < hi]
            if not sel:
                continue
            ok = [float(r["err_z_m"]) for r in sel if r["status"] == "ok"]
            a = np.abs(ok)
            rp = [float(r["reproj_px"]) for r in sel]
            print(f"| {cam} | {proj} | {lo}-{hi if hi < 1e9 else ''} m | {len(sel)} | {len(sel) - len(ok)} | "
                  + (f"{np.percentile(a, 50):.3f} / {np.percentile(a, 90):.3f} | {np.median(ok):+.3f} | "
                     f"{np.mean(a <= 0.10) * 100:.0f}% | " if ok else "- | - | - | ")
                  + f"{np.percentile(rp, 90):.1f} |")
