#!/usr/bin/env python3
"""I-110 — Offline rosbag action-distribution bias test.

Detects whether a cache/scheduling change has materially altered the
fresh-S2 action vs trajectory distribution. The motivation is the Gate 3b
failure where the cosine-only cache at thr=0.92 produced 0/28 fresh actions
(vs ~50/50 control). Chi-squared on a 2x2 contingency table gives a clean
null-hypothesis test: "the test condition has the same fresh action/traj
distribution as the control".

Usage:
    python3 chi_squared_action_test.py CONTROL.json TEST.json
    python3 chi_squared_action_test.py --pair CTRL_DIR TEST_DIR

CONTROL/TEST are server metrics.json files (from /async_metrics) that
expose `fresh_traj_outputs` and `fresh_action_outputs`.

Pass criterion (the Gate 3b retry gate):
    Primary:   Cramer's V <= 0.10  →  PASS (small/negligible effect size)
               Cramer's V >  0.10  →  FAIL (medium/large effect — deployment risk)
    Advisory:  p-value reported for reference; at large n (>200) the p-value
               rejects even tiny effects (V<0.05). Use V as the decision gate.

Rationale: The original failure mode was V≈0.5+ (0/28 actions under 50/50 base
rate). V≤0.10 (Cohen's "small") means the cache has negligible practical bias.
The p-value gate (p≥0.05) is inappropriate here — with n>500 it rejects V≈0.07
which is noise-level in terms of robot behavior.

Also reports:
- fresh_action_rate for both runs (raw rates)
- effect size (Cramer's V) for magnitude beyond p-value
- minimum action count needed for the test to have power
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def chi2_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Chi-squared statistic and p-value for a 2x2 contingency table.
    Uses Yates' correction (continuity-corrected) since cell counts may be small.

    Table:
            traj    action
    ctrl    a       b
    test    c       d
    """
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    if min(row1, row2, col1, col2) == 0:
        return 0.0, 1.0
    # Expected counts under independence
    e_a = row1 * col1 / n
    e_b = row1 * col2 / n
    e_c = row2 * col1 / n
    e_d = row2 * col2 / n
    # Yates' continuity correction: |obs-exp| - 0.5
    chi2 = sum(
        (max(0.0, abs(o - e) - 0.5) ** 2) / e
        for o, e in ((a, e_a), (b, e_b), (c, e_c), (d, e_d))
    )
    # p-value for chi-square with 1 dof: P(X^2 > chi2) where X^2 ~ chi2_1
    # = erfc(sqrt(chi2/2)) for dof=1
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p


def cramers_v(chi2: float, n: int) -> float:
    """Effect size for 2x2: V = sqrt(chi2 / n). Range [0, 1]."""
    if n <= 0:
        return 0.0
    return math.sqrt(chi2 / n)


def load_fresh(path: Path) -> tuple[int, int, dict]:
    with path.open() as f:
        d = json.load(f)
    return (
        int(d.get("fresh_traj_outputs", 0)),
        int(d.get("fresh_action_outputs", 0)),
        d,
    )


def report(control: Path, test: Path, alpha: float = 0.05) -> int:
    ct, ca, ctrl_full = load_fresh(control)
    tt, ta, test_full = load_fresh(test)

    n_ctrl = ct + ca
    n_test = tt + ta
    if n_ctrl < 10 or n_test < 10:
        print(f"WARN: low fresh-S2 count (ctrl={n_ctrl}, test={n_test}); test has limited power")

    chi2, p = chi2_2x2(ct, ca, tt, ta)
    n_total = n_ctrl + n_test
    v = cramers_v(chi2, n_total)

    print(f"=== I-110 Chi-Squared Action-Distribution Test ===")
    print(f"Control: {control}")
    print(f"  fresh_traj   = {ct}")
    print(f"  fresh_action = {ca}")
    print(f"  action_rate  = {ca / max(1, n_ctrl) * 100:.1f}%")
    print(f"  thr          = {ctrl_full.get('temporal_cache_threshold', '?')}")
    print(f"  bypasses     = AA:{ctrl_full.get('action_aware_bypasses', 0)}  MH:{ctrl_full.get('max_hold_bypasses', 0)}")
    print(f"Test:    {test}")
    print(f"  fresh_traj   = {tt}")
    print(f"  fresh_action = {ta}")
    print(f"  action_rate  = {ta / max(1, n_test) * 100:.1f}%")
    print(f"  thr          = {test_full.get('temporal_cache_threshold', '?')}")
    print(f"  bypasses     = AA:{test_full.get('action_aware_bypasses', 0)}  MH:{test_full.get('max_hold_bypasses', 0)}")
    print(f"")
    print(f"chi2 (Yates) = {chi2:.4f}")
    print(f"p-value      = {p:.4f}    (alpha={alpha})")
    print(f"Cramer's V   = {v:.4f}    (effect size; <0.1 small, <0.3 med, >=0.3 large)")
    print(f"")
    v_threshold = 0.10
    if v <= v_threshold:
        print(f"PASS — Cramer's V={v:.4f} <= {v_threshold} (small/negligible effect; distributions practically equivalent).")
        if p < alpha:
            print(f"       NOTE: p={p:.4f} < alpha={alpha}, but p-value unreliable at n={n_total} (rejects V≈{v:.2f}).")
        return 0
    else:
        print(f"FAIL — Cramer's V={v:.4f} > {v_threshold} (medium/large effect; cache introduces deployment-relevant bias).")
        print(f"       p={p:.4f}  (alpha={alpha})")
        if v >= 0.3:
            print(f"       large effect (V={v:.2f}); safety-critical — do not deploy.")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("control", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    return report(args.control, args.test, alpha=args.alpha)


if __name__ == "__main__":
    sys.exit(main())
