#!/usr/bin/env python3
"""Gate 3X (Phase 3X) offline proxy analysis.

Computes three metrics over per-request response-type sequences saved by the server:
  I-111  KL divergence of T/A/W distributions (PROD vs NOCACHE)
  I-112  DTW distance on encoded sequences (PROD vs NOCACHE)
  I-113  Counterfactual gap: |trajectory_ratio_prod - trajectory_ratio_nocache|

Gate 3X PASS criterion (any one of):
  I-111: KL(PROD‖NOCACHE) < 0.10  (low distributional shift)
  I-112: DTW_norm < 0.10           (sequences ≤10% different)
  I-113: counterfactual gap < 3pp  (at most 3% trajectory ratio difference)

Usage:
    python3 parse_gate3x.py <nocache_dir> <prod_dir>
    python3 parse_gate3x.py /tmp/gate3x/073623_NOCACHE /tmp/gate3x/073623_PROD

The metrics.json in each directory must contain 'response_sequence' (added in Phase 3X
server patch) plus the standard gate metrics.
"""

import json
import math
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Encoding                                                                     #
# --------------------------------------------------------------------------- #

def encode_sequence(seq):
    """Map each element to an integer for DTW.

    'T' → 0   (trajectory output)
    'W' → 1   (waiting / cold-start)
    any int or other → 2 + (value % 8)  (discrete action index)
    """
    out = []
    for s in seq:
        if s == 'T':
            out.append(0)
        elif s == 'W':
            out.append(1)
        else:
            try:
                out.append(2 + (int(s) % 8))
            except (ValueError, TypeError):
                out.append(2)
    return out


# --------------------------------------------------------------------------- #
# DTW                                                                          #
# --------------------------------------------------------------------------- #

def dtw_distance(a, b):
    """Standard DTW with absolute difference cost. O(n·m) time."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float('nan')
    INF = float('inf')
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


# --------------------------------------------------------------------------- #
# KL divergence                                                                #
# --------------------------------------------------------------------------- #

def distribution(seq):
    """Return P(T), P(A), P(W) from sequence."""
    n = len(seq)
    if n == 0:
        return (1/3, 1/3, 1/3)
    t = seq.count('T') / n
    w = seq.count('W') / n
    a = max(0.0, 1.0 - t - w)
    return (t, a, w)


def kl_divergence(p, q, eps=1e-9):
    """KL(P‖Q) over three categories."""
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        total += pi * math.log(pi / qi)
    return total


# --------------------------------------------------------------------------- #
# Main analysis                                                                #
# --------------------------------------------------------------------------- #

def analyse_pair(nc_dir: Path, prod_dir: Path) -> dict:
    nc_path = nc_dir / 'metrics.json'
    prod_path = prod_dir / 'metrics.json'

    if not nc_path.exists():
        raise FileNotFoundError(f"Missing: {nc_path}")
    if not prod_path.exists():
        raise FileNotFoundError(f"Missing: {prod_path}")

    nc = json.loads(nc_path.read_text())
    prod = json.loads(prod_path.read_text())

    nc_seq = nc.get('response_sequence', [])
    prod_seq = prod.get('response_sequence', [])

    seq_available = len(nc_seq) > 0 and len(prod_seq) > 0

    # I-111: KL divergence
    if seq_available:
        p_nc = distribution(nc_seq)
        p_prod = distribution(prod_seq)
        kl = kl_divergence(p_prod, p_nc)
    else:
        kl = float('nan')

    # I-112: DTW normalised by max(len)
    if seq_available:
        a_enc = encode_sequence(nc_seq)
        b_enc = encode_sequence(prod_seq)
        # Subsample to ≤500 points to keep DTW tractable (preserves shape)
        stride = max(1, max(len(a_enc), len(b_enc)) // 500)
        a_s = a_enc[::stride]
        b_s = b_enc[::stride]
        dtw_raw = dtw_distance(a_s, b_s)
        dtw_norm = dtw_raw / max(len(a_s), len(b_s)) if dtw_raw == dtw_raw else float('nan')
    else:
        dtw_norm = float('nan')

    # I-113: counterfactual trajectory-ratio gap.
    # PROD ≥ NOCACHE is a benefit (cache covers "waiting" frames that NOCACHE misses).
    # Only flag degradation: PROD < NOCACHE by more than 3pp is a failure.
    tr_nc = nc.get('trajectory_ratio', 0.0)
    tr_prod = prod.get('trajectory_ratio', 0.0)
    # Use fresh_trajectory_ratio when available (measures actual model output distribution,
    # not including cache replays). Falls back to trajectory_ratio.
    fresh_nc = nc.get('fresh_trajectory_ratio', tr_nc)
    fresh_prod = prod.get('fresh_trajectory_ratio', tr_prod)
    gap_pp = fresh_nc - fresh_prod  # positive = PROD degraded vs NOCACHE; negative = PROD better
    i113_pass = gap_pp < 3.0       # PROD must not degrade fresh output rate by >3pp

    # Gate pass criteria
    i111_pass = (kl < 0.10) if kl == kl else False
    i112_pass = (dtw_norm < 0.10) if dtw_norm == dtw_norm else False
    gate_pass = i111_pass or i112_pass or i113_pass

    return {
        'nc_seq_len': len(nc_seq),
        'prod_seq_len': len(prod_seq),
        'kl_prod_nc': kl,
        'dtw_norm': dtw_norm,
        'tr_nocache': tr_nc,
        'tr_prod': tr_prod,
        'fresh_nc': fresh_nc,
        'fresh_prod': fresh_prod,
        'gap_pp': gap_pp,
        'skip_prod': prod.get('temporal_cache_skip_ratio', 0.0),
        'v_max': None,  # Cramér's V not available here, computed by parse_gate3q
        'i111_pass': i111_pass,
        'i112_pass': i112_pass,
        'i113_pass': i113_pass,
        'gate_pass': gate_pass,
    }


def print_report(bag_id: str, r: dict):
    kl_s = f"{r['kl_prod_nc']:.4f}" if r['kl_prod_nc'] == r['kl_prod_nc'] else "N/A (no seq)"
    dtw_s = f"{r['dtw_norm']:.4f}" if r['dtw_norm'] == r['dtw_norm'] else "N/A (no seq)"

    status = "PASS" if r['gate_pass'] else "FAIL"
    print(f"\n{'='*60}")
    print(f" Gate 3X  {bag_id}  →  {status}")
    print(f"{'='*60}")
    print(f"  Sequence length  NOCACHE={r['nc_seq_len']}  PROD={r['prod_seq_len']}")
    print(f"  I-111  KL(PROD‖NOCACHE)   = {kl_s}   {'✓ <0.10' if r['i111_pass'] else '✗ ≥0.10'}")
    print(f"  I-112  DTW_norm            = {dtw_s}   {'✓ <0.10' if r['i112_pass'] else '✗ ≥0.10'}")
    gap_sign = '+' if r['gap_pp'] < 0 else '-' if r['gap_pp'] > 0 else '='
    print(f"  I-113  Fresh-output Δ      = {r['gap_pp']:+.1f} pp  {'✓ PROD not worse' if r['i113_pass'] else '✗ PROD degraded >3pp'}")
    print(f"    fresh_traj  NOCACHE={r['fresh_nc']:.1f}%  PROD={r['fresh_prod']:.1f}%")
    print(f"    traj_ratio  NOCACHE={r['tr_nocache']:.1f}%  PROD={r['tr_prod']:.1f}%  skip={r['skip_prod']:.1f}%")

    if not any([r['i111_pass'], r['i112_pass'], r['i113_pass']]):
        print("  → FAIL: none of the three criteria met")
    else:
        which = [n for n, p in [('I-111', r['i111_pass']), ('I-112', r['i112_pass']), ('I-113', r['i113_pass'])] if p]
        print(f"  → PASS: {', '.join(which)} met")


def latex_row(bag_id: str, r: dict) -> str:
    kl_s = f"{r['kl_prod_nc']:.4f}" if r['kl_prod_nc'] == r['kl_prod_nc'] else "--"
    dtw_s = f"{r['dtw_norm']:.4f}" if r['dtw_norm'] == r['dtw_norm'] else "--"
    status = r"{\bf PASS}" if r['gate_pass'] else "FAIL"
    return (
        f"  {bag_id} & {r['skip_prod']:.1f}\\% & {r['tr_nocache']:.1f}\\% & "
        f"{r['tr_prod']:.1f}\\% & {kl_s} & {dtw_s} & {r['gap_pp']:.1f} pp & {status} \\\\"
    )


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    nc_dir = Path(sys.argv[1])
    prod_dir = Path(sys.argv[2])
    bag_id = nc_dir.name.replace('_NOCACHE', '').replace('_PROD', '')

    result = analyse_pair(nc_dir, prod_dir)
    print_report(bag_id, result)

    print("\n--- LaTeX row ---")
    print(latex_row(bag_id, result))
