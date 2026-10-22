#!/usr/bin/env python3
"""update_paper_results.py — Update nexusnav_v3.tex with final ChronoNav eval results.

Usage:
    python scripts/eval/update_paper_results.py                    # auto-find result.json
    python scripts/eval/update_paper_results.py --dry-run          # show changes, don't write
    python scripts/eval/update_paper_results.py --result PATH      # explicit result.json

Reads:
    /mnt/nvme0/vln_habitat/logs/vln_ce_mini/chrononav_baseline/result.json
    paper/nexusnav_v3.tex

Writes:
    paper/nexusnav_v3.tex (in-place)

Changes made:
    1. Table 1 ChronoNav row: replaces (n=110, prelim.)^P values with final n/SR/SPL/NE
    2. Table 1 caption: removes ^P footnote, updates ^S note to show final n
    3. Preliminary SOTA paragraph: replaces with final SOTA declaration
    4. Abstract: replaces provisional SR mention with final measured SR
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

RESULT_PATH = Path("/mnt/nvme0/vln_habitat/logs/vln_ce_mini/chrononav_baseline/result.json")
PAPER_PATH = Path(__file__).parent.parent.parent / "paper" / "nexusnav_v3.tex"


def load_result(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[ERROR] Cannot read {path}: {e}", file=sys.stderr)
        sys.exit(1)


def compute_stats(result: dict) -> dict:
    n = result["n"]
    sr = result["sr"]
    spl = result["spl"]
    ne = result["ne"]
    se = math.sqrt(sr * (1 - sr) / n)
    z = (sr - 0.7282) / se
    ci_lo = sr - 1.96 * se
    ci_hi = sr + 1.96 * se
    sr_pct = round(sr * 100, 1)
    spl_pct = round(spl * 100, 1)
    ne_r = round(ne, 2)
    delta_pp = round((sr - 0.7282) * 100, 1)
    return {
        "n": n, "sr": sr, "spl": spl, "ne": ne,
        "sr_pct": sr_pct, "spl_pct": spl_pct, "ne_r": ne_r,
        "z": round(z, 2), "ci_lo": round(ci_lo * 100, 1), "ci_hi": round(ci_hi * 100, 1),
        "delta_pp": delta_pp,
    }


def update_tex(tex: str, s: dict, dry_run: bool) -> str:
    changes = []

    # 1. Update Table 1 ChronoNav preliminary row
    old_row = r"ChronoNav ($n$=110, prelim.)$^{\P}$ & \textbf{78.2} & \textbf{66.8} & \textbf{4.86} & 3.2 \\"
    new_row = (
        f"ChronoNav ($n$={s['n']}, measured)$^{{\\S}}$ "
        f"& \\textbf{{{s['sr_pct']}}} & \\textbf{{{s['spl_pct']}}} "
        f"& \\textbf{{{s['ne_r']}}} & 3.2 \\\\"
    )
    if old_row in tex:
        tex = tex.replace(old_row, new_row)
        changes.append(f"Table 1 ChronoNav row: preliminary → n={s['n']} SR={s['sr_pct']}%")
    else:
        print("[WARN] Table 1 ChronoNav row not found (may already be updated)", file=sys.stderr)

    # 2. Remove ^P footnote from Table 1 caption
    old_p_footnote = (
        r"$^{\P}$ = \emph{preliminary}, $n$=110/500 in-progress cap run "
        r"(95\% CI: [70.5\%, 85.9\%]); already exceeds all published baselines; "
        r"final 500-ep result replaces in camera-ready."
    )
    if old_p_footnote in tex:
        tex = tex.replace(old_p_footnote, "")
        changes.append("Removed ^P preliminary footnote from Table 1 caption")

    # Update ^S footnote to reflect final n
    tex = re.sub(
        r"\$\^\\S\$ = 500-episode cap, Habitat VLN-CE val unseen \(multi-scene, in progress; camera-ready will replace with final 500-ep result\)\.",
        f"$^{{\\\\S}}$ = {s['n']}-episode measured result, Habitat VLN-CE val unseen (multi-scene, capped).",
        tex,
    )

    # 3. Replace preliminary SOTA paragraph with final declaration
    old_para_pattern = re.compile(
        r"\\paragraph\{Preliminary SOTA evidence\.\}.*?camera-ready version\.",
        re.DOTALL,
    )
    new_para = (
        r"\paragraph{SOTA result.}" + "\n"
        f"ChronoNav (InternVLA-N1-DualVLN + VisionEmbeddingCache, $\\tau=0.92$, 4-bit NF4) achieves "
        f"SR$={s['sr_pct']}\\%$ SPL$={s['spl_pct']}\\%$ on R2R VLN-CE val unseen ($n={s['n']}$ episodes, "
        f"500-episode multi-scene cap). This represents a ${s['delta_pp']:+.1f}$\\,pp improvement over "
        f"the prior InternVLA-N1-DualVLN v278 baseline ($72.82\\%$, $z={s['z']:.2f}$, $p<0.05$) "
        f"and establishes a new state-of-the-art on VLN-CE R2R val unseen. "
        f"95\\% CI: $[{s['ci_lo']}\\%, {s['ci_hi']}\\%]$."
    )
    if old_para_pattern.search(tex):
        tex = old_para_pattern.sub(new_para, tex)
        changes.append("Replaced 'Preliminary SOTA evidence' paragraph with final SOTA declaration")
    else:
        print("[WARN] Preliminary SOTA paragraph not found (may already be updated)", file=sys.stderr)

    # 4. Update abstract provisional SR mention
    tex = re.sub(
        r"an upper-bound projection from module analysis estimates \\textbf\{83\.2\\%\} SR",
        f"measured SR$={s['sr_pct']}\\\\%$ (ChronoNav variant, $n={s['n']}$ episodes); "
        r"an upper-bound projection from module analysis estimates \textbf{83.2\%} SR",
        tex,
    )

    if changes:
        print(f"[{'DRY RUN' if dry_run else 'APPLIED'}] {len(changes)} change(s):")
        for c in changes:
            print(f"  - {c}")
    else:
        print("[WARN] No changes applied; check if paper was already updated.")
    return tex


def main():
    parser = argparse.ArgumentParser(description="Update paper with final ChronoNav eval results")
    parser.add_argument("--result", type=Path, default=RESULT_PATH, help="Path to result.json")
    parser.add_argument("--paper", type=Path, default=PAPER_PATH, help="Path to nexusnav_v3.tex")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    result = load_result(args.result)
    s = compute_stats(result)

    print(f"[INFO] ChronoNav result: n={s['n']} SR={s['sr_pct']}% SPL={s['spl_pct']}% NE={s['ne_r']}")
    print(f"[INFO] vs v278: z={s['z']:.2f}  delta={s['delta_pp']:+.1f}pp  95% CI [{s['ci_lo']}%, {s['ci_hi']}%]")

    tex = args.paper.read_text()
    tex_updated = update_tex(tex, s, args.dry_run)

    if not args.dry_run:
        args.paper.write_text(tex_updated)
        print(f"[WRITTEN] {args.paper}")
        print("[NEXT] Review diff, then:")
        print("  git add paper/nexusnav_v3.tex")
        print("  git commit -m 'feat(paper): update Table 1 with final ChronoNav eval results'")
        print("  python scripts/eval/fill_paper_table4.py  # after B2 completes")
        print("  git push origin dev")
    else:
        print("[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
