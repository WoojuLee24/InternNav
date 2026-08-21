"""DualVLN paper reproduction — Stage 2 initialized from a LOCALLY TRAINED stage1 checkpoint.

Same recipe as `dualvln_stage2_full.py` (PARAMS imported verbatim); only `system2_ckpt` differs.
Stage1 ckpt resolution at import: STAGE1_CKPT env var > newest
`dualvln_stage1_internvla-n1-system2_*` under checkpoints_root > FileNotFoundError.

    # train + auto-eval (--no-eval to train only)
    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200

    # evaluate an existing checkpoint
    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-train --model-path <checkpoints_root>/baseline/dualvln_stage2_from_stage1_<TIMESTAMP>
"""

import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import dualvln_stage2_full (eval.py doesn't add this dir)

import default_config as base  # noqa: E402
from dualvln_stage2_full import FULL_VLN_CE, PARAMS as _STAGE2_PARAMS  # noqa: E402

_RELEASED_SYSTEM2 = base.TRAIN_MACHINE["h200"]["system2_ckpt"]  # InternVLA-N1-System2 (released)
_STAGE1_GLOB = f"{base.TRAIN_MACHINE['h200']['checkpoints_root']}/baseline/dualvln_stage1_internvla-n1-system2_*"

STAGE1_CKPT = os.environ.get("STAGE1_CKPT") or max(glob.glob(_STAGE1_GLOB), default=None)
if not STAGE1_CKPT or not os.path.isfile(os.path.join(STAGE1_CKPT, "config.json")):
    raise FileNotFoundError(
        f"No usable stage1 checkpoint (looked at STAGE1_CKPT env var, then newest {_STAGE1_GLOB}; "
        f"got {STAGE1_CKPT!r}, which lacks a top-level config.json). Train one first:\n"
        "  python scripts/train_eval/qwenvl_train/runner.py "
        "--config scripts/train_eval/qwenvl_train/baseline/dualvln_stage1_full.py --machine h200 --no-eval\n"
        "(or set STAGE1_CKPT to a specific stage1 output dir / intermediate checkpoint-<step>)"
    )

# legacy safety: stage1 outputs saved before SaveAuxFilesCallback lack chat_template.json
_CHAT_TEMPLATE = os.path.join(STAGE1_CKPT, "chat_template.json")
if not os.path.isfile(_CHAT_TEMPLATE):
    shutil.copy(os.path.join(_RELEASED_SYSTEM2, "chat_template.json"), _CHAT_TEMPLATE)

TRAIN_MACHINE = {
    "h200": {**base.TRAIN_MACHINE["h200"], "data_root": FULL_VLN_CE, "system2_ckpt": STAGE1_CKPT},
}

EXP_NAME = "baseline/dualvln_stage2_from_stage1"

PARAMS = _STAGE2_PARAMS  # same recipe; system2_ckpt comes from TRAIN_MACHINE above

eval_cfg = base.make_eval_cfg(PARAMS)  # inherits eval_config_path=vln_r2r_full_ld30.yaml
