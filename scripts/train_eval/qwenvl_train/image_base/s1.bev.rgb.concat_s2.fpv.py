"""S1 view=bev, type=rgb, mode=raw, combine=concat (S2 stays fpv no-op).

BEV is fed as a 3rd image slot alongside [goal, cur] (not a substitution — that's
'replace'), folded into the token dim by internvla_n1.py's _encode_s1_memory_tokens.
MemoryEncoder.max_len widens 512->768 automatically (see internvla_n1_arch.py's
_s1_bev_concat) since S1 is trained from scratch against system2_ckpt in this repo's
training flow — no checkpoint-compatibility handling needed. See
.claude/tasks/ for the implementation writeup.

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine 5090 --no-eval --max-steps 3 --debug-dir logs/input_v0.1/image_base/s1.bev.rgb.concat_s2.fpv
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "image_base/s1.bev.rgb.concat_s2.fpv"
PARAMS = replace(
    base.PARAMS,
    image_provider=True,
    s1_image_view='bev',
    s1_image_type='rgb',
    s1_image_mode='raw',
    s1_combine_mode='concat',
    bev_depth_source='gt',
)
eval_cfg = base.make_eval_cfg(PARAMS)
