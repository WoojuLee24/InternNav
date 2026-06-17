"""S1-BEV train + Habitat-BEV eval config. == bev/bev_s1_bev.sh (v2 train==eval sync).

BEV pipeline ON: training uses internvla_n1_bev_provider_trainer.py with
--bev_s1_mode bev --bev_image_type rgb --bev_depth_source gt; eval uses the
habitat_vln_bev evaluator with the SAME bev_s1_mode (S2 stays fpv — S2 BEV is not
trained). All other hyperparameters inherit the b4 baseline (eff batch 128).

    # debug on a single 5090 (1 GPU), keep torchrun:
    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --eval-target 5090 --nproc 1
    # eval an existing BEV checkpoint only:
    python .../runner.py --config <this file> --eval-target 5090 --nproc 1 --no-train --model-path <ckpt>
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "bev/s1.bev_s2.fpv_occ.prob_zmin.0.1_gt"
PARAMS = replace(
    base.PARAMS,
    bev=True,
    bev_s1_mode="bev",       # S1 trained with BEV
    bev_s2_mode_eval="fpv",  # eval S2 stays fpv (S2 BEV not trained)
    bev_image_type="occ.prob",
    bev_depth_source="gt",
    bev_z_min=0.1,          # ROS2 standard: exclude floor/ceiling
    bev_z_max=2.0,
)
eval_cfg = base.make_eval_cfg(PARAMS)
