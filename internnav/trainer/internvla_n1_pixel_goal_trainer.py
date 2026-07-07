"""pixel_goal trainer: BEV image processing (optional) + pixel-coord S1 conditioning.

Monkey-patches InternVLAN1ForCausalLM with InternVLAN1BEVProviderPixelGoalForCausalLM,
which handles both bev=True (bev_s1_mode != 'fpv') and bev=False (bev_s1_mode='fpv').
--bev_* flags are peeled off before HfArgumentParser sees them (same as bev_provider_trainer).

Usage (via runner.py with use_pixel_goal=True):
    python internnav/trainer/internvla_n1_pixel_goal_trainer.py \
        --bev_s1_mode bev --bev_image_type rgb --bev_depth_source gt \
        --use_pixel_goal True --model_name_or_path ...
"""

import sys

import internnav.trainer.internvla_n1_trainer as _base
from internnav.model.basemodel.internvla_n1.internvla_n1_bev_provider import (
    parse_bev_cli_args,
    set_pending_bev_settings,
)
from internnav.model.basemodel.internvla_n1.internvla_n1_pixel_goal import (
    InternVLAN1BEVProviderPixelGoalForCausalLM,
)


def main():
    bev_settings, remaining = parse_bev_cli_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    set_pending_bev_settings(bev_settings)
    _base.InternVLAN1ForCausalLM = InternVLAN1BEVProviderPixelGoalForCausalLM

    print(f"[pixel_goal_trainer] BEV settings: {bev_settings}", flush=True)
    _base.train()


if __name__ == '__main__':
    main()
