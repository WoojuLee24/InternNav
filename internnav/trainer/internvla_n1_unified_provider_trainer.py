"""Unified image-provider trainer — same monkey-patch precedent as
internvla_n1_bev_provider_trainer.py, for the independent view/type/mode/combine
axes (see unified_image_provider.py). BEV keeps using its own trainer script
unchanged; this one is selected when `image_provider=True` in default_config.py.

Baseline ``internvla_n1_trainer.py`` is NOT modified.

Usage:
    python internnav/trainer/internvla_n1_unified_provider_trainer.py \
        --s1_image_view fpv --s1_image_type depth --s1_image_mode raw --s1_combine_mode replace \
        --model_name_or_path checkpoints/InternVLA-N1-System2 ...
"""

import sys

import internnav.trainer.internvla_n1_trainer as _base
from internnav.model.basemodel.internvla_n1.internvla_n1_unified_provider import (
    InternVLAN1UnifiedProviderForCausalLM,
    parse_unified_cli_args,
    set_pending_unified_settings,
)


def main():
    # 1. peel off --s{1,2}_image_*/--s{1,2}_combine_mode/--bev_depth_* flags so
    #    the base HfArgumentParser never sees them
    unified_settings, remaining = parse_unified_cli_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    # 2. make the settings reachable from the model __init__
    set_pending_unified_settings(unified_settings)

    # 3. redirect the base trainer's model class (bev_mode stays 'none', so
    #    train() takes the InternVLAN1ForCausalLM branch — which we override)
    _base.InternVLAN1ForCausalLM = InternVLAN1UnifiedProviderForCausalLM

    print(f"[unified_provider_trainer] settings: {unified_settings}", flush=True)
    _base.train()


if __name__ == '__main__':
    main()
