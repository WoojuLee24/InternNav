"""BEV-provider trainer — training BEV via the inference-side module.

Baseline ``internvla_n1_trainer.py`` and the legacy ``internvla_n1_bev_trainer.py``
are NOT modified. Same monkey-patch precedent as the legacy file: swap the
module-level model class so ``train()`` instantiates the provider-based model.

Adds ``--bev_*`` flags WITHOUT editing internvla_n1_argument.py: they are
pre-parsed and stripped from sys.argv, then stashed onto the model via
``set_pending_bev_settings`` so they reach ``self.config`` (and the saved
checkpoint's config.json) at construction time.

Usage (see scripts/train/qwenvl_train/bev/train_bev_provider_debug.sh):
    python internnav/trainer/internvla_n1_bev_provider_trainer.py \
        --bev_s1_mode bev --bev_image_type rgb --bev_depth_source gt \
        --model_name_or_path checkpoints/InternVLA-N1-System2 ...
"""

import sys

import internnav.trainer.internvla_n1_trainer as _base
from internnav.model.basemodel.internvla_n1.internvla_n1_bev_provider import (
    InternVLAN1BEVProviderForCausalLM,
    parse_bev_cli_args,
    set_pending_bev_settings,
)


def main():
    # 1. peel off --bev_* flags so the base HfArgumentParser never sees them
    bev_settings, remaining = parse_bev_cli_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    # 2. make the settings reachable from the model __init__
    set_pending_bev_settings(bev_settings)

    # 3. redirect the base trainer's model class (bev_mode stays 'none', so
    #    train() takes the InternVLAN1ForCausalLM branch — which we override)
    _base.InternVLAN1ForCausalLM = InternVLAN1BEVProviderForCausalLM

    print(f"[bev_provider_trainer] BEV settings: {bev_settings}", flush=True)
    _base.train()


if __name__ == '__main__':
    main()
