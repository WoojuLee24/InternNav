#!/usr/bin/env python
"""Verify the BEV Python configs produce the SAME trainer flags as the original
shell scripts in scripts/train/qwenvl_train/bev/, and that eval BEV modes are
train-synced. Reuses the shell-parsing helpers from batch_size2/verify_params.py.

    python scripts/train_eval/qwenvl_train/bev/verify_params.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QWEN = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, QWEN)
sys.path.insert(0, os.path.join(QWEN, "batch_size2"))

import runner  # noqa: E402
import verify_params as V  # noqa: E402  (batch_size2/verify_params.py — shell parsing helpers)

SH_DIR = os.path.join(runner.REPO_ROOT, "scripts", "train", "qwenvl_train", "bev")
BASE_TRAINER = "internnav/trainer/internvla_n1_trainer.py"
BEV_TRAINER = "internnav/trainer/internvla_n1_bev_provider_trainer.py"

# (python config, shell script, expected eval bev modes)
VARIANTS = [
    ("bev_s1_bev.py", "bev_s1_bev.sh"),
    ("bev_s1_fpv_base.py", "bev_s1_fpv_base.sh"),
]


def _sh_flags(sh_path):
    text = open(sh_path).read()
    vars_ = V._parse_assignments(text)
    # V._parse_trainer_flags anchors on the base trainer path; the bev scripts launch
    # the provider trainer, so alias it so the same parser finds the arg block.
    text = text.replace(BEV_TRAINER, BASE_TRAINER)
    return V._parse_trainer_flags(text, vars_)


def main():
    all_ok = True
    for py_name, sh_name in VARIANTS:
        mod, _ = runner._load_config(os.path.join(HERE, py_name))
        p = mod.PARAMS

        # (a) trainer flags (incl. --bev_*) match the shell script
        sh_flags = _sh_flags(os.path.join(SH_DIR, sh_name))
        py_flags = V._tokens_to_dict(p.train_argv("<output_dir>", "<run_name>"))
        diffs = V._compare(sh_name, sh_flags, py_flags)
        print(f"[{'OK' if not diffs else 'MISMATCH'}] {sh_name}  (bev_s1_mode={p.bev_s1_mode}, "
              f"trainer={'provider' if p.bev else 'base'})")
        for k, a, b in diffs:
            print(f"    {k}: shell={a!r}  python={b!r}")
            all_ok = False

        # (b) provider trainer selected
        if p.trainer != BEV_TRAINER:
            print(f"    trainer: expected {BEV_TRAINER}, got {p.trainer}")
            all_ok = False

        # (c) eval BEV modes are train-synced (habitat_vln_bev + bev_s1 == train mode)
        ms = mod.eval_cfg.agent.model_settings
        checks = {
            "eval_type": (mod.eval_cfg.eval_type, "habitat_vln_bev"),
            "visual_provider": (ms.get("visual_provider"), "bev_image"),
            "bev_s1_mode": (ms.get("bev_s1_mode"), p.bev_s1_mode),
            "bev_s2_mode": (ms.get("bev_s2_mode"), p.bev_s2_mode_eval),
        }
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        print(f"    [{'OK' if not bad else 'MISMATCH'}] eval-sync: " +
              ", ".join(f"{k}={checks[k][0]}" for k in checks))
        if bad:
            all_ok = False

    print("\n" + ("ALL PARAMETERS MATCH ✅" if all_ok else "PARAMETER MISMATCH ❌"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
