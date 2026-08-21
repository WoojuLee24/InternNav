#!/usr/bin/env python
"""Verify baseline/dualvln_stage1_full.py reproduces main's train_system2.sh.

Three checks (no eval checks: stage1 is System-2-only and always runs --no-eval):
  [train-argv]  generated trainer flags vs `git show main:.../train_system2.sh`
  [eff-batch]   64 GPUs x 2 x 1 (main) == 8 GPUs x 16 x 1 (ours)
  [dataset]     every dataset-setting in vln_datasets resolves to real data on disk

Exits non-zero on any unexplained difference.

    python scripts/train_eval/qwenvl_train/baseline/verify_params_stage1.py
"""

import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import runner  # noqa: E402
from verify_params import _norm, _resolve, _tokens_to_dict  # noqa: E402

REPO_ROOT = runner.REPO_ROOT
MAIN_SH = "scripts/train/qwenvl_train/train_system2.sh"

# Flags whose difference from main is intended, with the reason. Anything NOT listed here and
# not equal is a real mismatch.
EXPECTED_DIFFS = {
    "per_device_train_batch_size": "8 GPUs x 4 x accum 4 == 64 GPUs x 2 -> same effective batch 128",
    "gradient_accumulation_steps": "4 (with batch 4) == main's 1 (with batch 2 x 64 GPUs); full-VLM finetune OOMs above batch 4/GPU on H200",
    "per_device_eval_batch_size": "derived (batch_size*2); inert, eval_strategy=no",
    "data_root": "main relies on CWD; runner passes the dataset root explicitly",
    "output_dir": "timestamped run dir instead of a fixed one",
    "run_name": "timestamped run name instead of a fixed one",
    "eval_steps": "emitted unconditionally by train_argv; inert under eval_strategy=no",
    "metric_for_best_model": "emitted unconditionally; inert under load_best_model_at_end=False",
    "greater_is_better": "emitted unconditionally; inert under load_best_model_at_end=False",
    "load_best_model_at_end": "explicit False == HF default == main (which omits the flag)",
    "save_only_model": "explicit False == HF default == main (which omits the flag)",
    "val_ratio": "explicit 0.0 == trainer default == main's no-validation setup",
}


def _parse_main_sh():
    """Trainer flags from main's train_system2.sh, with ${var}/$((expr)) resolved."""
    text = subprocess.run(["git", "show", f"main:{MAIN_SH}"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout

    vars_ = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0]  # `vln_datasets=... #,scalevln_...` — strip the comment
        m = re.match(r"^([A-Za-z_]\w*)=(.*)$", line)
        if m:
            vars_[m.group(1)] = _resolve(m.group(2).strip(), vars_)

    block = text.split("internnav/trainer/internvla_n1_trainer.py", 1)[1]
    block = block.replace("\\\n", " ")
    return _tokens_to_dict(shlex.split(_resolve(block, vars_)))


def main():
    ok = True
    mod, _ = runner._load_config(os.path.join(HERE, "dualvln_stage1_full.py"))
    # runner applies the config's own TRAIN_MACHINE["h200"] on top of PARAMS; do the same here
    from dataclasses import replace
    m = mod.TRAIN_MACHINE["h200"]
    p = replace(mod.PARAMS, data_root=m["data_root"], system2_ckpt=m["system2_ckpt"],
                nproc_per_node=m["nproc_per_node"])

    # ---------------------------------------------------------------- train-argv
    print(f"[train-argv] generated flags vs main:{MAIN_SH}")
    sh = _parse_main_sh()
    py = _tokens_to_dict(p.train_argv("<output_dir>", "<run_name>"))

    same, explained, bad = [], [], []
    for k in sorted(set(sh) | set(py)):
        a, b = sh.get(k, "<missing>"), py.get(k, "<missing>")
        if a != "<missing>" and b != "<missing>" and _norm(k, a) == _norm(k, b):
            same.append(k)
        elif k in EXPECTED_DIFFS:
            explained.append((k, a, b))
        else:
            bad.append((k, a, b))

    print(f"    [OK] {len(same)} flags identical: {', '.join(same)}")
    print(f"    [intended] {len(explained)} flags differ by design:")
    for k, a, b in explained:
        print(f"        {k}: main={a!r}  generated={b!r}   # {EXPECTED_DIFFS[k]}")
    if bad:
        ok = False
        print(f"    [MISMATCH] {len(bad)} unexplained:")
        for k, a, b in bad:
            print(f"        {k}: main={a!r}  generated={b!r}")

    # effective batch must match main's 64 GPUs x 2 x 1
    eff_main = int(sh["per_device_train_batch_size"]) * 64 * int(sh["gradient_accumulation_steps"])
    eff_py = p.batch_size * p.nproc_per_node * p.grad_accum_steps
    print(f"    [{'OK' if eff_main == eff_py else 'MISMATCH'}] effective batch: "
          f"main={eff_main} (64 GPU x {sh['per_device_train_batch_size']}), "
          f"generated={eff_py} ({p.nproc_per_node} GPU x {p.batch_size} x {p.grad_accum_steps})")
    ok = ok and eff_main == eff_py

    # stage2 loader routing: the output dir name must hit the InternVLAN1 branch
    routed = "internvla-n1-system2" in mod.EXP_NAME.lower()
    print(f"    [{'OK' if routed else 'MISMATCH'}] EXP_NAME contains 'internvla-n1-system2' "
          f"(stage2 loader routing): {mod.EXP_NAME}")
    ok = ok and routed

    # ---------------------------------------------------------------- dataset
    print("\n[dataset] every vln_datasets entry resolves to real data")
    sys.path.insert(0, REPO_ROOT)
    from internnav.dataset.internvla_n1_lerobot_dataset import data_list  # noqa: E402
    for name, cfg in zip(p.vln_datasets.split(","), data_list(p.vln_datasets.split(","))):
        d = os.path.join(p.data_root, cfg["data_path"])
        setting = f"{cfg['height']}cm_{cfg['pitch_2']}deg"   # == the dataset loader's key
        scenes = sorted(os.listdir(d)) if os.path.isdir(d) else []
        status, detail = "MISSING", f"{d} not found"
        if scenes:
            info = os.path.join(d, scenes[0], "meta", "info.json")
            feats = json.load(open(info))["features"] if os.path.exists(info) else {}
            has = f"pose.{setting}" in feats
            status = "OK" if has else "MISSING"
            detail = (f"{len(scenes)} scenes, pose.{setting} "
                      f"{'present' if has else 'ABSENT in ' + scenes[0]}")
        print(f"    [{status}] {name:<30} sampling={cfg['sampling_rate']:.2f}  {detail}")
        ok = ok and status == "OK"

    print("\n" + ("ALL PARAMETERS MATCH ✅" if ok else "PARAMETER MISMATCH ❌"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
