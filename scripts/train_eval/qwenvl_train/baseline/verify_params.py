#!/usr/bin/env python
"""Verify baseline/dualvln_stage2_full.py reproduces main's train_dual_system.sh.

Four checks:
  [train-argv]  generated trainer flags vs `git show main:.../train_dual_system.sh`
  [eval-sync]   train PARAMS == eval model_settings (the train==eval single-source guarantee)
  [eval-yaml]   the eval yaml exists and its look_up/look_down tilt_angle is what we think
  [dataset]     every dataset-setting in vln_datasets resolves to real data on disk

Exits non-zero on any unexplained difference.

    python scripts/train_eval/qwenvl_train/baseline/verify_params.py
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

import default_config as base  # noqa: E402
import runner  # noqa: E402

REPO_ROOT = runner.REPO_ROOT
MAIN_SH = "scripts/train/qwenvl_train/train_dual_system.sh"

# Flags whose difference from main is intended, with the reason. Anything NOT listed here and
# not equal is a real mismatch.
EXPECTED_DIFFS = {
    "per_device_train_batch_size": "8 GPUs x 16 == 64 GPUs x 2 -> same effective batch 128",
    "per_device_eval_batch_size": "derived (batch_size*2); inert, val_ratio=0 means no eval dataset",
    "model_name_or_path": "absolute path to the same InternVLA-N1-System2 checkpoint",
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

IGNORE = set()  # nothing silently ignored; every key is classified


def _tokens_to_dict(tokens):
    out, i = {}, 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            i += 1
            continue
        key = t[2:]
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[key] = tokens[i + 1]
            i += 2
        else:
            out[key] = True  # bare flag, e.g. --bf16
            i += 1
    return out


def _parse_main_sh():
    """Trainer flags from main's train_dual_system.sh, with ${var}/$((expr)) resolved."""
    text = subprocess.run(["git", "show", f"main:{MAIN_SH}"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout

    vars_ = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_]\w*)=(.*)$", line)  # no leading space => top-level assignment
        if m:
            vars_[m.group(1)] = _resolve(m.group(2).strip(), vars_)

    block = text.split("internnav/trainer/internvla_n1_trainer.py", 1)[1]
    block = block.replace("\\\n", " ")
    return _tokens_to_dict(shlex.split(_resolve(block, vars_)))


def _resolve(s, vars_):
    def _arith(m):
        ns = {k: int(v) for k, v in vars_.items() if re.fullmatch(r"-?\d+", str(v))}
        try:
            return str(int(eval(m.group(1), {"__builtins__": {}}, ns)))  # noqa: S307
        except Exception:
            return m.group(0)  # launcher-only (e.g. RANDOM); not a trainer flag

    s = re.sub(r"\$\(\((.*?)\)\)", _arith, s)
    s = re.sub(r"\$\{(\w+)\}", lambda m: str(vars_.get(m.group(1), m.group(0))), s)
    s = re.sub(r"\$(\w+)", lambda m: str(vars_.get(m.group(1), m.group(0))), s)
    return s


def _norm(key, val):
    if val is True:
        return True
    if key == "lr_scheduler_kwargs":
        return json.loads(val)
    try:
        return float(val)  # 1e-4 == 0.0001
    except (TypeError, ValueError):
        return val


def main():
    ok = True
    mod, _ = runner._load_config(os.path.join(HERE, "dualvln_stage2_full.py"))
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
        if k in IGNORE:
            continue
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

    # ---------------------------------------------------------------- eval-sync
    print("\n[eval-sync] train PARAMS == eval model_settings (single source)")
    ms = base.build_eval_cfg(p, "h200").agent.model_settings
    checks = {"num_history": p.num_history, "resize_w": p.resize_w,
              "resize_h": p.resize_h, "predict_step_num": p.predict_step_num}
    bad_sync = {k: (ms.get(k), v) for k, v in checks.items() if ms.get(k) != v}
    print(f"    [{'OK' if not bad_sync else 'MISMATCH'}] " +
          ", ".join(f"{k}={ms.get(k)}" for k in checks))
    for k, (got, want) in bad_sync.items():
        print(f"        {k}: eval={got!r}  train={want!r}")
    ok = ok and not bad_sync

    # ---------------------------------------------------------------- eval-yaml
    print("\n[eval-yaml] look-down geometry of the selected eval config")
    yaml_rel = base.build_eval_cfg(p, "h200").env.env_settings["config_path"]
    yaml_abs = os.path.join(REPO_ROOT, yaml_rel)
    if not os.path.exists(yaml_abs):
        ok = False
        print(f"    [MISSING] {yaml_rel}")
    else:
        import yaml as _yaml
        acts = _yaml.safe_load(open(yaml_abs))["habitat"]["task"]["actions"]
        # habitat 0.3.3 LookUp/LookDownActionConfig default tilt_angle is 15
        down = acts.get("look_down", {}).get("tilt_angle", 15)
        up = acts.get("look_up", {}).get("tilt_angle", 15)
        # habitat_vln_evaluator.py steps LOOKDOWN twice (and LOOKUP twice to restore)
        print(f"    [{'OK' if down == up else 'MISMATCH'}] {yaml_rel}: "
              f"look_down={down} look_up={up} -> effective look-down {2 * down} deg "
              f"(main = 30 deg; training data pitch_1->pitch_2 max delta = 30 deg)")
        ok = ok and down == up

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
