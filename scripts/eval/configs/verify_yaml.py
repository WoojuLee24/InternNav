#!/usr/bin/env python
"""Verify the {full,mini} x {ld30,ld60} R2R eval yamls.

Every yaml under scripts/eval/configs/ is a standalone copy of the others, so the risk is
that they silently drift apart (that is exactly how vln_r2r.yaml ended up pointing at a
non-existent scenes_dir, and how vln_rxr.yaml / objectnav_hm3d.yaml stopped composing at
all during the habitat 0.3.x port). This script pins down what must stay equal.

Checks:
  [legacy]  vln_r2r{,_mini,_mini_5090}.yaml still compose (they must not be touched)
  [parity]  vln_r2r_mini_ld60.yaml == vln_r2r_mini.yaml, key for key
  [axes]    full_ld30 vs full_ld60 differ ONLY in tilt_angle
            full_ld30 vs mini_ld30 differ ONLY in scenes_dir / data_path
  [paths]   each variant's scenes_dir and data_path actually resolve on disk
  [episodes] each variant loads 1839 episodes / 11 scenes via the real dataset class

    python scripts/eval/configs/verify_yaml.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# This directory contains a `habitat/` subdir (holding .py eval configs). Python puts the
# script's own dir first on sys.path, so it would shadow the real habitat package as a
# namespace package -> "ImportError: cannot import name 'VectorEnv' from 'habitat'".
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
CFG = "scripts/eval/configs"

LEGACY = ["vln_r2r.yaml", "vln_r2r_mini.yaml", "vln_r2r_mini_5090.yaml"]
VARIANTS = ["vln_r2r_full_ld30.yaml", "vln_r2r_full_ld60.yaml",
            "vln_r2r_mini_ld30.yaml", "vln_r2r_mini_ld60.yaml"]

# habitat_vln_evaluator.py steps LOOKDOWN twice (and LOOKUP twice to restore)
LOOKDOWN_STEPS = 2
EXPECTED_EPISODES, EXPECTED_SCENES = 1839, 11
PROBE_SCENE = "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"  # from val_unseen.json.gz episodes[0]

TILT_KEYS = {"habitat.task.actions.look_up.tilt_angle",
             "habitat.task.actions.look_down.tilt_angle"}
PATH_KEYS = {"habitat.dataset.scenes_dir", "habitat.dataset.data_path"}


def flat(cfg):
    from omegaconf import OmegaConf
    out = {}

    def rec(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                rec(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                rec(v, f"{prefix}[{i}]")
        else:
            out[prefix] = node

    rec(OmegaConf.to_container(cfg, resolve=True))
    return out


def diff_keys(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k, "<missing>") != b.get(k, "<missing>"))


def main():
    os.chdir(REPO_ROOT)  # yaml paths are relative to the repo root
    from habitat_baselines.config.default import get_config as get_habitat_config

    ok = True
    cfgs, raw = {}, {}
    for name in LEGACY + VARIANTS:
        try:
            raw[name] = get_habitat_config(f"{CFG}/{name}")
            cfgs[name] = flat(raw[name])
        except Exception as e:
            ok = False
            print(f"[compose] MISMATCH {name}: {type(e).__name__}: {str(e).splitlines()[0]}")

    # ---------------------------------------------------------------- legacy
    print("[legacy] untouched yamls still compose")
    for name in LEGACY:
        n = len(cfgs.get(name, {}))
        print(f"    [{'OK' if n else 'MISMATCH'}] {name}: {n} keys")
        ok = ok and bool(n)

    # ---------------------------------------------------------------- parity
    print("\n[parity] vln_r2r_mini_ld60.yaml == vln_r2r_mini.yaml (legacy reproduced exactly)")
    d = diff_keys(cfgs["vln_r2r_mini_ld60.yaml"], cfgs["vln_r2r_mini.yaml"])
    print(f"    [{'OK' if not d else 'MISMATCH'}] {len(d)} differing keys")
    for k in d:
        print(f"        {k}: new={cfgs['vln_r2r_mini_ld60.yaml'].get(k)!r}  "
              f"legacy={cfgs['vln_r2r_mini.yaml'].get(k)!r}")
    ok = ok and not d

    # ---------------------------------------------------------------- axes
    print("\n[axes] each pair differs on exactly one axis")
    for a, b, expect, label in [
        ("vln_r2r_full_ld30.yaml", "vln_r2r_full_ld60.yaml", TILT_KEYS, "look-down only"),
        ("vln_r2r_mini_ld30.yaml", "vln_r2r_mini_ld60.yaml", TILT_KEYS, "look-down only"),
        ("vln_r2r_full_ld30.yaml", "vln_r2r_mini_ld30.yaml", PATH_KEYS, "data root only"),
        ("vln_r2r_full_ld60.yaml", "vln_r2r_mini_ld60.yaml", PATH_KEYS, "data root only"),
    ]:
        d = set(diff_keys(cfgs[a], cfgs[b]))
        good = d == expect
        print(f"    [{'OK' if good else 'MISMATCH'}] {a} vs {b}: {label} "
              f"({len(d)} keys: {', '.join(sorted(k.split('.')[-1] for k in d))})")
        for k in sorted(d - expect):
            print(f"        unexpected {k}: {cfgs[a].get(k)!r} vs {cfgs[b].get(k)!r}")
        for k in sorted(expect - d):
            print(f"        expected to differ but identical: {k}")
        ok = ok and good

    # ---------------------------------------------------------------- paths + look-down
    print("\n[paths] scenes_dir / data_path resolve on disk; effective look-down")
    for name in VARIANTS:
        c = cfgs[name]
        scenes_dir = c["habitat.dataset.scenes_dir"]
        data_path = c["habitat.dataset.data_path"].format(split=c["habitat.dataset.split"])
        glb = os.path.join(scenes_dir, PROBE_SCENE)
        tilt = c["habitat.task.actions.look_down.tilt_angle"]
        good = os.path.isfile(glb) and os.path.isfile(data_path)
        print(f"    [{'OK' if good else 'MISMATCH'}] {name:<26} look-down "
              f"{tilt}x{LOOKDOWN_STEPS}={tilt * LOOKDOWN_STEPS}deg")
        if not os.path.isfile(glb):
            print(f"        scenes_dir wrong: {glb} does not exist")
        if not os.path.isfile(data_path):
            print(f"        data_path wrong: {data_path} does not exist")
        ok = ok and good

    # ---------------------------------------------------------------- episodes
    print(f"\n[episodes] real dataset load == {EXPECTED_EPISODES} episodes / {EXPECTED_SCENES} scenes")
    from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1
    for name in VARIANTS:
        ds = VLNDatasetV1(raw[name].habitat.dataset)
        scenes = {e.scene_id for e in ds.episodes}
        good = len(ds.episodes) == EXPECTED_EPISODES and len(scenes) == EXPECTED_SCENES
        print(f"    [{'OK' if good else 'MISMATCH'}] {name:<26} "
              f"{len(ds.episodes)} episodes / {len(scenes)} scenes")
        ok = ok and good

    print("\n" + ("ALL YAML CHECKS PASS ✅" if ok else "YAML CHECK FAILED ❌"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
