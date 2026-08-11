"""
Build the small evaluation dataset shipped with the quantization kit.

VLN-PE mode (--dataset vln-pe, default):
  Inputs:  `data/InternData-N1-v0.5-mini/vln-pe/traj_data/r2r/<scene>/`
  Outputs: sample_XXXX/{current.jpg, history_*.jpg, instruction.txt, meta.json}

VLN-CE mode (--dataset vln-ce):
  Inputs:  `data/InternData-N1-v0.5-mini/vln-ce/traj_data/r2r/<scene>/`
  Outputs: sample_XXXX/{current.jpg, lookdown.jpg, history_*.jpg, instruction.txt, meta.json}
  - current.jpg  = FPV  (observation.images.rgb.125cm_0deg)
  - lookdown.jpg = look-down (observation.images.rgb.125cm_30deg)
  - Only steps where goal.125cm_30deg is valid (not [-1,-1]) are included.
  - meta.json carries look_down=True and goal_pixel=[y,x].

Usage (VLN-PE):
    python prepare_dataset.py --dataset vln-pe --out_root quantization_kit/data

Usage (VLN-CE):
    python prepare_dataset.py --dataset vln-ce --out_root quantization_kit/data
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_SRC_DEFAULTS = {
    "vln-pe": "data/InternData-N1-v0.5-mini/vln-pe/traj_data/r2r",
    "vln-ce": "data/InternData-N1-v0.5-mini/vln-ce/traj_data/r2r",
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_instructions(scene_dir: Path):
    eps_jsonl = scene_dir / "meta" / "episodes.jsonl"
    out = {}
    with open(eps_jsonl) as f:
        for line in f:
            row = json.loads(line)
            out[int(row["episode_index"])] = row["tasks"]
    return out


def pick_history_indices(current_step: int, num_history: int):
    """Mirror policy.s2_step: np.unique(np.linspace(0, current-1, num_history))."""
    if current_step <= 0:
        return []
    return np.unique(
        np.linspace(0, current_step - 1, num_history, dtype=np.int32)
    ).tolist()


# ---------------------------------------------------------------------------
# VLN-PE helpers
# ---------------------------------------------------------------------------

def list_pe_episodes(scene_dir: Path):
    parquet_dir = scene_dir / "data" / "chunk-000"
    rgb_dir = scene_dir / "videos" / "chunk-000" / "observation.images.rgb"
    if not parquet_dir.exists() or not rgb_dir.exists():
        return []
    out = []
    for pq in sorted(parquet_dir.glob("episode_*.parquet")):
        ep_idx = int(pq.stem.replace("episode_", ""))
        rgb_path = rgb_dir / f"episode_{ep_idx:06d}.npy"
        if rgb_path.exists():
            out.append((ep_idx, pq, rgb_path))
    return out


def save_pe_sample(out_dir: Path, rgb_arr, current_step, history_idx,
                   instruction, meta, jpeg_quality=85):
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_arr[current_step]).convert("RGB").save(
        out_dir / "current.jpg", quality=jpeg_quality)
    for i, h_idx in enumerate(history_idx):
        Image.fromarray(rgb_arr[h_idx]).convert("RGB").save(
            out_dir / f"history_{i}.jpg", quality=jpeg_quality)
    (out_dir / "instruction.txt").write_text(instruction)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# VLN-CE helpers
# ---------------------------------------------------------------------------

def list_ce_episodes(scene_dir: Path):
    """Find VLN-CE episodes that have both FPV and look-down JPEG frames."""
    parquet_dir = scene_dir / "data" / "chunk-000"
    fpv_dir = scene_dir / "videos" / "chunk-000" / "observation.images.rgb.125cm_0deg"
    lookdown_dir = scene_dir / "videos" / "chunk-000" / "observation.images.rgb.125cm_30deg"
    if not all(d.exists() for d in [parquet_dir, fpv_dir, lookdown_dir]):
        return []
    out = []
    for pq in sorted(parquet_dir.glob("episode_*.parquet")):
        ep_idx = int(pq.stem.replace("episode_", ""))
        if (fpv_dir / f"episode_{ep_idx:06d}_0.jpg").exists():
            out.append((ep_idx, pq, fpv_dir, lookdown_dir))
    return out


def load_ce_goal_pixels(parquet_path: Path) -> dict:
    """Return {frame_index: [y, x]} for steps where goal.125cm_30deg is visible."""
    df = pd.read_parquet(parquet_path, columns=["frame_index", "goal.125cm_30deg"])
    result = {}
    for _, row in df.iterrows():
        g = row["goal.125cm_30deg"]
        if int(g[0]) != -1:
            result[int(row["frame_index"])] = [int(g[0]), int(g[1])]
    return result


def save_ce_sample(out_dir: Path, fpv_dir: Path, lookdown_dir: Path,
                   ep_idx: int, cur_step: int, history_idx,
                   instruction: str, meta: dict, jpeg_quality=85):
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.open(fpv_dir / f"episode_{ep_idx:06d}_{cur_step}.jpg").convert("RGB").save(
        out_dir / "current.jpg", quality=jpeg_quality)
    Image.open(lookdown_dir / f"episode_{ep_idx:06d}_{cur_step}.jpg").convert("RGB").save(
        out_dir / "lookdown.jpg", quality=jpeg_quality)
    for i, h_idx in enumerate(history_idx):
        Image.open(fpv_dir / f"episode_{ep_idx:06d}_{h_idx}.jpg").convert("RGB").save(
            out_dir / f"history_{i}.jpg", quality=jpeg_quality)
    (out_dir / "instruction.txt").write_text(instruction)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vln-pe", "vln-ce"], default="vln-pe",
                    help="vln-pe: Isaac Sim LeRobot data; vln-ce: Habitat multi-view JPEG data")
    ap.add_argument("--src_root", default=None,
                    help="Override source root (default depends on --dataset)")
    ap.add_argument("--out_root", default="quantization_kit/data")
    ap.add_argument("--num_scenes", type=int, default=6)
    ap.add_argument("--episodes_per_scene", type=int, default=7)
    ap.add_argument("--samples_per_episode", type=int, default=8)
    ap.add_argument("--num_history", type=int, default=8)
    ap.add_argument("--copy_episode_scenes", type=int, default=4,
                    help="[vln-pe only] how many scenes to copy verbatim into data/episodes/")
    ap.add_argument("--min_current_step", type=int, default=4,
                    help="skip samples whose current step is below this (need history)")
    ap.add_argument("--jpeg_quality", type=int, default=85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_root = Path(args.src_root or _SRC_DEFAULTS[args.dataset]).resolve()
    out_root = Path(args.out_root).resolve() / args.dataset
    samples_root = out_root / "samples"
    episodes_root = out_root / "episodes"
    samples_root.mkdir(parents=True, exist_ok=True)
    episodes_root.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        sys.exit(f"src_root does not exist: {src_root}")

    rng = random.Random(args.seed)

    all_scenes = sorted([p for p in src_root.iterdir() if p.is_dir()])
    if len(all_scenes) < args.num_scenes:
        sys.exit(f"Only {len(all_scenes)} scenes available, requested {args.num_scenes}")
    picked_scenes = rng.sample(all_scenes, args.num_scenes)
    picked_scenes_sorted = sorted(picked_scenes, key=lambda p: p.name)
    print(f"[scenes] picked {len(picked_scenes_sorted)}: {[p.name for p in picked_scenes_sorted]}")

    sample_counter = 0
    manifest = []

    # ------------------------------------------------------------------ VLN-PE
    if args.dataset == "vln-pe":
        for scene_dir in picked_scenes_sorted:
            episodes = list_pe_episodes(scene_dir)
            if not episodes:
                print(f"  [skip] no episodes under {scene_dir.name}")
                continue
            instructions_map = load_instructions(scene_dir)

            n_ep = min(args.episodes_per_scene, len(episodes))
            chosen_eps = rng.sample(episodes, n_ep)
            chosen_eps.sort(key=lambda x: x[0])

            for ep_idx, parquet_path, rgb_path in chosen_eps:
                rgb_arr = np.load(rgb_path)
                n_steps = rgb_arr.shape[0]
                if n_steps <= args.min_current_step:
                    continue
                instr_list = instructions_map.get(ep_idx, [])
                if not instr_list:
                    continue
                instruction = instr_list[0].strip()

                candidate_steps = np.unique(
                    np.linspace(args.min_current_step, n_steps - 1,
                                args.samples_per_episode, dtype=np.int32)
                ).tolist()

                for cur_step in candidate_steps:
                    history_idx = pick_history_indices(cur_step, args.num_history)
                    sample_id = f"sample_{sample_counter:05d}"
                    save_pe_sample(
                        samples_root / sample_id,
                        rgb_arr, cur_step, history_idx, instruction,
                        meta={
                            "sample_id": sample_id,
                            "scene": scene_dir.name,
                            "episode_index": int(ep_idx),
                            "current_step": int(cur_step),
                            "history_indices": [int(h) for h in history_idx],
                            "num_history_provided": len(history_idx),
                            "instruction": instruction,
                            "look_down": False,
                        },
                        jpeg_quality=args.jpeg_quality,
                    )
                    manifest.append({
                        "sample_id": sample_id,
                        "scene": scene_dir.name,
                        "episode_index": int(ep_idx),
                        "current_step": int(cur_step),
                    })
                    sample_counter += 1
            print(f"  [scene] {scene_dir.name}: total samples so far = {sample_counter}")

        if args.copy_episode_scenes > 0:
            copy_scenes_pool = list(picked_scenes_sorted)
            extra_scenes = [p for p in all_scenes if p not in copy_scenes_pool]
            rng.shuffle(extra_scenes)
            copy_scenes_pool += extra_scenes
            for scene_dir in copy_scenes_pool[: args.copy_episode_scenes]:
                dst = episodes_root / scene_dir.name
                if dst.exists():
                    print(f"  [episodes] skip existing {dst.name}")
                    continue
                print(f"  [episodes] copying {scene_dir.name} ...")
                shutil.copytree(scene_dir, dst, symlinks=False)
            print(f"[episodes] copied scenes to {episodes_root}")

    # ------------------------------------------------------------------ VLN-CE
    else:
        for scene_dir in picked_scenes_sorted:
            episodes = list_ce_episodes(scene_dir)
            if not episodes:
                print(f"  [skip] no episodes under {scene_dir.name}")
                continue
            instructions_map = load_instructions(scene_dir)

            n_ep = min(args.episodes_per_scene, len(episodes))
            chosen_eps = rng.sample(episodes, n_ep)
            chosen_eps.sort(key=lambda x: x[0])

            for ep_idx, parquet_path, fpv_dir, lookdown_dir in chosen_eps:
                instr_list = instructions_map.get(ep_idx, [])
                if not instr_list:
                    continue
                instruction = instr_list[0].strip()

                # Only steps where the waypoint is visible in the look-down view.
                goal_pixels = load_ce_goal_pixels(parquet_path)
                valid_steps = sorted(s for s in goal_pixels if s >= args.min_current_step)
                if not valid_steps:
                    continue

                n_samples = min(args.samples_per_episode, len(valid_steps))
                indices = np.unique(
                    np.linspace(0, len(valid_steps) - 1, n_samples, dtype=np.int32)
                )
                chosen_steps = [valid_steps[i] for i in indices]

                for cur_step in chosen_steps:
                    history_idx = pick_history_indices(cur_step, args.num_history)
                    sample_id = f"sample_{sample_counter:05d}"
                    save_ce_sample(
                        samples_root / sample_id,
                        fpv_dir, lookdown_dir, ep_idx, cur_step, history_idx,
                        instruction,
                        meta={
                            "sample_id": sample_id,
                            "scene": scene_dir.name,
                            "episode_index": int(ep_idx),
                            "current_step": int(cur_step),
                            "history_indices": [int(h) for h in history_idx],
                            "num_history_provided": len(history_idx),
                            "instruction": instruction,
                            "look_down": True,
                            "goal_pixel": goal_pixels[cur_step],
                        },
                        jpeg_quality=args.jpeg_quality,
                    )
                    manifest.append({
                        "sample_id": sample_id,
                        "scene": scene_dir.name,
                        "episode_index": int(ep_idx),
                        "current_step": int(cur_step),
                    })
                    sample_counter += 1
            print(f"  [scene] {scene_dir.name}: total samples so far = {sample_counter}")

    (out_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest) + "\n"
    )
    print(f"[samples] wrote {sample_counter} samples to {samples_root}")
    print("Done.")


if __name__ == "__main__":
    main()
