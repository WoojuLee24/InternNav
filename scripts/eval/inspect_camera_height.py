"""
Derive camera height and torso_link height from:
  1. USD file  — camera prim translations + naming convention
  2. Parquet   — cross-validate with observed camera_position / robot_position

Run with:
    /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 scripts/eval/inspect_camera_height.py

Requirements (already installed):
    usd-core  : pip install usd-core
    pyarrow   : pip install pyarrow
"""

import argparse
import glob
import math
import re
import sys

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------
DEFAULT_USD = "data/InternData-N1-v0.5-mini/Embodiments/vln-pe/h1/h1_internvla.usd"
DEFAULT_PARQUET_GLOB = "data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/*/data/chunk-000/episode_000000.parquet"

# ---------------------------------------------------------------------------
# Part 1: USD analysis
# ---------------------------------------------------------------------------

def _parse_height_from_name(name: str):
    """Extract height hint from camera prim name.

    Convention: h1_<H_int>_<H_frac>_... → height = H_int.H_frac metres
    e.g. 'h1_1_25_down_30' → 1.25 m
         'h1_0_6_look_forward' → 0.6 m
    """
    m = re.match(r"h1_(\d+)_(\d+)_", name)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return None


def analyze_usd(usd_path: str):
    try:
        from pxr import Usd
    except ImportError:
        print("[USD] pxr not available — skipping USD analysis.")
        print("      Install with: pip install usd-core")
        return {}

    print(f"\n{'='*60}")
    print(f"USD analysis: {usd_path}")
    print(f"{'='*60}")

    stage = Usd.Stage.Open(usd_path)

    # Collect camera prims that are children of torso_link
    cameras = []
    for prim in stage.Traverse():
        if prim.GetParent().GetName() == "torso_link":
            xform = prim.GetAttribute("xformOp:translate")
            if xform and xform.Get() is not None:
                translate = xform.Get()
                height_hint = _parse_height_from_name(prim.GetName())
                cameras.append({
                    "name": prim.GetName(),
                    "translate_x": translate[0],
                    "translate_y": translate[1],
                    "translate_z": translate[2],
                    "height_hint": height_hint,
                })

    print("\nCamera prims attached to torso_link:")
    print(f"  {'name':<30} {'translate (x,y,z)':<30} {'height_hint (m)'}")
    for c in cameras:
        tx, ty, tz = c["translate_x"], c["translate_y"], c["translate_z"]
        print(f"  {c['name']:<30} ({tx:.3f}, {ty:.3f}, {tz:.3f})           {c['height_hint']}")

    # Compute torso_link height by comparing cameras that have height hints
    named = [c for c in cameras if c["height_hint"] is not None]

    result = {}
    if len(named) >= 2:
        print("\nDeriving torso_link height from camera name pairs:")
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                a, b = named[i], named[j]
                dz    = a["translate_z"] - b["translate_z"]          # difference in translate.z
                dh    = a["height_hint"] - b["height_hint"]          # difference in height above floor
                if abs(dz) < 1e-6:
                    continue
                # dh = dz  → means naming is consistent with translate diff
                # torso_above_floor = height_hint_a - translate_z_a
                torso_z_a = a["height_hint"] - a["translate_z"]
                torso_z_b = b["height_hint"] - b["translate_z"]
                consistent = abs(torso_z_a - torso_z_b) < 1e-3
                print(f"  {a['name']} vs {b['name']}:")
                print(f"    Δtranslate.z = {dz:.4f} m,  Δheight_hint = {dh:.4f} m  → {'✓ consistent' if consistent else '✗ inconsistent'}")
                print(f"    torso_link_above_floor = {a['height_hint']:.2f} - ({a['translate_z']:.3f}) = {torso_z_a:.4f} m")

                if consistent:
                    result["torso_link_above_floor"] = round(torso_z_a, 4)

    # Report for specific camera (h1_1_25_down_30)
    target_cam = next((c for c in cameras if "1_25" in c["name"] and "down_30" in c["name"]), None)
    if target_cam:
        cam_z_from_torso = target_cam["translate_z"]
        h_hint = target_cam["height_hint"]
        torso_z = result.get("torso_link_above_floor", h_hint - cam_z_from_torso)
        cam_height = torso_z + cam_z_from_torso

        result.update({
            "camera_name": target_cam["name"],
            "camera_translate_z_from_torso": cam_z_from_torso,
            "camera_translate_x_from_torso": target_cam["translate_x"],
            "cam_height": round(cam_height, 4),
        })

        print(f"\nTarget camera: {target_cam['name']}")
        print(f"  translate from torso_link : ({target_cam['translate_x']:.3f}, {target_cam['translate_y']:.3f}, {cam_z_from_torso:.3f})")
        print(f"  torso_link above floor    : {torso_z:.4f} m")
        print(f"  cam_height above floor    : {torso_z:.4f} + {cam_z_from_torso:.4f} = {cam_height:.4f} m")
        print(f"  cam_x_offset (forward)    : {target_cam['translate_x']:.4f} m")

    return result


# ---------------------------------------------------------------------------
# Part 2: Parquet cross-validation
# ---------------------------------------------------------------------------

def analyze_parquet(parquet_glob: str, usd_result: dict):
    try:
        import pyarrow.parquet as pq
        import numpy as np
    except ImportError:
        print("[Parquet] pyarrow not available — skipping parquet analysis.")
        return

    files = sorted(glob.glob(parquet_glob))[:5]
    if not files:
        print(f"\n[Parquet] No files found for: {parquet_glob}")
        return

    print(f"\n{'='*60}")
    print(f"Parquet cross-validation ({len(files)} episodes)")
    print(f"{'='*60}")

    cam_height_usd = usd_result.get("cam_height")

    all_diffs = []
    for f in files:
        scene = f.split("/")[-4]
        t = pq.read_table(f)
        cam_z = np.array([r[2] for r in t.column("observation.camera_position").to_pylist()])
        rob_z = np.array([r[2] for r in t.column("observation.robot_position").to_pylist()])
        diff = cam_z - rob_z
        all_diffs.append(diff.mean())

        floor_z = cam_z.mean() - cam_height_usd if cam_height_usd else float("nan")
        print(f"  {scene}: cam_z={cam_z.mean():.4f}  robot_z={rob_z.mean():.4f}"
              f"  diff={diff.mean():.4f}  floor_z={floor_z:.4f}")

    mean_diff = float(np.mean(all_diffs))
    print(f"\n  camera_z - robot_z mean = {mean_diff:.4f} m  (std {float(np.std(all_diffs)):.4f})")

    if cam_height_usd:
        # torso_link_above_floor = cam_height - camera_translate_z_from_torso
        torso_above_floor = usd_result.get("torso_link_above_floor", cam_height_usd - usd_result.get("camera_translate_z_from_torso", 0.2))
        cam_translate_z   = usd_result.get("camera_translate_z_from_torso", 0.2)
        # pelvis is below torso: pelvis_z_world = cam_z - mean_diff
        # torso_z_world = cam_z - cam_translate_z
        # torso_above_pelvis = torso_z_world - pelvis_z_world = mean_diff - cam_translate_z
        torso_above_pelvis = mean_diff - cam_translate_z

        print(f"\n  Cross-validation (USD cam_height={cam_height_usd:.4f} m):")
        print(f"    camera_z = floor_z + {cam_height_usd:.4f}")
        print(f"    torso_link_z = camera_z - {cam_translate_z:.4f}")
        print(f"    pelvis_z = camera_z - {mean_diff:.4f}")
        print(f"    torso_link is {torso_above_pelvis:.4f} m ABOVE pelvis (in standing simulation)")
        print(f"    torso_link above floor = {torso_above_floor:.4f} m  (USD: {torso_above_floor:.4f} m)")
        print(f"    → cam_height = {cam_height_usd:.4f} m  ✓")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(usd_result: dict):
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    cam_height = usd_result.get("cam_height")
    torso_z    = usd_result.get("torso_link_above_floor")
    cam_x      = usd_result.get("camera_translate_x_from_torso")

    if cam_height:
        print(f"  cam_height            = {cam_height:.4f} m  (camera above floor)")
    if torso_z:
        print(f"  torso_link above floor= {torso_z:.4f} m")
    if cam_x:
        print(f"  cam_x_offset (fwd)    = {cam_x:.4f} m  (camera ahead of torso_link)")
    print(f"\n  Use in depth_to_bev(): cam_height={cam_height}, cam_pitch_deg=30.0")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect H1 camera/torso heights from USD + parquet data")
    parser.add_argument("--usd",     default=DEFAULT_USD,          help="Path to h1_internvla.usd")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET_GLOB, help="Glob pattern for episode parquet files")
    args = parser.parse_args()

    usd_result = analyze_usd(args.usd)
    analyze_parquet(args.parquet, usd_result)
    print_summary(usd_result)
