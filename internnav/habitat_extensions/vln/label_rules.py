"""Reproduction of the InternData-N1 VLN-CE System2 labels, for reference ("GT-like") decisions
during evaluation. Findings, verified against the r2r parquet labels (see
.claude/memory/understanding_s2_label_generation.md and scripts/eval_dashboard/validate_reference_labels.py):

- Dataset poses (`pose.<H>cm_<P>deg`) are camera-to-world 4x4, OpenCV camera (x right, y down,
  z forward), world z-up in an EPISODE-LOCAL frame (start at the origin, heading +x).
  habitat world = start_position + R(start_rotation) @ [-y_local, 0, -x_local]   (to_habitat)
- pixel goal of frame t = the floor point under the camera of frame j = t+k+1, projected into
  frame t's look-down camera: exact projection (median 0.35 px, 92-95% within 2 px @125cm).
  Camera: 640x480, fx = fy = 388.2 (125cm, ~79 deg HFOV) / 465.5 (60cm, ~69 deg), c = (W-1)/2, (H-1)/2.
- The choice of k ("farthest visible point", InternVLA-N1 paper Sec. 4.2) is NOT fully recovered.
  Best empirical rule (EMPIRICAL_RULE): walk the future frames, keep the farthest one within
  3.25 m (planar Euclidean) that projects inside a 40/48/30 px margin with the depth surface at
  least 5 cm behind it, stop at the first failure after a success, drop k < 3. It reproduces
  ~58% of frames exactly (label/no-label + same k), ~70% of pixel labels within 10 px, and is
  almost never absent where a label exists -- use it as a REFERENCE, not as ground truth.
- STOP labels exist only at the last frame of an episode; turns are labelled where no pixel goal
  exists and the next action is not forward. Forward step 0.25 m, turn 15 deg.
"""
import math

import numpy as np

W, H = 640, 480
CAMERAS = {125: {"fx": 388.2, "height": 1.25}, 60: {"fx": 465.5, "height": 0.60}}
EMPIRICAL_RULE = {"max_dist": 3.25, "margin_u": 40, "margin_top": 48, "margin_bottom": 30,
                  "min_depth_margin": 0.05, "min_k": 3}


def camera(height_cm: int):
    c = CAMERAS[height_cm]
    return {"fx": c["fx"], "fy": c["fx"], "cx": (W - 1) / 2, "cy": (H - 1) / 2, "height": c["height"]}


def project(pose_t: np.ndarray, p_world: np.ndarray, cam: dict):
    """World point -> (u, v) pixel and camera depth z in frame t (None if behind the camera)."""
    pc = np.linalg.inv(pose_t) @ np.append(p_world, 1.0)
    if pc[2] <= 1e-6:
        return None, float(pc[2])
    return (cam["fx"] * pc[0] / pc[2] + cam["cx"], cam["fy"] * pc[1] / pc[2] + cam["cy"]), float(pc[2])


def floor_point(pose: np.ndarray, cam: dict) -> np.ndarray:
    """Floor point under a camera pose (world z-up)."""
    return pose[:3, 3] - np.array([0.0, 0.0, cam["height"]])


def reference_pixel_goal(poses, t: int, cam: dict, depth=None, rule=EMPIRICAL_RULE, points=None):
    """(k, (u, v)) of the reference pixel goal for frame t, or None.

    poses: camera poses of the trajectory (list of 4x4, dataset frame). `points` optionally replaces
    the future floor points (e.g. a navmesh geodesic path already in the dataset frame; then k is
    the index into `points`). depth: frame t's look-down depth in metres (HxW) for visibility."""
    pts = [floor_point(p, cam) for p in poses[t + 1:]] if points is None else list(points)
    origin = poses[t][:3, 3]
    best, started = None, False
    for i, p in enumerate(pts):
        if math.dist(p[:2], origin[:2]) > rule["max_dist"] + 1e-6:
            continue
        uv, z = project(poses[t], p, cam)
        ok = (uv is not None and rule["margin_u"] <= uv[0] < W - rule["margin_u"]
              and rule["margin_top"] <= uv[1] < H - rule["margin_bottom"])
        if ok and depth is not None:
            ok = float(depth[int(uv[1]), int(uv[0])]) - z >= rule["min_depth_margin"]
        if ok:
            started, best = True, (i, uv)
        elif started:
            break
    if best is None or (points is None and best[0] < rule["min_k"]):
        return None
    return best


# v2 (2026-10-05, scripts/eval_dashboard/validate_reference_labels.py): found from the labels themselves --
# every label point is unoccluded at its own pixel and has PATH length <= 3.25 m (float32 poses give
# 3.25005..., hence the tolerance); turning in place repeats a position, and the label takes the FIRST
# frame at it. Held-out episodes (61 scenes x episodes 3-4, not used for tuning), pixel <= 10 px v1 -> v2:
# 125cm train 66.8 -> 75.2%, val 63.5 -> 70.5%, 60cm train 75.1 -> 79.3%, val 72.2 -> 76.9%.
EMPIRICAL_RULE_V2 = {"max_path": 3.25, "path_tol": 0.01, "margin_u": 40, "margin_top": 48, "margin_bottom": 30,
                     "min_depth_margin": 0.05, "min_k": 3}


def reference_pixel_goal_v2(poses, t: int, cam: dict, depth=None, rule=EMPIRICAL_RULE_V2):
    """(k, (u, v)) or None: the farthest future frame within `max_path` m of TRAJECTORY length whose floor
    point is inside the margins and (with depth) unoccluded; among frames at the same position the first
    one. Unlike v1 there is no contiguity stop and the cap is path length, not straight-line distance."""
    origin = floor_point(poses[t], cam)
    best, path, prev = None, 0.0, origin
    for i, pose in enumerate(poses[t + 1:]):
        p = floor_point(pose, cam)
        step = math.dist(p[:2], prev[:2])
        path, prev = path + step, p
        if path > rule["max_path"] + rule["path_tol"]:
            break
        uv, z = project(poses[t], p, cam)
        ok = (uv is not None and rule["margin_u"] <= uv[0] < W - rule["margin_u"]
              and rule["margin_top"] <= uv[1] < H - rule["margin_bottom"])
        if ok and depth is not None:
            ok = float(depth[int(uv[1]), int(uv[0])]) - z >= rule["min_depth_margin"]
        if ok and not (best is not None and step < 1e-3):  # same position as the kept goal: keep the first
            best = (i, uv)
    if best is None or best[0] < rule["min_k"]:
        return None
    return best


# v3 (2026-10-05): a learned candidate selector. Hand rules plateau (v2); which feasible future frame
# the generator took depends jointly on how far it is in the farthest-first order, the depth margin, the
# heading at t relative to the candidate, path length, the action after the candidate (frames right
# before a turn are avoided) and its place among turn-in-place duplicates. A gradient-boosted classifier
# over these features (scripts/eval_dashboard/fit_label_rule_v3.py) scores every feasible candidate;
# the best one is the goal if its score >= threshold, else "no pixel goal". Held-out (episodes 3-4,
# frame match / pixel <= 10 px, v2 -> v3): 125cm train 62.7/75.2 -> 88.3/88.2%, val 61.3/70.5 -> 85.9/85.1%,
# 60cm train 65.5/79.3 -> 82.5/86.0%, val 63.1/76.9 -> 80.0/82.7%. Mesh line-of-sight features (Embree raycast
# on the scene .glb) were tested and add nothing beyond the depth image, which is rendered from the
# same mesh. Offline only: the features need the GT trajectory and actions.
V3_FEATURES = ("L", "u", "v", "z", "dd", "dev", "dev_rel", "head_j", "head_t", "act_after", "grp_size", "grp_pos",
               "rank_from_far", "is_last_frame", "frac_L")


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def v3_candidates(poses, actions, t: int, cam: dict, depth, rule=EMPIRICAL_RULE_V2):
    """Feasible candidates of frame t with their V3_FEATURES vectors: [(k, (u, v), [features])].

    poses: dataset camera poses; actions: the parquet `action` column (actions[i] led to frame i);
    feasible = path <= max_path (+tol), inside the margins, depth surface not in front of the point, k >= min_k."""
    n = len(poses)
    floor = [floor_point(p, cam) for p in poses]
    path, frames = 0.0, []
    for j in range(t + 1, n):
        path += math.dist(floor[j][:2], floor[j - 1][:2])
        if path > rule["max_path"] + rule["path_tol"]:
            break
        frames.append((j, path))
    feas = []
    for j, L in frames:
        uv, z = project(poses[t], floor[j], cam)
        if uv is None or not (rule["margin_u"] <= uv[0] < W - rule["margin_u"]
                              and rule["margin_top"] <= uv[1] < H - rule["margin_bottom"]):
            continue
        dd = float(depth[int(uv[1]), int(uv[0])]) - z
        if dd < 0.0 or j - t - 1 < rule["min_k"]:
            continue
        feas.append((j, L, uv, z, dd))
    out, f0 = [], floor[t][:2]
    for i, (j, L, uv, z, dd) in enumerate(feas):
        d = floor[j][:2] - f0
        nd = float(np.linalg.norm(d))
        pts = np.array([floor[x][:2] for x in range(t + 1, j)])
        dev = float(np.max(np.abs(np.cross(d / nd, pts - f0)))) if (nd > 1e-6 and len(pts)) else 0.0
        dirtj = math.atan2(d[1], d[0]) if nd > 1e-6 else 0.0
        grp = [x for x, Lx in frames if abs(Lx - L) < 1e-3]
        f = {"L": L, "u": uv[0], "v": uv[1], "z": z, "dd": dd, "dev": dev, "dev_rel": dev / max(nd, 0.25),
             "head_j": abs(_wrap(local_yaw(poses[j]) - dirtj)), "head_t": abs(_wrap(local_yaw(poses[t]) - dirtj)),
             "act_after": actions[j + 1] if j + 1 < n else 0, "grp_size": len(grp),
             "grp_pos": grp.index(j) / max(len(grp) - 1, 1), "rank_from_far": len(feas) - 1 - i,
             "is_last_frame": float(j == n - 1), "frac_L": L / rule["max_path"]}
        out.append((j - t - 1, uv, [f[k] for k in V3_FEATURES]))
    return out


def reference_pixel_goal_v3(poses, actions, t: int, cam: dict, depth, model, threshold: float = 0.2):
    """(k, (u, v)) or None using a model fitted by scripts/eval_dashboard/fit_label_rule_v3.py."""
    cands = v3_candidates(poses, actions, t, cam, depth)
    if not cands:
        return None
    p = model.predict_proba(np.array([f for _, _, f in cands], dtype=float))[:, 1]
    i = int(np.argmax(p))
    return (cands[i][0], cands[i][1]) if p[i] >= threshold else None


def to_habitat(p_local: np.ndarray, start_position, start_rotation) -> np.ndarray:
    """Dataset episode-local point (x fwd, y left, z up) -> habitat world (y up)."""
    import quaternion

    R = quaternion.as_rotation_matrix(np.quaternion(start_rotation[3], *start_rotation[:3]))  # [x,y,z,w]
    return np.asarray(start_position) + R @ np.array([-p_local[1], p_local[2], -p_local[0]])


def to_local(p_hab: np.ndarray, start_position, start_rotation) -> np.ndarray:
    """Inverse of to_habitat."""
    import quaternion

    R = quaternion.as_rotation_matrix(np.quaternion(start_rotation[3], *start_rotation[:3]))
    a = R.T @ (np.asarray(p_hab) - np.asarray(start_position))
    return np.array([-a[2], -a[0], a[1]])


def reference_pixel_habitat(points, cam_pos, cam_rot, fx, cx, cy, agent_pos, depth=None, rule=EMPIRICAL_RULE):
    """EMPIRICAL_RULE on habitat-world floor `points` (e.g. the geodesic path to the goal) for a
    habitat camera (position, 3x3 rotation; camera looks along -z, y up). Returns (i, (u, v)) or None.
    depth: metric z-depth image of that camera (HxW) for the visibility test."""
    Rt = np.asarray(cam_rot).T
    best, started = None, False
    for i, p in enumerate(points):
        if math.dist((p[0], p[2]), (agent_pos[0], agent_pos[2])) > rule["max_dist"] + 1e-6:
            continue
        pc = Rt @ (np.asarray(p) - np.asarray(cam_pos))
        z = -pc[2]
        ok = z > 1e-6
        if ok:
            u, v = fx * pc[0] / z + cx, cy - fx * pc[1] / z
            ok = rule["margin_u"] <= u < W - rule["margin_u"] and rule["margin_top"] <= v < H - rule["margin_bottom"]
            if ok and depth is not None:
                ok = float(depth[int(v), int(u)]) - z >= rule["min_depth_margin"]
        if ok:
            started, best = True, (i, (u, v))
        elif started:
            break
    return best


def local_yaw(pose: np.ndarray) -> float:
    """Heading of a dataset camera pose in the episode-local frame (0 = +x, CCW = left)."""
    f = pose[:3, 2]  # camera forward (OpenCV z) in world
    return math.atan2(f[1], f[0])


if __name__ == "__main__":
    # self-check: projection of a point straight ahead, frame round trip, rule on a synthetic path
    cam = camera(125)
    pitch = math.radians(30)
    # camera at 1.25 m looking along +x, pitched down 30 deg (OpenCV axes in a z-up world)
    fwd = np.array([math.cos(pitch), 0, -math.sin(pitch)])
    down = np.array([math.sin(pitch), 0, math.cos(pitch)])
    right = np.array([0, -1.0, 0])
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, fwd], axis=1)
    pose[:3, 3] = [0, 0, 1.25]
    uv, z = project(pose, np.array([1.25 / math.tan(pitch), 0, 0]), cam)  # floor point on the optical axis
    assert abs(uv[0] - cam["cx"]) < 1e-6 and abs(uv[1] - cam["cy"]) < 1e-6
    sp, sr = [1.0, 0.1, 2.0], [0.0, 0.3826834, 0.0, 0.9238795]  # 45 deg about +y
    p = np.array([2.0, -1.0, 0.5])
    assert np.allclose(to_local(to_habitat(p, sp, sr), sp, sr), p)
    assert abs(local_yaw(pose)) < 1e-9
    path = []
    for i in range(1, 30):
        q = pose.copy()
        q[:3, 3] = [0.25 * i, 0, 1.25]
        path.append(q)
    k, uv = reference_pixel_goal([pose] + path, 0, cam)
    assert 0.25 * (k + 1) <= 3.25 and uv[1] >= EMPIRICAL_RULE["margin_top"]
    # habitat projection == dataset projection for the same camera (converted with to_habitat)
    import quaternion

    R0 = quaternion.as_rotation_matrix(np.quaternion(sr[3], *sr[:3]))
    M = lambda v: R0 @ np.array([-v[1], v[2], -v[0]])  # noqa: E731  local direction -> habitat direction
    cam_rot = np.stack([M(pose[:3, 0]), -M(pose[:3, 1]), -M(pose[:3, 2])], axis=1)  # right, up, back
    cam_pos = to_habitat(pose[:3, 3], sp, sr)
    pts_h = [to_habitat(floor_point(q, cam), sp, sr) for q in path]
    kh, uvh = reference_pixel_habitat(pts_h, cam_pos, cam_rot, cam["fx"], cam["cx"], cam["cy"],
                                      agent_pos=to_habitat(floor_point(pose, cam), sp, sr))
    assert kh == k and abs(uvh[0] - uv[0]) < 1e-6 and abs(uvh[1] - uv[1]) < 1e-6, (kh, uvh, k, uv)
    # v2: path cap (0.25 m steps -> 13 steps = 3.25 m) and the first frame among turn-in-place duplicates
    k2, uv2 = reference_pixel_goal_v2([pose] + path, 0, cam)
    assert k2 == 12, k2
    dup = path[:5] + [path[4]] + path[5:]  # frame 5 repeats frame 4's position (a turn in place)
    k3, _ = reference_pixel_goal_v2([pose] + dup[:6], 0, cam)
    assert k3 == 4, k3  # the first of the two frames at that position
    # v3 candidates: same feasible set as v2's search space, farthest has rank_from_far 0
    flat_depth = np.full((H, W), 50.0)  # nothing occludes
    acts = [-1] + [1] * len(path)
    cands = v3_candidates([pose] + path, acts, 0, cam, flat_depth)
    assert cands and cands[-1][0] == k2 and cands[-1][2][V3_FEATURES.index("rank_from_far")] == 0
    print("label_rules self-check OK", k, [round(x, 1) for x in uv], "v2", k2, "v3 candidates", len(cands))
