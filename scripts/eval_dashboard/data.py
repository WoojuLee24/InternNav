"""Eval dashboard data layer: discover eval runs under the checkpoints root, compute metrics,
build the per-episode views. Ported from VLN-Challenge dev/jay dashboard/{tasks,metrics,episodes}.py
and adapted to InternNav's layout (see README.md).

A *task* is one eval log dir = one (checkpoint, experiment config) pair:
    <root>/<group>/<run>[/checkpoint-N]/logs/<exp_slug>/
        result_<machine>.json   one row per finished eval run (append-only, distributed_base.py)
        progress.json           one row per episode (resume index)
        raw/<run_stamp>/        per-episode JSON (eval_recorder.py) -- optional
Episodes are merged across raw/<stamp>/ dirs (a resumed eval spreads over several stamps); the
latest stamp wins per episode. Tasks without raw/ (older evals) fall back to result rows and
progress.json (metrics only, no trajectory).

Coordinates: raw files are habitat world metres (y up); the viewer uses 2D (x, -z).
"""
import glob
import json
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "config.json"
_DEFAULTS = {
    "checkpoints_root": "/home/irteam/data-vol2/checkpoints",
    "host": "0.0.0.0",
    "port": 8088,
    "cache_ttl": 30.0,
    "metric_radii": [3.0, 1.5, 0.5],
    "max_steps_per_episode": 500,
    "default_fail_reasons": ["wrong room", "stopped early", "overshoot", "instruction misread"],
    "split_point": {"threshold_factor": 0.12, "threshold_min": 0.6, "threshold_max": 1.5, "sustain_window": 4},
}


def _load_settings():
    s = dict(_DEFAULTS)
    if CONFIG_FILE.exists():
        s.update(json.loads(CONFIG_FILE.read_text()))
    return s


SETTINGS = _load_settings()
ACTION_LABELS = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}


def root() -> Path:
    return Path(SETTINGS["checkpoints_root"])


def radius_key(r):
    return "%g" % r


def metric_order():
    keys = []
    for r in sorted(SETTINGS["metric_radii"], reverse=True):
        k = radius_key(r)
        keys += [f"SR@{k}", f"SPL@{k}", f"OSR@{k}"]
    return keys + ["NDTW", "NE", "TL", "CR", "CFSR"]


# ----------------------------------------------------------------------------- geometry
def xy(p):
    """habitat world [x, y, z] -> viewer 2D [x, -z]"""
    return [p[0], -p[2]] if p and len(p) >= 3 else None


def _cum(p):
    L = [0.0]
    for i in range(1, len(p)):
        L.append(L[-1] + math.dist(p[i], p[i - 1]))
    return L


def _pt_at_frac(p, L, f):
    total = L[-1]
    if total <= 0:
        return p[0]
    target = f * total
    for k in range(1, len(p)):
        if L[k] >= target:
            seg = (L[k] - L[k - 1]) or 1.0
            t = (target - L[k - 1]) / seg
            return [p[k - 1][0] + t * (p[k][0] - p[k - 1][0]), p[k - 1][1] + t * (p[k][1] - p[k - 1][1])]
    return p[-1]


def compute_split(gt, pred):
    """Frame where the agent path forks away from the GT path for good (jay's split point)."""
    if len(gt) < 2 or len(pred) < 2:
        return None
    gtL, predL = _cum(gt), _cum(pred)
    if predL[-1] <= 0:
        return None
    sp = SETTINGS["split_point"]
    T = min(sp["threshold_max"], max(sp["threshold_min"], sp["threshold_factor"] * gtL[-1]))
    W = sp["sustain_window"]

    def dist(i):
        return math.dist(pred[i], _pt_at_frac(gt, gtL, predL[i] / predL[-1]))

    fork, i = -1, 0
    while i < len(pred):
        if dist(i) <= T:
            fork = i
        elif all(dist(k) > T for k in range(i, min(len(pred), i + W))):
            break
        i += 1
    if fork < 0:
        return None
    return {"before": fork, "after": min(fork + 1, len(pred) - 1), "point": pred[fork]}


# ----------------------------------------------------------------------------- per-episode
def failure_case(success, os_, stopped):
    """success | no_stop (hit the step limit) | passed_goal (was within the success radius
    but ended outside) | never_reached. stopped=None for legacy rows (unknown)."""
    if success:
        return "success"
    if stopped is False:
        return "no_stop"
    if os_:
        return "passed_goal"
    return "never_reached"


def episode_metrics(d):
    """SR/SPL/OSR at each radius + NDTW/NE/TL from one raw episode file.

    SR@r = STOP was called and final distance <= r (r = success distance reproduces habitat
    Success); SPL@r uses habitat's geodesic start->goal distance; OSR@r = min distance <= r."""
    meta, frames, m = d["meta"], d["frames"], d.get("metrics", {})
    d2g = [f["dist_to_goal"] for f in frames if isinstance(f.get("dist_to_goal"), (int, float))]
    ne = m.get("ne", d2g[-1] if d2g else None)
    mind = min(d2g) if d2g else None
    stopped = bool(frames) and frames[-1].get("action") == 0
    tl = m.get("tl") or 0.0
    gd = meta.get("geodesic_distance") or 0.0
    out = {}
    for r in sorted(SETTINGS["metric_radii"], reverse=True):
        k = radius_key(r)
        sr = 1.0 if (stopped and ne is not None and ne <= r) else 0.0
        out[f"SR@{k}"] = sr
        out[f"SPL@{k}"] = round(sr * gd / max(tl, gd), 4) if max(tl, gd) > 0 else 0.0
        out[f"OSR@{k}"] = 1.0 if (mind is not None and mind <= r) else 0.0
    out["NDTW"] = m.get("ndtw_ref")
    out["NE"] = round(ne, 3) if isinstance(ne, (int, float)) else None
    out["TL"] = round(tl, 3)
    return out, stopped


def _row_from_raw(ep_id, d):
    meta, frames, m = d["meta"], d["frames"], d.get("metrics", {})
    em, stopped = episode_metrics(d)
    success = bool(m.get("success"))
    fc = failure_case(success, m.get("os"), stopped)
    ref2 = [xy(p) for p in meta.get("reference_path", [])]
    pred2 = [xy(f["pose"]["position"]) for f in frames]
    return {
        "episode_id": ep_id, "scene": meta["scene_id"], "result": fc, "success": success,
        "failure_case": "-" if success else fc,
        "instruction": meta.get("instruction", ""),
        "steps": len(frames) - 1, "gt_steps": None,
        "NE": em["NE"], "TL": em["TL"], "gt_tl": meta.get("geodesic_distance"),
        "metrics": em, "split": None if success else compute_split(ref2, pred2),
        "collisions": m.get("collisions"),
        "has_raw": True, "run_stamp": meta.get("run_stamp"),
    }


def _row_from_progress(ep_id, r):
    success = bool(r.get("success"))
    fc = failure_case(success, r.get("os"), None)
    em = {"NE": r.get("ne"), "TL": r.get("tl"), "NDTW": r.get("ndtw_ref", r.get("ndtw"))}
    return {
        "episode_id": ep_id, "scene": r.get("scene_id"), "result": fc, "success": success,
        "failure_case": "-" if success else fc, "instruction": r.get("episode_instruction", ""),
        "steps": r.get("steps"), "gt_steps": None, "NE": r.get("ne"), "TL": r.get("tl"), "gt_tl": None,
        "metrics": em, "split": None, "collisions": r.get("collisions"), "has_raw": False,
        "run_stamp": r.get("run_stamp"),
    }


def ep_key(scene_id, episode_id):
    return f"{scene_id}_{int(episode_id):04d}"


# ----------------------------------------------------------------------------- per-task
def _read_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass  # a row being appended right now
    except OSError:
        pass
    return rows


def raw_episode_files(task_dir: Path):
    """{episode_key: path}, latest run_stamp wins (stamps sort chronologically)."""
    files = {}
    for stamp_dir in sorted(glob.glob(str(task_dir / "raw" / "*"))):
        for f in glob.glob(os.path.join(stamp_dir, "episodes", "*.json")):
            files[Path(f).stem] = f
    return files


def _stamp_time(stamp):
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").timestamp()
    except (TypeError, ValueError):
        return None


def build_task(task_dir: Path):
    rel = task_dir.relative_to(root()).as_posix()
    parts = rel.split("/")
    result_rows = []
    for f in sorted(task_dir.glob("result_*.json")):
        result_rows += [dict(r, _machine=f.stem[len("result_"):]) for r in _read_jsonl(f)]
    result_rows.sort(key=lambda r: str(r.get("timestamp", "")))
    last = result_rows[-1] if result_rows else {}
    raw_stamps = sorted(os.path.basename(p) for p in glob.glob(str(task_dir / "raw" / "*")))
    raw_files = raw_episode_files(task_dir)

    episodes = []
    if raw_files:
        for k in sorted(raw_files):
            try:
                episodes.append(_row_from_raw(k, json.loads(Path(raw_files[k]).read_text())))
            except (OSError, ValueError, KeyError):
                continue
    else:
        prog = {}
        for r in _read_jsonl(task_dir / "progress.json"):
            if "scene_id" in r:
                prog[ep_key(r["scene_id"], r["episode_id"])] = r
        episodes = [_row_from_progress(k, prog[k]) for k in sorted(prog)]

    metrics = {}
    if raw_files and episodes:
        acc = {}
        for e in episodes:
            for k, v in e["metrics"].items():
                if isinstance(v, (int, float)):
                    acc.setdefault(k, []).append(v)
        metrics = {k: round(sum(v) / len(v), 4) for k, v in acc.items()}
        metrics["Count"] = len(episodes)
        coll = [e for e in episodes if isinstance(e.get("collisions"), (int, float))]
        if coll:  # Isaac definitions (metrics_schema): CR = collisions / steps, CFSR = success w/o collision
            steps = sum(e["steps"] or 0 for e in coll)
            metrics["CR"] = round(sum(e["collisions"] for e in coll) / steps, 4) if steps else 0.0
            metrics["CFSR"] = round(sum(1 for e in coll if e["success"] and e["collisions"] == 0) / len(coll), 4)
    elif last:
        # legacy eval (no raw/): single-radius metrics from the latest result row
        sd = radius_key(3.0)
        for key, src in ((f"SR@{sd}", "sucs_all"), (f"SPL@{sd}", "spls_all"), (f"OSR@{sd}", "oss_all"),
                         ("NE", "nes_all"), ("TL", "tls_all"), ("NDTW", "ndtw_refs_all"),
                         ("CR", "crs_all"), ("CFSR", "cfsrs_all")):
            if last.get(src) is not None:
                metrics[key] = last[src]
        metrics["Count"] = last.get("length")

    # running = newest raw stamp has no result row yet (the eval has not written its final row)
    row_stamps = {str(r.get("timestamp")) for r in result_rows}
    running = bool(raw_stamps) and raw_stamps[-1] not in row_stamps
    run_meta = {}
    if raw_stamps:
        try:
            run_meta = json.loads((task_dir / "raw" / raw_stamps[-1] / "run.json").read_text())
        except (OSError, ValueError):
            pass
    ckpt_step = last.get("ckpt_step", run_meta.get("ckpt_step"))
    if ckpt_step is None:  # older rows: take it from the checkpoint-N dir name
        ckpt_step = next((int(p.split("-")[1]) for p in parts if p.startswith("checkpoint-") and p[11:].isdigit()), None)
    stamp = raw_stamps[-1] if running else str(last.get("timestamp") or (raw_stamps[-1] if raw_stamps else ""))
    date = _stamp_time(stamp) or task_dir.stat().st_mtime
    fc_counts = {}
    for e in episodes:
        if not e["success"]:
            fc_counts[e["result"]] = fc_counts.get(e["result"], 0) + 1
    li = parts.index("logs") if "logs" in parts else len(parts)
    return {
        "task_id": rel.replace("/", "~"),
        "group": parts[0],
        "run": "/".join(parts[1:li]) or parts[0],      # <run>[/checkpoint-N]
        "config": "/".join(parts[li + 1:]) or "-",    # exp slug under logs/
        "dir": str(task_dir),
        "ckpt_step": ckpt_step,
        "ckpt": last.get("ckpt", run_meta.get("ckpt")),
        "git_commit": last.get("git_commit", run_meta.get("git_commit")),
        "date": date,
        "date_str": datetime.fromtimestamp(date).strftime("%Y-%m-%d %H:%M"),
        "status": "running" if running else "completed",
        "metrics": metrics,
        "result_rows": [{k: v for k, v in r.items()} for r in result_rows],
        "raw_stamps": raw_stamps,
        "has_raw": bool(raw_files),
        "n_episodes": len(episodes),
        "failure_cases": fc_counts,
        "_episodes": episodes,
    }


# ----------------------------------------------------------------------------- discovery + cache
_lock = threading.Lock()
_cache = {"ts": 0.0, "tasks": {}}
_task_cache = {}  # task_dir -> (signature, task)


def _signature(task_dir: Path):
    def mt(p):
        try:
            return os.stat(p).st_mtime
        except OSError:
            return 0.0
    res = tuple(mt(f) for f in sorted(glob.glob(str(task_dir / "result_*.json"))))
    raw = tuple((os.path.basename(d), len(os.listdir(os.path.join(d, "episodes"))) if os.path.isdir(os.path.join(d, "episodes")) else 0)
                for d in sorted(glob.glob(str(task_dir / "raw" / "*"))))
    return res, raw, mt(task_dir / "progress.json")


def find_task_dirs():
    """Every eval log dir under the root (depth: group/run[/ckpt][/...]/logs/<slug>)."""
    r = str(root())
    dirs = set()
    for depth in range(1, 4):
        mid = "/".join(["*"] * depth)
        for pat in (f"{r}/{mid}/logs/*/result_*.json", f"{r}/{mid}/logs/result_*.json"):
            dirs.update(os.path.dirname(p) for p in glob.glob(pat))
        dirs.update(os.path.dirname(os.path.dirname(os.path.dirname(p)))
                    for p in glob.glob(f"{r}/{mid}/logs/*/raw/*/run.json"))
    return sorted(Path(d) for d in dirs)


def discover(force=False):
    now = time.time()
    with _lock:
        if not force and _cache["tasks"] and now - _cache["ts"] < SETTINGS["cache_ttl"]:
            return _cache["tasks"]
    tasks = {}
    for d in find_task_dirs():
        sig = _signature(d)
        hit = _task_cache.get(str(d))
        if hit and hit[0] == sig:
            t = hit[1]
        else:
            try:
                t = build_task(d)
            except Exception as e:  # one broken dir must not take the dashboard down
                print(f"[dashboard] skip {d}: {e}", flush=True)
                continue
            _task_cache[str(d)] = (sig, t)
        tasks[t["task_id"]] = t
    with _lock:
        _cache.update(ts=now, tasks=tasks)
    return tasks


def get_task(task_id):
    return discover().get(task_id)


def public(t):
    return {k: v for k, v in t.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------- episode detail
def episode_detail(t, episode_id):
    f = raw_episode_files(Path(t["dir"])).get(episode_id)
    row = next((e for e in t["_episodes"] if e["episode_id"] == episode_id), None)
    if f is None:
        return {**(row or {"episode_id": episode_id}), "num_frames": 0, "frame_info": [], "bookmarks": [],
                "result_metrics": (row or {}).get("metrics", {}), "has_raw": False}
    d = json.loads(Path(f).read_text())
    meta, frames = d["meta"], d["frames"]
    em, stopped = episode_metrics(d)
    success = bool(d.get("metrics", {}).get("success"))
    fc = failure_case(success, d.get("metrics", {}).get("os"), stopped)
    ref2 = [xy(p) for p in meta.get("reference_path", [])]
    pred2 = [xy(fr["pose"]["position"]) for fr in frames]
    frame_info, bookmarks = [], []
    for idx, fr in enumerate(frames):
        a = fr.get("action")
        lbl = ACTION_LABELS.get(a, "start" if a is None else str(a))
        frame_info.append({"i": idx, "label": lbl, "dist_to_goal": fr.get("dist_to_goal"),
                           "yaw": fr["pose"].get("yaw"), "gen": fr.get("gen"), "decision": fr.get("decision"),
                           "collision": bool(fr.get("collision"))})
        if fr.get("collision"):
            bookmarks.append({"frame": idx, "type": "collision"})
        if fr.get("gen") is not None:
            bookmarks.append({"frame": idx, "type": "s2"})
        if a in (2, 3):
            bookmarks.append({"frame": idx, "type": "left" if a == 2 else "right"})
    fd = meta.get("frames_dir")
    stamp_dir = os.path.dirname(os.path.dirname(f))
    images = [os.path.join(stamp_dir, fd, fr["image"]) if (fd and fr.get("image")) else None for fr in frames]
    return {
        "_image_paths": images,  # absolute JPEG path per frame (None = not saved); app.py turns these into URLs
        "episode_id": episode_id, "scene": meta["scene_id"], "result": fc, "success": success,
        "failure_case": "-" if success else fc, "instruction": meta.get("instruction", ""),
        "reference_path": ref2, "pred_path": pred2,
        "gt_goal": xy(meta.get("goal_position")) or (ref2[-1] if ref2 else None),
        "pred_goal": pred2[-1] if pred2 else None, "start": xy(meta.get("start_position")),
        "yaw": [fr["pose"].get("yaw") for fr in frames],
        "topdown": meta.get("topdown"),
        "fps": 5, "num_frames": len(frames), "frame_info": frame_info, "bookmarks": bookmarks,
        "metrics": {"NE": em["NE"], "TL": em["TL"], "gt_tl": meta.get("geodesic_distance"),
                    "collisions": d.get("metrics", {}).get("collisions"),
                    "steps": len(frames) - 1, "stopped": stopped, **d.get("metrics", {})},
        "result_metrics": em, "metric_order": metric_order(),
        "goal_radii": sorted(SETTINGS["metric_radii"], reverse=True),
        "split": None if success else compute_split(ref2, pred2),
        "meta": {k: meta.get(k) for k in ("ckpt", "ckpt_step", "config", "run_stamp", "git_commit", "rank",
                                           "trajectory_id", "geodesic_distance", "success_distance")},
        "has_raw": True,
    }


def topdown(t, key):
    """Navmesh top-down grid for `key`, converted to the viewer frame (x, -z)."""
    for stamp_dir in sorted(glob.glob(os.path.join(t["dir"], "raw", "*")), reverse=True):
        p = os.path.join(stamp_dir, "topdown", key + ".json")
        if os.path.exists(p):
            g = json.loads(Path(p).read_text())
            mpp = g["meters_per_pixel"]
            # grid row 0 = min z = max viewer-y, so the image is drawn top-left at (x0, -z0), unflipped
            g["bounds"] = [g["origin_x"], -(g["origin_z"] + g["rows"] * mpp),
                           g["origin_x"] + g["cols"] * mpp, -g["origin_z"]]
            return g
    return None


if __name__ == "__main__":
    # self-check for the pure helpers
    assert xy([1.0, 2.0, 3.0]) == [1.0, -3.0]
    assert failure_case(True, 1, True) == "success" and failure_case(False, 1, True) == "passed_goal"
    assert failure_case(False, 0, False) == "no_stop" and failure_case(False, 0, None) == "never_reached"
    gt = [[0, 0], [10, 0]]
    sp = compute_split(gt, [[0, 0], [1, 0], [2, 0], [2, 3], [2, 6], [2, 9], [2, 12], [2, 15]])
    assert sp and sp["before"] == 2, sp
    d = {"meta": {"geodesic_distance": 4.0}, "metrics": {"ne": 1.0, "tl": 5.0, "ndtw_ref": 0.5},
         "frames": [{"dist_to_goal": 4.0, "action": None}, {"dist_to_goal": 1.0, "action": 0}]}
    em, stopped = episode_metrics(d)
    assert stopped and em["SR@3"] == 1.0 and em["SR@0.5"] == 0.0 and em["SPL@3"] == 0.8 and em["OSR@1.5"] == 1.0
    print("data self-check OK")
