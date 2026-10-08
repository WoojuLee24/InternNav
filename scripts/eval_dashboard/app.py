#!/usr/bin/env python3
"""InternNav eval dashboard: Flask server over the raw eval outputs (ported from VLN-Challenge
dev/jay dashboard/app.py). Read-only over eval results; writes only the shared fail-analysis
tags and error points (annotations.py / errors.py).

    python scripts/eval_dashboard/app.py [--root <checkpoints_root>] [--port 8088]

Routes mirror jay's API so the frontend stays close to the original; see data.py for the layout.
"""
import argparse
import os
import sys
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annotations as ann  # noqa: E402
import data  # noqa: E402
import errors  # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / "static"), static_url_path="")


def _task_or_404(task_id):
    t = data.get_task(task_id)
    if not t:
        abort(404, "unknown task")
    return t


# ----------------------------------------------------------------------------- payload builders
# (shared with build_static.py, which calls them to pre-render every URL)
def leaderboard_payload(tasks):
    groups = {}
    for t in tasks.values():
        row = {k: v for k, v in data.public(t).items() if k != "result_rows"}
        groups.setdefault(t["group"], []).append(row)
    return {"groups": [{"name": g, "rows": groups[g]} for g in sorted(groups)]}


def task_payload(t):
    episodes = [dict(e) for e in t["_episodes"]]
    snap = ann.snapshot(t["dir"])
    err = errors._load(t["dir"])
    fa_counts = {}
    for e in episodes:
        labels = snap["episodes"].get(e["episode_id"], {}).get("labels", [])
        e["fail_analysis"] = labels
        e["has_error_point"] = e["episode_id"] in err
        for label in labels:
            fa_counts[label] = fa_counts.get(label, 0) + 1
    return {"task": data.public(t), "episodes": episodes, "failure_cases": t["failure_cases"],
            "fail_analysis": fa_counts, "buttons": snap["buttons"]}


def episode_payload(t, episode_id, image_url=None):
    """image_url(i, path) -> URL of frame i's front-view JPEG (default: the Flask route below)."""
    det = data.episode_detail(t, episode_id)
    paths = det.pop("_image_paths", None) or []
    url = image_url or (lambda i, _p: f"/api/frame/{t['task_id']}/{episode_id}/{i}")
    det["frame_images"] = [url(i, p) if p else None for i, p in enumerate(paths)]
    snap = ann.snapshot(t["dir"])
    det.update(task_id=t["task_id"], group=t["group"], buttons=snap["buttons"],
               selected=snap["episodes"].get(episode_id, {}).get("labels", []),
               error_point=errors.get(t["dir"], episode_id))
    return det


def neighbors_payload(t, episode_id):
    ids = sorted(e["episode_id"] for e in t["_episodes"])
    if episode_id not in ids:
        return {"prev": None, "next": None}
    i = ids.index(episode_id)
    return {"prev": ids[i - 1] if i > 0 else None, "next": ids[i + 1] if i < len(ids) - 1 else None}


def config_payload():
    s = data.SETTINGS
    return {"split_point": s["split_point"], "metric_radii": sorted(s["metric_radii"], reverse=True),
            "metric_order": data.metric_order()}


# ----------------------------------------------------------------------------- read API
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/leaderboard")
def api_leaderboard():
    p = leaderboard_payload(data.discover(force=request.args.get("refresh") == "1"))
    p["host"] = request.host
    return jsonify(p)


@app.route("/api/task/<task_id>")
def api_task(task_id):
    return jsonify(task_payload(_task_or_404(task_id)))


@app.route("/api/task/<task_id>/episode/<episode_id>")
def api_episode(task_id, episode_id):
    return jsonify(episode_payload(_task_or_404(task_id), episode_id))


@app.route("/api/task/<task_id>/neighbors/<episode_id>")
def api_neighbors(task_id, episode_id):
    return jsonify(neighbors_payload(_task_or_404(task_id), episode_id))


@app.route("/api/topdown/<task_id>/<key>")
def api_topdown(task_id, key):
    g = data.topdown(_task_or_404(task_id), key)
    if g is None:
        abort(404)
    return jsonify(g)


@app.route("/api/frame/<task_id>/<episode_id>/<int:i>")
def api_frame(task_id, episode_id, i):
    paths = data.episode_detail(_task_or_404(task_id), episode_id).get("_image_paths") or []
    if not (0 <= i < len(paths)) or not paths[i]:
        abort(404)
    return send_file(paths[i], mimetype="image/jpeg", conditional=True)


@app.route("/api/config")
def api_config():
    return jsonify(config_payload())


# ----------------------------------------------------------------------------- write API
def _json_body(*required):
    p = request.get_json(force=True)
    if any(p.get(k) in (None, "") for k in required):
        abort(400, f"{', '.join(required)} required")
    return p


@app.route("/api/annotation/toggle", methods=["POST"])
def api_toggle():
    p = _json_body("task_id", "episode_id", "label")
    t = _task_or_404(p["task_id"])
    return jsonify(ann.toggle(t["dir"], p["episode_id"], p["label"], request.remote_addr))


@app.route("/api/annotation/buttons", methods=["POST"])
def api_add_button():
    p = _json_body("label")
    return jsonify(ann.add_button(p["label"], request.remote_addr))


def _all_task_dirs():
    return [t["dir"] for t in data.discover().values()]


@app.route("/api/annotation/rename", methods=["POST"])
def api_rename_button():
    p = _json_body("old", "new")
    return jsonify(ann.rename_button(p["old"], p["new"], _all_task_dirs(), request.remote_addr))


@app.route("/api/annotation/delete", methods=["POST"])
def api_delete_button():
    p = _json_body("label")
    return jsonify(ann.delete_button(p["label"], _all_task_dirs(), request.remote_addr))


@app.route("/api/error/set", methods=["POST"])
def api_error_set():
    p = _json_body("task_id", "episode_id", "frame")
    t = _task_or_404(p["task_id"])
    return jsonify(errors.set_point(t["dir"], p["episode_id"], p["frame"], request.remote_addr))


@app.route("/api/error/clear", methods=["POST"])
def api_error_clear():
    p = _json_body("task_id", "episode_id")
    errors.clear(_task_or_404(p["task_id"])["dir"], p["episode_id"])
    return jsonify({"ok": True})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="InternNav eval dashboard")
    ap.add_argument("--root", help=f"checkpoints root to scan (default: {data.root()})")
    ap.add_argument("--host", default=data.SETTINGS["host"])
    ap.add_argument("--port", type=int, default=data.SETTINGS["port"])
    args = ap.parse_args()
    if args.root:
        data.SETTINGS["checkpoints_root"] = os.path.abspath(args.root)
    print(f"\n  eval dashboard -> http://{args.host}:{args.port}")
    print(f"  root           :  {data.root()}  ({'ok' if data.root().is_dir() else 'MISSING'})")

    def _warm():  # scan + metrics in the background so the server is reachable immediately
        n = len(data.discover(force=True))
        print(f"  [warm] {n} eval tasks indexed", flush=True)

    threading.Thread(target=_warm, daemon=True).start()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
