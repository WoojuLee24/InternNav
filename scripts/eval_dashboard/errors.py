"""User-set error points (from VLN-Challenge dev/jay dashboard/errors.py), shared per task in
`<task_dir>/error_points.json` (travels with the logs, so anyone hitting the same logs sees them). One frame per episode:

    { "<episode_id>": {"frame": 12, "client": "1.2.3.4", "ts": "..."} }

When an episode has an error point it overrides the auto-computed split point in the viewer.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

_lock = threading.Lock()
FILENAME = "error_points.json"


def _file(task_dir):
    return Path(task_dir) / FILENAME


def _load(task_dir):
    f = _file(task_dir)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {}


def _save(task_dir, d):
    f = _file(task_dir)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    os.replace(tmp, f)


def get(task_dir, episode_id):
    return _load(task_dir).get(episode_id)


def set_point(task_dir, episode_id, frame, client):
    with _lock:
        d = _load(task_dir)
        d[episode_id] = {"frame": int(frame), "client": client,
                         "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        _save(task_dir, d)
        return d[episode_id]


def clear(task_dir, episode_id):
    with _lock:
        d = _load(task_dir)
        d.pop(episode_id, None)
        _save(task_dir, d)
        return True
