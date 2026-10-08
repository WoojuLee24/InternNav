"""Shared fail-analysis store (from VLN-Challenge dev/jay dashboard/annotations.py).

  - reason TYPES (buttons) are GLOBAL: one file (global_buttons_file()),
    shared across every task. Adding / renaming / deleting a type applies everywhere.
  - SELECTIONS are PER TASK: `<task_dir>/fail_analysis.json` = {episodes, log}, travelling
    with the logs. episodes maps episode_id -> {labels:[...]}, log is append-only history.

Renaming / deleting a type rewrites the global buttons AND every task's selections + log.
All writes are atomic, under one process lock.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

import data

ANNOT_FILENAME = "fail_analysis.json"


def global_buttons_file():
    """Reason types are shared by every task: <checkpoints_root>/_eval_dashboard/fail_reasons.json"""
    return data.root() / "_eval_dashboard" / "fail_reasons.json"


_lock = threading.Lock()


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def _dedupe(seq):
    return list(dict.fromkeys(seq))


# ---- global reason types (stored at the logs root, outside any task folder) -------------
def _read_list(path):
    if path.exists():
        try:
            b = json.loads(path.read_text())
            if isinstance(b, list):
                return b
        except Exception:
            pass
    return None


def _load_buttons():
    b = _read_list(global_buttons_file())
    if b is not None:
        return b
    return list(data.SETTINGS["default_fail_reasons"])


def _save_buttons(lst):
    global_buttons_file().parent.mkdir(exist_ok=True)  # only on write: reads must not touch the root
    _atomic_write(global_buttons_file(), _dedupe(lst))


def load_buttons():
    return _load_buttons()


# ---- per-task selections ----------------------------------------------------
def _task_file(task_dir):
    return Path(task_dir) / ANNOT_FILENAME


def _load_task(task_dir):
    f = _task_file(task_dir)
    if f.exists():
        try:
            d = json.loads(f.read_text())
            d.setdefault("episodes", {})
            d.setdefault("log", [])
            return d
        except Exception:
            pass
    return {"episodes": {}, "log": []}


def _save_task(task_dir, a):
    _atomic_write(_task_file(task_dir), a)


def snapshot(task_dir):
    a = _load_task(task_dir)
    return {"buttons": _load_buttons(), "episodes": a["episodes"]}


# ---- mutations --------------------------------------------------------------
def toggle(task_dir, episode_id, label, client):
    label = (label or "").strip()
    with _lock:
        buttons = _load_buttons()
        if label and label not in buttons:           # typing a new reason registers it globally
            buttons.append(label)
            _save_buttons(buttons)
        a = _load_task(task_dir)
        ent = a["episodes"].setdefault(episode_id, {"labels": []})
        if label in ent["labels"]:
            ent["labels"].remove(label)
            action = "off"
        else:
            ent["labels"].append(label)
            action = "on"
        a["log"].append({"ts": _now(), "client": client, "episode": episode_id,
                         "label": label, "action": action})
        _save_task(task_dir, a)
        return {"selected": ent["labels"], "buttons": buttons, "action": action}


def add_button(label, client):
    label = (label or "").strip()
    with _lock:
        buttons = _load_buttons()
        if label and label not in buttons:
            buttons.append(label)
            _save_buttons(buttons)
        return {"buttons": buttons}


def rename_button(old, new, task_dirs, client):
    old, new = (old or "").strip(), (new or "").strip()
    with _lock:
        if not (old and new and old != new):
            return {"buttons": _load_buttons()}
        _save_buttons([new if x == old else x for x in _load_buttons()])
        for td in task_dirs:                          # propagate to every task's selections
            a = _load_task(td)
            changed = False
            for ent in a["episodes"].values():
                if old in ent["labels"]:
                    ent["labels"] = _dedupe(new if l == old else l for l in ent["labels"])
                    changed = True
            if changed:
                a["log"].append({"ts": _now(), "client": client, "action": "rename",
                                 "from": old, "to": new})
                _save_task(td, a)
        return {"buttons": _load_buttons()}


def delete_button(label, task_dirs, client):
    label = (label or "").strip()
    with _lock:
        _save_buttons([x for x in _load_buttons() if x != label])
        for td in task_dirs:                          # remove the label from every task
            a = _load_task(td)
            changed = False
            for ent in a["episodes"].values():
                if label in ent["labels"]:
                    ent["labels"] = [l for l in ent["labels"] if l != label]
                    changed = True
            if changed:
                a["log"].append({"ts": _now(), "client": client, "action": "delete", "label": label})
                _save_task(td, a)
        return {"buttons": _load_buttons()}
