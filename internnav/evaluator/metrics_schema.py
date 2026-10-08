"""One definition of the navigation metrics, shared by habitat, Isaac (h1), and analysis code.

Canonical names: SR, SPL, OS, NE, TL, nDTW (+ Isaac-only FR, StR, CR, CFSR, and Count).
It re-derives every metric from per-episode records and is checked against each evaluator's
existing output with exact equality (enabled by eval_settings['metrics_schema'], off by default).
To be bit-identical it reproduces the existing aggregation: same order, dtype and rounding.
    habitat  HabitatVLNEvaluator.calc_metrics: float32 torch mean in gather order; SPL nan/inf -> 0;
             NE mean over finite values only; tls_all / ndtw_refs_all = Python mean over progress
             rows deduped by (scene_id, episode_id); crs_all / cfsrs_all likewise
             (eval_recorder.aggregate_progress / collision_aggregate)
    isaac    ResultLogger.finalize_all_results: Python-float sums, rank -> split_map order,
             NE<0 -> 0, osr<0 -> 0, round(., 4)
Existing keys stay untouched; the maps below only pair them with the canonical names.
"""
import collections
import json
import os
from datetime import datetime

HABITAT_KEYS = {"SR": "sucs_all", "SPL": "spls_all", "OS": "oss_all", "NE": "nes_all",
                "nDTW": "ndtws_all", "TL": "tls_all", "nDTW_ref": "ndtw_refs_all", "CR": "crs_all",
                "CFSR": "cfsrs_all", "Count": "length"}
ISAAC_KEYS = {k: k for k in ("TL", "NE", "FR", "StR", "OS", "SR", "SPL", "CR", "CFSR", "Count")}


# ----------------------------------------------------------------------------- habitat
def habitat_aggregate(per_ep: dict) -> dict:
    """per_ep: {'SR','SPL','OS','NE'[, 'nDTW']: 1-D float32 tensors in gather order}."""
    import torch

    spl = torch.nan_to_num(per_ep["SPL"].clone(), nan=0.0, posinf=0.0, neginf=0.0)
    ne = per_ep["NE"][torch.isfinite(per_ep["NE"])]
    n = len(per_ep["SR"])
    out = {"SR": float(per_ep["SR"].mean().item()) if n else 0.0,
           "SPL": float(spl.mean().item()) if n else 0.0,
           "OS": float(per_ep["OS"].mean().item()) if n else 0.0,
           "NE": float(ne.mean().item()) if n else 0.0,
           "Count": n}
    if "nDTW" in per_ep:
        out["nDTW"] = float(per_ep["nDTW"].mean().item()) if n else 0.0
    return out


def progress_aggregate(rows) -> dict:
    """TL / nDTW_ref as Python means over progress rows, last row per (scene_id, episode_id)."""
    dedup = {}
    for r in rows:
        dedup[(r.get("scene_id"), r.get("episode_id"))] = r
    out = {}
    for key, name in (("tl", "TL"), ("ndtw_ref", "nDTW_ref")):
        vals = [r[key] for r in dedup.values() if isinstance(r.get(key), (int, float))]
        if vals:
            out[name] = sum(vals) / len(vals)
    # collisions (Isaac definitions): CR = total collisions / total steps, CFSR = success with 0 collisions
    coll = [r for r in dedup.values() if isinstance(r.get("collisions"), (int, float))]
    if coll:
        steps = sum(r.get("steps", 0) for r in coll)
        out["CR"] = sum(r["collisions"] for r in coll) / steps if steps else 0.0
        out["CFSR"] = sum(1 for r in coll if r.get("success", 0) > 0 and r["collisions"] == 0) / len(coll)
    return out


def habitat_check(result_all: dict, global_metrics: dict, progress_rows) -> dict:
    """Recompute from the gathered per-episode tensors (+ progress rows) and compare exactly."""
    import torch

    per_ep = {"SR": global_metrics["sucs"], "SPL": global_metrics["spls"], "OS": global_metrics["oss"],
              "NE": global_metrics["nes"]}
    if "ndtws" in global_metrics:
        per_ep["nDTW"] = global_metrics["ndtws"]
    schema = habitat_aggregate(per_ep)
    schema.update(progress_aggregate(progress_rows))
    existing = {canon: result_all[key] for canon, key in HABITAT_KEYS.items() if key in result_all}
    mismatches = [k for k in schema if k in existing and schema[k] != existing[k]]
    # per-episode: the gathered values must be exactly the deduped progress rows (as multisets)
    dedup = {}
    for r in progress_rows:
        dedup[(r.get("scene_id"), r.get("episode_id"))] = r
    for canon, key in (("SR", "success"), ("OS", "os"), ("NE", "ne"), ("SPL", "spl")):
        a = per_ep[canon].float()
        b = torch.tensor([float(r[key]) for r in dedup.values()], dtype=torch.float32)
        if canon == "SPL":
            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        if len(a) != len(b) or not torch.equal(torch.sort(a).values, torch.sort(b).values):
            mismatches.append(f"per_episode:{canon}")
    return {"schema": schema, "existing": existing, "mismatches": mismatches,
            "status": "pass" if not mismatches else "FAIL"}


# ----------------------------------------------------------------------------- isaac
def isaac_records(lmdb_path: str, split_map: dict, world_size: int):
    """(split, info, fail_reason) per episode in finalize_all_results' iteration order."""
    import lmdb
    import msgpack_numpy

    for i in range(world_size):
        d = f"{lmdb_path}/sample_data{i}.lmdb"
        if not os.path.exists(d):
            continue
        env = lmdb.open(d, readonly=True, lock=False, max_readers=256)
        for split, keys in split_map.items():
            for k in keys:
                with env.begin() as txn:
                    v = txn.get(k.encode())
                if v is not None:
                    data = msgpack_numpy.unpackb(v)
                    yield split, data["info"], data.get("fail_reason", "")
        env.close()


def isaac_aggregate(records) -> dict:
    acc = collections.defaultdict(lambda: {"TL": 0.0, "NE": 0.0, "OS": 0.0, "SR": 0.0, "SPL": 0.0,
                                           "coll": 0, "steps": 0, "cfs": 0, "fall": 0, "stuck": 0, "n": 0})
    for split, info, fail_reason in records:
        a = acc[split]
        a["TL"] += info["TL"]
        a["NE"] += 0 if info["NE"] < 0 else info["NE"]
        a["OS"] += 0 if info["osr"] < 0 else info["osr"]
        a["SR"] += info["success"]
        a["SPL"] += info["spl"]
        coll, steps = info.get("collision_count", 0), info.get("steps", 0)
        a["coll"] += coll
        a["steps"] += steps
        a["cfs"] += int(info["success"] > 0 and coll == 0)
        reason = fail_reason or "success"
        a["fall"] += reason == "fall"
        a["stuck"] += reason == "stuck"
        a["n"] += 1
    out = {}
    for split, a in acc.items():
        n = a["n"]
        out[split] = {"TL": round(a["TL"] / n, 4), "NE": round(a["NE"] / n, 4), "FR": round(a["fall"] / n, 4),
                      "StR": round(a["stuck"] / n, 4), "OS": round(a["OS"] / n, 4), "SR": round(a["SR"] / n, 4),
                      "SPL": round(a["SPL"] / n, 4),
                      "CR": round(a["coll"] / a["steps"], 4) if a["steps"] > 0 else 0.0,
                      "CFSR": round(a["cfs"] / n, 4), "Count": n}
    return out


def isaac_check(existing: dict, lmdb_path: str, split_map: dict, world_size: int) -> dict:
    schema = isaac_aggregate(isaac_records(lmdb_path, split_map, world_size))
    mismatches = []
    for split in sorted(set(schema) | set(existing)):
        for canon, key in ISAAC_KEYS.items():
            if schema.get(split, {}).get(canon) != existing.get(split, {}).get(key):
                mismatches.append(f"{split}:{canon}")
    return {"schema": schema, "existing": existing, "mismatches": mismatches,
            "status": "pass" if not mismatches else "FAIL"}


# ----------------------------------------------------------------------------- report
def write_report(output_dir: str, report: dict, append: bool = True) -> None:
    """schema_check_<machine>.jsonl next to result_<machine>.json (separate file: result rows and
    wandb keys stay exactly as before)."""
    machine = os.environ.get("TRAIN_EVAL_TARGET", "")
    path = os.path.join(output_dir, f"schema_check_{machine}.jsonl" if machine else "schema_check.jsonl")
    row = {"timestamp": os.environ.get("EVAL_RUN_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S"), **report}
    with open(path, "a" if append else "w") as f:
        f.write(json.dumps(row) + "\n")
    tag = "OK" if report["status"] == "pass" else "!!! MISMATCH " + ", ".join(report["mismatches"])
    print(f"[metrics_schema] {tag} -> {path}", flush=True)
