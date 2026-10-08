"""metrics_schema must reproduce the evaluators' own numbers bit-for-bit (GPU-free).

Runs the REAL HabitatVLNEvaluator.calc_metrics and ResultLogger.finalize_all_results on synthetic
episodes (NaN SPL, inf NE, duplicated progress rows, several Isaac ranks/splits) and checks that
the schema report passes, and fails when one value is perturbed.

    pytest tests/unit_test/test_metrics_schema.py -q
"""
import json
import math
import random
import types

import torch

from internnav.evaluator import metrics_schema


def _habitat_eval_cls():
    import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: F401 (registers)
    from internnav.evaluator import Evaluator

    return Evaluator.evaluators["habitat_vln"]  # register() returns None, take it from the registry


def _episodes(n, seed=0):
    rnd = random.Random(seed)
    eps = []
    for i in range(n):
        succ = float(rnd.random() < 0.5)
        ne = rnd.uniform(0, 12) if rnd.random() > 0.01 else math.inf
        spl = succ * rnd.random() if rnd.random() > 0.01 else math.nan
        eps.append({"scene_id": f"s{i % 11}", "episode_id": i, "success": succ, "spl": spl,
                    "os": float(succ or rnd.random() < 0.2), "ne": ne, "steps": rnd.randint(5, 300),
                    "tl": rnd.uniform(0, 20), "ndtw_ref": rnd.random(), "collisions": rnd.choice([0, 0, 1, 4])})
    return eps


def _run_habitat(tmp_path, eps, progress_rows):
    cls = _habitat_eval_cls()
    (tmp_path / "progress.json").write_text("".join(json.dumps(r) + "\n" for r in progress_rows))
    order = eps[:]
    random.Random(1).shuffle(order)  # gather order differs from file order
    gm = {"sucs": torch.tensor([e["success"] for e in order]), "spls": torch.tensor([e["spl"] for e in order]),
          "oss": torch.tensor([e["os"] for e in order]), "nes": torch.tensor([e["ne"] for e in order])}
    fake = types.SimpleNamespace(output_path=str(tmp_path), _recorder=object(),
                                 eval_config=types.SimpleNamespace(eval_settings={"metrics_schema": True}))
    cls.calc_metrics(fake, gm)
    return json.loads((tmp_path / "schema_check.jsonl").read_text().splitlines()[-1])


def test_habitat_parity(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAIN_EVAL_TARGET", raising=False)
    eps = _episodes(1839)
    stale = [dict(e, success=1.0 - e["success"]) for e in eps[:30]]  # older duplicate rows, overwritten below
    rep = _run_habitat(tmp_path, eps, stale + eps)
    assert rep["status"] == "pass", rep["mismatches"]
    assert set(rep["schema"]) >= {"SR", "SPL", "OS", "NE", "TL", "nDTW_ref", "CR", "CFSR"}
    assert {"CR", "CFSR"} <= set(rep["existing"])  # compared, not just computed


def test_habitat_detects_mismatch(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAIN_EVAL_TARGET", raising=False)
    eps = _episodes(200)
    bad = [dict(e) for e in eps]
    bad[7]["os"] = 1.0 - bad[7]["os"]  # progress disagrees with the gathered tensors
    rep = _run_habitat(tmp_path, eps, bad)
    assert rep["status"] == "FAIL" and "per_episode:OS" in rep["mismatches"]


def _write_isaac_lmdb(root, world_size, split_map, rnd):
    import lmdb
    import msgpack_numpy

    for r in range(world_size):
        env = lmdb.open(f"{root}/sample_data{r}.lmdb", map_size=1 << 26)
        with env.begin(write=True) as txn:
            for keys in split_map.values():
                for k in keys[r::world_size]:
                    succ = float(rnd.random() < 0.5)
                    info = {"TL": rnd.uniform(0, 20), "NE": rnd.uniform(-1, 10), "osr": rnd.choice([-1, 0.0, 1.0]),
                            "success": succ, "spl": succ * rnd.random(), "steps": rnd.randint(1, 400),
                            "collision_count": rnd.randint(0, 3)}
                    reason = rnd.choice(["", "fall", "stuck", "timeout"])
                    txn.put(k.encode(), msgpack_numpy.packb({"info": info, "finish_status": "x", "fail_reason": reason},
                                                            use_bin_type=True))
        env.close()


def test_isaac_parity(tmp_path, monkeypatch):
    from internnav.evaluator.utils.result_logger import ResultLogger

    monkeypatch.setenv("EVAL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("TRAIN_EVAL_TARGET", raising=False)
    split_map = {"val_seen": [f"a_{i}" for i in range(97)], "val_unseen": [f"b_{i}" for i in range(131)]}
    _write_isaac_lmdb(tmp_path, 3, split_map, random.Random(2))
    rl = ResultLogger.__new__(ResultLogger)  # skip dataset loading; finalize only needs these
    rl.lmdb_path, rl.split_map, rl.name = str(tmp_path), split_map, "t"
    existing = rl.finalize_all_results(0, 3)
    rep = metrics_schema.isaac_check(existing, str(tmp_path), split_map, 3)
    assert rep["status"] == "pass", rep["mismatches"]
    existing["val_unseen"]["SPL"] += 1e-4
    assert metrics_schema.isaac_check(existing, str(tmp_path), split_map, 3)["mismatches"] == ["val_unseen:SPL"]
