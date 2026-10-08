"""GPU-free end-to-end check of eval_recorder.py: a REAL HabitatEnv (no sensors, no renderer) driven by a
stub policy -- ShortestPathFollower, with random turns on odd episodes (-> failures) -- instead of the VLM.
Mimics the evaluator: LOOKDOWN/LOOKUP tilts between moves (must not become frames), a fake S2
generate() every 3 moves, and HabitatVLNEvaluator._write_progress at episode end. The output is a
valid eval log dir, handy for developing the dashboard (app.py --root <out root>).

One scene per process (habitat-sim without a renderer segfaults on a scene switch):
    CUDA_VISIBLE_DEVICES="" python scripts/eval_dashboard/check_recorder.py <out>/harness/run/checkpoint-123/logs/harness 4 2azQ1b91cZZ
    CUDA_VISIBLE_DEVICES="" python scripts/eval_dashboard/check_recorder.py <same dir> 4 8194nk5LbLH final
('final' writes the result_h200.json row; CHECK_DECISION=1 also records decision metrics;
CHECK_FRAMES=1 injects a synthetic rgb so failed episodes save front-view frames.)
"""
import json
import os
import random
import sys
import types

import numpy as np

OUT = sys.argv[1]
N_EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SCENE = sys.argv[3]  # one scene per process: habitat-sim without a renderer segfaults on scene switch
FINAL = len(sys.argv) > 4 and sys.argv[4] == "final"  # last process writes the result row

import habitat  # noqa: E402
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower  # noqa: E402
from habitat_baselines.config.default import get_config as get_habitat_config  # noqa: E402

from internnav.configs.evaluator import EnvCfg  # noqa: E402
from internnav.env.habitat_env import HabitatEnv  # noqa: E402
from internnav.habitat_extensions.vln.eval_recorder import EvalRecorder, aggregate_progress  # noqa: E402
import internnav.habitat_extensions.vln.habitat_vln_evaluator  # noqa: E402,F401  (registers habitat_vln)
from internnav.evaluator import Evaluator  # noqa: E402
HabitatVLNEvaluator = Evaluator.evaluators["habitat_vln"]  # register() returns None

cfg = get_habitat_config("scripts/eval/configs/vln_r2r_full_ld30.yaml")
with habitat.config.read_write(cfg):
    # no GPU: no sensors, no renderer (the recorder only needs agent state + pathfinder)
    cfg.habitat.simulator.agents.main_agent.sim_sensors = {}
    cfg.habitat.simulator.create_renderer = False
    cfg.habitat.dataset.content_scenes = [SCENE]
    # the real evaluator adds this measure in HabitatVLNEvaluator.__init__
    from habitat.config.default_structured_configs import CollisionsMeasurementConfig

    cfg.habitat.task.measurements.update({"collisions": CollisionsMeasurementConfig()})
os.makedirs(OUT, exist_ok=True)
# rank/world_size sharding picks every k-th episode of the scene
env = HabitatEnv(EnvCfg(env_type="habitat", env_settings={
    "habitat_config": cfg, "rank": 0, "world_size": 7, "output_path": OUT, "max_episodes": N_EPS}))


class FakeModel:
    def generate(self, input_ids=None, **kw):
        return types.SimpleNamespace(sequences=[list(range(input_ids.shape[1] + 3))])


class FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return random.choice(["↑↑↑↑", "←←", "→→→", f"{random.randint(0, 640)} {random.randint(0, 480)}", "STOP"])


ev = types.SimpleNamespace(env=env, model=FakeModel(), processor=types.SimpleNamespace(tokenizer=FakeTok()),
                           config=cfg, rank=0, output_path=OUT)
run_meta = {"run_stamp": "20260930_000000", "ckpt": "/fake/ckpt/checkpoint-123", "ckpt_step": 123,
            "config": "harness/stub_policy", "machine": "h200", "git_commit": "harness"}
if os.environ.get("CHECK_FRAMES") == "1":
    # no camera here: inject a synthetic 640x480 "rgb" so the front-view frame path runs end to end
    _n, _reset0, _step0 = [0], env.reset, env.step

    def _fake(obs):
        if obs is not None:
            img = np.zeros((480, 640, 3), np.uint8)
            img[..., 0] = (_n[0] * 7) % 255
            _n[0] += 1
            obs["rgb"] = img
        return obs

    env.reset = lambda: _fake(_reset0())

    def _step(a):
        ret = _step0(a)
        _fake(ret[0])
        return ret

    env.step = _step
ev._recorder = EvalRecorder(ev, os.path.join(OUT, "raw", run_meta["run_stamp"]), run_meta,
                            decision_metrics=os.environ.get("CHECK_DECISION") == "1")
ev._recorder.install()

random.seed(0)
n_ep, frames_expected = 0, {}
while env.is_running:
    obs = env.reset()
    if not env.is_running or obs is None:
        break
    ep = env.get_current_episode()
    follower = ShortestPathFollower(env._env.sim, goal_radius=0.5, return_one_hot=False)
    goal = ep.goals[0].position
    noisy = n_ep % 2 == 1
    done, moves = False, 0
    while not done and moves < 200:
        env.step(5); env.step(5); env.step(4); env.step(4)  # depth-gather tilts: not frames
        if moves % 3 == 0:
            ev.model.generate(input_ids=np.zeros((1, 10)))
        a = follower.get_next_action(goal)
        a = 0 if a is None else int(a)
        if noisy and a != 0 and random.random() < 0.35:
            a = random.choice([1, 2, 3])
        if noisy and moves > 60 and random.random() < 0.05:
            a = 0  # early stop
        _, _, done, _ = env.step(a)
        moves += 1
    m = env.get_metrics()
    scene_id, episode_id = ep.scene_id.split("/")[-2], int(ep.episode_id)
    result = {"scene_id": scene_id, "episode_id": episode_id, "success": m["success"], "spl": m["spl"],
              "os": m["oracle_success"], "ne": m["distance_to_goal"], "steps": moves,
              "episode_instruction": ep.instruction.instruction_text}
    HabitatVLNEvaluator._write_progress(ev, result)
    frames_expected[f"{scene_id}_{episode_id:04d}"] = moves + 1
    n_ep += 1
    print(f"[{n_ep}] {scene_id}_{episode_id:04d} success={m['success']:.0f} ne={m['distance_to_goal']:.2f} moves={moves}")
env.close()

# ---- checks ----
raw = os.path.join(OUT, "raw", run_meta["run_stamp"])
rows = [json.loads(l) for l in open(os.path.join(OUT, "progress.json"))]
assert len(rows) >= n_ep and all("tl" in r and "ndtw_ref" in r and r["run_stamp"] == run_meta["run_stamp"] for r in rows)
for name, n_frames in frames_expected.items():
    d = json.load(open(os.path.join(raw, "episodes", name + ".json")))
    assert len(d["frames"]) == n_frames, (name, len(d["frames"]), n_frames)  # tilts excluded
    assert d["frames"][0]["action"] is None and all(f["action"] in (0, 1, 2, 3) for f in d["frames"][1:])
    assert any("gen" in f for f in d["frames"]), name
    assert os.path.exists(os.path.join(raw, "topdown", d["meta"]["topdown"] + ".json"))
    # recorder TL == habitat-independent recomputation; d2g of last frame == habitat NE
    assert abs(d["frames"][-1]["dist_to_goal"] - d["metrics"]["ne"]) < 1e-3, name
agg = aggregate_progress(os.path.join(OUT, "progress.json"))
assert {"tls_all", "ndtw_refs_all"} <= set(agg)
# result row exactly as distributed_base builds it: the REAL calc_metrics over float32 per-episode
# tensors in gather order (single rank here = progress file order), + length / stamp / run meta
import torch  # noqa: E402

gm = {k: torch.tensor([float(r[src]) for r in rows]) for k, src in
      (("sucs", "success"), ("spls", "spl"), ("oss", "os"), ("nes", "ne"))}
ev.eval_config = types.SimpleNamespace(eval_settings={"metrics_schema": FINAL})
res = HabitatVLNEvaluator.calc_metrics(ev, gm)
res.update(length=len(rows), timestamp=run_meta["run_stamp"], **run_meta)
if FINAL:
    with open(os.path.join(OUT, "result_h200.json"), "a") as f:
        f.write(json.dumps(res) + "\n")
print("aggregate:", {k: round(v, 3) for k, v in agg.items()}, "SR", round(res["sucs_all"], 3))
print("HARNESS OK")
