"""Raw per-episode eval output for later analysis and the eval dashboard.

Enabled by ``eval_settings['save_raw']`` (default False -> evaluator behaves exactly as before).
The recorder never edits the evaluator's episode loops: it wraps ``env.reset``/``env.step`` and
``model.generate`` and is finalized from ``HabitatVLNEvaluator._write_progress``.

Layout (JSON only), under ``<output_path>/raw/<run_stamp>/``::

    run.json                           run meta: ckpt, ckpt_step, config, git commit, machine
    episodes/<scene>_<episode:04d>.json
        {"meta":    {scene_id, episode_id, instruction, reference_path, start_position,
                     goal_position, success_distance, topdown, ckpt, run_stamp, rank, ...},
         "frames":  [{i, pose:{position:[x,y,z], yaw}, action, dist_to_goal, gen?, collision?}],
         "metrics": {success, spl, os, ne, steps, tl, ndtw_ref, collisions, ...}}
    topdown/<scene>_y<floor>.json      navmesh top-down occupancy (row-major RLE)
    frames/<scene>_<episode:04d>/<i:04d>.jpg   front-view image of frame i (downscaled; by default only
                                       for failed episodes, eval_settings['raw_frames']); the frame
                                       carries "image" and meta "frames_dir" when saved

Coordinates are habitat world metres (y up). A frame is one movement action (STOP/FORWARD/LEFT/
RIGHT); the look-down/up camera tilts the evaluator issues to gather depth are not frames.
``gen`` is the S2 text generated right before that frame's action (several calls joined by " | ").
With ``eval_settings['decision_metrics']`` each S2 call also leaves a ``decision`` record on the frame
(EvalRecorder._decide: exact 3 m STOP oracle, reference ShortestPathFollower action and reference
pixel goal) and per-episode ``dec_*`` counters in metrics/progress (aggregate_progress -> result row).

``ndtw_ref`` is nDTW against the episode's R2R ``reference_path`` resampled every 0.25 m (d_th =
success distance), with repeated agent positions dropped. It approximates, but is NOT, the official
VLN-CE nDTW, which uses dense shortest-path GT locations (only shipped for RxR here, see
measures.NDTW); use it to compare our own runs, not against paper numbers.
"""
import json
import math
import os
import zlib

MOVE_ACTIONS = (0, 1, 2, 3)  # STOP, FORWARD, LEFT, RIGHT (action_code in the evaluator)
TOPDOWN_MPP = 0.05  # metres per top-down pixel


def _yaw(rotation) -> float:
    """Heading in the dashboard's 2D frame (x, -z): 0 = +x, counter-clockwise positive."""
    import quaternion  # numpy-quaternion, habitat dependency

    fwd = quaternion.rotate_vectors(rotation, [0.0, 0.0, -1.0])  # habitat forward is -z
    return math.atan2(-fwd[2], fwd[0])


def path_length(points) -> float:
    return sum(math.dist(points[i - 1], points[i]) for i in range(1, len(points)))


def densify(path, step: float = 0.25):
    """Resample a polyline every `step` m (R2R reference paths are ~2 m apart graph nodes)."""
    if len(path) < 2:
        return [list(p) for p in path]
    out = [list(path[0])]
    for a, b in zip(path, path[1:]):
        n = max(1, int(math.dist(a, b) // step))
        out += [[a[k] + (b[k] - a[k]) * i / n for k in range(len(a))] for i in range(1, n + 1)]
    return out


def dedupe(path):
    """Drop consecutive repeated positions (turn actions), as measures.NDTW does."""
    return [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]


def ndtw(pred, ref, success_distance: float) -> float:
    """nDTW = exp(-DTW(pred, ref) / (|ref| * d_th)) (Ilharco et al. 2019), exact O(n*m) DTW.

    Callers pass dedupe(agent positions) and densify(reference_path) so both sides have the
    ~0.25 m spacing the formula assumes."""
    if not pred or not ref:
        return 0.0
    inf = float("inf")
    prev = [0.0] + [inf] * len(ref)
    for p in pred:
        cur = [inf] * (len(ref) + 1)
        for j, r in enumerate(ref, 1):
            cur[j] = math.dist(p, r) + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return math.exp(-prev[-1] / (len(ref) * success_distance))


def floor_height(ys, bin_m: float = 0.5) -> float:
    """Height of the floor the agent spent most frames on: the most common `bin_m` bin, then the
    median inside it. Not the start (R2R often starts on a stair) and not the overall median (a
    stair-crossing episode's median lands mid-staircase, where the map is nearly empty)."""
    bins = {}
    for y in ys:
        bins.setdefault(round(y / bin_m), []).append(y)
    best = sorted(max(bins.values(), key=len))
    return best[len(best) // 2]


def rle(flat):
    """[v, v, v, w] -> [v, 3, w, 1] (row-major occupancy grids are long runs)."""
    out, prev, n = [], None, 0
    for v in flat:
        if v == prev:
            n += 1
        else:
            if prev is not None:
                out += [prev, n]
            prev, n = v, 1
    if prev is not None:
        out += [prev, n]
    return out


def _atomic_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


class EvalRecorder:
    def __init__(self, evaluator, out_dir: str, run_meta: dict, decision_metrics: bool = False,
                 frames: str = "fail", frame_scale: float = 0.5, frame_sample: float = 0.05):
        self.ev = evaluator
        self.out_dir = out_dir
        self.run_meta = run_meta
        # front-view JPEGs per frame (save_frames): which episodes keep them, at what scale
        assert frames in FRAME_POLICIES, f"unknown frames policy {frames!r} (expected one of {FRAME_POLICIES})"
        self.frames_policy, self.frame_scale, self.frame_sample = frames, frame_scale, frame_sample
        self.tilt, self.jpgs = 0, {}
        self.run_meta = {**run_meta, "raw_frames": frames, "raw_frame_scale": frame_scale, "raw_frame_sample": frame_sample}
        # per-S2-call decision records (_decide), off by default; see the module docstring
        self.decision_metrics = decision_metrics
        self.last_obs, self.follower, self.pending_dec = None, None, []
        self.success_distance = float(
            evaluator.config.habitat.task.measurements.success.success_distance
        )
        self.meta, self.frames, self.pending_gen = None, [], None
        os.makedirs(os.path.join(out_dir, "episodes"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "topdown"), exist_ok=True)
        if evaluator.rank == 0:
            _atomic_json(os.path.join(out_dir, "run.json"), self.run_meta)

    # ------------------------------------------------------------------ hooks
    def install(self):
        env, model = self.ev.env, self.ev.model
        orig_reset, orig_step, orig_generate = env.reset, env.step, model.generate

        def reset():
            obs = orig_reset()
            if obs is not None:
                self.last_obs = obs
                self._start_episode()
            return obs

        def step(action):
            ret = orig_step(action)
            self.last_obs = ret[0]
            # camera tilt in look steps (LOOKDOWN=5 / LOOKUP=4): front-view frames are tilt 0
            self.tilt += {5: 1, 4: -1}.get(int(action), 0)
            self._on_step(action, ret[3])
            return ret

        def generate(*args, **kwargs):
            out = orig_generate(*args, **kwargs)
            self._on_generate(kwargs.get("input_ids"), out)
            return out

        env.reset, env.step, model.generate = reset, step, generate

    def _sim(self):
        return self.ev.env._env.sim

    def _pose(self):
        st = self._sim().get_agent_state()
        return {"position": [round(float(v), 4) for v in st.position], "yaw": round(_yaw(st.rotation), 4)}

    def _start_episode(self):
        ep = self.ev.env.get_current_episode()
        scene_id = ep.scene_id.split("/")[-2]
        pose = self._pose()
        self.meta = {
            "scene_id": scene_id,
            "episode_id": int(ep.episode_id),
            "trajectory_id": getattr(ep, "trajectory_id", None),
            "instruction": ep.instruction.instruction_text,
            "reference_path": [[float(v) for v in p] for p in (ep.reference_path or [])],
            "start_position": [float(v) for v in ep.start_position],
            "start_rotation": [float(v) for v in ep.start_rotation],
            "goal_position": [float(v) for v in ep.goals[0].position] if ep.goals else None,
            "geodesic_distance": (getattr(ep, "info", None) or {}).get("geodesic_distance"),
            "success_distance": self.success_distance,
            "topdown": None,  # set in finish(): the floor the agent spent most time on
            "rank": self.ev.rank,
            **self.run_meta,
        }
        d2g = self.ev.env.get_metrics().get("distance_to_goal")
        self.frames = [{"i": 0, "pose": pose, "action": None, "dist_to_goal": _num(d2g)}]
        self.pending_gen, self.pending_dec = None, []
        self.tilt, self.jpgs = 0, {}
        self._capture(0)
        if self.decision_metrics:
            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

            self.follower = ShortestPathFollower(self._sim(), goal_radius=0.25, return_one_hot=False)

    def _on_step(self, action, info):
        if self.meta is None:
            return
        a = int(action)
        if a not in MOVE_ACTIONS:
            return
        # the action is attached to the frame it produced (pose after the step)
        frame = {"i": len(self.frames), "pose": self._pose(), "action": a,
                 "dist_to_goal": _num((info or {}).get("distance_to_goal"))}
        coll = (info or {}).get("collisions")  # habitat Collisions measure: this move bumped into geometry
        if isinstance(coll, dict) and coll.get("is_collision"):
            frame["collision"] = True
        if self.pending_gen is not None:
            frame["gen"], self.pending_gen = self.pending_gen, None
        if self.pending_dec:
            frame["decision"], self.pending_dec = self.pending_dec, []
        self.frames.append(frame)
        self._capture(frame["i"])

    def _capture(self, i: int):
        """Buffer frame i's front view as a downscaled JPEG; written in finish() if the policy keeps it."""
        if self.frames_policy == "none" or self.tilt != 0 or self.last_obs is None or "rgb" not in self.last_obs:
            return
        import io

        from PIL import Image

        img = Image.fromarray(self.last_obs["rgb"][..., :3])
        if self.frame_scale != 1.0:
            img = img.resize((round(img.width * self.frame_scale), round(img.height * self.frame_scale)), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        self.jpgs[i] = buf.getvalue()

    def _keep_frames(self, success) -> bool:
        if self.frames_policy == "none":
            return False
        elif self.frames_policy == "all":
            return True
        elif self.frames_policy == "fail":
            return not success
        elif self.frames_policy == "fail+sample":
            # failures + a fixed pseudo-random share of successes as a control group (same set every run)
            key = f"{self.meta['scene_id']}_{self.meta['episode_id']}".encode()
            return not success or zlib.crc32(key) % 1000 < self.frame_sample * 1000
        else:
            assert False, f"unreachable frames_policy={self.frames_policy!r}"

    def _on_generate(self, input_ids, out):
        seq = getattr(out, "sequences", out)
        n_in = input_ids.shape[1] if input_ids is not None else 0
        text = self.ev.processor.tokenizer.decode(seq[0][n_in:], skip_special_tokens=True)
        # several S2 calls before one move ("↓" then "x y") are all kept
        self.pending_gen = text if self.pending_gen is None else f"{self.pending_gen} | {text}"
        if self.decision_metrics and self.meta is not None:
            self.pending_dec.append(self._decide(text))

    # ------------------------------------------------------------------ decision metrics
    def _decide(self, text: str) -> dict:
        """One S2 call, parsed like the evaluator (digits -> pixel goal [row, col]; else STOP / arrows).

        exact:      oracle_stop = geodesic distance to goal < success distance (the Success criterion)
        reference:  spf_action (ShortestPathFollower to the goal) and ref_pixel (farthest visible point of
                    the geodesic path, label_rules.EMPIRICAL_RULE) -- validated as weak references,
                    see .claude/memory/understanding_s2_label_generation.md"""
        import re

        d2g = self.ev.env.get_metrics().get("distance_to_goal")
        t = text.strip()
        if re.search(r"\d", t):
            kind = "pixel"
        elif t.startswith("STOP"):
            kind = "stop"
        elif t.startswith("↓"):
            kind = "lookdown"
        elif t[:1] in ("↑", "←", "→"):
            kind = "turn"
        else:
            kind = "other"
        rec = {"type": kind, "d2g": _num(d2g), "oracle_stop": bool(d2g is not None and d2g < self.success_distance)}
        goal = self.meta["goal_position"]
        spf = self.follower.get_next_action(goal) if (self.follower is not None and goal) else None
        rec["spf_action"] = None if spf is None else int(spf)
        first = {"STOP": 0, "↑": 1, "←": 2, "→": 3}.get(t[:4] if t.startswith("STOP") else t[:1])
        if kind in ("stop", "turn"):
            rec["pred_action"] = first
        if kind == "pixel":
            nums = [int(c) for c in re.findall(r"\d+", t)]
            if len(nums) >= 2:
                rec["pixel"] = [nums[0], nums[1]]  # model text is "x y" (col row)
                ref = self._ref_pixel()
                rec["ref_pixel"] = None if ref is None else [round(ref[0], 1), round(ref[1], 1)]
                if ref is not None:
                    rec["pixel_err"] = round(math.dist(ref, rec["pixel"]), 2)
        return rec

    def _ref_pixel(self):
        """Reference pixel in the current (look-down) rgb view, or None (out of view / no camera)."""
        import quaternion

        from internnav.habitat_extensions.vln import label_rules as LR

        st = self._sim().get_agent_state()
        cam = st.sensor_states.get("rgb")
        cfg = getattr(self.ev, "sim_sensors_config", None)
        if cam is None or cfg is None or (cfg.rgb_sensor.width, cfg.rgb_sensor.height) != (LR.W, LR.H):
            return None
        fx = cfg.rgb_sensor.width / (2 * math.tan(math.radians(cfg.rgb_sensor.hfov) / 2))
        path = _geodesic(self._sim(), st.position, self.meta["goal_position"])
        depth = None
        if self.last_obs is not None and "depth" in self.last_obs:
            depth = self.last_obs["depth"].reshape(LR.H, LR.W)
            if getattr(cfg.depth_sensor, "normalize_depth", True):
                depth = depth * (cfg.depth_sensor.max_depth - cfg.depth_sensor.min_depth) + cfg.depth_sensor.min_depth
        ref = LR.reference_pixel_habitat(densify([list(p) for p in path])[1:], cam.position,
                                         quaternion.as_rotation_matrix(cam.rotation), fx, (LR.W - 1) / 2,
                                         (LR.H - 1) / 2, st.position, depth=depth)
        return None if ref is None else ref[1]

    def _ensure_topdown(self, scene_id: str, height: float) -> str:
        from habitat.utils.visualizations import maps

        key = f"{scene_id}_y{round(height * 2) / 2:g}"  # 0.5 m floor bins
        path = os.path.join(self.out_dir, "topdown", key + ".json")
        if not os.path.exists(path):
            pf = self._sim().pathfinder
            grid = maps.get_topdown_map(pf, height, draw_border=True, meters_per_pixel=TOPDOWN_MPP)
            lower, _ = pf.get_bounds()
            _atomic_json(path, {
                "scene_id": scene_id, "height": float(height), "meters_per_pixel": TOPDOWN_MPP,
                # grid row r / col c = world z = origin_z + r*mpp, x = origin_x + c*mpp
                "origin_x": float(lower[0]), "origin_z": float(lower[2]),
                "rows": int(grid.shape[0]), "cols": int(grid.shape[1]),
                "values": {"0": "occupied", "1": "navigable", "2": "border"},
                "rle": rle(grid.ravel().tolist()),
            })
        return key

    # ------------------------------------------------------------------ episode end
    def finish(self, result: dict) -> dict:
        """Called with the evaluator's progress row; writes the episode file, returns tl/ndtw_ref."""
        if self.meta is None:
            return {}
        pts = [f["pose"]["position"] for f in self.frames]
        self.meta["topdown"] = self._ensure_topdown(self.meta["scene_id"], floor_height([p[1] for p in pts]))
        extra = {"tl": round(path_length(pts), 4),
                 "ndtw_ref": round(ndtw(dedupe(pts), densify(self.meta["reference_path"]), self.success_distance), 4),
                 "collisions": sum(1 for f in self.frames if f.get("collision"))}
        if self.decision_metrics:
            if self.pending_dec:  # S2 calls after the last move (episode ended): keep them on the last frame
                self.frames[-1].setdefault("decision", []).extend(self.pending_dec)
                self.pending_dec = []
            extra.update(decision_counts([d for f in self.frames for d in f.get("decision", [])]))
        metrics = {k: v for k, v in result.items() if k not in ("scene_id", "episode_id", "episode_instruction")}
        name = f"{self.meta['scene_id']}_{self.meta['episode_id']:04d}.json"
        if self.jpgs and self._keep_frames(bool(result.get("success"))):
            rel = os.path.join("frames", name[:-5])
            os.makedirs(os.path.join(self.out_dir, rel), exist_ok=True)
            for i, b in self.jpgs.items():
                with open(os.path.join(self.out_dir, rel, f"{i:04d}.jpg"), "wb") as f:
                    f.write(b)
                self.frames[i]["image"] = f"{i:04d}.jpg"
            self.meta["frames_dir"] = rel  # relative to raw/<run_stamp>/
        self.jpgs = {}
        _atomic_json(os.path.join(self.out_dir, "episodes", name),
                     {"meta": self.meta, "frames": self.frames, "metrics": {**metrics, **extra}})
        self.meta, self.frames = None, []
        return extra


def _geodesic(sim, a, b):
    import habitat_sim

    sp = habitat_sim.ShortestPath()
    sp.requested_start, sp.requested_end = a, b
    return [list(p) for p in sp.points] if sim.pathfinder.find_path(sp) else []


def decision_counts(decisions) -> dict:
    """Per-episode counters (summed over episodes by aggregate_progress). STOP is judged on every S2
    call: predicted STOP vs oracle (within the success distance)."""
    c = dict.fromkeys(DECISION_KEYS, 0)
    for d in decisions:
        pred_stop, oracle = d["type"] == "stop", d["oracle_stop"]
        c["dec_n"] += 1
        c["dec_stop_tp"] += pred_stop and oracle
        c["dec_stop_fp"] += pred_stop and not oracle
        c["dec_stop_fn"] += oracle and not pred_stop
        if d.get("pred_action") is not None and d.get("spf_action") is not None:
            c["dec_spf_n"] += 1
            c["dec_spf_agree"] += d["pred_action"] == d["spf_action"]
        if "pixel" in d:
            c["dec_pixel_n"] += 1
            if d.get("ref_pixel") is not None:
                c["dec_pixel_ref_n"] += 1
                c["dec_pixel_err_sum"] = round(c["dec_pixel_err_sum"] + d["pixel_err"], 3)
                c["dec_pixel_le30"] += d["pixel_err"] <= 30
    return c


FRAME_POLICIES = ("fail", "fail+sample", "all", "none")

DECISION_KEYS = ("dec_n", "dec_stop_tp", "dec_stop_fp", "dec_stop_fn", "dec_spf_n", "dec_spf_agree",
                 "dec_pixel_n", "dec_pixel_ref_n", "dec_pixel_err_sum", "dec_pixel_le30")


def _num(v):
    return round(float(v), 4) if isinstance(v, (int, float)) and math.isfinite(v) else None


def collision_aggregate(rows) -> dict:
    """Isaac's definitions (result_logger / metrics_schema): CR = total collisions / total steps,
    CFSR = share of episodes that succeed with no collision. Rows without `collisions` are skipped."""
    rows = [r for r in rows if isinstance(r.get("collisions"), (int, float))]
    if not rows:
        return {}
    steps = sum(r.get("steps", 0) for r in rows)
    return {"crs_all": sum(r["collisions"] for r in rows) / steps if steps else 0.0,
            "cfsrs_all": sum(1 for r in rows if r.get("success", 0) > 0 and r["collisions"] == 0) / len(rows)}


def aggregate_progress(progress_path: str) -> dict:
    """Mean tl / ndtw_ref, CR / CFSR, decision rates over progress.json rows that carry them (last row
    per episode wins)."""
    rows = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            for line in f:
                r = json.loads(line)
                rows[(r.get("scene_id"), r.get("episode_id"))] = r
    out = {}
    for key, name in (("tl", "tls_all"), ("ndtw_ref", "ndtw_refs_all")):
        vals = [r[key] for r in rows.values() if isinstance(r.get(key), (int, float))]
        if vals:
            out[name] = sum(vals) / len(vals)
    out.update(collision_aggregate(rows.values()))
    dec = [r for r in rows.values() if "dec_n" in r]
    if dec:  # decision metrics (eval_settings['decision_metrics']): exact STOP, reference SPF / pixel
        t = {k: sum(r.get(k, 0) for r in dec) for k in DECISION_KEYS}
        if t["dec_stop_tp"] + t["dec_stop_fp"]:
            out["dec_stop_precision"] = t["dec_stop_tp"] / (t["dec_stop_tp"] + t["dec_stop_fp"])
        if t["dec_stop_tp"] + t["dec_stop_fn"]:
            out["dec_stop_recall"] = t["dec_stop_tp"] / (t["dec_stop_tp"] + t["dec_stop_fn"])
        if t["dec_spf_n"]:
            out["dec_spf_agree_ref"] = t["dec_spf_agree"] / t["dec_spf_n"]
        if t["dec_pixel_ref_n"]:
            out["dec_pixel_err_ref"] = t["dec_pixel_err_sum"] / t["dec_pixel_ref_n"]
            out["dec_pixel_le30_ref"] = t["dec_pixel_le30"] / t["dec_pixel_ref_n"]
        if t["dec_pixel_n"]:
            out["dec_pixel_ref_found"] = t["dec_pixel_ref_n"] / t["dec_pixel_n"]
    return out


if __name__ == "__main__":
    # self-check for the pure helpers
    assert rle([0, 0, 1, 1, 1, 2]) == [0, 2, 1, 3, 2, 1] and rle([]) == []
    assert path_length([[0, 0, 0], [3, 0, 4], [3, 0, 4]]) == 5.0
    ref = [[0, 0, 0], [0, 0, -4]]
    assert ndtw(ref, ref, 3.0) == 1.0
    assert abs(ndtw([[0, 0, 0], [0, 0, -1]], ref, 3.0) - math.exp(-3 / 6)) < 1e-9
    assert ndtw([], ref, 3.0) == 0.0
    # 27 frames on the ground floor + a staircase: map the ground floor, not mid-stairs
    assert floor_height([0.1] * 27 + [0.75, 0.95, 1.24, 1.5, 2.0, 2.5, 2.9, 2.9, 2.9] * 3) == 0.1
    assert densify([[0, 0, 0], [0, 0, -1]]) == [[0, 0, 0], [0, 0, -0.25], [0, 0, -0.5], [0, 0, -0.75], [0, 0, -1]]
    assert dedupe([[1, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0]]) == [[1, 0, 0], [2, 0, 0], [1, 0, 0]]
    dense = densify(ref)
    assert ndtw(dedupe([p for p in dense for _ in (0, 1)]), dense, 3.0) == 1.0  # turns don't penalize
    ds = [{"type": "stop", "oracle_stop": True, "pred_action": 0, "spf_action": 0},
          {"type": "stop", "oracle_stop": False, "pred_action": 0, "spf_action": 1},
          {"type": "pixel", "oracle_stop": True, "pixel": [1, 2], "ref_pixel": [1, 32], "pixel_err": 30.0},
          {"type": "turn", "oracle_stop": False, "pred_action": 2, "spf_action": 2}]
    c = decision_counts(ds)
    assert (c["dec_stop_tp"], c["dec_stop_fp"], c["dec_stop_fn"]) == (1, 1, 1)
    assert (c["dec_spf_n"], c["dec_spf_agree"], c["dec_pixel_le30"]) == (3, 2, 1)
    ca = collision_aggregate([{"collisions": 2, "steps": 10, "success": 1.0}, {"collisions": 0, "steps": 30, "success": 1.0},
                              {"collisions": 0, "steps": 10, "success": 0.0}, {"tl": 1.0}])
    assert ca == {"crs_all": 2 / 50, "cfsrs_all": 1 / 3}, ca
    print("eval_recorder self-check OK")
