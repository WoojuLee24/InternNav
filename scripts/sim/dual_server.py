"""Sim adapter for the vln-deploy S1/S2 module split — same HTTP API as vln_server.py (bridge unchanged).

  vln_bridge_node --HTTP /eval_dual--> dual_server --ZMQ (vln-deploy deploy/common/interface.py S2Client/S1Client)
      --> vln-deploy S2 module (quantization_kit/async_zmq/s2_service.py, --s2-backend hf|gguf)
      --> vln-deploy S1 module (deploy/s1_module/service.py, --s1-backend hf|lite)

--mode sync   reproduces InternVLAN1AsyncAgent.step() + vln_server.py exactly (S2 every plan_step_gap+1 frames
              and when there is no output; blocking) -> used to verify the split gives identical results.
--mode async  never waits for S2 once a plan exists: each request is answered by S1 with the latest latent;
              S2 infers in its own process and its result is used as soon as it arrives. Frames arriving while
              S2 is busy go to S2 as history ("accumulate" = step_no_infer). After a discrete S2 action the plan is
              cleared, so the next request waits for S2 (as in sync).

The pixel goal refers to the frame S2 used: replies carry `pixel_frame_id` (the request `frame_id` of that frame),
the bridge projects it with that frame's depth / pose. Extra reply field: timing {s1_ms, s2_ms, wait_ms, ...}.

Usage: python3 scripts/sim/dual_server.py --mode async --port 5801     (modules: scripts/sim/run_dual.sh)
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

DEPLOY = Path(__file__).resolve().parents[2] / "vln-deploy"
sys.path.insert(0, str(DEPLOY))
from deploy.common.config import DeployConfig  # noqa: E402
from deploy.common.interface import S1Client, S2Client  # noqa: E402


class Orchestrator:
    def __init__(self, args):
        self.args = args
        cfg = DeployConfig.from_env()
        self.s2 = S2Client(cfg)
        self.s1 = S1Client(cfg)
        self.gap = args.plan_step_gap
        self.instruction = None
        self.reset_state()

    def reset_state(self):
        # mirrors InternVLAN1AsyncAgent counters (episode_idx / last_s2_idx) and outputs
        self.episode_idx = 0
        self.last_s2_idx = -100
        self.output_action = None
        self.latent = None
        # (rgb, depth, frame_id) S1 memory frame of the latent. With a 30° look-down camera it is that camera's frame
        # (S1 trains and is evaluated on 30° memory + current frames: internvla_n1_lerobot_dataset.py traj_images,
        # habitat_vln_evaluator.py images_dp/depths_dp), else the forward frame.
        self.goal = None
        self.goal_frame = "fpv"  # camera the S2 pixel refers to: "lookdown" when S2 answered from the 30° frame
        self.pixel = None
        self.inflight = None  # (seq, rgb, depth, frame_id, t_sent, look)
        self.fresh = None     # S2 result not yet reported to the client

    # ------------------------------------------------------------------ S2
    def s2_reset(self, instruction):
        self.instruction = instruction
        self.reset_state()
        self.s2.reset()

    def s2_due(self):
        no_output = self.output_action is None and self.latent is None
        return (self.episode_idx - self.last_s2_idx > self.gap) or no_output

    def s2_infer(self, rgb, depth, frame_id, look=None):
        # look = (rgb, depth) of the 30° look-down camera: S2 re-queries "↓" with it (as vln_server.py).
        # None -> the S2 module re-queries with the same frame (realworld semantics)
        seq = self.s2.fire(rgb, self.instruction, lookdown_rgb=None if look is None else look[0])
        self.inflight = (seq, rgb, depth, frame_id, time.time(), look)
        self.episode_idx += 1  # step_s2 (non look-down) increments episode_idx
        self.last_s2_idx = self.episode_idx

    def s2_observe(self, rgb):
        self.s2.accumulate(rgb)
        self.episode_idx += 1  # step_no_infer increments episode_idx

    def s2_poll(self, timeout_s):
        """Take the in-flight S2 result if it arrives within timeout_s (stale seqs are dropped)."""
        deadline = time.time() + timeout_s
        while self.inflight is not None:
            r = self.s2.try_recv(self.inflight[0])
            if r is not None:
                _, rgb, depth, frame_id, t_sent, look = self.inflight
                self.inflight = None
                if r.kind == "action":
                    self.output_action, self.latent, self.pixel = list(r.action), None, None
                elif r.kind == "latent":
                    self.output_action, self.latent, self.pixel = None, r.latent, r.pixel
                    # s2_service joins the two answers with "  >  " when it re-queried with lookdown_rgb
                    if look is not None:
                        self.goal = (look[0], look[1], frame_id)
                        self.goal_frame = "lookdown" if "  >  " in r.text else "fpv"
                    else:
                        self.goal, self.goal_frame = (rgb, depth, frame_id), "fpv"
                else:
                    assert False, f"unreachable kind={r.kind!r}"
                self.fresh = {"text": r.text, "t_ms": r.t_infer_ms, "e2e_ms": 1000 * (time.time() - t_sent)}
                return True
            if time.time() >= deadline:
                return False
            time.sleep(0.002)
        return False

    # ------------------------------------------------------------------ one request
    def step(self, rgb, depth, frame_id, seed, look=None):
        t_wait = 0.0
        if self.args.mode == "sync":
            if self.s2_due():
                self.s2_infer(rgb, depth, frame_id, look)
                t0 = time.time()
                self.s2_poll(600.0)
                t_wait = time.time() - t0
            else:
                self.s2_observe(rgb)
        elif self.args.mode == "async":
            self.s2_poll(0.0)
            if self.inflight is None and self.s2_due():
                self.s2_infer(rgb, depth, frame_id, look)
            else:
                self.s2_observe(rgb)
            if self.latent is None and self.output_action is None:  # nothing to act on yet -> wait for S2
                t0 = time.time()
                self.s2_poll(600.0)
                t_wait = time.time() - t0
        else:
            assert False, f"unreachable mode={self.args.mode!r}"

        fresh, self.fresh = self.fresh, None
        reply = {"is_s2_step": fresh is not None, "s2_img_size": [384, 384],
                 "llm_output": fresh["text"] if fresh else None, "pixel_frame": "fpv",
                 "timing": {"s2_ms": fresh["t_ms"] if fresh else None, "s2_e2e_ms": fresh["e2e_ms"] if fresh else None,
                            "wait_ms": 1000 * t_wait, "s1_ms": None, "s2_inflight": self.inflight is not None}}
        if self.output_action is not None:
            reply["discrete_action"] = list(self.output_action)
            self.output_action = None  # consumed, as in agent.step
        else:
            g_rgb, g_depth, g_fid = self.goal
            c_rgb, c_depth = (rgb, depth) if look is None else (look[0], look[1])  # S1 current frame: 30° if present
            r = self.s1.act(self.latent, g_rgb, c_rgb, depth_mem=g_depth, depth_cur=c_depth, seed=seed, timeout_s=60)
            assert r is not None, "S1 module did not answer"
            reply["trajectory"] = np.asarray(r.waypoints, np.float64).tolist()
            reply["pixel_goal"] = list(self.pixel) if (fresh is not None and self.pixel is not None) else None
            reply["pixel_frame_id"] = g_fid
            reply["pixel_frame"] = self.goal_frame
            reply["timing"]["s1_ms"] = r.t_infer_ms
        return reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="async", choices=["sync", "async"])
    ap.add_argument("--plan_step_gap", type=int, default=4)
    ap.add_argument("--port", type=int, default=5801)
    ap.add_argument("--s1_depth_clip", type=float, default=5.0, help="same clip as SimVLNAgent (<=0 disables)")
    ap.add_argument("--instruction", default="")
    ap.add_argument("--model_name", default="")
    args = ap.parse_args()
    orch = Orchestrator(args)
    lock = threading.Lock()
    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"ok": True, "model": args.model_name, "mode": args.mode, "server": "dual",
                        "instruction": orch.instruction})

    @app.route("/eval_dual", methods=["POST"])
    def eval_dual():
        image = np.asarray(Image.open(request.files["image"].stream).convert("RGB"))
        depth = np.asarray(Image.open(request.files["depth"].stream).convert("I")).astype(np.float32) / 10000.0
        data = json.loads(request.form.get("json", "{}"))
        instruction = data.get("instruction") or args.instruction
        look = None
        if "lookdown_image" in request.files:  # bridge lookdown_ns set (sim GUIDEDOG_LOOKDOWN_CAM=1)
            look = [np.asarray(Image.open(request.files["lookdown_image"].stream).convert("RGB")),
                    np.asarray(Image.open(request.files["lookdown_depth"].stream).convert("I")).astype(np.float32)
                    / 10000.0]
        if args.s1_depth_clip and args.s1_depth_clip > 0:
            depth = np.minimum(depth, args.s1_depth_clip).astype(np.float32)
            if look is not None:
                look[1] = np.minimum(look[1], args.s1_depth_clip).astype(np.float32)
        with lock:
            t0 = time.time()
            if data.get("reset", False) or instruction != orch.instruction:
                orch.s2_reset(instruction)
                print(f"[dual_server] reset, instruction: {instruction}", flush=True)
            reply = orch.step(image, depth, data.get("frame_id"), data.get("seed"), look)
            reply["t_infer"] = time.time() - t0
        t = reply["timing"]
        print(f"[dual_server] {reply['t_infer'] * 1000:.0f} ms s1={t['s1_ms'] and round(t['s1_ms'])} "
              f"s2={t['s2_ms'] and round(t['s2_ms'])} wait={t['wait_ms']:.0f} inflight={t['s2_inflight']} "
              + (f"llm={reply['llm_output']!r}" if reply["llm_output"] else ""), flush=True)
        return jsonify(reply)

    print(f"[dual_server] mode={args.mode} gap={args.plan_step_gap} on :{args.port}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
