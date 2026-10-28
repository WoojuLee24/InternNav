"""InternVLA-N1 HTTP server for the sim bridge (derived from scripts/realworld/http_internvla_server.py).

POST /eval_dual
  files: image (jpeg/png RGB), depth (uint16 png, metres * 10000 — same as realworld)
         optional lookdown_image / lookdown_depth: 30° look-down camera (sim GUIDEDOG_LOOKDOWN_CAM=1).
         When S2 answers "↓" it is re-queried with this frame (VLN-CE protocol); without it the same
         frame is reused (realworld behaviour). reply["pixel_frame"] = "lookdown" | "fpv" tells which
         image the pixel goal refers to.
  form json: {"reset": bool, "instruction": str, "intrinsic": [[3x3]] (optional, logged only)}
  reply: {"discrete_action": [...]}  or  {"trajectory": [[x, y], ...], "pixel_goal": [row, col] | null}
         + "is_s2_step", "s2_img_size": [w, h], "llm_output", "t_infer"
  Instruction change or reset=true -> agent.reset().
GET /health -> {"ok": true, "model": ..., "quant": ..., "system1": ...}

Run (inside the gd_vln container):
  python3 scripts/sim/vln_server.py --model_path /checkpoints/InternVLA-N1-DualVLN --quant bf16
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vln_agent import SimVLNAgent  # noqa: E402

app = Flask(__name__)
agent = None
args = None
lock = threading.Lock()  # one inference at a time; the bridge never sends concurrent requests anyway
state = {"instruction": None}


def _decode_pair(req, rgb_key, depth_key):
    image = np.asarray(Image.open(req.files[rgb_key].stream).convert('RGB'))
    depth = np.asarray(Image.open(req.files[depth_key].stream).convert('I')).astype(np.float32) / 10000.0
    return image, depth


def _decode(req):
    image, depth = _decode_pair(req, 'image', 'depth')
    look = _decode_pair(req, 'lookdown_image', 'lookdown_depth') if 'lookdown_image' in req.files else None
    data = json.loads(req.form.get('json', '{}'))
    return image, depth, look, data


@app.route("/health", methods=['GET'])
def health():
    return jsonify(
        {
            "ok": agent is not None,
            "model": args.model_path,
            "quant": args.quant,
            "system1": getattr(agent.model.config, "system1", None) if agent else None,
            "instruction": state["instruction"],
        }
    )


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    image, depth, look, data = _decode(request)
    instruction = data.get("instruction") or args.instruction
    if not instruction:
        return jsonify({"error": "no instruction"}), 400

    with lock:
        if data.get("reset", False) or instruction != state["instruction"]:
            agent.reset()
            state["instruction"] = instruction
            print(f"[vln_server] reset, instruction: {instruction}", flush=True)

        t0 = time.time()
        if data.get("seed") is not None:  # reproducible S1 noise (verify/bench_dual.py; same as s1_worker)
            import torch

            torch.manual_seed(int(data["seed"]))
        out = agent.step(image, depth, np.eye(4), instruction, intrinsic=None, look_down=False)
        is_s2 = out.is_s2_step
        llm = out.llm_output
        pixel_frame = "fpv"
        if out.output_action is not None and list(out.output_action) == [5]:
            if look is not None:  # 30° look-down camera: same protocol as VLN-CE (goal.125cm_30deg)
                out = agent.step(look[0], look[1], np.eye(4), instruction, intrinsic=None, look_down=True)
                pixel_frame = "lookdown"
            else:  # fixed-pitch camera only -> re-query with the same frame (realworld server behaviour)
                out = agent.step(image, depth, np.eye(4), instruction, intrinsic=None, look_down=True)
            is_s2, llm = True, out.llm_output
        dt = time.time() - t0

    reply = {
        "is_s2_step": bool(is_s2),
        "s2_img_size": [agent.resize_w, agent.resize_h],
        "llm_output": llm,
        "t_infer": dt,
    }
    if out.output_action is not None:
        reply["discrete_action"] = [int(a) for a in out.output_action]
    else:
        reply["trajectory"] = np.asarray(out.output_trajectory).tolist()
        reply["pixel_goal"] = [int(x) for x in out.output_pixel] if out.output_pixel is not None else None
        reply["pixel_frame"] = pixel_frame
        reply["pixel_frame_id"] = data.get("frame_id")  # monolithic: always the current frame
    print(f"[vln_server] {dt * 1000:.0f} ms s2={is_s2} " + (f"llm={llm!r}" if llm else ""), flush=True)
    return jsonify(reply)


def main():
    global agent, args
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--quant", default="bf16", choices=["bf16", "nf4", "int8"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=5801)
    ap.add_argument("--resize_w", type=int, default=384)
    ap.add_argument("--resize_h", type=int, default=384)
    ap.add_argument("--num_history", type=int, default=8)
    ap.add_argument("--plan_step_gap", type=int, default=4)
    ap.add_argument("--s1_depth_clip", type=float, default=5.0, help="metres; <=0 disables")
    ap.add_argument("--instruction", default="", help="default instruction when the request has none")
    ap.add_argument("--log_dir", default=None)
    args = ap.parse_args()
    if args.s1_depth_clip is not None and args.s1_depth_clip <= 0:
        args.s1_depth_clip = None

    import torch

    agent = SimVLNAgent(args)
    # warm-up (first call compiles kernels / allocates caches)
    agent.step(np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.float32), np.eye(4), "hello", None)
    agent.reset()
    if torch.cuda.is_available():
        print(f"[vln_server] ready. cuda mem allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
              f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)
    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
