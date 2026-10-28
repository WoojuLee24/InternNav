"""Model-free stand-in for vln_server.py / dual_server.py: answers /eval_dual with FIXED replies (frame_test.sh,
overlay_test.sh).

Reply = straight S1 path (33 waypoints, x = 0..3.2 m, y = 0, base frame) + an S2 pixel goal [row, col] taken in turn
from --pixels. --lag N answers for the frame sent N requests earlier (pixel_frame_id, like the async dual_server).
With a look-down image in the request the goal refers to it (pixel_frame "lookdown", llm "↓  >  x y").
--save_dir keeps every received image as <frame_id>.png / <frame_id>_ld.png and writes llm_output as
"fid=<goal frame id> <x> <y>" so a checker can find the exact frame the pixel refers to.

Usage: python3 scripts/sim/verify/stub_server.py --port 5891 --pixels 400,320 [--lag 2 --save_dir DIR]
"""
import argparse
import cgi
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5801)
ap.add_argument("--pixels", nargs="+", default=["400,320"], help="row,col in the 640x480 frame, used in turn")
ap.add_argument("--lag", type=int, default=0)
ap.add_argument("--save_dir", default="")
ap.add_argument("--once", action="store_true", help="report the pixel goal only in the first reply (one gdm goal)")
args = ap.parse_args()
PIXELS = [[int(v) for v in p.split(",")] for p in args.pixels]
if args.save_dir:
    os.makedirs(args.save_dir, exist_ok=True)
state = {"n": 0, "fids": []}


class H(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"ok": True, "model": "stub"})

    def do_POST(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        data = json.loads(form.getvalue("json") or "{}")
        fid = data.get("frame_id")
        look = "lookdown_image" in form
        if args.save_dir and fid is not None:
            Image.open(io.BytesIO(form["image"].value)).save(os.path.join(args.save_dir, f"{fid}.png"))
            if look:
                Image.open(io.BytesIO(form["lookdown_image"].value)).save(os.path.join(args.save_dir, f"{fid}_ld.png"))
        state["fids"] = (state["fids"] + [fid])[-(args.lag + 1):]
        g_fid = state["fids"][0]
        row, col = PIXELS[state["n"] % len(PIXELS)]
        state["n"] += 1
        if args.once and state["n"] > 1:  # S1-only replies: no new S2 answer -> the bridge sends no new goal
            self._send({"trajectory": [[0.1 * i, 0.0] for i in range(33)], "pixel_goal": None, "is_s2_step": False,
                        "pixel_frame": "fpv", "pixel_frame_id": g_fid, "s2_img_size": [640, 480],
                        "llm_output": None, "t_infer": 0.0})
            return
        text = f"fid={g_fid} {col} {row}" if args.save_dir else f"{col} {row}"
        self._send({
            "trajectory": [[0.1 * i, 0.0] for i in range(33)],
            "pixel_goal": [row, col],
            "is_s2_step": True,
            "pixel_frame": "lookdown" if look else "fpv",
            "pixel_frame_id": g_fid,
            "s2_img_size": [640, 480],
            "llm_output": ("↓  >  " if look else "") + text,
            "t_infer": 0.0,
        })

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", args.port), H).serve_forever()
