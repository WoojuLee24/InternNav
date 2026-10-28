"""Deploy viewer — live 3-panel visualizer for the deployed dual system.

Fork of quantization_kit/async_zmq/visualizer.py with the deploy corrections:

  * S2 pixel goal is interpreted in the policy's NATIVE [row, col] convention
    (internvla_n1_policy(_llamacpp)._parse_pixel returns [y, x]); the old viewer's
    default [x, y] reading transposed the marker — that mismatch is what broke the
    "S1 and S2 must agree on the goal pixel" check from the paper.
  * RIGHT panel (S1): background is the EXACT frame S1 computed the trajectories on
    (`s1_frame`, paired by the orchestrator) — never the newer live frame.
  * Trajectories are the RAW diffusion samples ([N, T+1, 2] metric, robot frame) and are
    PROJECTED into the image with the repo camera model (common/camproj.py: intrinsics
    from HFOV per get_intrinsic_matrix; ground-plane geometry per unproject_depth).
    Every sample is drawn — no averaging, no interpolation. The mean endpoint is compared
    against S2's pixel goal (Δpx readout) to visualize the paper's S1/S2 goal agreement.

Panels:  LEFT  = what S2 processed + its pixel goal
         CENTER= LIVE camera + action arrow + rates
         RIGHT = S1 input frame (synced) + projected raw trajectories + goal agreement
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.common.camproj import CamModel, pixel_to_ground, project_ground  # noqa: E402
from deploy.common.config import DeployConfig  # noqa: E402
from quantization_kit.async_zmq.zmq_proto import DEFAULT_PORTS, make_socket, recv_msg  # noqa: E402

ACT = {-1: "wait", 0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT", 5: "LOOK-DOWN"}
ARROW = {1: (0, -1), 2: (-1, 0), 3: (1, 0)}


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1, bg=(0, 0, 0)):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b + 1), bg, -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _wrap(s, n=42):
    out, line = [], ""
    for word in s.split():
        if len(line) + len(word) + 1 > n:
            out.append(line); line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out[:3]


def _pix_uv(pixel, W, H, pixel_xy=False):
    """S2 pixel -> drawable (u, v). Native convention: [row, col] (policy _parse_pixel)."""
    if pixel is None or len(pixel) < 2:
        return None
    py, px = (pixel[1], pixel[0]) if pixel_xy else (pixel[0], pixel[1])
    if px <= 1.0 and py <= 1.0:                      # normalized fallback
        px, py = px * W, py * H
    return int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))


def _draw_goal(img, uv, label="pixel goal"):
    cv2.circle(img, uv, 16, (0, 0, 255), 2)
    cv2.drawMarker(img, uv, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    _text(img, label, (max(0, uv[0] - 30), max(14, uv[1] - 20)), 0.45, (0, 0, 255))


def draw_s2(frame_rgb, size, pixel, pixel_xy=False):
    """LEFT: the exact frame S2 processed (goal_rgb) + the pixel goal it picked on it."""
    if frame_rgb is None:
        panel = np.full((size, size, 3), 20, np.uint8)
        _text(panel, "S2 input: waiting for first decision...", (8, 24), 0.5, (0, 200, 255))
        return panel
    panel = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    if panel.shape[0] != size or panel.shape[1] != size:
        panel = cv2.resize(panel, (size, size))
    cv2.rectangle(panel, (0, 0), (size - 1, size - 1), (0, 200, 255), 2)
    uv = _pix_uv(pixel, size, size, pixel_xy)
    if uv is not None:
        _draw_goal(panel, uv)
    _text(panel, "S2 processing (input frame)", (8, 24), 0.55, (0, 200, 255))
    return panel


def draw_camera(frame_rgb, ov, pixel_xy=False, cap_age_ms=None):
    """CENTER: the LIVE camera + action arrow + rates (unchanged from the original)."""
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = img.shape[:2]
    ov = ov or {}
    uv = _pix_uv(ov.get("pixel"), W, H, pixel_xy)
    if uv is not None:
        _draw_goal(img, uv)
    action = ov.get("action", -1)
    if action in ARROW:
        cx, cy = W // 2, H - 60
        dx, dy = ARROW[action]
        cv2.arrowedLine(img, (cx, cy), (cx + dx * 46, cy + dy * 46), (0, 255, 0), 4, tipLength=0.4)
    elif action == 0:
        cv2.circle(img, (W // 2, H - 60), 22, (0, 0, 255), 3)
    _text(img, f"action: {ACT.get(action, '?')}", (8, 22), 0.6, (0, 255, 0))
    s2_state = "S2: thinking..." if ov.get("inflight") else f"S2: plan age={ov.get('latent_age', -1)}"
    _text(img, s2_state, (8, 44), 0.5, (0, 200, 255))
    hz_str = f"S1 {ov.get('s1_hz', 0):.0f}Hz  S2 {ov.get('s2_hz', 0):.2f}Hz"
    if cap_age_ms is not None:
        hz_str += f"  frame {cap_age_ms:.0f}ms"
    _text(img, hz_str, (8, 64), 0.5, (200, 200, 200))
    _text(img, "LIVE camera", (W - 118, 22), 0.5, (0, 255, 0))
    for i, ln in enumerate(_wrap("instr: " + ov.get("instr", ""), 44)):
        _text(img, ln, (8, H - 12 - (2 - i) * 18), 0.45, (255, 255, 0))
    return img


def _bev_inset(panel, waypoints, size, goal_xy=None, inset=None):
    """Always-visible top-down inset of the denoised path (autoscaled) — guarantees the
    waypoints can be seen even when the image-plane projection puts them out of frame
    (e.g. a miscalibrated cam height/pitch). Robot at the bottom, x=fwd up, y=left."""
    inset = inset or max(96, size // 3)
    x0, y0 = size - inset - 6, 32
    sub = panel[y0:y0 + inset, x0:x0 + inset]
    sub[:] = (sub.astype(np.float32) * 0.25).astype(np.uint8)          # darken the patch
    cv2.rectangle(panel, (x0, y0), (x0 + inset, y0 + inset), (90, 90, 90), 1)
    pts_all = np.abs(np.asarray(waypoints, np.float32)).max()
    if goal_xy is not None:
        pts_all = max(pts_all, abs(goal_xy[0]), abs(goal_xy[1]))
    maxr = max(0.5, float(pts_all))
    scale = (inset * 0.44) / maxr
    cx, cy = x0 + inset // 2, y0 + inset - 8
    pts = [(int(cx - y * scale), int(cy - x * scale)) for x, y in waypoints]
    for i in range(1, len(pts)):
        cv2.line(panel, pts[i - 1], pts[i], (0, 165, 255), 2, cv2.LINE_AA)
    for p in pts:                                                       # waypoint dots
        cv2.circle(panel, p, 1, (0, 220, 255), -1)
    if goal_xy is not None:                                             # S2 goal on the ground
        cv2.circle(panel, (int(cx - goal_xy[1] * scale), int(cy - goal_xy[0] * scale)),
                   4, (0, 0, 255), -1)
    cv2.drawMarker(panel, (cx, cy), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 10, 2)
    _text(panel, f"BEV {maxr:.1f}m", (x0 + 3, y0 + 13), 0.38, (170, 170, 170))


def draw_s1(s1_frame, waypoints, size, ov, cam: CamModel, pixel_xy=False):
    """RIGHT: S1 panel — the SYNCED input frame with the DENOISED robot-facing waypoint
    path PROJECTED onto it via the repo camera model, plus the S2 goal-pixel agreement."""
    if s1_frame is None or waypoints is None:
        panel = np.full((size, size, 3), 28, np.uint8)
        _text(panel, "S1: waiting for first waypoints...", (8, 24), 0.5, (0, 200, 255))
        return panel
    panel = cv2.cvtColor(s1_frame, cv2.COLOR_RGB2BGR).copy()
    if panel.shape[0] != size or panel.shape[1] != size:
        panel = cv2.resize(panel, (size, size))

    waypoints = np.asarray(waypoints, np.float32)    # [T+1, 2] — THE path the robot executes
    uv, in_front, in_frame = project_ground(waypoints, cam)
    pts = uv.astype(int)
    for i in range(1, len(pts)):
        if in_front[i - 1] and in_front[i]:          # ahead of the camera (cv2 clips the rest)
            cv2.line(panel, tuple(pts[i - 1]), tuple(pts[i]), (0, 165, 255), 2, cv2.LINE_AA)
    for i in range(len(pts)):                        # each denoised waypoint as a dot
        if in_frame[i]:
            cv2.circle(panel, tuple(pts[i]), 3, (0, 220, 255), -1)
    if int(in_frame.sum()) == 0:                     # geometry fine but nothing lands in-frame
        _text(panel, "waypoints below/outside frame ->", (8, 66), 0.45, (0, 165, 255))
        _text(panel, "calibrate VLN_CAM_HEIGHT / VLN_CAM_PITCH", (8, 84), 0.45, (0, 165, 255))

    end_xy = waypoints[-1]
    end_uv, end_front, _ = project_ground(end_xy[None, :], cam)
    if end_front[0]:
        cv2.circle(panel, tuple(end_uv[0].astype(int)), 6, (0, 255, 255), -1)

    # --- S1 <-> S2 goal-pixel agreement (paper: the two systems share one pixel goal) ---
    goal_ground = None
    uv_goal = _pix_uv(ov.get("pixel"), size, size, pixel_xy)
    if uv_goal is not None:
        _draw_goal(panel, uv_goal, "S2 goal")
        goal_ground = pixel_to_ground(uv_goal[1], uv_goal[0], cam)   # (row, col) -> ground (x, y)
        if end_front[0]:
            d_px = float(np.hypot(end_uv[0][0] - uv_goal[0], end_uv[0][1] - uv_goal[1]))
            g = goal_ground
            d_m = float(np.hypot(g[0] - end_xy[0], g[1] - end_xy[1])) if g else None
            agree = f"S1-S2 goal: {d_px:.0f}px" + (f" / {d_m:.2f}m" if d_m is not None else "")
            _text(panel, agree, (8, 46), 0.5, (0, 255, 255))
            cv2.line(panel, tuple(end_uv[0].astype(int)), uv_goal, (0, 255, 255), 1, cv2.LINE_AA)

    _bev_inset(panel, waypoints, size, goal_xy=goal_ground)   # the path, ALWAYS visible
    cv2.drawMarker(panel, (size // 2, size - 8), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 14, 2)
    _text(panel, f"S1 input (synced) + denoised waypoints ({len(waypoints)})", (8, 24), 0.5, (0, 165, 255))
    _text(panel, f"cam h={cam.height_m}m pitch={cam.pitch_deg}deg hfov={cam.hfov_deg}deg",
          (8, size - 10), 0.4, (150, 150, 150))
    return panel


def render(frame_rgb, ov, waypoints, s1_frame, cam, s2_frame=None, s2_pixel=None,
           pixel_xy=False, cap_age_ms=None):
    center = draw_camera(frame_rgb, ov, pixel_xy, cap_age_ms)
    size = center.shape[0]
    left = draw_s2(s2_frame, size, s2_pixel, pixel_xy)
    right = draw_s1(s1_frame, waypoints, size, ov, cam, pixel_xy)
    return cv2.hconcat([left, center, right])


# ---- backends (verbatim from the original visualizer) ------------------------------
def run_tk(get_image, title):
    import tkinter as tk

    from PIL import Image, ImageTk
    root = tk.Tk(); root.title(title); root.configure(bg="#111")
    lbl = tk.Label(root, bg="#111"); lbl.pack()
    stop = {"q": False}
    root.protocol("WM_DELETE_WINDOW", lambda: stop.__setitem__("q", True))
    root.bind("<KeyPress-q>", lambda e: stop.__setitem__("q", True))
    print(f"[viz] tk window open: {title!r}  (press q or close to quit)", flush=True)
    while not stop["q"]:
        img = get_image()
        if img is not None:
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            lbl.configure(image=photo); lbl.image = photo
        root.update_idletasks(); root.update()
        time.sleep(0.02)
    root.destroy()


def run_mjpeg(get_image, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    latest = {"buf": None}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
                self.wfile.write(b"<html><body style='margin:0;background:#111;text-align:center'>"
                                 b"<img src='/stream' style='max-width:100vw;max-height:100vh'></body></html>")
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
                try:
                    while True:
                        buf = latest["buf"]
                        if buf:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[viz] MJPEG viewer: open  http://<this-host-ip>:{port}/  in a browser  (Ctrl-C to stop)", flush=True)
    while True:
        img = get_image()
        if img is not None:
            ok, jpg = cv2.imencode(".jpg", img)
            if ok:
                latest["buf"] = jpg.tobytes()
        time.sleep(0.04)


def run_save(get_image, snap):
    print(f"[viz] HEADLESS -> writing {snap} ~2 Hz (no DISPLAY). Use --view mjpeg for a browser view.", flush=True)
    last = 0.0
    while True:
        img = get_image()
        if img is not None and time.time() - last > 0.5:
            cv2.imwrite(snap, img); last = time.time()
        time.sleep(0.05)


def main():
    dcfg = DeployConfig.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cam-host", default=None)
    ap.add_argument("--viz-host", default=None)
    ap.add_argument("--s2-host", default=None)
    ap.add_argument("--pixel-xy", action="store_true",
                    help="treat S2 pixel as [x, y] (NON-standard; the policy emits [row, col])")
    ap.add_argument("--view", choices=["auto", "tk", "mjpeg", "save"], default="auto")
    ap.add_argument("--mjpeg-port", type=int, default=dcfg.mjpeg_port)
    ap.add_argument("--snapshot", default=str(_ROOT / "quantization_kit" / "dbg" / "viz_live.png"))
    ap.add_argument("--cam-hfov", type=float, default=dcfg.cam_hfov_deg,
                    help="HFOV of the published square frame (deg)")
    ap.add_argument("--cam-height", type=float, default=dcfg.cam_height_m, help="camera height (m)")
    ap.add_argument("--cam-pitch", type=float, default=dcfg.cam_pitch_deg, help="camera pitch, +down (deg)")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    cam = make_socket(ctx, zmq.SUB, DEFAULT_PORTS["cam"], bind=False, host=args.cam_host or args.host)
    viz = make_socket(ctx, zmq.SUB, DEFAULT_PORTS["viz"], bind=False, host=args.viz_host or args.host)
    s2 = make_socket(ctx, zmq.SUB, DEFAULT_PORTS["s2_out"], bind=False, host=args.s2_host or args.host)
    poll = zmq.Poller()
    for sk in (cam, viz, s2):
        poll.register(sk, zmq.POLLIN)
    time.sleep(0.3)
    print(f"[viz] cam SUB:{DEFAULT_PORTS['cam']}  viz SUB:{DEFAULT_PORTS['viz']}  "
          f"s2 SUB:{DEFAULT_PORTS['s2_out']}  | projection: hfov={args.cam_hfov} "
          f"h={args.cam_height} pitch={args.cam_pitch}", flush=True)

    state = {"frame": None, "ov": {}, "wp": None, "s1_frame": None,
             "s2_frame": None, "s2_pixel": None, "cap_t": None}
    cam_model = {"m": None}

    def get_image():
        poll.poll(30)
        while cam.poll(0):
            m, a = recv_msg(cam); state["frame"] = a["rgb"]; state["cap_t"] = m.get("t_cap")
        while viz.poll(0):
            ov, a = recv_msg(viz)
            state["ov"] = ov
            wp = a.get("waypoints")
            if wp is None and "trajs" in a:           # legacy raw-fan overlay -> same mean
                wp = a["trajs"].mean(axis=0)
            if wp is not None and "s1_frame" in a:    # the path arrives PAIRED with its frame
                state["wp"] = wp; state["s1_frame"] = a["s1_frame"]
        while s2.poll(0):
            m, a = recv_msg(s2)
            if "goal_rgb" in a:
                state["s2_frame"] = a["goal_rgb"]; state["s2_pixel"] = m.get("pixel")
        if state["frame"] is None:
            return None
        if cam_model["m"] is None:                    # camera model at the published frame size
            cam_model["m"] = CamModel(size=int(state["frame"].shape[0]), hfov_deg=args.cam_hfov,
                                      height_m=args.cam_height, pitch_deg=args.cam_pitch)
        age_ms = (time.time() - state["cap_t"]) * 1000.0 if state["cap_t"] else None
        return render(state["frame"], state["ov"], state["wp"], state["s1_frame"],
                      cam_model["m"], state["s2_frame"], state["s2_pixel"],
                      args.pixel_xy, age_ms)

    view = args.view
    if view == "auto":
        view = "tk" if os.environ.get("DISPLAY") else "save"
    try:
        if view == "tk":
            run_tk(get_image, "VLN-deploy  (S2 input | LIVE camera | S1 raw trajs, synced+projected)")
        elif view == "mjpeg":
            run_mjpeg(get_image, args.mjpeg_port)
        else:
            run_save(get_image, args.snapshot)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[viz] '{view}' backend failed ({e}); falling back to --view save", flush=True)
        run_save(get_image, args.snapshot)


if __name__ == "__main__":
    main()
