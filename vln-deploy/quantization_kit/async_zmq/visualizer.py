"""Live visualizer: subscribe to the camera stream (port cam), the orchestrator's overlay (port viz),
and the S2 service output (port s2_out), and show three panels side by side:
  * LEFT   - the exact frame System-2 is processing (its goal_rgb) + the pixel goal it picked on it
  * CENTER - the LIVE camera frame with the action arrow, instruction, S2/S1 rates + frame age (ms)
  * RIGHT  - a top-down bird's-eye (BEV) panel of the S1 trajectory (robot at bottom, x=fwd, y=left, goal=red)

convenv ships opencv-python-HEADLESS (no cv2.imshow), so the window uses **tkinter** (always available)
by default. Backends (--view):
  * tk     - native Tk window (default when a display is present)            [recommended on the GUI box]
  * mjpeg  - serve the view over HTTP; open http://<jetson-ip>:8089/ in a browser  [works on Wayland/remote]
  * save   - headless: write the annotated frame to --snapshot ~2 Hz        [auto when no DISPLAY]

  PYTHONPATH=/usr/lib/python3.12/dist-packages \
    convenv/bin/python -m quantization_kit.async_zmq.visualizer --view tk
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

_QK = Path(__file__).resolve().parents[1]
_ROOT = _QK.parent
sys.path[:0] = [str(_ROOT), str(_QK)]

from quantization_kit.async_zmq.zmq_proto import DEFAULT_PORTS, make_socket, recv_msg  # noqa: E402

ACT = {-1: "wait", 0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT", 5: "LOOK-DOWN"}
ARROW = {1: (0, -1), 2: (-1, 0), 3: (1, 0)}   # forward/left/right unit dirs on the image


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


def draw_s2(frame_rgb, size, pixel, pixel_yx=False):
    """LEFT panel: the exact RGB frame System-2 is processing (its goal_rgb), with the pixel goal it
    picked on that frame. Updates each time S2 finishes a decision (~1 Hz), so you can SEE what S2 sees."""
    if frame_rgb is None:
        panel = np.full((size, size, 3), 20, np.uint8)
        _text(panel, "S2 input: waiting for first decision...", (8, 24), 0.5, (0, 200, 255))
        return panel
    panel = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    if panel.shape[0] != size or panel.shape[1] != size:
        panel = cv2.resize(panel, (size, size))
    H, W = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (W - 1, H - 1), (0, 200, 255), 2)   # cyan frame = "this is what S2 sees"
    if pixel is not None and len(pixel) >= 2:
        px, py = (pixel[1], pixel[0]) if pixel_yx else (pixel[0], pixel[1])
        if px <= 1.0 and py <= 1.0:
            px, py = px * W, py * H
        px, py = int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))
        cv2.circle(panel, (px, py), 16, (0, 0, 255), 2)
        cv2.drawMarker(panel, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        _text(panel, "pixel goal", (max(0, px - 30), max(14, py - 20)), 0.45, (0, 0, 255))
    _text(panel, "S2 processing (input frame)", (8, 24), 0.55, (0, 200, 255))
    return panel


def draw_camera(frame_rgb, ov, pixel_yx=False, cap_age_ms=None):
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = img.shape[:2]
    ov = ov or {}

    pixel = ov.get("pixel")
    if pixel is not None and len(pixel) >= 2:
        px, py = (pixel[1], pixel[0]) if pixel_yx else (pixel[0], pixel[1])
        if px <= 1.0 and py <= 1.0:
            px, py = px * W, py * H
        px, py = int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))
        cv2.circle(img, (px, py), 16, (0, 0, 255), 2)
        cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        _text(img, "pixel goal", (max(0, px - 30), max(14, py - 20)), 0.45, (0, 0, 255))

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


def draw_bev(waypoints, size, ov, bg=None):
    if bg is not None:
        panel = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        if panel.shape[0] != size or panel.shape[1] != size:
            panel = cv2.resize(panel, (size, size))
        panel = (panel.astype(np.float32) * 0.55).astype(np.uint8)   # darken for overlay contrast
    else:
        panel = np.full((size, size, 3), 28, np.uint8)
    cx, cy = size // 2, size - 40
    _text(panel, "BEV  (robot=triangle, goal=red)  x=fwd up", (6, 18), 0.42, (180, 180, 180))
    if waypoints is not None and len(waypoints) >= 2:
        wp = np.asarray(waypoints, np.float32)
        maxr = max(0.5, float(np.abs(wp).max()))
        scale = (size * 0.42) / maxr
        pts = [(int(cx - y * scale), int(cy - x * scale)) for x, y in wp]
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i - 1], pts[i], (0, 200, 255), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(panel, p, 2, (0, 150, 255), -1)
        cv2.circle(panel, pts[-1], 7, (60, 60, 255), -1)
        _text(panel, f"reach ~{maxr:.1f} (model units)", (6, size - 10), 0.42, (150, 150, 150))
    cv2.drawMarker(panel, (cx, cy), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 16, 2)
    return panel


def render(frame_rgb, ov, waypoints, s2_frame=None, s2_pixel=None, pixel_yx=False, cap_age_ms=None):
    cam = draw_camera(frame_rgb, ov, pixel_yx, cap_age_ms)
    size = cam.shape[0]
    s2 = draw_s2(s2_frame, size, s2_pixel, pixel_yx)         # LEFT: what S2 is processing
    bev = draw_bev(waypoints, size, ov, bg=frame_rgb)        # RIGHT: S1 trajectory BEV
    return cv2.hconcat([s2, cam, bev])


# ---- backends (each consumes get_image() -> BGR frame or None; called on the MAIN thread) ----------
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
    while True:                                   # ZMQ recv stays on the main thread; handlers only read latest
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="default host for both cam and viz SUB")
    ap.add_argument("--cam-host", default=None, help="host publishing camera frames (default: --host). "
                    "Cross-machine: point at the camera/publisher box while --viz-host stays on the orchestrator.")
    ap.add_argument("--viz-host", default=None, help="host publishing the overlay (default: --host)")
    ap.add_argument("--s2-host", default=None, help="host running the S2 service (default: --host); "
                    "subscribed to show the exact frame S2 is processing in the left panel")
    ap.add_argument("--pixel-yx", action="store_true", help="treat S2 pixel as [row,col] instead of [x,y]")
    ap.add_argument("--view", choices=["auto", "tk", "mjpeg", "save"], default="auto",
                    help="auto = tk if a display is present else save")
    ap.add_argument("--mjpeg-port", type=int, default=8089)
    ap.add_argument("--snapshot", default=str(_QK / "dbg" / "viz_live.png"))
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
          f"s2 SUB:{DEFAULT_PORTS['s2_out']}", flush=True)

    state = {"frame": None, "ov": {}, "wp": None, "s2_frame": None, "s2_pixel": None, "cap_t": None}

    def get_image():
        poll.poll(30)
        while cam.poll(0):                              # drain to the freshest camera frame only
            m, a = recv_msg(cam); state["frame"] = a["rgb"]; state["cap_t"] = m.get("t_cap")
        while viz.poll(0):
            ov, a = recv_msg(viz); state["ov"] = ov; state["wp"] = a.get("waypoints", state["wp"])
        while s2.poll(0):                               # grab the frame S2 just processed (goal_rgb)
            m, a = recv_msg(s2)
            if "goal_rgb" in a:
                state["s2_frame"] = a["goal_rgb"]; state["s2_pixel"] = m.get("pixel")
        if state["frame"] is None:
            return None
        age_ms = (time.time() - state["cap_t"]) * 1000.0 if state["cap_t"] else None
        return render(state["frame"], state["ov"], state["wp"],
                      state["s2_frame"], state["s2_pixel"], args.pixel_yx, age_ms)

    view = args.view
    if view == "auto":
        view = "tk" if os.environ.get("DISPLAY") else "save"
    try:
        if view == "tk":
            run_tk(get_image, "VLN-field  (S2 input | LIVE camera | S1 BEV)")
        elif view == "mjpeg":
            run_mjpeg(get_image, args.mjpeg_port)
        else:
            run_save(get_image, args.snapshot)
    except KeyboardInterrupt:
        pass
    except Exception as e:                          # e.g. Tk cannot open the display -> degrade to snapshots
        print(f"[viz] '{view}' backend failed ({e}); falling back to --view save", flush=True)
        run_save(get_image, args.snapshot)


if __name__ == "__main__":
    main()
