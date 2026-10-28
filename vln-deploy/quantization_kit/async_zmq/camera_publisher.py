"""Camera publisher: grab frames from a V4L2 camera and PUB them over ZMQ (port `cam`:5605) for
the orchestrator (-> S2/S1) and the visualizer to consume. Center-crops to a square and resizes to
--size (default 336 = the S2 input resolution, so S2's pixel-goal coordinates map 1:1 onto the
published frame). Emits RGB uint8 [size,size,3].

Standalone:  convenv/bin/python -m quantization_kit.async_zmq.camera_publisher --device 0
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

_QK = Path(__file__).resolve().parents[1]
_ROOT = _QK.parent
sys.path[:0] = [str(_ROOT), str(_QK)]

from quantization_kit.async_zmq.zmq_proto import DEFAULT_PORTS, make_socket, send_msg  # noqa: E402


def center_square(img, size):
    """Center-crop to a square, then resize to size x size (no aspect distortion)."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="/dev/videoN index (0 = the 640x480 RGB cam)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORTS["cam"])
    ap.add_argument("--size", type=int, default=336, help="published square size (336 = S2 input res, 3x112)")
    ap.add_argument("--fps", type=float, default=30.0, help="publish rate cap")
    ap.add_argument("--squash", action="store_true",
                    help="resize the full frame to square (keep FOV, distort aspect) instead of center-crop")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[cam] ERROR: cannot open /dev/video{args.device}", flush=True)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always serve the freshest frame (low latency)

    ctx = zmq.Context.instance()
    pub = make_socket(ctx, zmq.PUB, args.port, bind=True)   # default HWM: never drop the NEWEST on send
    print(f"CAMERA READY  /dev/video{args.device} -> PUB:{args.port}  size={args.size} "
          f"({'squash' if args.squash else 'center-crop'})", flush=True)

    seq = 0
    dt = 1.0 / args.fps
    miss = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()                       # BGR
        if not ok:
            miss += 1
            if miss % 100 == 1:
                print(f"[cam] read() failed ({miss})", flush=True)
            time.sleep(0.01)
            continue
        miss = 0
        sq = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_AREA) if args.squash \
            else center_square(frame, args.size)
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        # t_cap is wall-clock (time.time), so consumers on any process can measure frame age.
        send_msg(pub, {"seq": seq, "t_cap": time.time()}, {"rgb": np.ascontiguousarray(rgb, np.uint8)})
        seq += 1
        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
