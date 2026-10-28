"""② visualisation check for vln_bridge_node with stub_server.py --save_dir (overlay_test.sh).

Pairs each /vln/s2_image with the /vln/s2 JSON of the same stamp and checks
  position : centroid of the green goal circle  vs  the pixel the stub answered (0 px expected, <= 1 px PASS)
  frame    : the overlay background  vs  the frame the stub says the pixel refers to (llm "fid=<id>", saved by the
             stub as <id>.png / <id>_ld.png). Pixels changed by the drawings are excluded (|diff| > 40); the rest
             must match up to JPEG noise (mean <= 4). The same score against the newest saved frame is printed as a
             reference: it must be clearly worse when the robot moves (frame lag visible).
Usage: python3 scripts/sim/verify/overlay_check.py --ros-args -p save_dir:=DIR -p seconds:=20.0
"""
import json
import os
import re

import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

POS_TOL, BG_TOL = 1.0, 4.0


def bg_diff(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    keep = d <= 40
    return float(d[keep].mean()), float(1.0 - keep.mean())


class OverlayCheck(Node):
    def __init__(self):
        super().__init__("overlay_check")
        self.save_dir = self.declare_parameter("save_dir", "").value
        self.seconds = self.declare_parameter("seconds", 20.0).value
        self.bridge = CvBridge()
        self.imgs, self.s2 = {}, {}
        self.rows = []
        self.create_subscription(Image, "/vln/s2_image", self.on_img, 10)
        self.create_subscription(String, "/vln/s2", self.on_s2, 10)
        self.t0 = self.get_clock().now()

    def on_img(self, m):
        self.imgs[round(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, 3)] = self.bridge.imgmsg_to_cv2(m, "rgb8")
        self.match()

    def on_s2(self, m):
        j = json.loads(m.data)
        self.s2[round(j["stamp"], 3)] = j
        self.match()

    def match(self):
        for st in sorted(set(self.imgs) & set(self.s2)):
            img, j = self.imgs.pop(st), self.s2.pop(st)
            if j.get("pixel") is None:
                continue
            row, col = j["pixel"]
            g = (img[:, :, 1] >= 200) & (img[:, :, 0] <= 80) & (img[:, :, 2] <= 80)
            ys, xs = np.nonzero(g)
            pos_err = float(np.hypot(xs.mean() - col, ys.mean() - row)) if len(xs) else float("inf")
            fid = int(re.search(r"fid=(\d+)", j["llm"]).group(1))
            suffix = "_ld" if j.get("pixel_frame") == "lookdown" else ""
            saved = sorted(int(f.split("_")[0].split(".")[0]) for f in os.listdir(self.save_dir) if f.endswith(f"{suffix}.png")
                           and (suffix or "_" not in f))
            bg, drawn = bg_diff(img, np.asarray(PILImage.open(f"{self.save_dir}/{fid}{suffix}.png").convert("RGB")))
            newest = max(saved)
            bg_new = bg_diff(img, np.asarray(PILImage.open(f"{self.save_dir}/{newest}{suffix}.png").convert("RGB")))[0]
            self.rows.append((pos_err, bg, drawn, bg_new, fid, newest, j.get("pixel_frame")))
            self.get_logger().info(f"pixel=({row},{col}) frame={j.get('pixel_frame')} pos_err={pos_err:.2f}px "
                                   f"bg(goal fid {fid})={bg:.2f} drawn={drawn * 100:.1f}% bg(newest fid {newest})={bg_new:.2f}")

    def done(self):
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9 > self.seconds


def main():
    rclpy.init()
    n = OverlayCheck()
    while rclpy.ok() and not n.done():
        rclpy.spin_once(n, timeout_sec=0.1)
    r = n.rows
    if not r:
        print("[overlay_check] NO SAMPLES -> FAIL")
    else:
        pos = max(x[0] for x in r)
        bg = max(x[1] for x in r)
        moved = [x[3] for x in r if x[5] != x[4]]
        ok = pos <= POS_TOL and bg <= BG_TOL
        print(f"[overlay_check] n={len(r)} frames={sorted(set(x[6] for x in r))} pos_err max={pos:.2f}px "
              f"bg(goal frame) max={bg:.2f} | bg(newest frame) median={np.median(moved) if moved else float('nan'):.2f} "
              f"-> {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
