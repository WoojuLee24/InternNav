"""Tool 2: live per-episode inspector (S1 path / gdm plan -> controller -> robot). Run next to the bridge
(run_e2e.py --inspect launches one per episode) or by hand; stop with Ctrl-C / SIGINT -> analysis + summary png.

Records (all as they come, <out>/raw/):
  odom.csv   t, x, y, yaw           /gdq/msg/gdq_odom POSE only (twist unused: world frame in sim, zero on the robot)
  cmd.csv    t, topic, v, w         /gdq/msg/cmd_vel (gdm or MPC) and /cmd_vel (bridge discrete PID, Mode A)
  refs.jsonl {t, src, pts}          /vln/trajectory (S1 path = MPC reference) and /gdm/local_plan_map, in odom
  s2.jsonl / gp.jsonl / status.csv  /vln/s2 answers, /gdg/data/gp decoded to odom, /vln/status
  pubs.csv   t, n_gdq_cmd, n_cmd    publisher counts (sampled 1 Hz)
Analysis (analyze(), also offline: python3 live_inspect.py --analyze DIR [--mode M] [--goal x,y]):
  cross-track  robot distance to the controller's reference active at that time (Mode B: S1 path, Mode A: gdm plan),
               while the controller commands motion       -> RMSE <= 0.15 m
  limits       |v| > v_max or |w| > w_max on the controller topic  -> 0
  curvature    w_cmd vs v_cmd * kappa(reference at the robot), v_cmd >= 0.1 -> median |w - v kappa|
  execution    actual v / w from POSE differences vs the command (best lag 0..1 s) -> RMSE, lag
  publishers   max count on /gdq/msg/cmd_vel -> 1
  stop         final pose vs episode goal / last S2 goal, S2 answer count
Velocity is NEVER read from odom twist (see module note above).
Usage: python3 scripts/sim/verify/live_inspect.py --ros-args -p use_sim_time:=true -p out:=DIR -p mode:=trajectory
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

MAPS = Path(__file__).resolve().parents[1] / "maps"
LIMITS = {"trajectory": (0.4, 0.4), "pixel_goal": (0.8, 1.0)}  # bridge v_max/w_max ; gdm rbq.launch.py max speeds
CTRL_TOPIC = "/gdq/msg/cmd_vel"
PID_TOPIC = "/cmd_vel"


def yaw_q(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


# ---------------------------------------------------------------- recording (ROS)
def record():
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry, Path as PathMsg
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frame_check import tf_to_T

    class Inspect(Node):
        def __init__(self):
            super().__init__("vln_live_inspect")
            p = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
            self.out = Path(p("out", "scripts/sim/verify/out/inspect"))
            self.mode = p("mode", "trajectory")
            self.goal = p("goal", "")  # "x,y" episode goal in odom, optional
            self.world = p("world_frame", "odom")
            self.map_name = p("map", "etri_indoor")
            raw = self.out / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            self.f = {k: open(raw / k, "a", newline="") for k in
                      ("odom.csv", "cmd.csv", "refs.jsonl", "s2.jsonl", "gp.jsonl", "status.csv", "pubs.csv")}
            self.w = {k: csv.writer(v) for k, v in self.f.items() if k.endswith(".csv")}
            self.buf = Buffer(cache_time=Duration(seconds=30))
            TransformListener(self.buf, self)
            self.tf_to_T, self.Time = tf_to_T, Time
            self.create_subscription(Odometry, "/gdq/msg/gdq_odom", self.on_odom, 50)
            for topic in (CTRL_TOPIC, PID_TOPIC):
                self.create_subscription(Twist, topic, lambda m, tp=topic: self.on_cmd(tp, m), 50)
            self.create_subscription(PathMsg, "/vln/trajectory", lambda m: self.on_path("s1", m), 10)
            self.create_subscription(PathMsg, "/gdm/local_plan_map", lambda m: self.on_path("gdm", m), 10)
            self.create_subscription(String, "/vln/s2", self.on_s2, 10)
            self.create_subscription(String, "/vln/status", lambda m: self.row("status.csv", [self.now(), m.data]), 10)
            try:
                from gd_ifc_pkg.msg import GDGGuidancePoint
                self.create_subscription(GDGGuidancePoint, "/gdg/data/gp", self.on_gp, 10)
            except ImportError:
                self.get_logger().warn("gd_ifc_pkg missing: /gdg/data/gp not recorded")
            self.create_timer(1.0, lambda: self.row("pubs.csv", [self.now(), self.count_publishers(CTRL_TOPIC),
                                                                  self.count_publishers(PID_TOPIC)]))

        def now(self):
            return round(self.get_clock().now().nanoseconds * 1e-9, 4)

        def row(self, k, r):
            self.w[k].writerow(r)

        def jl(self, k, obj):
            self.f[k].write(json.dumps(obj) + "\n")

        def on_odom(self, m):
            st = m.header.stamp
            t = st.sec + st.nanosec * 1e-9 or self.now()
            p = m.pose.pose
            self.row("odom.csv", [f"{t:.4f}", f"{p.position.x:.4f}", f"{p.position.y:.4f}",
                                  f"{yaw_q(p.orientation):.5f}"])

        def on_cmd(self, topic, m):
            self.row("cmd.csv", [self.now(), topic, f"{m.linear.x:.4f}", f"{m.angular.z:.4f}"])

        def on_path(self, src, m):
            if not m.poses:
                pts = []
            else:
                pts = np.array([[q.pose.position.x, q.pose.position.y, q.pose.position.z, 1.0] for q in m.poses])
                fr = m.header.frame_id or self.world
                if fr != self.world:
                    try:
                        T = self.tf_to_T(self.buf.lookup_transform(self.world, fr, self.Time()))
                    except Exception as e:
                        self.get_logger().warn(f"{src} path TF {fr}->{self.world}: {e}", throttle_duration_sec=5.0)
                        return
                    pts = pts @ T.T
                pts = np.round(pts[:, :2], 4).tolist()
            self.jl("refs.jsonl", {"t": self.now(), "src": src, "pts": pts})

        def on_s2(self, m):
            j = json.loads(m.data)
            j["t_rx"] = self.now()
            self.jl("s2.jsonl", j)

        def on_gp(self, m):
            o = m.odometry.pose.pose
            yaw = yaw_q(o.orientation)
            xy = [[o.position.x + math.cos(yaw) * q.x - math.sin(yaw) * q.y,
                   o.position.y + math.sin(yaw) * q.x + math.cos(yaw) * q.y] for q in m.position]
            self.jl("gp.jsonl", {"t": self.now(), "type": list(m.gp_type), "xy": xy})

    rclpy.init()
    n = Inspect()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, Exception):
        pass
    for f in n.f.values():
        f.close()
    res = analyze(n.out, n.mode, n.goal, n.map_name)
    print("[live_inspect] " + json.dumps(res), flush=True)
    rclpy.try_shutdown()


# ---------------------------------------------------------------- analysis (offline)
def load(raw):
    def rows(name):
        p = raw / name
        return list(csv.reader(open(p))) if p.exists() else []

    def jl(name):
        p = raw / name
        return [json.loads(line) for line in open(p) if line.strip()] if p.exists() else []

    od = np.array([[float(v) for v in r] for r in rows("odom.csv")]).reshape(-1, 4)
    od = od[np.argsort(od[:, 0], kind="stable")]
    cmd = [(float(r[0]), r[1], float(r[2]), float(r[3])) for r in rows("cmd.csv")]
    pubs = np.array([[float(v) for v in r] for r in rows("pubs.csv")]).reshape(-1, 3)
    status = [(float(r[0]), r[1]) for r in rows("status.csv")]
    return od, cmd, jl("refs.jsonl"), jl("s2.jsonl"), jl("gp.jsonl"), status, pubs


def seg_dist(p, pts):
    """Distance from p to the polyline; the first segment extends backwards (the path starts traj_skip waypoints
    ahead of the robot) and the last one forwards, so being behind/past the path is not counted as cross-track."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return float("nan"), 0
    best, bi = float("inf"), 0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            continue
        s = float((p - a) @ ab) / L2
        lo = -np.inf if i == 0 else 0.0
        hi = np.inf if i == len(pts) - 2 else 1.0
        s = min(max(s, lo), hi)
        d = float(np.linalg.norm(a + s * ab - p))
        if d < best:
            best, bi = d, i
    return best, bi


def curvature(pts, i, span=0.3):
    """Signed curvature of the polyline around vertex i from points ~span metres before / after."""
    pts = np.asarray(pts, float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s0 = cum[i]
    j0 = int(np.searchsorted(cum, s0 - span))
    j0 = max(0, min(j0, i))
    j1 = min(len(pts) - 1, int(np.searchsorted(cum, s0 + span)))
    a, b, c = pts[j0], pts[i], pts[j1]
    if j0 == i:
        b = pts[min(i + 1, len(pts) - 1)] if j1 > i + 1 else b
    ab, bc, ac = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(c - a)
    if min(ab, bc, ac) < 1e-3:
        return 0.0
    cross = (b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]
    return float(2 * cross / (ab * bc * ac))


def body_vel(od, win=0.2):
    """Forward speed / yaw rate from pose differences over ~win s (central)."""
    t = od[:, 0]
    i0 = np.searchsorted(t, t - win / 2)
    i1 = np.clip(np.searchsorted(t, t + win / 2), 0, len(t) - 1)
    dt = t[i1] - t[i0]
    ok = dt > 1e-3
    dx, dy = od[i1, 1] - od[i0, 1], od[i1, 2] - od[i0, 2]
    yaw = od[:, 3]
    v = np.where(ok, (dx * np.cos(yaw) + dy * np.sin(yaw)) / np.where(ok, dt, 1), np.nan)
    dyaw = np.arctan2(np.sin(od[i1, 3] - od[i0, 3]), np.cos(od[i1, 3] - od[i0, 3]))
    w = np.where(ok, dyaw / np.where(ok, dt, 1), np.nan)
    return v, w


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return round(float(np.percentile(a, q)), 3) if a.size else None


def analyze(out, mode, goal="", map_name="etri_indoor", ref_max_age=3.0):
    out = Path(out)
    assert mode in LIMITS, f"unknown mode {mode!r}"
    od, cmd, refs, s2, gps, status, pubs = load(out / "raw")
    res = {"mode": mode, "n_odom": len(od), "n_cmd": len(cmd), "n_s2": len(s2)}
    if len(od) < 2:
        res["error"] = "no odom"
        (out / "inspect.json").write_text(json.dumps(res, indent=1))
        return res
    v_max, w_max = LIMITS[mode]
    ref_src = "s1" if mode == "trajectory" else "gdm"
    rlist = [r for r in refs if r["src"] == ref_src]
    rt = np.array([r["t"] for r in rlist])
    ctrl = [c for c in cmd if c[1] == CTRL_TOPIC]
    pid = [c for c in cmd if c[1] == PID_TOPIC]
    ct = np.array([c[0] for c in ctrl])

    def ref_at(t):
        k = int(np.searchsorted(rt, t, side="right")) - 1
        if k < 0 or t - rt[k] > ref_max_age or len(rlist[k]["pts"]) < 2:
            return None
        return rlist[k]["pts"]

    def moving_cmd(t):  # the controller asked for motion within the last 0.5 s, and no PID override
        k = int(np.searchsorted(ct, t, side="right")) - 1
        if k < 0 or t - ct[k] > 0.5 or (abs(ctrl[k][2]) < 1e-3 and abs(ctrl[k][3]) < 1e-3):
            return False
        return not any(t - 0.5 <= c[0] <= t and (abs(c[2]) > 1e-3 or abs(c[3]) > 1e-3) for c in pid)

    # cross-track
    xt = []
    for t, x, y, _ in od[::2]:
        ref = ref_at(t)
        if ref is not None and moving_cmd(t):
            xt.append((t, seg_dist(np.array([x, y]), ref)[0]))
    xa = np.array([d for _, d in xt])
    res["cross_track"] = {"n": len(xa), "rmse_m": round(float(np.sqrt(np.mean(xa ** 2))), 3) if len(xa) else None,
                          "p90_m": pct(xa, 90), "max_m": pct(xa, 100), "ref": ref_src}
    # limits
    over = [c for c in ctrl if abs(c[2]) > v_max + 1e-3 or abs(c[3]) > w_max + 1e-3]
    res["limits"] = {"v_max": v_max, "w_max": w_max, "n_over": len(over),
                     "v_abs_max": pct([abs(c[2]) for c in ctrl], 100), "w_abs_max": pct([abs(c[3]) for c in ctrl], 100),
                     "n_pid_cmd": len(pid)}
    # curvature consistency
    oti = od[:, 0]
    cv = []
    for t, _, v, w in ctrl:
        if v < 0.1:
            continue
        ref = ref_at(t)
        if ref is None:
            continue
        k = min(int(np.searchsorted(oti, t)), len(od) - 1)
        _, i = seg_dist(od[k, 1:3], ref)
        kap = curvature(ref, i + 1 if i + 1 < len(ref) - 1 else i)
        cv.append((t, w, v * kap))
    ca = np.array([[w, vk] for _, w, vk in cv]).reshape(-1, 2)
    res["curvature"] = {"n": len(ca), "abs_err_p50": pct(np.abs(ca[:, 0] - ca[:, 1]), 50),
                        "abs_err_p90": pct(np.abs(ca[:, 0] - ca[:, 1]), 90),
                        "corr": round(float(np.corrcoef(ca.T)[0, 1]), 3) if len(ca) > 5 and ca.std(0).min() > 1e-6
                        else None}
    # execution: actual (pose differences) vs command on the topic actually driving (PID overrides when active)
    va, wa = body_vel(od)
    allc = sorted(ctrl + pid)
    if len(allc) > 10:
        tc = np.array([c[0] for c in allc])
        vc, wc = np.array([c[2] for c in allc]), np.array([c[3] for c in allc])
        best = None
        for lag in np.arange(0.0, 1.01, 0.1):
            vi = np.interp(tc + lag, oti, va)
            wi = np.interp(tc + lag, oti, wa)
            ok = np.isfinite(vi) & np.isfinite(wi) & (tc + lag <= oti[-1])
            if ok.sum() < 10:
                continue
            e = float(np.sqrt(np.mean((vi[ok] - vc[ok]) ** 2)) + np.sqrt(np.mean((wi[ok] - wc[ok]) ** 2)))
            if best is None or e < best[0]:
                best = (e, lag, float(np.sqrt(np.mean((vi[ok] - vc[ok]) ** 2))),
                        float(np.sqrt(np.mean((wi[ok] - wc[ok]) ** 2))))
        if best is not None:
            res["execution"] = {"lag_s": round(best[1], 1), "v_rmse": round(best[2], 3), "w_rmse": round(best[3], 3)}
    # publishers
    res["publishers"] = {"gdq_cmd_max": int(pubs[:, 1].max()) if len(pubs) else None,
                         "cmd_vel_max": int(pubs[:, 2].max()) if len(pubs) else None}
    # stop
    fin = od[-1, 1:3]
    path_len = float(np.sum(np.linalg.norm(np.diff(od[:, 1:3], axis=0), axis=1)))
    s2g = [j for j in s2 if j.get("goal_xy")]
    stop_t = next((t for t, s in status if s == "STOP"), None)
    res["stop"] = {"final_xy": np.round(fin, 3).tolist(), "path_len_m": round(path_len, 2),
                   "stop_status_t": stop_t, "n_s2_goal": len(s2g),
                   "n_s2_action": sum(1 for j in s2 if j.get("action") is not None)}
    if s2g:
        res["stop"]["dist_last_s2_goal_m"] = round(float(np.hypot(*(fin - np.array(s2g[-1]["goal_xy"][:2])))), 3)
    gxy = None
    if goal:
        gxy = np.array([float(v) for v in goal.split(",")])
        res["stop"]["dist_goal_m"] = round(float(np.hypot(*(fin - gxy))), 3)
    res["pass"] = {"cross_track": res["cross_track"]["rmse_m"] is not None and res["cross_track"]["rmse_m"] <= 0.15,
                   "limits": len(over) == 0,
                   "publishers": res["publishers"]["gdq_cmd_max"] == 1}
    (out / "inspect.json").write_text(json.dumps(res, indent=1))
    try:
        plot(out, od, rlist, refs, s2g, gps, gxy, ctrl, pid, oti, va, wa, xt, cv, map_name, res)
    except Exception as e:  # the png is a convenience; never lose the numbers because of it
        print(f"[live_inspect] plot failed: {e}", flush=True)
    return res


def plot(out, od, rlist, refs, s2g, gps, gxy, ctrl, pid, oti, va, wa, xt, cv, map_name, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 2, 1)
    mj = MAPS / f"{map_name}.json"
    xy = od[:, 1:3]
    if mj.exists():
        m = json.loads(mj.read_text())
        img = plt.imread(str(MAPS / f"{map_name}.png"))
        ext = [m["x0"], m["x0"] + m["width"] * m["res"], m["y0"], m["y0"] + m["height"] * m["res"]]
        ax.imshow(img, extent=ext, cmap="gray", origin="upper")
    for r in [r for r in refs if r["src"] == "gdm"][::3]:
        if len(r["pts"]) > 1:
            p = np.array(r["pts"])
            ax.plot(p[:, 0], p[:, 1], "-", color="#00b400", lw=0.6, alpha=0.5)
    for r in [r for r in refs if r["src"] == "s1"][::2]:
        if len(r["pts"]) > 1:
            p = np.array(r["pts"])
            ax.plot(p[:, 0], p[:, 1], "-", color="#0078ff", lw=0.6, alpha=0.5)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=od[:, 0] - od[0, 0], s=3, cmap="autumn", zorder=3)
    fig.colorbar(sc, ax=ax, fraction=0.03, label="t [s]")
    for j in s2g:
        g = j["goal_xy"]
        ax.plot(g[0], g[1], "x", color="m", ms=7, mew=2, zorder=4)
    for g in gps:
        for p in g["xy"]:
            ax.plot(p[0], p[1], "s", mfc="none", mec="c", ms=9, zorder=4)
    ax.plot(*xy[0], "go", ms=10, label="start")
    ax.plot(*xy[-1], "ks", ms=8, label="final")
    if gxy is not None:
        ax.add_patch(plt.Circle(gxy, 1.5, fill=False, color="r", lw=2))
        ax.plot(*gxy, "r*", ms=14, label="episode goal")
    lo, hi = xy.min(0) - 3, xy.max(0) + 3
    if gxy is not None:
        lo, hi = np.minimum(lo, gxy - 3), np.maximum(hi, gxy + 3)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("path (time colour)  blue=S1 path  green=gdm plan  x=S2 goal  cyan=gp", fontsize=9)

    t0 = od[0, 0]
    a1 = fig.add_subplot(3, 2, 2)
    if ctrl:
        c = np.array([[t, v] for t, _, v, _ in ctrl])
        a1.step(c[:, 0] - t0, c[:, 1], where="post", lw=0.8, label="v cmd (gdq)")
    if pid:
        c = np.array([[t, v] for t, _, v, _ in pid])
        a1.plot(c[:, 0] - t0, c[:, 1], ".", ms=2, label="v cmd (PID)")
    a1.plot(oti - t0, va, lw=0.8, label="v actual (pose diff)")
    a1.set_ylabel("v [m/s]")
    a1.legend(fontsize=7)
    a2 = fig.add_subplot(3, 2, 4, sharex=a1)
    if ctrl:
        c = np.array([[t, w] for t, _, _, w in ctrl])
        a2.step(c[:, 0] - t0, c[:, 1], where="post", lw=0.8, label="w cmd (gdq)")
    if pid:
        c = np.array([[t, w] for t, _, _, w in pid])
        a2.plot(c[:, 0] - t0, c[:, 1], ".", ms=2, label="w cmd (PID)")
    a2.plot(oti - t0, wa, lw=0.8, label="w actual")
    if cv:
        c = np.array(cv)
        a2.plot(c[:, 0] - t0, c[:, 2], ".", ms=2, label="v*kappa(ref)")
    a2.set_ylabel("w [rad/s]")
    a2.legend(fontsize=7)
    a3 = fig.add_subplot(3, 2, 6, sharex=a1)
    if xt:
        c = np.array(xt)
        a3.plot(c[:, 0] - t0, c[:, 1], lw=0.8)
    a3.axhline(0.15, color="r", ls="--", lw=0.8)
    a3.set_ylabel("cross-track [m]")
    a3.set_xlabel("t [s]")
    fig.suptitle(json.dumps({k: res[k] for k in ("cross_track", "limits", "execution", "stop") if k in res})[:300],
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "inspect.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--analyze", required=True)
        ap.add_argument("--mode", default="trajectory")
        ap.add_argument("--goal", default="")
        ap.add_argument("--map", default="etri_indoor")
        a = ap.parse_args()
        print(json.dumps(analyze(a.analyze, a.mode, a.goal, a.map), indent=1))
    else:
        record()
