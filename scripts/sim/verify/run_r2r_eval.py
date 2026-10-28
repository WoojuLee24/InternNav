"""R2R 형식 val (`raw_data/r2r/<split>/<split>.json.gz`) 을 live sim(RBQ10, sim_docker) 에서 평가한다 — run_e2e.py 의 ep 루프 + VLN 지표.

ep 마다: 출발 자세로 순간이동(`/sim/teleport_pose`, 층 높이 포함) → 사람 있는 ep 면 녹화 재생 시각 t0 를 보냄(`/sim/hunav_t0`)
→ vln_bridge_node 실행(지시문) → STOP 또는 시간 초과까지 odom 기록 → 지표 → `<out>/<tag>/<mode>/<ep>/episode.json` · `summary.json`.

| 지표 | 뜻 |
|---|---|
| `ne_m` | 멈춘 곳 ↔ 도착점 3D 거리 (층이 다르면 크다) |
| `success` | 모델이 STOP 했고 `ne_m` ≤ goal 반경 (3 m, 릴리스와 같음) |
| `oracle_success` | 지나간 점 중 하나라도 도착점 반경 안 |
| `spl` | success × L / max(L, 걸은 길이), L = ep 의 `info.geodesic_distance` |
| `human_min_dist_m` · `human_close_s` | 사람 있는 ep: 같은 층 사람 중심까지 최소 수평 거리 · 그 거리가 0.5 m 미만이었던 시간 (녹화 npz 로 계산) |

좌표: ep 는 habitat Y-up (릴리스 규약) → sim world = [x, -z, y] (InternNav isaac 로더와 같음, nav3d/noeun/52_export_raw.py 가 씀).

sim 쪽 (사람 있는 val):
  GUIDEDOG_HUNAV_REPLAY=/isaac-sim/workspace/NoeunStation/hunav/humans_noeun_v3.npz ./run_rbq10_noeun_sim.sh
사용 (gd_vln 컨테이너, sim + gdm(Mode A: ./lpp-sim, Mode B: ./lpp-sim-vln) + 모델 스택이 떠 있어야 한다):
  python3 scripts/sim/verify/run_r2r_eval.py --mode pixel_goal --tag n1_noeun --episodes data/InternData-N1-v0.5-mini/vln_pe_noeun_nav3d_rbq_v6/raw_data/r2r/val_seen/val_seen.json.gz
"""
import argparse
import gzip
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_e2e import ROOT, Recorder, spin_for  # noqa: E402

BASE_H = 0.55   # RBQ10 서 있는 trunk 높이 − 바닥 (run_rbq10_sim STAND_Z_MIN)


def to_world(p):
    x, y, z = p
    return np.array([x, -z, y], np.float64)


def hab_yaw_to_world(rot):
    """habitat start_rotation [x,y,z,w] → sim yaw. InternNav 로더 변환(transform_rotation_z_90degrees)과 같은 결과."""
    return 2.0 * math.atan2(rot[1], rot[3]) + math.pi / 2


class EvalRecorder(Recorder):
    """odom 위치를 z · sim 시각까지 (지표에 필요)."""

    def __init__(self):
        super().__init__()
        self.tp3 = self.create_publisher(Pose, "/sim/teleport_pose", 1)
        self.t0_pub = self.create_publisher(Float64, "/sim/hunav_t0", 1)

    def on_odom(self, m):
        if self.recording:
            p = m.pose.pose.position
            self.path.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, p.x, p.y, p.z))


def teleport(rec, xyz_floor, yaw):
    while rec.tp3.get_subscription_count() == 0:
        spin_for(rec, 0.2)
    rec.cmd.publish(Twist())
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = float(xyz_floor[0]), float(xyz_floor[1]), float(xyz_floor[2] + BASE_H)
    msg.orientation.z, msg.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    for _ in range(3):   # 새 publisher 의 첫 메시지는 discovery 전에 사라질 수 있다 (run_e2e 와 같은 확인)
        rec.tp3.publish(msg)
        rec.path, rec.recording = [], True
        spin_for(rec, 6.0)
        rec.recording = False
        if rec.path and math.hypot(rec.path[-1][1] - xyz_floor[0], rec.path[-1][2] - xyz_floor[1]) < 0.5:
            return True
    return False


def human_dist(P, t_rel, path, t0, close_m=0.5):
    """P: 녹화 npz, path: [(stamp, x, y, z)], t_rel 시작 stamp → 사람 시각 t0 + (stamp − t_rel)."""
    dmin, close_s = float("inf"), 0.0
    for i, (s, x, y, z) in enumerate(path):
        k = int(np.clip(np.searchsorted(P["t"], t0 + s - t_rel) - 1, 0, len(P["t"]) - 1))
        same = np.abs(P["z"][k] - (z - BASE_H)) < 1.5
        if not same.any():
            continue
        d = float(np.hypot(P["x"][k, same] - x, P["y"][k, same] - y).min())
        dmin = min(dmin, d)
        if d < close_m and i:
            close_s += s - path[i - 1][0]
    return (None if math.isinf(dmin) else round(dmin, 3)), round(close_s, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["pixel_goal", "trajectory"], help="Mode A = pixel_goal (./lpp-sim) · Mode B = trajectory (./lpp-sim-vln)")
    ap.add_argument("--tag", required=True, help="모델/양자화 이름 (출력 폴더)")
    ap.add_argument("--episodes", required=True, help="raw_data/r2r/<split>/<split>.json.gz")
    ap.add_argument("--humans_dir", default=None, help="사람 있는 ep 의 녹화 npz 폴더 (기본: --episodes 의 raw_data/r2r)")
    ap.add_argument("--only", default="", help="쉼표로 episode_id")
    ap.add_argument("--timeout_base_s", type=float, default=60.0)
    ap.add_argument("--timeout_per_m", type=float, default=4.0, help="시간 초과 = base + 경로 길이 × 이 값 (RBQ ≈ 0.3~0.5 m/s)")
    ap.add_argument("--out", default="scripts/sim/verify/out/r2r_eval")
    ap.add_argument("--bridge_args", default="", help='추가 ros 인자, 예 "lookdown_ns:=/camera_lookdown/camera"')
    args = ap.parse_args()

    eps = json.load(gzip.open(args.episodes))["episodes"]
    if args.only:
        eps = [e for e in eps if str(e["episode_id"]) in args.only.split(",")]
    hdir = Path(args.humans_dir) if args.humans_dir else Path(args.episodes).parents[1]
    npz = {}
    rclpy.init()
    rec = EvalRecorder()
    summary = []
    for ep in eps:
        eid = str(ep["episode_id"])
        d = Path(args.out) / args.tag / args.mode / eid
        d.mkdir(parents=True, exist_ok=True)
        start, goal = to_world(ep["start_position"]), to_world(ep["goals"][0]["position"])
        yaw = hab_yaw_to_world(ep["start_rotation"])
        if not teleport(rec, start, yaw):
            print(f"[r2r_eval] WARN ep {eid} 순간이동 확인 안 됨", flush=True)
        hv = ep.get("hunav")
        rec.path, rec.min_range, rec.status = [], float("inf"), []
        rec.recording = True
        if hv:
            rec.t0_pub.publish(Float64(data=float(hv["t0"])))
        instr = ep["instruction"]["instruction_text"]
        cmd = ["python3", "scripts/sim/vln_bridge_node.py", "--ros-args", "-p", "use_sim_time:=true",
               "-p", f"mode:={args.mode}", "-p", f"log_dir:={d}", "-p", f"instruction:={instr}"]
        for kv in args.bridge_args.split():
            cmd += ["-p", kv]
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=open(d / "bridge.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        L = float(ep["info"]["geodesic_distance"])
        timeout = args.timeout_base_s + args.timeout_per_m * L
        t_wall, stopped = time.time(), False
        while time.time() - t_wall < timeout:
            spin_for(rec, 0.5)
            if any(s == "STOP" for _, s in rec.status):
                stopped = True
                spin_for(rec, 3.0)   # 멈출 때까지
                break
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        rec.recording = False
        rec.cmd.publish(Twist())
        P = np.array([p[1:] for p in rec.path]) if rec.path else start[None] + [0, 0, BASE_H]
        feet = P - [0, 0, BASE_H]
        walked = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
        dist = np.linalg.norm(feet - goal, axis=1)
        r = float(ep["goals"][0].get("radius", 3.0))
        res = {"episode_id": ep["episode_id"], "key": ep.get("key"), "kind": ep.get("kind"), "tag": args.tag, "mode": args.mode,
               "instruction": instr, "stopped_by_model": stopped, "duration_s": round(time.time() - t_wall, 1), "timeout_s": timeout,
               "geodesic_distance_m": L, "path_length_m": round(walked, 3), "ne_m": round(float(dist[-1]), 3),
               "success": bool(stopped and dist[-1] <= r), "oracle_success": bool((dist <= r).any()),
               "min_scan_range_m": rec.min_range if math.isfinite(rec.min_range) else None}
        res["spl"] = round(float(res["success"]) * L / max(L, walked, 1e-9), 4)
        if hv and rec.path:
            f = str(hdir / hv["humans"])
            npz.setdefault(f, dict(np.load(f)))
            res["human_min_dist_m"], res["human_close_s"] = human_dist(npz[f], rec.path[0][0], rec.path, float(hv["t0"]))
        np.savetxt(d / "path.csv", np.array([p for p in rec.path]) if rec.path else np.zeros((0, 4)), delimiter=",",
                   header="stamp,x,y,z", comments="")
        (d / "episode.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps(res, ensure_ascii=False), flush=True)
        summary.append(res)
    n = max(len(summary), 1)
    agg = {k: round(sum(float(s[k]) for s in summary) / n, 4) for k in ("success", "oracle_success", "spl", "ne_m")}
    out = Path(args.out) / args.tag / args.mode / "summary.json"
    out.write_text(json.dumps({"episodes_file": args.episodes, "n": len(summary), "mean": agg, "episodes": summary}, indent=1, ensure_ascii=False))
    print(f"[r2r_eval] {len(summary)} ep · {agg} -> {out}", flush=True)
    rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
