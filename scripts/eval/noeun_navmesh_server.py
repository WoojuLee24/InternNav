"""노은역 navmesh 경로 서버 — S2 에이전트의 pixel goal 추종을 habitat 처럼 navmesh 최단 경로로 (follower='navmesh').

habitat `_run_eval_system2` 는 pixel goal 을 ShortestPathFollower(navmesh, goal_radius 0.25) 로 따라간다. Isaac · sim 에는
navmesh 가 없어 이 서버(habitat_sim PathFinder, CPU)가 대신 경로를 구한다. 다음 동작 규칙은 GT 재생
(`scripts/eval/noeun_gt_replay_check.py`, VLN-PE 37/40 · 막힘 0 검증)과 같다:
  목표(navmesh 로 snap)까지 xy < goal_radius 면 STOP, 아니면 경로의 다음 점 방향 오차 > rotation/2 면 회전, 아니면 전진.
habitat 과 다른 점: ShortestPathFollower 는 동작을 navmesh 위에서 미리 굴려 보고 고르는 greedy 추종이고, 여기는
경로점 방향 규칙이다 (경로를 매 step 다시 구한다).

좌표: 요청은 sim world (Isaac / sim_docker, Z-up) — habitat = [x, z, -y]. 바닥 z = base z - base_height.

    python3 scripts/eval/noeun_navmesh_server.py --navmesh data/data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh --port 8040
POST /next {"pos": [x,y,z], "yaw": rad, "goal": [x,y,z], "base_height": 1.05} ->
     {"action": 0|1|2|3, "found": bool, "dist_xy": m, "geodesic": m, "goal_snapped": [x,y,z] (world), "next": [x,y,z] (world)}
"""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import habitat_sim
import numpy as np

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3


def w2h(p):
    return np.array([p[0], p[2], -p[1]], dtype=np.float32)


def h2w(p):
    return [float(p[0]), float(-p[2]), float(p[1])]


class Planner:
    def __init__(self, navmesh, goal_radius, rotation_deg, min_seg):
        self.pf = habitat_sim.PathFinder()
        self.pf.load_nav_mesh(navmesh)
        assert self.pf.is_loaded, navmesh
        self.goal_radius, self.half_rot, self.min_seg = goal_radius, math.radians(rotation_deg) / 2, min_seg

    def next(self, req):
        pos, goal = np.asarray(req["pos"], float), np.asarray(req["goal"], float)
        floor = pos.copy()
        floor[2] -= float(req["base_height"])
        s = np.array(self.pf.snap_point(w2h(floor)))
        g = w2h(goal)
        if not self.pf.is_navigable(g):  # habitat 과 같다: navmesh 밖 목표는 snap
            g = np.array(self.pf.snap_point(g))
        out = {"goal_snapped": h2w(g), "found": False, "geodesic": None, "next": None}
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(g))):
            return {**out, "action": STOP, "dist_xy": None}
        out["dist_xy"] = float(math.hypot(pos[0] - h2w(g)[0], pos[1] - h2w(g)[1]))
        if out["dist_xy"] < self.goal_radius:
            return {**out, "action": STOP}
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = s, g
        if not self.pf.find_path(path) or len(path.points) < 2:
            return {**out, "action": STOP}  # 갈 수 없음 -> STOP (에이전트가 재질의)
        out["found"], out["geodesic"] = True, float(path.geodesic_distance)
        nxt = next((p for p in path.points[1:] if math.hypot(p[0] - s[0], p[2] - s[2]) > self.min_seg), path.points[-1])
        nw = h2w(nxt)
        out["next"] = nw
        err = math.atan2(nw[1] - pos[1], nw[0] - pos[0]) - float(req["yaw"])
        err = (err + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > self.half_rot:
            return {**out, "action": LEFT if err > 0 else RIGHT}
        return {**out, "action": FORWARD}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navmesh", default="data/data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh")
    ap.add_argument("--port", type=int, default=8040)
    ap.add_argument("--goal_radius", type=float, default=0.25, help="habitat ShortestPathFollower goal_radius")
    ap.add_argument("--rotation_deg", type=float, default=15.0)
    ap.add_argument("--min_seg", type=float, default=0.05, help="이보다 가까운 경로점은 건너뛴다 (시작점 snap 오차)")
    args = ap.parse_args()
    planner = Planner(args.navmesh, args.goal_radius, args.rotation_deg, args.min_seg)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/next":
                self.send_error(404)
                return
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(planner.next(req)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"[navmesh] {args.navmesh} area {planner.pf.navigable_area:.0f} m2 · :{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
