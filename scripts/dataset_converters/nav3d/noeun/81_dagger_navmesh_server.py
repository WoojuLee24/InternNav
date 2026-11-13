"""noeun 81 · DAgger navmesh 서버 — 롤아웃 이동(/step)과 전문가 연속 궤적(/expert)을 navmesh 위에서 계산한다 (habitat_sim, CPU).

`scripts/eval/noeun_navmesh_server.py` 의 Planner(/next) 를 그대로 쓰고 두 가지를 더한다:
  /step   {"pos", "yaw", "base_height", "dist"} -> 전진 dist 를 navmesh 위로 (try_step, 벽에서 미끄러짐) -> {"pos": base 위치}
  /expert {"pos", "yaw", "base_height", "ref", "hint", ...} -> 전문가가 여기서부터 걷는 궤적 [{pos, yaw, action}] (렌더 없음)
  STOP 은 **끝에서 stop_radius_m 안 + 목표가 stop_yaw_deg(기본 45°) 안에 보일 때만**. 등 뒤·옆이면 목표 쪽으로 회전한다
  (거리만 보면 목표를 등지고 선 자세에 STOP 이 붙는다).
전문가 = 정답 경로(ref, 문장이 말하는 길)로 복귀: ref 위 가장 가까운 점(hint 이후, 되돌아가지 않음)에서 lookahead_m 앞 점을
navmesh 최단 경로로 향한다. ref 끝에서 stop_radius_m 안이면 STOP. 회전 15° · 전진 0.25 m (학습 데이터 vlnce_discretize 와 같음).

좌표: 요청 · 응답은 sim world (Z-up). 바닥 = base z - base_height.

    python3 scripts/dataset_converters/nav3d/noeun/81_dagger_navmesh_server.py --port 8041
"""
import argparse
import importlib.util
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('noeun_navmesh_server', Path(__file__).resolve().parents[4] / 'scripts/eval/noeun_navmesh_server.py')
NS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NS)
STOP, FORWARD, LEFT, RIGHT = NS.STOP, NS.FORWARD, NS.LEFT, NS.RIGHT


class DaggerPlanner(NS.Planner):
    def __init__(self, navmesh, rotation_deg, min_seg, forward_m, stop_yaw_deg=45.0):
        super().__init__(navmesh, 0.0, rotation_deg, min_seg)
        self.rot, self.forward_m = math.radians(rotation_deg), forward_m
        self.stop_yaw_deg = float(stop_yaw_deg)  # 목표가 이 각도 안에 있을 때만 STOP (요청의 stop_yaw_deg 로 덮어쓸 수 있다)

    def step(self, pos, yaw, base_height, dist):
        """base 위치에서 yaw 방향 dist 전진 (navmesh 위). -> 새 base 위치."""
        floor = np.asarray(pos, float) - [0, 0, base_height]
        s = np.array(self.pf.snap_point(NS.w2h(floor)))
        e = s + NS.w2h([math.cos(yaw) * dist, math.sin(yaw) * dist, 0.0])
        n = np.array(self.pf.try_step(s, e))
        return (np.array(NS.h2w(n)) + [0, 0, base_height]).tolist()

    @staticmethod
    def progress(pos_floor, ref, hint):
        """ref 위 가장 가까운 점 번호 (hint 이전으로 되돌아가지 않음, 층이 다른 점은 높이 차로 밀어냄)."""
        d = np.linalg.norm((ref[hint:] - pos_floor) * [1, 1, 3], axis=1)
        return hint + int(np.argmin(d)), float(np.min(np.linalg.norm(ref[hint:, :2] - pos_floor[:2], axis=1)))

    @staticmethod
    def target(ref, i, lookahead_m):
        """ref 의 i 에서 경로 길이 lookahead_m 앞 점 번호."""
        L = np.cumsum(np.linalg.norm(np.diff(ref[i:], axis=0), axis=1))
        return min(i + int(np.searchsorted(L, lookahead_m)) + 1, len(ref) - 1)

    def expert(self, req):
        ref = np.asarray(req['ref'], float)  # 바닥 높이 world 점들
        pos, yaw, bh = np.asarray(req['pos'], float), float(req['yaw']), float(req['base_height'])
        hint, out, t = int(req.get('hint', 0)), [], None
        for _ in range(int(req.get('max_steps', 60))):
            floor = pos - [0, 0, bh]
            i, dev = self.progress(floor, ref, hint)
            hint = i
            d_goal = float(np.linalg.norm(floor[:2] - ref[-1, :2]))
            if d_goal < req.get('stop_radius_m', 0.35) and abs(floor[2] - ref[-1, 2]) < 1.0:
                # 반경 안이어도 **목표가 앞에 있을 때만** STOP. 등 뒤/옆이면 목표 쪽으로 돈다.
                #   거리만 보면 목표를 등지고 선 자세에 STOP 이 붙는다 (84 smoke: B 가지 90°+ 구간 STOP 13 개).
                #   정답 경로 위 데이터(반경 STOP 학습)에서는 "남은 행동이 전진뿐" 조건이 각도 역할을 대신했지만,
                #   82 · 84 는 일부러 비켜서고 지나치므로 그 조건이 성립하지 않는다.
                #   아주 가까우면 목표 방향이 정의되지 않으므로(84 view() 와 같은 0.3 m) 그대로 STOP.
                if d_goal < req.get('min_goal_dir_m', 0.3):
                    err = 0.0
                else:
                    err = (math.atan2(ref[-1, 1] - floor[1], ref[-1, 0] - floor[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(err) <= math.radians(req.get('stop_yaw_deg', self.stop_yaw_deg)):
                    out.append({'pos': pos.tolist(), 'yaw': yaw, 'action': STOP, 'ref_i': i, 'dev': dev})
                    break
                a = LEFT if err > 0 else RIGHT
                out.append({'pos': pos.tolist(), 'yaw': yaw, 'action': a, 'ref_i': i, 'dev': dev})
                yaw += self.rot if a == LEFT else -self.rot
                continue
            # 목표 점은 가까워질 때(retarget_m)까지 유지 — 매 step 바꾸면 15° 양자화 때문에 좌우로 흔들린다 (vlnce_discretize 의 waypoint 추종과 같은 방식)
            if t is None or t <= i or np.linalg.norm(floor[:2] - ref[t, :2]) < req.get('retarget_m', 0.5):
                t = self.target(ref, i, req.get('lookahead_m', 2.0))
            g = ref[t]
            nx = self.next({'pos': pos.tolist(), 'goal': (g + [0, 0, bh]).tolist(), 'base_height': bh, 'yaw': yaw})
            a = nx['action']
            if a == STOP:  # 목표 점이 가깝거나 경로 없음 -> 끝 방향으로 직접
                g = ref[-1]
                nx = self.next({'pos': pos.tolist(), 'goal': (g + [0, 0, bh]).tolist(), 'base_height': bh, 'yaw': yaw})
                a = nx['action'] if nx['action'] != STOP else FORWARD
            out.append({'pos': pos.tolist(), 'yaw': yaw, 'action': a, 'ref_i': i, 'dev': dev})
            if a == FORWARD:
                # 전진하면서 남은 각도(< rotation/2)를 흡수한다 — 학습 데이터(RBQ 물리 걷기)도 걸으며 방향을 연속으로 고친다
                if nx.get('next') is not None:
                    yaw = math.atan2(nx['next'][1] - pos[1], nx['next'][0] - pos[0])
                pos = np.asarray(self.step(pos, yaw, bh, self.forward_m))
            elif a == LEFT:
                yaw += self.rot
            elif a == RIGHT:
                yaw -= self.rot
            else:
                assert False, f'unreachable action={a!r}'
        return {'steps': out, 'hint': out[0]['ref_i'] if out else hint}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--navmesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh')
    ap.add_argument('--port', type=int, default=8041)
    ap.add_argument('--rotation_deg', type=float, default=15.0)
    ap.add_argument('--min_seg', type=float, default=0.05)
    ap.add_argument('--forward_m', type=float, default=0.25)
    ap.add_argument('--stop_yaw_deg', type=float, default=45.0,
                    help='반경 안이어도 목표가 이 각도 밖이면 STOP 대신 회전. 180 = 예전처럼 거리만 봄')
    args = ap.parse_args()
    P = DaggerPlanner(args.navmesh, args.rotation_deg, args.min_seg, args.forward_m, args.stop_yaw_deg)
    routes = {
        '/next': lambda r: P.next(r),
        '/step': lambda r: {'pos': P.step(r['pos'], float(r['yaw']), float(r['base_height']), float(r.get('dist', args.forward_m)))},
        '/expert': lambda r: P.expert(r),
    }

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path not in routes:
                self.send_error(404)
                return
            body = json.dumps(routes[self.path](json.loads(self.rfile.read(int(self.headers['Content-Length']))))).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f'[dagger navmesh] {args.navmesh} · :{args.port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
