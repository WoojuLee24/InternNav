"""06o — action 하나가 실제로 얼마나 움직이는지(action 의미)를 릴리스 vln_ce · vln_pe 와 우리 노은역 RBQ(06k walk) 에서 같은 자로 잰다.

왜: 06m 은 RBQ 가 물리로 걸은 자세로 렌더한다. 릴리스 vln_ce 는 운동학(전진 = 정확히 0.25 m · 회전 = 정확히 15°),
릴리스 vln_pe 는 H1 물리(`move_by_discrete`). 로봇·속도 같은 파라미터는 달라도 되지만, **action 라벨이 뜻하는 움직임**
(전진이면 앞으로 ≈0.25 m 가고 방향은 거의 안 바뀜, 회전이면 제자리에서 ≈15°)이 같아야 같은 알고리즘의 데이터다.

## 재는 것 (action 마다, 그 action 전후 프레임)
- step: 몸(vln_pe·RBQ) 또는 카메라(vln_ce) 수평 이동 거리 [m]
- dyaw: 방향 변화 [°] (회전 action 은 크기, 전진 action 은 절댓값)

## 판정 기준 (결과를 보기 전에 정함) — 우리 RBQ vs 각 릴리스
| | 기준 | 근거 |
|---|---|---|
| A1 전진 거리 | 우리 전진 step 중앙이 릴리스 중앙의 0.8 ~ 1.2 배 | 걸음 길이가 같아야 '전진 1 번' 의 뜻이 같다 |
| A2 전진 중 방향 | 우리 전진 |dyaw| p90 ≤ 릴리스 p90 + 5° | 06k 는 걸으면서 방향 보정(steer) — 전진에 회전이 섞이면 라벨이 틀린다 |
| A3 회전 각 | 우리 회전 |dyaw| 중앙 15° ± 3° · 릴리스 중앙과 차 ≤ 3° | 회전 1 번 = 15° |
| A4 회전 중 이동 | 우리 회전 step p90 ≤ 릴리스 p90 + 0.05 m | 제자리 회전이어야 한다 (vln_ce 는 카메라가 몸 앞이라 0.095 m 가 정상) |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/06o_action_semantics_check.py
"""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

D = 'data/InternData-N1-v0.5-mini'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walk_json', default='scripts/dataset_converters/nav3d/logs/06k_rbq_walk/walk_noeun_v4.json')
    ap.add_argument('--n_scenes', type=int, default=10, help='릴리스에서 쓸 씬 수 (앞에서부터)')
    ap.add_argument('--eps_per_scene', type=int, default=20)
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/06o_action_semantics/summary.json')
    return ap.parse_args()


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def stats(acts, steps, dyaws):
    acts, steps, dyaws = map(np.asarray, (acts, steps, dyaws))
    out = {}
    for name, m in (('forward', acts == 1), ('turn', (acts == 2) | (acts == 3))):
        if m.any():
            out[name] = {'n': int(m.sum()), 'step_med': float(np.median(steps[m])), 'step_p90': float(np.percentile(steps[m], 90)),
                         'dyaw_med': float(np.median(np.abs(dyaws[m]))), 'dyaw_p90': float(np.percentile(np.abs(dyaws[m]), 90))}
    return out


def release_vlnce(n_scenes, k):
    import pandas as pd
    A, S, Y = [], [], []
    for sc in sorted(glob.glob(f'{D}/vln_ce/traj_data/r2r/*'))[:n_scenes]:
        for p in sorted(glob.glob(f'{sc}/data/chunk-000/*.parquet'))[:k]:
            d = pd.read_parquet(p, columns=['action', 'pose.125cm_0deg'])
            P = np.stack([np.stack(x).astype(np.float64) for x in d['pose.125cm_0deg']])
            fw = P[:, :3, 2]                                   # 카메라 앞 (pitch 0 rig)
            yaw = np.degrees(np.arctan2(fw[:, 1], fw[:, 0]))
            a = d['action'].to_numpy()
            for i in range(1, len(P)):                         # vln_ce: action[i] = i-1 -> i 로 온 action
                A.append(a[i])
                S.append(float(np.linalg.norm(P[i, :2, 3] - P[i - 1, :2, 3])))
                Y.append(wrap(yaw[i] - yaw[i - 1]))
    return A, S, Y


def release_vlnpe(n_scenes, k):
    import pandas as pd
    A, S, Y = [], [], []
    for sc in sorted(glob.glob(f'{D}/vln_pe/traj_data/r2r/*'))[:n_scenes]:
        for p in sorted(glob.glob(f'{sc}/data/chunk-000/*.parquet'))[:k]:
            d = pd.read_parquet(p, columns=['observation.action', 'observation.robot_position', 'observation.robot_yaw'])
            X = np.stack(d['observation.robot_position'].to_numpy())
            yaw = np.degrees(d['observation.robot_yaw'].to_numpy())
            a = d['observation.action'].to_numpy()
            for i in range(len(X) - 1):                        # vln_pe: action[i] = i 에서 한 action
                A.append(a[i])
                S.append(float(np.linalg.norm(X[i + 1, :2] - X[i, :2])))
                Y.append(wrap(yaw[i + 1] - yaw[i]))
    return A, S, Y


def ours(walk_json):
    A, S, Y = [], [], []
    for r in json.loads(Path(walk_json).read_text())['results']:
        if not r['success']:
            continue
        fr = r['frames']
        for i in range(1, len(fr)):                            # walk: frames[i]['action'] = i-1 -> i 로 온 action
            w, x, y, z = fr[i - 1]['quat_wxyz']
            y0 = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            w, x, y, z = fr[i]['quat_wxyz']
            y1 = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            A.append(fr[i]['action'])
            S.append(float(np.linalg.norm(np.subtract(fr[i]['pos'][:2], fr[i - 1]['pos'][:2]))))
            Y.append(wrap(y1 - y0))
    return A, S, Y


def judge(us, rel):
    f, t, rf, rt = us['forward'], us['turn'], rel['forward'], rel['turn']
    return {'A1': {'ratio': f['step_med'] / rf['step_med'], 'pass': 0.8 <= f['step_med'] / rf['step_med'] <= 1.2},
            'A2': {'ours_p90': f['dyaw_p90'], 'rel_p90': rf['dyaw_p90'], 'pass': f['dyaw_p90'] <= rf['dyaw_p90'] + 5.0},
            'A3': {'ours_med': t['dyaw_med'], 'rel_med': rt['dyaw_med'], 'pass': abs(t['dyaw_med'] - 15) <= 3 and abs(t['dyaw_med'] - rt['dyaw_med']) <= 3},
            'A4': {'ours_p90': t['step_p90'], 'rel_p90': rt['step_p90'], 'pass': t['step_p90'] <= rt['step_p90'] + 0.05}}


def main():
    args = parse()
    R = {'release_vlnce': stats(*release_vlnce(args.n_scenes, args.eps_per_scene)),
         'release_vlnpe': stats(*release_vlnpe(args.n_scenes, args.eps_per_scene)),
         'ours_rbq': stats(*ours(args.walk_json))}
    R['vs_vlnce'] = judge(R['ours_rbq'], R['release_vlnce'])
    R['vs_vlnpe'] = judge(R['ours_rbq'], R['release_vlnpe'])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=1))
    for k in ('release_vlnce', 'release_vlnpe', 'ours_rbq'):
        print(k, json.dumps({a: {b: round(c, 3) for b, c in v.items()} for a, v in R[k].items()}, ensure_ascii=False))
    for k in ('vs_vlnce', 'vs_vlnpe'):
        print(k, ' · '.join(f"{a} {'PASS' if v['pass'] else 'FAIL'} {({b: round(c, 3) for b, c in v.items() if b != 'pass'})}" for a, v in R[k].items()))
    print('->', args.out)


if __name__ == '__main__':
    main()
