"""연속 경로 -> VLN-CE 이산 action 시퀀스 (전진 0.25 m / 제자리 회전 15°).

왜 필요한가
-----------
`vln_ce`는 **프레임 하나 = 이산 action 하나**다. 디스크 데이터를 직접 재서 확인했다
(`17DRP5sb8fy` 20 에피소드, 806 전이):

```
action=1 (전진)    이동 0.2500 m (min 0.2499 max 0.2538) · 회전  0.00°
action=2/3 (회전)  이동 0.0000 m                          · 회전 15.00°
action[0] = -1     에피소드당 정확히 1회, 프레임 0의 센티널
action=0 (STOP)    parquet에 없음 — 로더가 뒤에 붙인다
```

원 VLN-CE 규격(MOVE_FORWARD 0.25 m, TURN_LEFT/RIGHT 15°, STOP)과 정확히 일치한다.

우리 `03_sample_gt_paths.py`가 만드는 경로는 **연속**(0.14 m 균등 리샘플)이라 그대로는 못 쓴다.
**제자리 회전 프레임이 실제로 존재**하므로 단순 리샘플로는 재현이 안 된다 — 회전 중에는 위치가
멈춰 있어야 한다.

(`vln_pe`에는 이 제약이 없다. 거기서는 `max_step=200` 절단 때문에 프레임 간격만 맞추면 된다.)

규칙
----
`scripts/visualization/eval_gt_collect.py:56-73`의 `navigate_to_waypoint` 판정을 그대로 쓴다.
env를 스텝하는 대신 **우리 경로 위에서** 같은 결정을 내린다.

```
목표 웨이포인트까지 delta = wrap(bearing - theta)
  |delta| > 7.5°  ->  theta += ±15°, 위치 그대로   (action 2=좌 / 3=우)
  그 외           ->  위치 += 0.25 * [cos theta, sin theta]  (action 1)
목표에 0.15 m 안으로 들어오면 다음 웨이포인트로
```

**전진은 경로가 아니라 현재 heading 방향으로 간다.** 그래서 실제 궤적이 원 경로에서 조금씩
벗어나는데, 그게 시뮬레이터가 실제로 하는 일이고 릴리스 데이터도 그렇게 만들어졌다.

self-check: `/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/vlnce_discretize.py --self_test`
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCRIPT_NAME = 'vlnce_discretize'

# VLN-CE 규격 (원 논문 + 릴리스 데이터 실측이 일치)
FORWARD_DIST_M = 0.25
TURN_ANGLE_RAD = math.radians(15.0)
# `eval_gt_collect.py:58` 와 같은 값 — 목표에 이만큼 들어오면 다음 웨이포인트로
CLOSE_ENOUGH_M = FORWARD_DIST_M * 0.6
# `eval_gt_collect.py:73` 와 같은 판정 — 이보다 크게 틀어져 있으면 먼저 회전한다
TURN_TRIGGER_RAD = TURN_ANGLE_RAD * 0.5

ACTION_START = -1   # 프레임 0 센티널 (릴리스에 에피소드당 정확히 1회)
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# 한 웨이포인트에 쓸 수 있는 최대 스텝 — 제자리 맴돌이 방지
MAX_STEPS_PER_WP = 400


def wrap_pi(a):
    """각도를 (-pi, pi]로."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def thin_waypoints(xy: np.ndarray, spacing_m: float = 0.5) -> np.ndarray:
    """조밀한 경로를 웨이포인트로 솎는다. 첫 점과 끝 점은 항상 남긴다.

    0.14 m 간격 원본을 그대로 목표로 쓰면 매 스텝 목표가 이미 `CLOSE_ENOUGH`(0.15 m) 안이라
    전진이 한 번도 안 나온다. `FORWARD_DIST_M`보다 넉넉히 큰 간격으로 솎아야 한다.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) <= 2:
        return xy.copy()
    keep = [0]
    acc = 0.0
    for i in range(1, len(xy)):
        acc += float(np.linalg.norm(xy[i] - xy[i - 1]))
        if acc >= spacing_m:
            keep.append(i)
            acc = 0.0
    if keep[-1] != len(xy) - 1:
        keep.append(len(xy) - 1)
    return xy[keep]


def discretize(xy_path: np.ndarray, initial_yaw: float = None,
               waypoint_spacing_m: float = 0.5) -> dict:
    """연속 2D 경로 -> 프레임별 `(x, y, yaw, action)`.

    반환
    ----
    `{'xy': (N,2), 'yaw': (N,), 'action': (N,), 'n_forward', 'n_turn', 'end_gap_m'}`

    - 프레임 0은 시작 pose이고 `action[0] = -1`이다(릴리스 규약).
    - `action[i]`는 **프레임 i로 들어온 동작**이다 — 릴리스 parquet과 같은 의미.
      (학습 로더가 `actions[1:] + [0]`으로 한 칸 밀어 "프레임 i에서 취할 동작"으로 바꾼다.)
    - 높이·pitch는 여기서 다루지 않는다. rig마다 다르므로 렌더 단계에서 얹는다.
    """
    xy_path = np.asarray(xy_path, dtype=np.float64)[:, :2]
    assert len(xy_path) >= 2, f'{SCRIPT_NAME}: 경로가 너무 짧다 ({len(xy_path)}점)'

    wps = thin_waypoints(xy_path, waypoint_spacing_m)
    pos = xy_path[0].copy()
    if initial_yaw is None:
        d = wps[1] - wps[0] if len(wps) > 1 else xy_path[-1] - xy_path[0]
        initial_yaw = float(np.arctan2(d[1], d[0]))
    theta = float(initial_yaw)

    out_xy = [pos.copy()]
    out_yaw = [theta]
    out_act = [ACTION_START]

    for wp in wps[1:]:
        for _ in range(MAX_STEPS_PER_WP):
            if float(np.linalg.norm(wp - pos)) < CLOSE_ENOUGH_M:
                break
            dp = wp - pos
            delta = float(wrap_pi(math.atan2(dp[1], dp[0]) - theta))
            if abs(delta) > TURN_TRIGGER_RAD:
                # 제자리 회전 — 위치는 그대로
                sgn = 1.0 if delta > 0 else -1.0
                theta = float(wrap_pi(theta + sgn * TURN_ANGLE_RAD))
                out_act.append(ACTION_LEFT if sgn > 0 else ACTION_RIGHT)
            else:
                # 전진 — **경로가 아니라 현재 heading 방향으로** 간다
                pos = pos + FORWARD_DIST_M * np.array([math.cos(theta), math.sin(theta)])
                out_act.append(ACTION_FORWARD)
            out_xy.append(pos.copy())
            out_yaw.append(theta)

    act = np.asarray(out_act, dtype=np.int32)
    return {
        'xy': np.stack(out_xy),
        'yaw': np.asarray(out_yaw, dtype=np.float64),
        'action': act,
        'n_forward': int((act == ACTION_FORWARD).sum()),
        'n_turn': int(((act == ACTION_LEFT) | (act == ACTION_RIGHT)).sum()),
        'end_gap_m': float(np.linalg.norm(np.stack(out_xy)[-1] - xy_path[-1])),
    }


# ---------------------------------------------------------------------------
# pose 합성 — rig(높이·pitch)를 얹어 4x4로
# ---------------------------------------------------------------------------

def poses_from_yaw(xy: np.ndarray, yaw: np.ndarray, floor_z: float, h_b: float,
                   pitch_down_deg: float) -> np.ndarray:
    """`(x, y, yaw)` + rig -> `action[t]` 포맷 (N,4,4).

    `geometry_utils.synthesize_action_poses`와 **같은 회전 합성 공식**을 쓰되, yaw를 경로
    접선에서 유도하지 않고 **인자로 받는다** — 제자리 회전 프레임은 위치가 안 변해서
    접선을 구할 수 없기 때문이다(중앙차분이 0 벡터가 된다).

    공식은 그쪽 docstring에 실측 검증 기록이 있는 것을 그대로 옮겼다:
        forward_world(yaw, pitch) = Rz(yaw - 90°) @ mount_forward(pitch)
    """
    from geometry_utils import CAM_CV_TO_GL, action_to_c2w, compose_camera_extrinsic

    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    yaw = np.asarray(yaw, dtype=np.float64)
    n = len(xy)
    assert len(yaw) == n, f'{SCRIPT_NAME}: xy {n} != yaw {len(yaw)}'

    mount_cv_rot = action_to_c2w(compose_camera_extrinsic(h_b, pitch_down_deg), 'cam2world_gl')[:3, :3]
    delta = yaw - np.pi / 2.0
    c, s = np.cos(delta), np.sin(delta)
    rz = np.zeros((n, 3, 3), dtype=np.float64)
    rz[:, 0, 0], rz[:, 0, 1] = c, -s
    rz[:, 1, 0], rz[:, 1, 1] = s, c
    rz[:, 2, 2] = 1.0

    c2w_cv = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w_cv[:, :3, :3] = rz @ mount_cv_rot
    c2w_cv[:, :3, 3] = np.stack([xy[:, 0], xy[:, 1], np.full(n, floor_z + h_b)], axis=1)
    return c2w_cv @ CAM_CV_TO_GL


def to_start_relative(c2w: np.ndarray, floor_z: float = 0.0) -> np.ndarray:
    """절대 cam2world (N,4,4) -> **에피소드 시작 기준 상대** (릴리스 `pose.<rig>` 규약).

    릴리스 실측 (`17DRP5sb8fy` ep0, `125cm_30deg`):
    ```
    pose[0] translation = (0, 0, 1.25)          = (0, 0, rig 높이)
    pose[0] rotation    = [[0,-0.5,0.866],[-1,0,0],[0,-0.866,-0.5]]   = pitch 마운트
    ```

    즉 제거하는 것은 **시작 프레임의 yaw와 xy**뿐이다. pitch와 높이는 남는다.

    ⚠️ **전체 회전을 제거하면 안 된다.** 처음에 `t0[:3,:3] = c2w[0,:3,:3]`로 두었더니
    pitch까지 상쇄돼 `pose[0]` 회전이 단위행렬이 되고, translation이 회전을 먹어
    `(0, -1.04, -0.6)`이 나왔다(실측). 그러면 `goal_world_from_poses`가 바닥 높이를
    `z - rig높이`로 유도할 때 1.85 m 아래를 바닥으로 잡아 **픽셀 goal이 전부 화면 밖**이 된다
    (goal 있는 비율 0% — 그러면 `pixel_goal_only=True`에서 학습 샘플이 0개다).

    `floor_z`는 절대 좌표의 바닥 높이다. 릴리스는 상대 프레임에서 **바닥이 z=0**이므로
    (`pose[0].z == rig 높이`), 여기서도 바닥을 0으로 옮긴다.
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    # 시작 프레임의 heading — OpenCV c2w의 3열이 카메라 forward다. pitch가 섞여 있어도
    # xy 성분의 각도가 yaw다.
    fwd0 = c2w[0, :3, 2]
    yaw0 = float(np.arctan2(fwd0[1], fwd0[0]))
    c, s = np.cos(yaw0), np.sin(yaw0)
    t0 = np.eye(4)
    t0[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]   # yaw만
    t0[:3, 3] = [c2w[0, 0, 3], c2w[0, 1, 3], float(floor_z)]     # xy + 바닥
    return np.linalg.inv(t0) @ c2w


# ---------------------------------------------------------------------------
# 자기검사 — 릴리스 데이터로 규칙이 맞는지 확인 (GPU 불필요)
# ---------------------------------------------------------------------------

def release_episode_state(scene_dir: Path, episode: int, rig: str = '125cm_30deg') -> dict:
    """릴리스 vln_ce 에피소드의 `pose`에서 `(x, y, yaw)`와 `action`을 복원한다."""
    import pyarrow.parquet as pq

    pq_path = scene_dir / 'data' / f'chunk-{episode // 1000:03d}' / f'episode_{episode:06d}.parquet'
    t = pq.read_table(pq_path, columns=['action', f'pose.{rig}']).to_pydict()
    P = np.stack([np.asarray(p, dtype=np.float64).reshape(4, 4) for p in t[f'pose.{rig}']])
    xy = P[:, :2, 3]
    # 카메라 forward = c2w 회전의 3열(OpenCV +Z). 그 xy 성분이 heading.
    fwd = P[:, :3, 2]
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    return {'xy': xy, 'yaw': yaw, 'action': np.asarray(t['action'], dtype=np.int32), 'pose': P}


def self_test(ce_root: Path, scenes: list, max_eps: int) -> int:
    """두 가지를 확인한다.

    ① **릴리스 불변식** — 저장된 action과 pose가 우리가 쓰는 규격과 정말 맞는가.
       (전진 0.25 m/회전 0°, 회전 15°/이동 0, action[0]=-1)
    ② **우리 이산화기의 출력이 같은 불변식을 만족하는가**, 그리고 원 경로를 얼마나 잘 따라가는가.
    """
    fails = []

    # --- ① 릴리스 불변식 ---
    fwd_d, fwd_r, turn_d, turn_r, n_ep, start_ok = [], [], [], [], 0, 0
    for scene in scenes:
        sd = ce_root / scene
        if not (sd / 'data' / 'chunk-000').is_dir():
            continue
        for pqf in sorted((sd / 'data' / 'chunk-000').glob('*.parquet'))[:max_eps]:
            ep = int(pqf.stem.split('_')[1])
            st = release_episode_state(sd, ep)
            a, xy, yaw = st['action'], st['xy'], st['yaw']
            n_ep += 1
            if len(a) and a[0] == ACTION_START and (a[1:] == ACTION_START).sum() == 0:
                start_ok += 1
            for i in range(1, len(a)):
                d = float(np.linalg.norm(xy[i] - xy[i - 1]))
                r = abs(float(np.degrees(wrap_pi(yaw[i] - yaw[i - 1]))))
                if a[i] == ACTION_FORWARD:
                    fwd_d.append(d); fwd_r.append(r)
                elif a[i] in (ACTION_LEFT, ACTION_RIGHT):
                    turn_d.append(d); turn_r.append(r)

    if not n_ep:
        print(f'[{SCRIPT_NAME}] 릴리스 에피소드를 못 찾았다: {ce_root}', file=sys.stderr)
        return 2
    fwd_d, fwd_r, turn_d, turn_r = map(np.asarray, (fwd_d, fwd_r, turn_d, turn_r))
    print(f'[{SCRIPT_NAME}] ① 릴리스 불변식 — 에피소드 {n_ep}개')
    print(f'    전진 {len(fwd_d):5d}건  이동 {np.median(fwd_d):.4f} m (max {fwd_d.max():.4f}) · '
          f'회전 {np.median(fwd_r):.2f}° (max {fwd_r.max():.2f}°)')
    print(f'    회전 {len(turn_d):5d}건  이동 {np.median(turn_d):.4f} m (max {turn_d.max():.4f}) · '
          f'회전 {np.median(turn_r):.2f}° (max {turn_r.max():.2f}°)')
    print(f'    action[0]=-1 이고 그 뒤 없음: {start_ok}/{n_ep}')
    if abs(np.median(fwd_d) - FORWARD_DIST_M) > 1e-3:
        fails.append(f'릴리스 전진 거리 {np.median(fwd_d):.4f} != {FORWARD_DIST_M}')
    if turn_d.max() > 1e-6:
        fails.append(f'릴리스 회전 중 이동 {turn_d.max():.6f} m (0이어야)')
    if abs(np.median(turn_r) - 15.0) > 1e-2:
        fails.append(f'릴리스 회전각 {np.median(turn_r):.3f}° != 15°')
    if start_ok != n_ep:
        fails.append(f'action[0]=-1 규약 위반 {n_ep - start_ok}건')

    # --- ② 우리 이산화기 ---
    print(f'\n[{SCRIPT_NAME}] ② 우리 이산화기 — 릴리스 궤적을 입력 경로로 넣어 재이산화')
    dev, ratio, ends = [], [], []
    for scene in scenes:
        sd = ce_root / scene
        if not (sd / 'data' / 'chunk-000').is_dir():
            continue
        for pqf in sorted((sd / 'data' / 'chunk-000').glob('*.parquet'))[:max_eps]:
            ep = int(pqf.stem.split('_')[1])
            st = release_episode_state(sd, ep)
            path = st['xy']
            if len(path) < 3:
                continue
            r = discretize(path, initial_yaw=float(st['yaw'][0]))
            a, xy, yaw = r['action'], r['xy'], r['yaw']
            # 불변식 재확인
            for i in range(1, len(a)):
                d = float(np.linalg.norm(xy[i] - xy[i - 1]))
                rot = abs(float(np.degrees(wrap_pi(yaw[i] - yaw[i - 1]))))
                if a[i] == ACTION_FORWARD and (abs(d - FORWARD_DIST_M) > 1e-9 or rot > 1e-9):
                    fails.append(f'우리 전진 프레임 이상: d={d:.6f} rot={rot:.4f}')
                    break
                if a[i] in (ACTION_LEFT, ACTION_RIGHT) and (d > 1e-9 or abs(rot - 15.0) > 1e-9):
                    fails.append(f'우리 회전 프레임 이상: d={d:.6f} rot={rot:.4f}')
                    break
            if a[0] != ACTION_START:
                fails.append('우리 action[0] != -1')
            # 원 경로를 얼마나 따라갔나 (각 원 경로 점에서 우리 궤적까지 최근접 거리)
            dmin = np.linalg.norm(path[:, None, :] - xy[None, :, :], axis=2).min(axis=1)
            dev.append(float(np.median(dmin)))
            ratio.append(len(a) / len(path))
            ends.append(r['end_gap_m'])
    dev, ratio, ends = map(np.asarray, (dev, ratio, ends))
    print(f'    에피소드 {len(dev)}개')
    print(f'    원 경로 이탈   중앙 {np.median(dev):.4f} m · p90 {np.percentile(dev, 90):.4f} m')
    print(f'    프레임 수 비율 중앙 {np.median(ratio):.3f} (1.0이면 릴리스와 같은 길이)')
    print(f'    종점 오차     중앙 {np.median(ends):.4f} m · max {ends.max():.4f} m')
    if np.median(dev) > 0.15:
        fails.append(f'원 경로 이탈이 크다: 중앙 {np.median(dev):.4f} m')

    if fails:
        print(f'\n[{SCRIPT_NAME}] 자기검사 FAIL {len(fails)}건')
        for f in fails[:10]:
            print(f'  - {f}')
        return 1
    print(f'\n[{SCRIPT_NAME}] 자기검사 전부 PASS')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self_test', action='store_true')
    ap.add_argument('--ce_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--scenes', default='17DRP5sb8fy,1LXtFkjw3qL,29hnd4uzFmX')
    ap.add_argument('--max_eps', type=int, default=10, help='씬당 검사할 에피소드 수')
    args = ap.parse_args()
    if not args.self_test:
        ap.print_help()
        return 0
    return self_test(Path(args.ce_root), [s for s in args.scenes.split(',') if s], args.max_eps)


if __name__ == '__main__':
    sys.exit(main())
