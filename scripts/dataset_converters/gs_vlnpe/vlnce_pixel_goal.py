"""프레임별 픽셀 목표 `goal.<rig>` · `relative_goal_frame_id.<rig>` 생성.

왜 필요한가
-----------
InternVLA-N1 기본 학습 설정이 `pixel_goal_only=True`라서, `goal`이 전부 `[-1,-1]`이면
**학습 샘플이 0개**가 된다 (`internvla_n1_lerobot_dataset.py:917, 977`). 반드시 만들어야 한다.

정의 (릴리스 데이터로 역산해 확인)
----------------------------------
> frame `i`의 pixel goal = **frame `i+goal_len+1`의 카메라 위치 아래 바닥점**을
> frame `i`의 **룩다운(pitch_2) 카메라**에 투영한 `[u, v]` (640×480 정수).

`3dloader_vlnce/pixel_goal_utils.py`가 이 투영을 이미 구현해 뒀고, 릴리스 610 프레임에
역산해 확인했다 — **u 오차 중앙 1.28 px · v 1.04 px · 5px 초과 0.00%**.

`goal_len` 선택 규칙 (이 파일에서 역산)
--------------------------------------
기존 코드에는 **투영·가시성만** 있고 "`goal_len`을 어떻게 고르나"는 없었다
(`07_validate_pixel_goal.py`는 릴리스의 값을 읽어 검증만 한다). 그래서 릴리스에서 역산했다.

저장된 goal 구간의 **전진(action=1) 스텝 수**를 세어 보니 **정확히 13에서 끊긴다**:

```
{3:2, 4:8, 5:25, 6:33, 7:36, 8:37, 9:43, 10:36, 11:42, 12:57, 13:291}
                                                              ↑ 하드 상한, 최빈
```

`13 × 0.25 m = 3.25 m`이고, 저장 goal까지의 수평거리 최대값도 정확히 **3.25 m**였다.
13 미만인 것들은 가림·화면밖으로 잘린 경우다.

그래서 규칙은 **"전진 13스텝 이내에서, 보이는 가장 큰 `goal_len`"**이다. 재현율:

| | |
|---|---|
| 정확 일치 | **62.9 %** (278/442) |
| ±1 이내 | **81.7 %** |
| `-1`(goal 없음) 예측 일치 | 41.1 % |

**완전 재현은 아니다.** 오차는 한쪽으로 치우쳐 있다(우리가 더 큰 `goal_len`을 고른다, p95 = +4) —
원본이 우리 `check_visible` 기본값(`depth_tol_m=0.35`, `min_forward_m=0.3`)보다 엄격한 기준을
썼다는 뜻이다. 원본 생성 코드가 없어 그 이상은 좁히지 못했다.

**우리 목적에는 이걸로 충분하다.** 노은역은 새 씬이라 릴리스를 비트 단위로 재현할 필요가 없고,
필요한 것은 "정의를 만족하는 유효한 goal"이다 — 실제로 보이는, 3.25 m 이내 앞쪽 바닥점.
다만 분포가 릴리스와 다를 수 있다는 것은 provenance에 남긴다.

self-check: `/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/vlnce_pixel_goal.py --validate_against_release`
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_3D = REPO / 'scripts' / 'dataset_converters' / '3dloader_vlnce'
for p in (str(HERE), str(_3D)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pixel_goal_utils as pg  # noqa: E402

SCRIPT_NAME = 'vlnce_pixel_goal'

# 릴리스 역산으로 확정한 상한 — 전진 13스텝(= 13 × 0.25 m = 3.25 m)
MAX_FORWARD_STEPS = 13
ACTION_FORWARD = 1
NO_GOAL = -1
GOAL_NONE = (NO_GOAL, NO_GOAL)
# 로더가 버리는 하한 (`internvla_n1_lerobot_dataset.py:943`)
MIN_USEFUL_GOAL_LEN = 3
# 탐색 상한 — 회전이 많이 섞여도 13 전진을 넘기려면 이 정도면 충분하다(릴리스 최대 goal_len 26)
SEARCH_LIMIT = 60


def select_goal(poses: np.ndarray, actions: np.ndarray, i: int, rig: str,
                depth_img: np.ndarray = None,
                max_forward_steps: int = MAX_FORWARD_STEPS) -> tuple:
    """프레임 `i`의 `(goal_len, u, v)`. 목표가 없으면 `(-1, -1, -1)`.

    `poses`는 **에피소드 시작 기준 상대** cam2world (N,4,4) — 릴리스 `pose.<rig>`와 같은 규약.
    `actions`는 프레임별 이산 action (N,), `action[k]`는 프레임 `k`로 들어온 동작.
    `depth_img`는 프레임 `i`의 깊이[m] (가림 검사용, 없으면 기하 검사만).
    """
    n = len(poses)
    best_len, best_uv = NO_GOAL, GOAL_NONE
    for L in range(1, SEARCH_LIMIT + 1):
        j = i + L + 1
        if j >= n:
            break
        # 이 구간에서 실제로 전진한 횟수 — 제자리 회전은 거리를 안 늘리므로 세지 않는다
        if int(np.count_nonzero(actions[i + 1:j + 1] == ACTION_FORWARD)) > max_forward_steps:
            break
        G = pg.goal_world_from_poses(poses[i:j + 1], rig=rig)
        u, v, Z = pg.project_to_pixel(G, poses[i], rig=rig)
        ok, _ = pg.check_visible(u, v, Z, depth_img)
        if ok:
            best_len, best_uv = L, (int(round(u)), int(round(v)))
    return best_len, best_uv[0], best_uv[1]


def build_episode_goals(poses: np.ndarray, actions: np.ndarray, rig: str,
                        depth_imgs=None, max_forward_steps: int = MAX_FORWARD_STEPS) -> dict:
    """에피소드 전체의 `goal`·`relative_goal_frame_id`.

    `depth_imgs`는 프레임별 깊이[m] 목록이거나, 프레임 번호를 받아 깊이를 돌려주는 callable.
    `None`이면 가림 검사를 건너뛴다(goal이 벽 너머로 잡힐 수 있다 — 권장하지 않는다).
    """
    n = len(poses)
    goals = np.full((n, 2), NO_GOAL, dtype=np.int32)
    rels = np.full(n, NO_GOAL, dtype=np.int32)
    for i in range(n):
        d = None
        if callable(depth_imgs):
            d = depth_imgs(i)
        elif depth_imgs is not None:
            d = depth_imgs[i]
        L, u, v = select_goal(poses, actions, i, rig, d, max_forward_steps)
        rels[i] = L
        goals[i] = (u, v)
    n_goal = int((rels >= 0).sum())
    return {
        'goal': goals, 'relative_goal_frame_id': rels,
        'n_with_goal': n_goal, 'frac_with_goal': n_goal / max(n, 1),
        'n_useful': int((rels >= MIN_USEFUL_GOAL_LEN).sum()),
    }


# ---------------------------------------------------------------------------
# 릴리스 대조 — 규칙이 맞는지 확인
# ---------------------------------------------------------------------------

def load_release_episode(scene_dir: Path, episode: int, rig: str) -> dict:
    import pyarrow.parquet as pq

    pqf = scene_dir / 'data' / f'chunk-{episode // 1000:03d}' / f'episode_{episode:06d}.parquet'
    t = pq.read_table(pqf, columns=['action', f'pose.{rig}', f'goal.{rig}',
                                    f'relative_goal_frame_id.{rig}']).to_pydict()
    return {
        'action': np.asarray(t['action'], dtype=np.int32),
        'pose': np.stack([np.asarray(p, dtype=np.float64).reshape(4, 4) for p in t[f'pose.{rig}']]),
        'goal': np.asarray(t[f'goal.{rig}'], dtype=np.int64),
        'rel': np.asarray(t[f'relative_goal_frame_id.{rig}'], dtype=np.int32),
    }


def depth_loader(scene_dir: Path, episode: int, rig: str):
    from PIL import Image

    d = scene_dir / 'videos' / 'chunk-000' / f'observation.images.depth.{rig}'

    def get(i):
        p = d / f'episode_{episode:06d}_{i}.png'
        if not p.is_file():
            return None
        return np.asarray(Image.open(p), dtype=np.float32) / 1000.0   # uint16 mm -> m

    return get


def validate(ce_root: Path, scenes: list, rig: str, max_eps: int) -> int:
    """① 정의 재현(저장 rel로 투영했을 때 저장 goal과 같은가) ② 규칙 재현율."""
    du, dv = [], []
    exact = tot = nm_hit = nm_tot = 0
    diffs = []
    fwd_counts = []

    for scene in scenes:
        sd = ce_root / scene
        if not (sd / 'data' / 'chunk-000').is_dir():
            print(f'[{SCRIPT_NAME}] 씬 없음: {scene}', file=sys.stderr)
            continue
        for f in sorted(glob.glob(str(sd / 'data' / 'chunk-000' / '*.parquet')))[:max_eps]:
            ep = int(Path(f).stem.split('_')[1])
            E = load_release_episode(sd, ep, rig)
            P, A, G, R = E['pose'], E['action'], E['goal'], E['rel']
            n = len(P)
            getd = depth_loader(sd, ep, rig)

            for i in range(n):
                # --- ① 정의 재현 ---
                if R[i] >= 0 and i + R[i] + 1 < n:
                    Gw = pg.goal_world_from_poses(P[i:i + R[i] + 2], rig=rig)
                    u, v, _ = pg.project_to_pixel(Gw, P[i], rig=rig)
                    if np.isfinite(u) and np.isfinite(v):
                        du.append(abs(u - G[i][0])); dv.append(abs(v - G[i][1]))
                    fwd_counts.append(int(np.count_nonzero(A[i + 1:i + R[i] + 2] == ACTION_FORWARD)))
                # --- ② 규칙 재현 ---
                d = getd(i)
                if d is None:
                    continue
                L, _, _ = select_goal(P, A, i, rig, d)
                if R[i] >= 0:
                    tot += 1
                    if L == R[i]:
                        exact += 1
                    diffs.append(L - int(R[i]))
                else:
                    nm_tot += 1
                    if L == NO_GOAL:
                        nm_hit += 1

    if not du and not tot:
        print(f'[{SCRIPT_NAME}] 검사할 데이터가 없다', file=sys.stderr)
        return 2

    fails = []
    du, dv = np.asarray(du), np.asarray(dv)
    print(f'[{SCRIPT_NAME}] ① 정의 재현 — {len(du)} 프레임')
    print(f'    u 오차 중앙 {np.median(du):.2f} px · p99 {np.percentile(du, 99):.2f} · max {du.max():.1f}')
    print(f'    v 오차 중앙 {np.median(dv):.2f} px · p99 {np.percentile(dv, 99):.2f} · max {dv.max():.1f}')
    print(f'    5 px 초과 비율 {((du > 5) | (dv > 5)).mean() * 100:.2f} %')
    if np.median(du) > 3 or np.median(dv) > 3:
        fails.append(f'정의 재현 오차가 크다: u {np.median(du):.2f} v {np.median(dv):.2f} px')

    fc = np.asarray(fwd_counts)
    print(f'\n[{SCRIPT_NAME}] 전진 스텝 상한 — 저장 goal 구간의 전진 횟수')
    print(f'    중앙 {np.median(fc):.0f} · max {fc.max()} (상한 가정 {MAX_FORWARD_STEPS})')
    if fc.max() > MAX_FORWARD_STEPS:
        fails.append(f'전진 스텝이 상한을 넘는다: max {fc.max()} > {MAX_FORWARD_STEPS}')

    d = np.asarray(diffs)
    print(f'\n[{SCRIPT_NAME}] ② 규칙 재현율 — goal 있는 프레임 {tot}')
    print(f'    정확 일치 {exact} ({exact / max(tot, 1) * 100:.1f} %) · '
          f'±1 이내 {(np.abs(d) <= 1).mean() * 100:.1f} %')
    print(f'    예측-저장 오차 p[5,50,95] = {np.percentile(d, [5, 50, 95]).astype(int).tolist()}')
    print(f'    goal 없는(-1) {nm_tot} 중 예측도 -1: {nm_hit} ({nm_hit / max(nm_tot, 1) * 100:.1f} %)')

    if fails:
        print(f'\n[{SCRIPT_NAME}] FAIL {len(fails)}건')
        for x in fails:
            print(f'  - {x}')
        return 1
    print(f'\n[{SCRIPT_NAME}] 정의·상한 확인 PASS '
          f'(규칙 정확 일치 {exact / max(tot, 1) * 100:.1f}%는 참고값 — docstring의 한계 참고)')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate_against_release', action='store_true')
    ap.add_argument('--ce_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--scenes', default='17DRP5sb8fy,1LXtFkjw3qL')
    ap.add_argument('--rig', default='125cm_30deg')
    ap.add_argument('--max_eps', type=int, default=6)
    args = ap.parse_args()
    if not args.validate_against_release:
        ap.print_help()
        return 0
    return validate(Path(args.ce_root), [s for s in args.scenes.split(',') if s], args.rig, args.max_eps)


if __name__ == '__main__':
    sys.exit(main())
