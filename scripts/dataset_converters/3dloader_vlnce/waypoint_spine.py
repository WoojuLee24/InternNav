"""사람 주석 waypoint(spine) 추출 — `reference_path`를 mesh 좌표로 놓고 GT 프레임에 앵커한다.

## 출처 (왜 `reference_path`가 사람 주석인가)
- **R2R** (Anderson et al., CVPR 2018, arXiv:1711.07280): 노드 = Matterport3D 파노라마 뷰포인트,
  엣지는 씬 mesh ray-trace로 장애물 검사 후 5 m 초과 제거 + 수동 검증. 경로는 region 주석으로
  **다른 방**의 start/goal을 뽑고 "5 m 미만 / 엣지 4개 미만·6개 초과" 경로를 버려 7,189개. 경로마다
  **AMT 3명**이 3D fly-through를 보고 지시문 작성 → 21,567개.
- **VLN-CE** (Krantz et al., ECCV 2020, arXiv:2004.02857): 파노라마 노드를 *"a ground-based agent
  represented by a 1.5m tall cylinder of diameter of 0.2m"* 가 점유 가능한 최근접 점으로 투영해
  **"waypoint locations"** 를 만들고(98.3% 성공), 궤적 검증은 *"We run this algorithm between each
  waypoint in a trajectory to the next … navigable if … the shortest path to **within 0.5 m** of the
  next waypoint"* → 77%만 통과. **우리가 하려는 leg별 계획이 데이터셋 생성 절차 그 자체다.**
  저장된 것은 *"a pre-computed shortest path following the waypoints via low-level actions"*.

즉 `r_b = 0.1`, `h = 1.5`, `leg_tol = 0.5 m`는 우리가 고른 값이 아니라 **데이터셋의 정의값**이다.

## 실측으로 확정한 사실 (11,597 raw / 159 traj 에피소드)
- `reference_path` = **4~7개** waypoint, 간격 median **1.79 m**. `start_position == reference_path[0]`,
  `goals[0].position == reference_path[-1]` (100%).
- 좌표: habitat Y-up → mesh Z-up은 `(x, −z, y)` — `vlnce_align.hmap_translation`과 **같은 매핑**이고
  `internnav/env/utils/episode_loader/dataset_utils.py:597`도 같은 변환을 쓴다.
- **`reference_path` y는 바닥 높이**다 — `cam_z − rp_z = 1.2493 m`(= rig 높이). 그래서 카메라 점은
  `rp_z + rig_height`. 이것이 `vlnce_align.Z_OFFSET_M = 0.0`이 맞다는 독립 근거다.
- waypoint→최근접 GT 카메라 프레임 XY 거리 **mean 0.100 m / max 0.388 m**, 프레임 순서 **159/159 단조**,
  마지막 waypoint의 최근접 프레임이 마지막 프레임 **159/159** → 프레임 타임라인 앵커가 안전하다.
- ⚠️ **waypoint 직선을 따라가면 안 된다**: 실제 GT와 직선 폴리라인 거리의 max가 median 0.447 m,
  30/159 에피소드가 1.0 m 초과(최악 4.36 m). 그래서 leg별로 **GT 서브궤적**을 회랑의 중심으로 쓴다.

self-check: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/waypoint_spine.py
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))

# VLN-CE가 궤적 navigability를 판정한 기준 (논문 §3.1). 우리도 같은 값을 쓴다.
LEG_TOL_M = 0.5
# waypoint를 GT 프레임에 앵커할 때 허용 오차. 실측 max 0.388 m라 넉넉하다.
ANCHOR_TOL_M = 0.60


def habitat_to_mesh(p_yup):
    """habitat Y-up 좌표 -> mesh Z-up. `(x, −z, y)`.

    `vlnce_align.hmap_translation`(Z_OFFSET_M=0.0)과 동일한 매핑이다. 여기서 따로 구현하지 않고
    같은 식을 쓰는 이유는 `reference_path`는 **배열 전체**를 변환해야 해서다(그 함수는 단일 점용).
    """
    p = np.asarray(p_yup, dtype=np.float64).reshape(-1, 3)
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def anchor_to_frames(waypoints_mesh, gt_xy):
    """각 waypoint를 **가장 가까운 GT 카메라 프레임**에 앵커. -> (frame_idx (T,), dist_m (T,))

    `gt_xy`는 mesh 좌표의 카메라 궤적 (N,2). waypoint는 바닥점이지만 xy만 쓰므로 높이는 무관하다.
    순서 단조성은 호출부가 `spine_ok`로 검사한다(실측 159/159 단조지만 보장은 아니다).
    """
    w = np.asarray(waypoints_mesh, dtype=np.float64)[:, :2]
    g = np.asarray(gt_xy, dtype=np.float64)[:, :2]
    d = np.linalg.norm(g[None, :, :] - w[:, None, :], axis=2)      # (T, N)
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(w)), idx]


def spine_ok(anchor_frame, anchor_dist, n_frames, tol_m=ANCHOR_TOL_M):
    """spine을 쓸 수 있는지. -> (bool, 이유 문자열)

    세 조건 모두 실측으로 100% 만족했으므로, 깨지면 그 에피소드는 조용히 쓰지 말고 걸러야 한다.
    """
    if len(anchor_frame) < 2:
        return False, f'waypoint {len(anchor_frame)}개'
    if np.any(np.diff(anchor_frame) <= 0):
        return False, f'앵커 프레임이 단조가 아님 {anchor_frame.tolist()}'
    if anchor_dist.max() > tol_m:
        return False, f'앵커 거리 {anchor_dist.max():.3f} m > {tol_m}'
    if anchor_frame[-1] != n_frames - 1:
        return False, f'마지막 waypoint가 마지막 프레임이 아님 ({anchor_frame[-1]} != {n_frames - 1})'
    return True, 'ok'


def build_spine(waypoints_yup, gt_xyz_mesh, tol_m=ANCHOR_TOL_M):
    """`reference_path`(habitat) + GT 카메라 궤적(mesh) -> spine dict.

    반환 dict:
      `waypoints_mesh` (T,3) 바닥점 · `anchor_frame` (T,) · `anchor_dist_m` (T,) ·
      `ok` bool · `reason` str · `legs` [(frame_lo, frame_hi)] — leg `i`가 덮는 **프레임 구간**

    `legs`가 핵심이다. leg `i` = waypoint `i` → `i+1` 구간이고, 그 구간의 **GT 서브궤적**
    `gt[frame_lo : frame_hi+1]`이 회랑의 중심선이 된다(직선이 아니라 실제 궤적).
    """
    w = habitat_to_mesh(waypoints_yup)
    gt = np.asarray(gt_xyz_mesh, dtype=np.float64)
    af, ad = anchor_to_frames(w, gt[:, :2])
    ok, reason = spine_ok(af, ad, len(gt), tol_m)
    legs = [(int(af[i]), int(af[i + 1])) for i in range(len(af) - 1)] if ok else []
    return dict(waypoints_mesh=w, anchor_frame=af, anchor_dist_m=ad, ok=ok, reason=reason, legs=legs)


def frame_flags(legs, leg_status, n_frames):
    """leg별 status -> **프레임별 플래그** (0 ok / 1 detour / 2 blocked). -> (n_frames,) uint8

    학습 샘플은 `poses[start_frame_id : start_frame_id+goal_len+1]`(≤3.25 m 국소 창)이므로,
    로더는 이 배열만 보고 창을 O(1)로 판정한다:
      - 제외 모드      : 창에 2가 있으면 샘플 기각
      - 도달불가 학습  : 창에서 2가 처음 나오는 프레임을 stop 라벨로
    blocked leg **이후 전부**를 blocked로 칠한다 — 그 지점을 못 지나가면 뒤쪽도 도달 불가다.
    """
    flags = np.zeros(int(n_frames), dtype=np.uint8)
    rank = {'ok': 0, 'detour': 1, 'blocked': 2}
    for (lo, hi), st in zip(legs, leg_status):
        v = rank[st]
        if v == 2:
            flags[lo:] = 2
            break
        flags[lo:hi + 1] = np.maximum(flags[lo:hi + 1], v)
    return flags


# ---------------------------------------------------------------------------
# self-check — 합성 데이터로 앵커·플래그 규약을 확인한다 (씬 불필요).
# ---------------------------------------------------------------------------
def _selfcheck():
    # habitat (x, y, z) -> mesh (x, -z, y)
    got = habitat_to_mesh([[1.0, 2.0, 3.0]])[0]
    assert np.allclose(got, [1.0, -3.0, 2.0]), f'좌표 매핑 틀림 {got}'

    # GT: x=0..4 직선, 카메라 높이 1.25. waypoint는 바닥(z=0)에 x=0,2,4
    n = 21
    gt = np.stack([np.linspace(0, 4, n), np.zeros(n), np.full(n, 1.25)], axis=1)
    wp_yup = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]   # (x, y=floor, z=0) -> mesh (x, 0, 0)
    sp = build_spine(wp_yup, gt)
    assert sp['ok'], f"spine 실패: {sp['reason']}"
    assert sp['anchor_frame'].tolist() == [0, 10, 20], f"앵커 {sp['anchor_frame'].tolist()}"
    assert sp['anchor_dist_m'].max() < 1e-9, '앵커 거리가 0이 아니다'
    assert sp['legs'] == [(0, 10), (10, 20)], f"legs {sp['legs']}"

    # 앵커가 단조가 아니면 거부 (waypoint 순서가 뒤바뀐 경우)
    bad = build_spine([[0.0, 0, 0], [4.0, 0, 0], [2.0, 0, 0]], gt)
    assert not bad['ok'] and '단조' in bad['reason'], f"단조 검사 실패: {bad['reason']}"
    # 마지막 waypoint가 끝 프레임이 아니면 거부
    short = build_spine([[0.0, 0, 0], [2.0, 0, 0]], gt)
    assert not short['ok'] and '마지막' in short['reason'], f"끝점 검사 실패: {short['reason']}"
    # 앵커가 멀면 거부
    far = build_spine([[0.0, 0, 0], [2.0, 0, 9.0], [4.0, 0, 0]], gt)
    assert not far['ok'] and '앵커 거리' in far['reason'], f"거리 검사 실패: {far['reason']}"

    # 프레임 플래그
    f = frame_flags([(0, 10), (10, 20)], ['ok', 'ok'], n)
    assert f.max() == 0, '전부 ok인데 플래그가 섰다'
    f = frame_flags([(0, 10), (10, 20)], ['ok', 'detour'], n)
    assert f[5] == 0 and f[15] == 1, f'detour 플래그 위치 틀림 {f.tolist()}'
    f = frame_flags([(0, 10), (10, 20)], ['detour', 'blocked'], n)
    assert f[5] == 1, 'blocked 앞 leg의 detour가 지워졌다'
    assert (f[10:] == 2).all(), 'blocked 이후가 전부 2가 아니다'
    f = frame_flags([(0, 10), (10, 20)], ['blocked', 'ok'], n)
    assert (f == 2).all(), '첫 leg가 막히면 전 구간이 도달 불가여야 한다'

    print('[selfcheck] waypoint_spine 10/10 통과 '
          '(좌표·앵커·legs·단조·끝점·거리·플래그 ok/detour/blocked/전구간)')


if __name__ == '__main__':
    _selfcheck()
