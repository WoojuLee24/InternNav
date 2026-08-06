"""M1.3 — GT path를 재현해 경로 생성 알고리즘을 검증한다.

    00: vln_n1 데이터 -> "어떻게 읽고 쓰나"  -> target_schema.json      (전체에 1개)
    01: 씬 mesh/USD    -> "이 씬 써도 되나"    -> scene_meta/<scene>.json (씬마다 1개)
    02: 씬 mesh        -> "어디로 갈 수 있나"  -> esdf/<scene>.npz        (씬마다 1개)
    03: occupancy      -> "어떤 경로로 가나"   -> paths/<scene>.json      (씬마다 1개)

계산은 전부 `esdf_utils.py`가 한다. 이 스크립트는 파일 입출력 + 검증 + 리포트만 담당한다.

## 두 모드

  - `reproduce` (**기본**) — 에피소드별 `h_b`·pitch·start·goal을 **전부 GT에서 복사**해 우리
    A*+refinement+spline이 정답지와 같은 경로를 내는지 본다. 분포 비교보다 날카로운 검증이다.
  - `random` — 논문처럼 무작위 생성 (C2에서 구현).

## 처리 단계별 데이터 (에피소드 1개당)

```
esdf/<scene>.npz (02)  +  GT parquet
  | camera_extrinsic 분해 / action[0], action[-1]
h_b, pitch, floor_z, start(x,y), goal(x,y)
  | derive_obstacle_2d(occupancy, floor_z, h_obs=h_b)     밴드 (floor+h_nav, floor+h_b]
(2) obstacle_2d  (Ny,Nx) bool
  | compute_esdf_2d
(3) esdf         (Ny,Nx) float32 [m]
  | truncate_navigable(esdf, r_b)                          논문 3단계
(4) navigable    (Ny,Nx) bool      0.05 m
  | downsample_navigable(factor 4)                         논문 4단계 (낙관적 any)
(5) nav_coarse   (Ny/4,Nx/4) bool  0.2 m
  | astar                                                  논문 5단계
(6) waypoints_astar   (N,2) 0.2 m 격자 -> world xy
  | greedy_refine(esdf 0.05 m, radius)                     논문 6단계
(7) waypoints_refined (N,2) world xy   <- r_b 제약은 여기에 걸린다
  | smooth_cubic_spline 또는 smooth_bezier                  논문 7단계
(8) trajectory        (M,2) world xy
```

## `refine_radius`를 0.10으로 정한 근거

논문은 refinement 범위를 "a local area"라고만 한다. 처음엔 chamfer 스윕으로 0.15를 골랐는데,
그건 지표를 보고 지표에 맞춘 순환논증이었다. `03b_verify_reproduction.py`가 **GT에서 역추정**하니
**R\* = 0.05**가 나왔고(두 씬 일치, 알려진 반경 복원으로 자기검증됨), 그 방향으로 다시 재니:

    radius  씬1 chamfer/Frechet/clr차     씬2 chamfer/Frechet/같은루트   판정
    0.05    0.127  0.271  -0.034          0.157  0.363  19/20          FAIL (r_b 위반 6/20)
    0.10    0.107  0.233  -0.014          0.154  0.366  19/20          <- 채택
    0.15    0.102  0.252  +0.020          0.177  0.383  19/20          (이전 기본값)
    0.20    0.097  0.250  +0.037          0.180  0.402  18/20

0.15 -> 0.10에서 **Fréchet가 두 씬 모두 개선**되고(0.252->0.233, 0.383->0.366) 씬2 chamfer가 13%
줄었다. 씬1 chamfer만 0.102->0.107로 소폭 악화한다 — GT의 회전 구간 촘촘한 점을 덜 맞추기 때문이다.
**clearance 차이(지표 ②)가 두 씬 모두 0에 가까워지는 것**이 결정 근거다: +0.020/+0.077 -> -0.014/+0.035.
0.15는 GT보다 벽에서 멀리 밀어내고 있었다.

**R\* = 0.05를 그대로 쓰지 못하는 이유**: 낙관적 0.2 m 다운샘플(`any()`)이 통과시킨 셀은 하위
0.05 m 셀에서 `r_b` 미만일 수 있는데, 0.05 m 창으로는 되끌어올리지 못해 6/20이 `r_b`를 위반한다.
**논문의 `r_b` 제약 자체가 refine 반경의 하한(~0.10)을 만든다.**

radius를 키우면 waypoint가 국소 ESDF 최대점(방 중앙)으로 끌려가 경로가 길어진다.
radius 0.0은 refinement가 없어 "refine이 clearance를 올렸는가" 게이트에서 걸린다.

## 검증

`r_b` 제약은 **refined waypoint**(7)에 걸고, 스무딩된 궤적(8)은 측정만 한다 — GT 궤적도 13%(7/52)가
min clearance < `r_b`이기 때문이다(cubic spline 오버슈트로 보인다). 하드 제약을 (8)에 걸면 GT조차 탈락한다.

재현 정확도는 **chamfer**(순서 무시)와 **discrete Fréchet**(순서 고려)로 잰다. 임계값을 미리 박지 않고
분포를 먼저 본다 — A* tie-breaking·refinement 스텝·spline 파라미터가 논문에 명시돼 있지 않다.

실행 예시:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03_sample_gt_paths.py \\
        --scene 17DRP5sb8fy --num_episodes 20

    ... --smooth bezier        # 오버슈트 없는 스무딩과 비교
    ... --r_b 1.0              # negative test: 통과 불가 -> A* 전부 실패
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esdf_utils import (  # noqa: E402
    ASTAR_CELL_M,
    DOWNSAMPLE_FACTOR,
    DOWNSAMPLE_MODES,
    ROBOT_RADIUS_M,
    WAYPOINT_SPACING_M,
    astar,
    cell_to_world,
    astar_cost_gap,
    chamfer_distance,
    check_path_navigable,
    compare_clearance_profile,
    compute_esdf_2d,
    compute_scan_coverage_mask,
    derive_obstacle_2d,
    discrete_frechet_distance,
    downsample_navigable,
    greedy_refine,
    heading_alignment,
    homotopy_obstacle_area,
    REFINERS,
    SMOOTHERS,
    metric_floors,
    pipeline_floor_path,
    path_smoothness,
    sample_esdf_at,
    thin_waypoints,
    truncate_navigable,
    world_to_cell,
)
from geometry_utils import action_to_c2w, decompose_camera_extrinsic, save_jpg  # noqa: E402
from viz_utils import (  # noqa: E402
    FLOOR_COLOR, GT_COLOR, OURS_COLOR, blink_widget_html, floorplan_canvas,
    reference_button_html, save_gallery,
)

DEFAULT_DATA_ROOT = 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i'
DEFAULT_OUT_DIR = 'scripts/dataset_converters/gs_vlnpe'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe'
DEFAULT_SCENE = '17DRP5sb8fy'
SCRIPT_NAME = '03_sample_gt_paths'

FLOORPLAN_MIN_PX = 700
DOOR_WIDTH_M = 0.8            # "같은 루트인가" 판정의 참고 스케일 (Fréchet가 이보다 작으면 같은 루트)
# 지표① 게이트는 **양쪽**이다. 상한은 A* 격자 꺾임이 spline에 새는 것을, 하한은 스무딩이 GT의 실제
# 회전까지 씻어내는 것을 잡는다. bezier가 0.67 / 0.86을 내면서 하한이 필요하다는 게 드러났다
# (convex hull 안에 머물며 코너를 자르므로 GT보다 매끈해진다). 실측: cubic 1.01/1.30, bezier 0.67/0.86.
TURN_RATIO_MIN = 0.5
TURN_RATIO_MAX = 2.0
# 기준선 ⓒ 대비 이 배수 이내면 "남은 차이가 거의 전부 이산화"로 본다 (실측 1.34 / 1.41배)
FLOOR_RATIO_AT_LIMIT = 1.5

# --- C2(--mode random) 전용 상수 ---
# 논문 범위: h_b(로봇 키)·pitch(카메라 하향각)를 균등분포에서 뽑는다.
RANDOM_H_B_RANGE_M = (0.25, 1.5)
RANDOM_PITCH_RANGE_DEG = (0.0, 30.0)
RANDOM_MIN_START_GOAL_DIST_M = 2.0   # 제자리 왕복 같은 무의미한 에피소드를 거르는 하한
# GT가 없어 chamfer로 못 재므로, reproduce 모드에서 실측한 GT 통계(두 씬)를 넓은 범위의 참고값으로
# 쓴다 — 무작위 start/goal은 GT 에피소드와 다른 목적지 쌍이라 정확히 일치할 이유는 없다(스모크 테스트).
RANDOM_REF_LENGTH_MEDIAN_M = 6.03
RANDOM_REF_CLEARANCE_RANGE_M = (0.46, 0.68)
RANDOM_REF_LEN_RATIO_MEDIAN = 1.08
START_COLOR = (255, 60, 180)   # 마젠타: 무작위 시작점
GOAL_COLOR = (60, 220, 255)    # 시안: 무작위 목표점
# nav_coarse의 낙관적 다운샘플("any") 때문에, 셀 중심은 navigable이어도 하위 0.05 m 셀 중 일부는
# r_b 미만일 수 있다(03의 refine_radius 설명 참고) — GT의 특정 경로는 이런 병목을 잘 안 지나가지만
# 무작위 start/goal은 지도 전체를 고르게 훑으므로 훨씬 자주 걸린다. 하드 게이트(충돌 0, r_b 위반 0)를
# reproduce와 똑같이 유지하려면 위반 에피소드를 통계에 넣지 말고 재추출해야 한다.
RANDOM_MAX_RESAMPLE_TRIES = 30


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default=DEFAULT_SCENE)
    parser.add_argument('--mode', default='reproduce', choices=['reproduce', 'random'],
                        help='reproduce는 GT에서 h_b/pitch/start/goal을 전부 복사한다. random은 C2에서 구현.')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--r_b', type=float, default=ROBOT_RADIUS_M)
    parser.add_argument('--h_nav_ratio', type=float, default=None,
                        help='주면 h_nav = ratio * h_b 로 쓴다(--h_nav 무시). 논문의 "h_nav는 로봇 키에 '
                             '의존" 서술 대응 — 고정 0.10과 재현 정확도가 같아 기본은 고정값이다.')
    parser.add_argument('--refine_radius', type=float, default=0.10,
                        help='논문 6단계 국소 greedy 탐색 반경. 논문은 "a local area"라고만 한다. '
                             '03b가 GT에서 역추정한 R*=0.05에 r_b 제약이 만드는 하한(~0.10)을 적용한 '
                             '값이다(위 표). 크게 잡으면 waypoint가 방 중앙으로 끌려가 GT보다 벽에서 '
                             '멀어진다.')
    parser.add_argument('--astar_cell_m', type=float, default=ASTAR_CELL_M,
                        help='논문 5단계 A* 격자. 논문이 0.2를 명시하므로 기본값은 고정이다 — '
                             '"to facilitate efficient A*"라는 효율 근거이므로 세분은 계산만 더 쓴다. '
                             '기준선 분해에서 이 스냅이 최대 성분(+0.034 m)이었다.')
    parser.add_argument('--downsample_mode', default='any', choices=list(DOWNSAMPLE_MODES),
                        help='논문 4단계 0.2 m 다운샘플. any는 블록에 하나라도 navigable이면 통과'
                             '(기본, 낙관적). majority는 절반 이상일 때만 — 연결성은 같고 navigable만 '
                             '16% 줄어든다. all은 문틈이 지워져 연결성이 붕괴한다.')
    parser.add_argument('--refine_mode', default='argmax', choices=list(REFINERS),
                        help='논문 6단계 해석. argmax는 문장 그대로 "장애물 거리 최대화"로 최대한 '
                             '밀어낸다(기본). min_move는 r_b를 만족하는 만큼만 옮긴다 — GT가 약하게 '
                             'refine된 것으로 측정되어(⑤ R*=0.05) 기준선이 GT에 더 붙는다.')
    parser.add_argument('--smooth', default='cubic', choices=list(SMOOTHERS),
                        help='cubic은 논문 그대로(waypoint 통과, 코너 오버슈트 가능). '
                             'bezier는 convex hull 안에 머물러 오버슈트가 없다.')
    parser.add_argument('--waypoint_spacing_m', type=float, default=WAYPOINT_SPACING_M,
                        help='스무딩 전 waypoint를 솎아낼 호길이 간격. A* 전점을 통과시키면(0) GT보다 '
                             '3배 많이 회전한다 — 0.8에서 GT 스무드니스와 일치. 0이면 솎지 않는다.')
    parser.add_argument('--smooth_step', type=float, default=0.05)
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--esdf_dir', default=None, help='기본값은 <out_dir>/esdf')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    parser.add_argument('--seed', type=int, default=0, help='--mode random 전용 시드 (reproduce는 무시)')
    return parser


def load_gt_episode(data_root: str, scene: str, episode: int):
    """(카메라 궤적 (T,3) world, h_b, pitch_down)"""
    path = Path(data_root) / scene / 'data' / 'chunk-000' / f'episode_{episode:06d}.parquet'
    table = pq.read_table(path, columns=['observation.camera_extrinsic', 'action'])
    extrinsic = np.asarray(table['observation.camera_extrinsic'].to_pylist()[0],
                           dtype=np.float64).reshape(4, 4)
    actions = np.stack([np.asarray(a, dtype=np.float64).reshape(4, 4) for a in table['action'].to_pylist()])
    h_b, pitch = decompose_camera_extrinsic(extrinsic)
    cam_xyz = np.stack([action_to_c2w(a, 'cam2world_gl')[:3, 3] for a in actions])
    return cam_xyz, h_b, pitch


# ---------------------------------------------------------------------------
# 에피소드 1개 — 논문 §3의 4~7단계
# ---------------------------------------------------------------------------

def plan_episode(occupancy, origin, floor_z, h_b, start_xy, goal_xy, args) -> dict:
    """논문 절차를 그대로 돈다. 실패하면 `status`에 이유를 담아 반환한다(조용히 버리지 않는다)."""
    cell = float(args.cell_m)
    h_nav = args.h_nav_ratio * h_b if args.h_nav_ratio else args.h_nav
    obstacle = derive_obstacle_2d(occupancy, origin, floor_z, h_b, cell, h_nav)
    esdf = compute_esdf_2d(obstacle, cell)
    # truncate_navigable만으로는 "장애물 없음"과 "mesh가 아예 없는 스캔 밖 허공"을 구분 못 한다 —
    # coverage mask로 후자를 걷어낸다(무작위 start/goal이 건물 밖으로 새는 것을 여기서 막는다).
    navigable = truncate_navigable(esdf, args.r_b) & compute_scan_coverage_mask(occupancy)
    nav_coarse = downsample_navigable(navigable, args.downsample_factor, args.downsample_mode)

    coarse_cell = cell * args.downsample_factor
    sg = np.array([start_xy, goal_xy], dtype=np.float64)
    ij = world_to_cell(sg, origin, coarse_cell)
    h_c, w_c = nav_coarse.shape
    ij[:, 0] = np.clip(ij[:, 0], 0, w_c - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, h_c - 1)

    result = {'h_b': h_b, 'floor_z': floor_z,
              'start': sg[0].tolist(), 'goal': sg[1].tolist(),
              'smooth': args.smooth, 'esdf': esdf}
    if not nav_coarse[ij[0, 1], ij[0, 0]] or not nav_coarse[ij[1, 1], ij[1, 0]]:
        result['status'] = 'start_or_goal_not_navigable'
        return result

    path_ij = astar(nav_coarse, ij[0], ij[1])
    if path_ij is None:
        result['status'] = 'astar_failed'
        return result

    wp_astar = cell_to_world(path_ij, origin, coarse_cell)
    # start/goal은 GT 값을 그대로 쓴다 (격자 스냅으로 인한 어긋남 제거)
    wp_astar[0], wp_astar[-1] = sg[0], sg[1]
    wp_refined = REFINERS[args.refine_mode](wp_astar, esdf, origin, cell,
                                            args.refine_radius, fix_endpoints=True)
    # r_b 제약은 솎기 전 wp_refined에 걸린다 — 솎기는 스무딩 곡선을 완만하게 만들기 위한 것이다.
    wp_thinned = thin_waypoints(wp_refined, args.waypoint_spacing_m)
    trajectory = SMOOTHERS[args.smooth](wp_thinned, args.smooth_step)

    result.update({
        'status': 'ok',
        # 아래 3개는 논문 5단계 역검증(④ 비용갭)과 ⑦ 위상에 필요하다. JSON에는 안 나간다.
        'path_ij': path_ij, 'nav_coarse': nav_coarse, 'obstacle': obstacle,
        'waypoints_astar': wp_astar,
        'waypoints_refined': wp_refined,
        'waypoints_thinned': wp_thinned,
        'trajectory': trajectory,
        'check_refined': check_path_navigable(wp_refined, esdf, origin, cell, args.r_b),
        'check_trajectory': check_path_navigable(trajectory, esdf, origin, cell, args.r_b),
        'clearance_astar_median': float(np.median(sample_esdf_at(esdf, wp_astar, origin, cell))),
    })
    return result


def sample_random_start_goal(occupancy, origin, floor_z, h_b, args, rng,
                             min_dist_m: float = RANDOM_MIN_START_GOAL_DIST_M,
                             max_tries: int = 200):
    """C2 — 그 `h_b`의 navigable 최대 연결 성분에서 무작위 두 점(최소 이동거리 이상)을 뽑는다.

    `plan_episode`가 내부에서 만드는 것과 완전히 같은 순서(obstacle->esdf->navigable->nav_coarse)를
    다시 밟는다 — start/goal 자체를 뽑으려면 그 h_b의 navigable 격자가 먼저 있어야 하기 때문이다.
    실패하면 `None`(재현 모드처럼 조용히 버리지 않고 호출부가 실패로 기록한다).
    """
    cell = float(args.cell_m)
    h_nav = args.h_nav_ratio * h_b if args.h_nav_ratio else args.h_nav
    obstacle = derive_obstacle_2d(occupancy, origin, floor_z, h_b, cell, h_nav)
    esdf = compute_esdf_2d(obstacle, cell)
    # plan_episode와 같은 이유로 coverage mask를 걸어야 한다 — 여기서 안 걸면 애초에 건물 밖 허공을
    # start/goal로 뽑아버려서, plan_episode에 넘어간 뒤에야 (다른 이유로) 걸러지길 바라는 셈이 된다.
    navigable = truncate_navigable(esdf, args.r_b) & compute_scan_coverage_mask(occupancy)
    nav_coarse = downsample_navigable(navigable, args.downsample_factor, args.downsample_mode)
    coarse_cell = cell * args.downsample_factor

    labels, n_labels = ndimage.label(nav_coarse)
    if n_labels == 0:
        return None
    sizes = ndimage.sum(nav_coarse, labels, index=np.arange(1, n_labels + 1))
    biggest = 1 + int(np.argmax(sizes))   # 가장 큰 성분에서만 뽑는다 — 작은 성분은 고립된 방일 확률이 높다
    ys, xs = np.nonzero(labels == biggest)
    if len(ys) < 2:
        return None

    for _ in range(max_tries):
        i1, i2 = rng.integers(0, len(ys), size=2)
        ij = np.array([[xs[i1], ys[i1]], [xs[i2], ys[i2]]])
        p1, p2 = cell_to_world(ij, origin, coarse_cell)
        if np.linalg.norm(p2 - p1) >= min_dist_m:
            return p1, p2
    return None


def sample_random_episode(occupancy, origin, floor_z, args, rng):
    """h_b~U(0.25,1.5), pitch~U(0,30°)를 뽑고 그 h_b의 navigable에서 start/goal을 표본한다."""
    lo_h, hi_h = RANDOM_H_B_RANGE_M
    lo_p, hi_p = RANDOM_PITCH_RANGE_DEG
    h_b = float(rng.uniform(lo_h, hi_h))
    pitch = float(rng.uniform(lo_p, hi_p))
    sg = sample_random_start_goal(occupancy, origin, floor_z, h_b, args, rng)
    if sg is None:
        return None
    start_xy, goal_xy = sg
    return h_b, pitch, start_xy, goal_xy


def path_vs_gt(path_xy, gt_xy: np.ndarray, esdf, origin, cell: float) -> dict:
    """한 경로를 GT와 재는 지표 묶음. **우리 궤적과 기준선(ⓒ)에 똑같이 쓴다.**

    기준선을 별도 코드로 재면 지표 정의가 갈려 비교가 무의미해진다 — 반드시 같은 함수를 통과시킨다.
    """
    sm_ours, sm_gt = path_smoothness(path_xy), path_smoothness(gt_xy)
    gt_len = float(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).sum())
    our_len = float(np.linalg.norm(np.diff(np.asarray(path_xy), axis=0), axis=1).sum())
    return {
        'chamfer_m': chamfer_distance(path_xy, gt_xy),
        'frechet_m': discrete_frechet_distance(path_xy, gt_xy),
        'len_ratio': our_len / gt_len if gt_len > 0 else float('nan'),
        'gt_length_m': gt_len, 'our_length_m': our_len,
        # ① 스무드니스 — 위치가 맞아도 모양이 틀린 것을 잡는다
        'turn_per_m_deg': sm_ours['turn_per_m_deg'],
        'gt_turn_per_m_deg': sm_gt['turn_per_m_deg'],
        'turn_ratio': sm_ours['turn_per_m_deg'] / max(sm_gt['turn_per_m_deg'], 1e-9),
        'kappa_p95': sm_ours['kappa_p95'], 'gt_kappa_p95': sm_gt['kappa_p95'],
        # ② clearance 프로파일 — 벽에서 GT와 같은 거리를 지나는가
        'clearance': compare_clearance_profile(path_xy, gt_xy, esdf, origin, cell),
        # ③ 방향 정합 — 04의 카메라 방향에 직접 영향
        'heading': heading_alignment(path_xy, gt_xy),
    }


def compare_with_gt(planned: dict, gt_xy: np.ndarray, origin, cell: float, args) -> dict:
    """주 지표(chamfer/Fréchet/길이비) + 보조 지표 ①②③ + ④⑦ + **기준선 ⓒ**.

    주 지표만으로는 부족했다 — chamfer/Fréchet가 **둘 다 통과했는데도** 경로가 GT보다 3배
    지그재그였다. 그래서 모양(①), 벽과의 거리(②), 진행 방향(③)을 따로 잰다.

    **절대값만 보면 안 된다.** 루트가 완벽해도 이산화 때문에 chamfer가 0.08~0.11 m 나온다.
    `floor_*`가 그 하한(ⓒ: GT 루트를 격자에 얹어 논문 6~7단계를 그대로 돌린 것)이고,
    `*_over_floor`가 "하한의 몇 배인가"다 — 판정은 이 배수로 읽어야 한다.
    """
    ours = planned['trajectory']
    esdf = planned['esdf']
    m = path_vs_gt(ours, gt_xy, esdf, origin, cell)
    m['our_length_m'] = planned['check_trajectory']['length_m']   # 선분 샘플 기준으로 통일
    m['same_route'] = bool(m['frechet_m'] < DOOR_WIDTH_M)

    # 기준선 ⓒ / ⓑ — 같은 `path_vs_gt`를 통과시켜야 비교가 성립한다
    coarse = cell * args.downsample_factor
    kw = dict(coarse_cell_m=coarse, refine_radius_m=args.refine_radius,
              spacing_m=args.waypoint_spacing_m, smooth_step_m=args.smooth_step,
              refine_mode=args.refine_mode, smooth_mode=args.smooth)
    floor_path = pipeline_floor_path(gt_xy, esdf, origin, cell, **kw)
    planned['floor_path'] = floor_path        # 시각화용 (JSON에는 안 나간다)
    floor_c = path_vs_gt(floor_path, gt_xy, esdf, origin, cell)
    floor_b = path_vs_gt(pipeline_floor_path(gt_xy, esdf, origin, cell, with_refine=False, **kw),
                         gt_xy, esdf, origin, cell)
    m.update({
        # --- ④ 논문 5단계 역검증: 루트가 갈렸을 때 맵 탓인가 동점인가 ---
        'astar_gap': astar_cost_gap(gt_xy, planned['path_ij'], planned['nav_coarse'], origin, coarse),
        # --- ⑦ 위상: 두 경로 사이에 갇힌 장애물 면적 (임계값 없는 "같은 통로" 판정) ---
        'homotopy_area_m2': homotopy_obstacle_area(ours, gt_xy, planned['obstacle'], origin, cell),
        # --- 기준선 ---
        'floor': floor_c, 'floor_no_refine': floor_b,
        'chamfer_over_floor': m['chamfer_m'] / max(floor_c['chamfer_m'], 1e-9),
        'frechet_over_floor': m['frechet_m'] / max(floor_c['frechet_m'], 1e-9),
        # 오차 원인 분해 (기준선이 아니라 진단용 — docstring 참고)
        'floors_f2f3': metric_floors(gt_xy, coarse, cell),
    })
    return m


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def render_states(navigable, origin, cell, episodes, out_dir: Path):
    """navigable 배경에 GT(노랑) / 기준선ⓒ(하늘) / 우리 경로(초록)를 같은 grid로 겹친다.

    **기준선을 같이 그리는 것이 핵심이다** — 하늘색이 노랑에서 벌어지는 폭이 "루트가 완벽해도 남는
    오차"이므로, 초록이 그 폭 안에 있으면 더 줄일 여지가 없다는 것을 눈으로 확인할 수 있다.
    """
    base, draw, emit = floorplan_canvas(navigable, origin, cell, out_dir, FLOORPLAN_MIN_PX)
    gt_stack = np.vstack([e['gt_xy'] for e in episodes])
    floor_stack = np.vstack([e['planned']['floor_path'] for e in episodes])
    ours_stack = np.vstack([e['planned']['trajectory'] for e in episodes])

    gt_only = draw(base.copy(), gt_stack, GT_COLOR)
    with_floor = draw(gt_only.copy(), floor_stack, FLOOR_COLOR)
    with_ours = draw(with_floor.copy(), ours_stack, OURS_COLOR)
    ours_only = draw(draw(base.copy(), gt_stack, GT_COLOR), ours_stack, OURS_COLOR)
    return [
        ('① navigable만', emit(base, 'navigable.jpg')),
        ('② GT 궤적 (노랑)', emit(gt_only, 'gt.jpg')),
        ('③ + 기준선 ⓒ (하늘) — GT 루트를 격자에 얹은 것. 이만큼은 못 줄인다',
         emit(with_floor, 'floor.jpg')),
        ('④ + 우리 경로 (초록) — 하늘색과 겹치면 한계 도달', emit(with_ours, 'all.jpg')),
        ('⑤ GT vs 우리만 — 같은 루트면 포개진다', emit(ours_only, 'overlay.jpg')),
    ]


def render_worst_episodes(navigable, origin, cell, episodes, out_dir: Path, n: int = 3):
    """기준선 대비 배수가 가장 큰 에피소드를 개별로 — 어디서 벌어지는지 보려면 겹쳐보기로는 안 된다."""
    base, draw, emit = floorplan_canvas(navigable, origin, cell, out_dir, FLOORPLAN_MIN_PX)
    worst = sorted(episodes, key=lambda e: -e['compare']['chamfer_over_floor'])[:n]
    out = []
    for e in worst:
        ep, c = e['episode'], e['compare']
        gt = draw(base.copy(), e['gt_xy'], GT_COLOR, 2)
        fl = draw(gt.copy(), e['planned']['floor_path'], FLOOR_COLOR, 2)
        al = draw(fl.copy(), e['planned']['trajectory'], OURS_COLOR, 2)
        out.append((ep, c, [
            ('① GT만 (노랑)', emit(gt, f'ep{ep}_gt.jpg')),
            (f'② + 기준선 ⓒ (하늘) — chamfer {c["floor"]["chamfer_m"]:.3f} m',
             emit(fl, f'ep{ep}_floor.jpg')),
            (f'③ + 우리 (초록) — chamfer {c["chamfer_m"]:.3f} m = 기준선의 '
             f'{c["chamfer_over_floor"]:.2f}배', emit(al, f'ep{ep}_all.jpg')),
        ]))
    return out


def render_states_random(navigable, origin, cell, episodes, out_dir: Path):
    """C2 — GT가 없으므로 navigable 위에 무작위 경로만 겹친다 (시작=마젠타 점, 목표=시안 점)."""
    base, draw, emit = floorplan_canvas(navigable, origin, cell, out_dir, FLOORPLAN_MIN_PX)
    traj_stack = np.vstack([e['planned']['trajectory'] for e in episodes])
    starts = np.array([e['planned']['start'] for e in episodes])
    goals = np.array([e['planned']['goal'] for e in episodes])

    with_paths = draw(base.copy(), traj_stack, OURS_COLOR)
    with_starts = draw(with_paths.copy(), starts, START_COLOR, 4)
    with_goals = draw(with_starts.copy(), goals, GOAL_COLOR, 4)
    return [
        ('① navigable만', emit(base, 'navigable.jpg')),
        ('② + 무작위 경로 (초록)', emit(with_paths, 'paths.jpg')),
        ('③ + start(마젠타)/goal(시안)', emit(with_goals, 'all.jpg')),
    ]


def build_summary_random(args, stats, episodes) -> str:
    lo, hi = RANDOM_REF_CLEARANCE_RANGE_M
    rows = ''.join(
        f'<tr><td>{e["episode"]}</td><td>{e["planned"]["h_b"]:.3f}</td>'
        f'<td>{e["pitch_deg"]:.1f}</td>'
        f'<td>{e["planned"]["check_trajectory"]["length_m"]:.2f}</td>'
        f'<td>{e["straight_m"]:.2f}</td><td>{e["len_ratio"]:.2f}</td>'
        f'<td>{e["planned"]["check_trajectory"]["points_median_m"]:.3f}</td>'
        f'<td>{e["planned"]["check_refined"]["points_min_m"]:.3f}</td>'
        f'<td>{e["planned"]["check_trajectory"]["points_min_m"]:.3f}</td>'
        f'<td>{e["smoothness"]["turn_per_m_deg"]:.1f}</td></tr>' for e in episodes)
    return f'''
<div class="stat-row">
  <div class="stat"><b>전체</b>{pill(stats["passed"])}</div>
  <div class="stat"><b>계획 성공</b><span class="pill {"good" if stats["success_rate"] == 1 else "warn"}">
      {stats["n_ok"]}/{stats["n_total"]}</span></div>
  <div class="stat"><b>길이 median</b><span class="pill">{stats["length_median_m"]:.2f} m</span></div>
  <div class="stat"><b>clearance median</b><span class="pill">{stats["clearance_median_m"]:.3f} m</span></div>
  <div class="stat"><b>길이/직선 median</b><span class="pill">{stats["len_ratio_median"]:.2f}</span></div>
  <div class="stat"><b>seed</b><span class="pill">{args.seed}</span></div>
</div>
<p>모드 <code>random</code> — 에피소드마다 <code>h_b~U({RANDOM_H_B_RANGE_M[0]},{RANDOM_H_B_RANGE_M[1]})</code>,
<code>pitch~U({RANDOM_PITCH_RANGE_DEG[0]:.0f}°,{RANDOM_PITCH_RANGE_DEG[1]:.0f}°)</code>를 뽑고, 그
<code>h_b</code>의 navigable 최대 연결 성분에서 서로 {RANDOM_MIN_START_GOAL_DIST_M} m 이상 떨어진 두
점을 start/goal로 삼는다. 이후 A*→refine→thin→{args.smooth} 스무딩은 <code>reproduce</code> 모드와
완전히 같은 코드 경로다. <b>GT가 없어 chamfer로 못 재므로</b>, reproduce 모드에서 실측한 GT 통계
(길이 median {RANDOM_REF_LENGTH_MEDIAN_M} m, clearance median 씬별 {lo}~{hi} m, 길이/직선 비율 median
{RANDOM_REF_LEN_RATIO_MEDIAN})와 넓은 범위로 분포만 비교한다 — 무작위 목적지 쌍은 GT 에피소드와
다른 거리/난이도라 정확히 일치할 이유는 없어서 <b>하드 게이트로 쓰지 않는다</b>. 하드 게이트
(refined r_b, 충돌)를 위반한 후보는 통계에 넣지 않고 <b>재추출</b>했다({stats["n_rejected_resamples"]}건
버림) — nav_coarse의 낙관적 다운샘플 때문에 무작위 start/goal이 GT보다 훨씬 자주 병목을 지나가서다.</p>
<table class="table">
  <tr><th>검증</th><th>결과</th><th>수치</th></tr>
  <tr><td>① refined waypoint clearance ≥ r_b={args.r_b}</td><td>{pill(stats["refined_ok"])}</td>
      <td>최소 {stats["refined_min_m"]:.3f} m ({stats["n_refined_ok"]}/{stats["n_ok"]} 에피소드 통과)</td></tr>
  <tr><td>② <b>하드 충돌</b> — 궤적이 장애물을 관통하는가 (clearance = 0)</td>
      <td>{pill(stats["no_collision"])}</td>
      <td>{stats["hard_collision_n"]}/{stats["collision_samples"]:,} 샘플 —
          {stats["hard_collision_eps"]}/{stats["n_ok"]} 에피소드</td></tr>
  <tr><td>③ 분포가 GT 참고 범위 안인가 <i>(스모크 테스트, 하드 게이트 아님)</i></td>
      <td><span class="pill {"good" if stats["distribution_ok"] else "warn"}">
          {"참고범위 내" if stats["distribution_ok"] else "참고범위 밖"}</span></td>
      <td>길이 {stats["length_median_m"]:.2f} m (참고 {RANDOM_REF_LENGTH_MEDIAN_M} m),
          clearance {stats["clearance_median_m"]:.3f} m (참고 {lo}~{hi} m),
          길이/직선 {stats["len_ratio_median"]:.2f} (참고 {RANDOM_REF_LEN_RATIO_MEDIAN})</td></tr>
</table>
<table class="table">
  <tr><th>ep</th><th>h_b</th><th>pitch</th><th>길이 m</th><th>직선 m</th><th>길이비</th>
      <th>clr median</th><th>refined min</th><th>traj min</th><th>회전 도/m</th></tr>{rows}</table>
'''


def pill(ok: bool) -> str:
    return f'<span class="pill {"good" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span>'


def build_summary(args, stats, episodes) -> str:
    rows = ''.join(
        f'<tr><td>{e["episode"]}</td><td>{e["planned"]["h_b"]:.3f}</td>'
        f'<td>{e["compare"]["chamfer_m"]:.3f}</td><td>{e["compare"]["frechet_m"]:.3f}</td>'
        f'<td>{e["compare"]["len_ratio"]:.2f}</td>'
        f'<td>{e["compare"]["turn_per_m_deg"]:.0f} / {e["compare"]["gt_turn_per_m_deg"]:.0f}</td>'
        f'<td>{e["compare"]["clearance"]["median_diff_m"]:+.3f}</td>'
        f'<td>{e["compare"]["heading"]["median_deg"]:.1f}</td>'
        f'<td>{e["compare"]["astar_gap"]["cost_ratio"]:.3f}</td>'
        f'<td>{e["compare"]["homotopy_area_m2"]:.3f}</td>'
        f'<td>{e["compare"]["floor"]["chamfer_m"]:.3f}</td>'
        f'<td>{e["compare"]["chamfer_over_floor"]:.2f}</td>'
        f'<td>{e["planned"]["check_refined"]["points_min_m"]:.3f}</td>'
        f'<td>{e["planned"]["check_trajectory"]["points_min_m"]:.3f}</td>'
        f'<td>{"O" if e["compare"]["same_route"] else "X"}</td></tr>' for e in episodes)
    return f'''
<div class="stat-row">
  <div class="stat"><b>전체</b>{pill(stats["passed"])}</div>
  <div class="stat"><b>계획 성공</b><span class="pill {"good" if stats["success_rate"] == 1 else "warn"}">
      {stats["n_ok"]}/{stats["n_total"]}</span></div>
  <div class="stat"><b>같은 루트</b><span class="pill">{stats["n_same_route"]}/{stats["n_ok"]}</span></div>
  <div class="stat"><b>Fréchet median</b><span class="pill">{stats["frechet_median_m"]:.3f} m</span></div>
  <div class="stat"><b>기준선 대비</b><span class="pill {"good" if stats["at_floor"] else "warn"}">
      chamfer {stats["chamfer_over_floor"]:.2f}× / Fréchet {stats["frechet_over_floor"]:.2f}×</span></div>
  <div class="stat"><b>스무딩</b><span class="pill">{args.smooth}</span></div>
  <div class="stat"><b>refine</b><span class="pill">{args.refine_mode} / {args.refine_radius}</span></div>
</div>
<p>모드 <code>{args.mode}</code> — 에피소드별 <code>h_b</code>·pitch·start·goal을 전부 GT에서 복사해
논문 §3의 A*(0.2 m) → greedy refine(0.05 m) → {args.smooth} 스무딩을 돌렸다.
같은 start/goal이므로 경로가 갈리면 <b>루트 선택이 다른 것</b>이다.</p>
<table class="table">
  <tr><th>검증</th><th>결과</th><th>수치</th></tr>
  <tr><td>① refined waypoint clearance ≥ r_b={args.r_b}</td><td>{pill(stats["refined_ok"])}</td>
      <td>최소 {stats["refined_min_m"]:.3f} m ({stats["n_refined_ok"]}/{stats["n_ok"]} 에피소드 통과)</td></tr>
  <tr><td>② refine이 clearance를 올렸는가</td><td>{pill(stats["refine_improves"])}</td>
      <td>A* {stats["astar_clearance_median_m"]:.3f} → refined {stats["refined_clearance_median_m"]:.3f} m</td></tr>
  <tr><td>③ 재현 — Fréchet &lt; 문 폭 {DOOR_WIDTH_M} m</td>
      <td><span class="pill {"good" if stats["n_same_route"] == stats["n_ok"] else "warn"}">
          {stats["n_same_route"]}/{stats["n_ok"]}</span></td>
      <td>chamfer median {stats["chamfer_median_m"]:.3f} m,
          Fréchet median {stats["frechet_median_m"]:.3f} m,
          길이 비율 median {stats["len_ratio_median"]:.2f}</td></tr>
  <tr><td>④ <b>하드 충돌</b> — 궤적이 장애물을 관통하는가 (clearance = 0)</td>
      <td>{pill(stats["no_collision"])}</td>
      <td>{stats["hard_collision_n"]}/{stats["collision_samples"]:,} 샘플 (0.01 m 간격, 점 + 점 사이 선분)
          — {stats["hard_collision_eps"]}/{stats["n_ok"]} 에피소드</td></tr>
  <tr><td>⑤ 몸통 침범 — clearance &lt; r_b <i>(측정만)</i></td><td><span class="pill warn">참고</span></td>
      <td>최소 {stats["traj_min_m"]:.3f} m, {stats["n_body_violation_eps"]}/{stats["n_ok"]} 에피소드
          — <b>GT도 위반한다</b>(씬1 5/20, 씬2 4/20)라 하드 제약으로 쓰지 않는다</td></tr>
</table>
<h3>보조 지표 — chamfer/Fréchet가 못 잡는 것</h3>
<p>chamfer·Fréchet가 <b>둘 다 통과했는데도</b> 경로가 GT보다 3배 지그재그였다(100 vs 34 도/m).
위치가 맞아도 <b>모양·거리·방향</b>은 따로 재야 한다.</p>
<table class="table">
  <tr><th>지표</th><th>결과</th><th>우리</th><th>GT</th><th>읽는 법</th></tr>
  <tr><td>① 스무드니스 — 미터당 회전각 [도/m]</td><td>{pill(stats["smooth_ok"])}</td>
      <td>{stats["turn_per_m_deg"]:.1f}</td><td>{stats["gt_turn_per_m_deg"]:.1f}</td>
      <td>배율 <b>{stats["turn_ratio"]:.2f}×</b> (게이트 {TURN_RATIO_MIN}~{TURN_RATIO_MAX},
          <b>기준선 {stats["floor_turn_ratio"]:.2f}×</b>). 곡률 p95
          {stats["kappa_p95"]:.2f} vs GT {stats["gt_kappa_p95"]:.2f} — A* 격자 꺾임이 남으면 여기서 먼저 뜬다</td></tr>
  <tr><td>② clearance 프로파일 [m]</td><td><span class="pill warn">참고</span></td>
      <td>{stats["clearance_ours_median_m"]:.3f}</td><td>{stats["clearance_gt_median_m"]:.3f}</td>
      <td>차이 <b>{stats["clearance_median_diff_m"]:+.3f} m</b>
          (<b>기준선 {stats["floor_clr_diff_m"]:+.3f}</b>). 양수면 우리가 벽에서 더 멀다 =
          refine이 과하다는 뜻</td></tr>
  <tr><td>③ 방향 정합 — 헤딩 차이 [도]</td><td><span class="pill warn">참고</span></td>
      <td>{stats["heading_median_deg"]:.1f}</td><td>0</td>
      <td>p90 {stats["heading_p90_deg"]:.1f}도, <b>기준선 {stats["floor_heading_deg"]:.1f}도</b>.
          격자에 얹는 순간 45도 꺾임이 생겨 <b>하한이 0이 아니다</b>.
          04가 이 경로로 카메라를 놓으므로 위치 5 cm보다 헤딩 10도가 렌더 결과를 더 바꾼다</td></tr>
</table>
<h3>논문 단계별 역검증 — GT가 그 단계의 산출물로서 성립하는가</h3>
<p>chamfer/Fréchet는 <b>최종 곡선</b>만 본다. 그래서 "chamfer {stats["chamfer_median_m"]:.3f} m가 좋은
값인가?"에 답할 수 없다. 아래는 논문 5·7단계를 GT에 거꾸로 적용한 것이다.</p>
<table class="table">
  <tr><th>검사</th><th>수치</th><th>읽는 법</th></tr>
  <tr><td><b>기준선 ⓒ</b> — A*가 GT 루트를 정확히 골랐을 때 우리 파이프라인이 내는 값</td>
      <td><span class="pill {"good" if stats["at_floor"] else "warn"}">
          chamfer {stats["chamfer_over_floor"]:.2f}×</span></td>
      <td>chamfer {stats["chamfer_median_m"]:.3f} / <b>기준선 {stats["floor_chamfer_m"]:.3f}</b> m<br>
          Fréchet {stats["frechet_median_m"]:.3f} / <b>{stats["floor_frechet_m"]:.3f}</b> m
          ({stats["frechet_over_floor"]:.2f}×)</td>
      <td>GT 루트를 0.2 m 격자에 얹어 논문 6~7단계를 <b>실제와 똑같이</b> 돌린 것과 GT를 비교한 값이다.
          루트가 완벽해도 이 값이 나오므로 <b>0을 목표로 삼을 수 없다.</b>
          배수가 1에 가까우면 남은 차이가 거의 전부 이산화라는 뜻
          (게이트 아님, {FLOOR_RATIO_AT_LIMIT}× 이내를 한계 도달로 본다).
          내역: 격자+스무딩만 {stats["floor_no_refine_chamfer_m"]:.3f} m -> refine 포함
          {stats["floor_chamfer_m"]:.3f} m. 원인 분해는 격자(F3) {stats["f3_chamfer_m"]:.3f} /
          점분포(F2) {stats["f2_chamfer_m"]:.3f} m</td></tr>
  <tr><td>④ <b>A* 최적성 갭</b> (논문 5단계)</td>
      <td>cost_ratio median {stats["cost_ratio_median"]:.3f}<br>
          GT가 우리 맵을 벗어난 에피소드 {stats["n_gt_blocked_eps"]}/{stats["n_ok"]}</td>
      <td>같은 맵·같은 start/goal에서 둘 다 A*면 비용이 같아야 한다(=1.0).
          <b>1.0인데 루트가 다르면 동점</b>(tie-breaking 차이 — 우리 잘못 아님),
          <b>&gt;1.05면 우리 맵에 GT 맵엔 없던 장애물</b>이 있다는 뜻이다</td></tr>
  <tr><td>⑦ 위상 — 두 경로 사이에 갇힌 장애물 면적</td>
      <td>median {stats["homotopy_area_median_m2"]:.3f} m²<br>
          정확히 0인 에피소드 {stats["n_homotopy_zero"]}/{stats["n_ok"]}</td>
      <td>같은 통로면 사이에 장애물이 없어 <b>정확히 0</b>이다. Fréchet 임계 {DOOR_WIDTH_M} m는
          임의값인데 이건 0/1로 갈린다 — 보조 지표로 함께 본다</td></tr>
</table>
<table class="table">
  <tr><th>ep</th><th>h_b</th><th>chamfer</th><th>Fréchet</th><th>길이비</th>
      <th>① 회전 우리/GT</th><th>② clr 차</th><th>③ 헤딩</th>
      <th>④ cost비</th><th>⑦ 위상 m²</th><th>기준선 ⓒ</th><th>배수</th>
      <th>refined min</th><th>traj min</th><th>같은루트</th></tr>{rows}</table>
'''


def load_esdf(args):
    """reproduce/random 공용 — `esdf/<scene>.npz`를 읽고 `args`에 파생 필드를 채운다."""
    esdf_dir = Path(args.esdf_dir or (Path(args.out_dir) / 'esdf'))
    npz_path = esdf_dir / f'{args.scene}.npz'
    if not npz_path.is_file():
        print(f'  [ERROR] {npz_path} 없음 — 02_build_freemap_esdf.py를 먼저 돌릴 것')
        return None
    data = np.load(npz_path)
    occupancy, origin = data['occupancy'], data['origin']
    floor_z = float(data['floor_z'])
    args.cell_m = float(data['voxel_size'])
    args.downsample_factor = max(1, int(round(args.astar_cell_m / args.cell_m)))
    args.h_nav = float(data['h_nav'])
    if args.h_nav_ratio is None and 'h_nav_ratio' in data and float(data['h_nav_ratio']) > 0:
        args.h_nav_ratio = float(data['h_nav_ratio'])   # 02가 비례형으로 만들었으면 그대로 따른다
    print(f'  esdf: grid={occupancy.shape} cell={args.cell_m} h_nav={args.h_nav} floor_z={floor_z:+.4f}')
    return occupancy, origin, floor_z


def main_random(args) -> int:
    """C2 — `--mode random`. reproduce와 계획 단계(`plan_episode`)는 완전히 동일한 코드 경로를
    타고, start/goal/h_b/pitch를 GT에서 복사하는 대신 `sample_random_episode`로 뽑는 것만 다르다.
    """
    loaded = load_esdf(args)
    if loaded is None:
        return 2
    occupancy, origin, floor_z = loaded

    rng = np.random.default_rng(args.seed)
    episodes, failures, n_rejected = [], {}, 0
    for ep in range(args.num_episodes):
        planned = h_b = pitch = start_xy = goal_xy = None
        last_status = 'no_valid_sample'
        for _attempt in range(RANDOM_MAX_RESAMPLE_TRIES):
            sample = sample_random_episode(occupancy, origin, floor_z, args, rng)
            if sample is None:
                last_status = 'start_goal_sampling_failed'
                continue
            h_b, pitch, start_xy, goal_xy = sample
            candidate = plan_episode(occupancy, origin, floor_z, h_b, start_xy, goal_xy, args)
            if candidate['status'] != 'ok':
                last_status = candidate['status']
                continue
            # 하드 게이트(reproduce와 동일) — 위반이면 통계에 넣지 않고 재추출한다.
            if candidate['check_refined']['points_min_m'] < args.r_b:
                last_status, n_rejected = 'refined_r_b_violation', n_rejected + 1
                continue
            if candidate['check_trajectory']['hard_n'] > 0:
                last_status, n_rejected = 'hard_collision', n_rejected + 1
                continue
            planned = candidate
            break
        if planned is None:
            failures[ep] = last_status
            print(f'    ep {ep:>3}: {last_status} ({RANDOM_MAX_RESAMPLE_TRIES}회 재추출 실패)')
            continue
        straight_m = float(np.linalg.norm(np.asarray(goal_xy) - np.asarray(start_xy)))
        len_ratio = (planned['check_trajectory']['length_m'] / straight_m
                    if straight_m > 0 else float('nan'))
        smoothness = path_smoothness(planned['trajectory'])
        episodes.append({'episode': ep, 'pitch_deg': pitch, 'planned': planned,
                         'straight_m': straight_m, 'len_ratio': len_ratio, 'smoothness': smoothness})
        print(f'    ep {ep:>3}: h_b={h_b:.3f} pitch={pitch:.1f} '
              f'len={planned["check_trajectory"]["length_m"]:.2f}m straight={straight_m:.2f}m '
              f'len_ratio={len_ratio:.2f} clr_med={planned["check_trajectory"]["points_median_m"]:.3f} '
              f'refined_min={planned["check_refined"]["points_min_m"]:.3f} '
              f'traj_min={planned["check_trajectory"]["points_min_m"]:.3f} '
              f'turn={smoothness["turn_per_m_deg"]:.1f}deg/m')

    if not episodes:
        print(f'  [ERROR] 계획 성공 0건 (실패 사유: {failures})')
        return 1
    print(f'  재추출(r_b 위반/하드충돌로 버려진 후보): {n_rejected}건')

    ref_min = np.array([e['planned']['check_refined']['points_min_m'] for e in episodes])
    stats = {
        'n_total': args.num_episodes, 'n_ok': len(episodes),
        'success_rate': len(episodes) / args.num_episodes,
        'failures': failures, 'n_rejected_resamples': n_rejected,
        'n_refined_ok': int((ref_min >= args.r_b).sum()),
        'refined_min_m': float(ref_min.min()),
        'refined_ok': bool((ref_min >= args.r_b).all()),
        'traj_min_m': float(min(e['planned']['check_trajectory']['points_min_m'] for e in episodes)),
        'hard_collision_n': int(sum(e['planned']['check_trajectory']['hard_n'] for e in episodes)),
        'hard_collision_eps': int(sum(not e['planned']['check_trajectory']['hard_ok'] for e in episodes)),
        'collision_samples': int(sum(e['planned']['check_trajectory']['n_samples'] for e in episodes)),
        'length_median_m': float(np.median(
            [e['planned']['check_trajectory']['length_m'] for e in episodes])),
        'straight_median_m': float(np.median([e['straight_m'] for e in episodes])),
        'len_ratio_median': float(np.median([e['len_ratio'] for e in episodes])),
        'clearance_median_m': float(np.median(
            [e['planned']['check_trajectory']['points_median_m'] for e in episodes])),
        'turn_per_m_deg_median': float(np.median(
            [e['smoothness']['turn_per_m_deg'] for e in episodes])),
    }
    stats['no_collision'] = bool(stats['hard_collision_n'] == 0)
    lo, hi = RANDOM_REF_CLEARANCE_RANGE_M
    # 스모크 테스트다 — 절대값 일치가 아니라 "터무니없이 벗어나지 않는가"만 넓은 배수로 본다.
    stats['distribution_ok'] = bool(
        0.3 * RANDOM_REF_LENGTH_MEDIAN_M <= stats['length_median_m'] <= 3.0 * RANDOM_REF_LENGTH_MEDIAN_M
        and 0.5 * lo <= stats['clearance_median_m'] <= 1.8 * hi
        and 1.0 <= stats['len_ratio_median'] <= 1.6)
    # 하드 게이트는 reproduce와 같다(충돌 0, r_b 위반 0) — 분포는 참고용이라 게이트에 넣지 않는다.
    stats['passed'] = bool(stats['refined_ok'] and stats['no_collision']
                           and len(episodes) == args.num_episodes)

    # --- 저장 ---
    paths_dir = Path(args.out_dir) / 'paths'
    paths_dir.mkdir(parents=True, exist_ok=True)
    out = {
        'scene_id': args.scene, 'mode': args.mode, 'smooth': args.smooth, 'seed': args.seed,
        'params': {'r_b': args.r_b, 'refine_radius': args.refine_radius, 'cell_m': args.cell_m,
                   'astar_cell_m': args.cell_m * args.downsample_factor, 'h_nav': args.h_nav,
                   'refine_mode': args.refine_mode, 'downsample_mode': args.downsample_mode,
                   'smooth_step': args.smooth_step, 'floor_z': floor_z,
                   'h_nav_ratio': args.h_nav_ratio,
                   'waypoint_spacing_m': args.waypoint_spacing_m},
        'stats': stats, 'passed': stats['passed'],
        'episodes': [{
            'episode_id': e['episode'], 'mode': args.mode, 'smooth': args.smooth,
            'h_b': e['planned']['h_b'], 'pitch_deg': e['pitch_deg'], 'floor_z': floor_z,
            'start': e['planned']['start'], 'goal': e['planned']['goal'],
            'waypoints_astar': e['planned']['waypoints_astar'].tolist(),
            'waypoints_refined': e['planned']['waypoints_refined'].tolist(),
            'waypoints_thinned': e['planned']['waypoints_thinned'].tolist(),
            'trajectory': e['planned']['trajectory'].tolist(),
            'path_length_m': e['planned']['check_trajectory']['length_m'],
            'straight_dist_m': e['straight_m'], 'len_ratio': e['len_ratio'],
            'check_refined': e['planned']['check_refined'],
            'check_trajectory': e['planned']['check_trajectory'],
        } for e in episodes],
    }
    json_path = paths_dir / f'{args.scene}_random.json'
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'  paths -> {json_path}')

    # --- 리포트 ---
    log_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.scene}_random'
    navigable = truncate_navigable(episodes[0]['planned']['esdf'], args.r_b) & compute_scan_coverage_mask(occupancy)
    states = render_states_random(navigable, origin, args.cell_m, episodes, log_dir)
    body = ('<h3>무작위 생성 경로 (C2, 같은 grid)</h3>'
            '<p class="note">‹ ›로 넘긴다. <b>초록 = 경로, 마젠타 = start, 시안 = goal.</b> '
            f'GT가 없어 개별 정답 비교는 못 하고, {args.num_episodes}개 에피소드를 한 grid에 겹쳐 '
            '분포(밀도·clearance)를 눈으로 확인한다.</p>'
            + blink_widget_html('paths', states, title=f'{args.scene} random/{args.smooth}'))
    report = save_gallery(log_dir, 'report.html',
                          f'03_sample_gt_paths — {args.scene} (random/{args.smooth})',
                          build_summary_random(args, stats, episodes), body)
    print(f'  report html -> {report}')
    print(f'  => {"PASS" if stats["passed"] else "FAIL"} '
          f'(성공 {stats["n_ok"]}/{args.num_episodes}, '
          f'길이 median {stats["length_median_m"]:.2f} m, '
          f'분포 {"참고범위 내" if stats["distribution_ok"] else "참고범위 밖"})')
    return 0 if stats['passed'] else 1


def main() -> int:
    args = build_argparser().parse_args()
    print(f'[{SCRIPT_NAME}] scene={args.scene} mode={args.mode} smooth={args.smooth} '
          f'refine={args.refine_mode}/{args.refine_radius} astar_cell={args.astar_cell_m}')

    if args.mode == 'random':
        return main_random(args)

    loaded = load_esdf(args)
    if loaded is None:
        return 2
    occupancy, origin, floor_z = loaded

    n_avail = len(sorted((Path(args.data_root) / args.scene / 'data' / 'chunk-000').glob('*.parquet')))
    episodes, failures = [], {}
    for ep in range(min(args.num_episodes, n_avail)):
        gt_xyz, h_b, pitch = load_gt_episode(args.data_root, args.scene, ep)
        gt_xy = gt_xyz[:, :2]
        planned = plan_episode(occupancy, origin, floor_z, h_b, gt_xy[0], gt_xy[-1], args)
        if planned['status'] != 'ok':
            failures[ep] = planned['status']
            print(f'    ep {ep:>3}: {planned["status"]}  (h_b={h_b:.3f})')
            continue
        compare = compare_with_gt(planned, gt_xy, origin, args.cell_m, args)
        episodes.append({'episode': ep, 'pitch_deg': pitch, 'gt_xy': gt_xy,
                         'planned': planned, 'compare': compare})
        print(f'    ep {ep:>3}: h_b={h_b:.3f}  chamfer={compare["chamfer_m"]:.3f} '
              f'frechet={compare["frechet_m"]:.3f} len_ratio={compare["len_ratio"]:.2f}  '
              f'refined_min={planned["check_refined"]["points_min_m"]:.3f} '
              f'traj_min={planned["check_trajectory"]["points_min_m"]:.3f} '
              f'turn={compare["turn_per_m_deg"]:.1f}/{compare["gt_turn_per_m_deg"]:.1f}deg/m '
              f'clr_diff={compare["clearance"]["median_diff_m"]:+.3f} '
              f'head={compare["heading"]["median_deg"]:.1f}deg '
              f'cost_ratio={compare["astar_gap"]["cost_ratio"]:.3f}'
              f'{"(blocked %d)" % compare["astar_gap"]["n_blocked"] if compare["astar_gap"]["n_blocked"] else ""} '
              f'homotopy={compare["homotopy_area_m2"]:.3f} '
              f'floor={compare["floor"]["chamfer_m"]:.3f}'
              f'({compare["chamfer_over_floor"]:.2f}x)  '
              f'{"same-route" if compare["same_route"] else "DIFFERENT-ROUTE"}')

    if not episodes:
        print(f'  [ERROR] 계획 성공 0건 (실패 사유: {failures})')
        return 1

    ref_min = np.array([e['planned']['check_refined']['points_min_m'] for e in episodes])
    n_total = min(args.num_episodes, n_avail)
    stats = {
        'n_total': n_total, 'n_ok': len(episodes), 'success_rate': len(episodes) / n_total,
        'failures': failures,
        'n_refined_ok': int((ref_min >= args.r_b).sum()),
        'refined_min_m': float(ref_min.min()),
        'refined_ok': bool((ref_min >= args.r_b).all()),
        'traj_min_m': float(min(e['planned']['check_trajectory']['points_min_m'] for e in episodes)),
        'hard_collision_n': int(sum(e['planned']['check_trajectory']['hard_n'] for e in episodes)),
        'hard_collision_eps': int(sum(not e['planned']['check_trajectory']['hard_ok'] for e in episodes)),
        'collision_samples': int(sum(e['planned']['check_trajectory']['n_samples'] for e in episodes)),
        'n_body_violation_eps': int(sum(
            not e['planned']['check_trajectory']['segments_ok'] for e in episodes)),
        'astar_clearance_median_m': float(np.median([e['planned']['clearance_astar_median'] for e in episodes])),
        'refined_clearance_median_m': float(np.median(
            [e['planned']['check_refined']['points_median_m'] for e in episodes])),
        'chamfer_median_m': float(np.median([e['compare']['chamfer_m'] for e in episodes])),
        'frechet_median_m': float(np.median([e['compare']['frechet_m'] for e in episodes])),
        'len_ratio_median': float(np.median([e['compare']['len_ratio'] for e in episodes])),
        'n_same_route': int(sum(e['compare']['same_route'] for e in episodes)),
        'door_width_m': DOOR_WIDTH_M,
        # --- 보조 지표 ①②③ (median) ---
        'turn_per_m_deg': float(np.median([e['compare']['turn_per_m_deg'] for e in episodes])),
        'gt_turn_per_m_deg': float(np.median([e['compare']['gt_turn_per_m_deg'] for e in episodes])),
        'turn_ratio': float(np.median([e['compare']['turn_ratio'] for e in episodes])),
        'kappa_p95': float(np.median([e['compare']['kappa_p95'] for e in episodes])),
        'gt_kappa_p95': float(np.median([e['compare']['gt_kappa_p95'] for e in episodes])),
        'clearance_median_diff_m': float(np.median(
            [e['compare']['clearance']['median_diff_m'] for e in episodes])),
        'clearance_ours_median_m': float(np.median(
            [e['compare']['clearance']['ours_median_m'] for e in episodes])),
        'clearance_gt_median_m': float(np.median(
            [e['compare']['clearance']['gt_median_m'] for e in episodes])),
        'heading_median_deg': float(np.median([e['compare']['heading']['median_deg'] for e in episodes])),
        'heading_p90_deg': float(np.median([e['compare']['heading']['p90_deg'] for e in episodes])),
        # --- ④ A* 최적성 갭 ---
        'cost_ratio_median': float(np.median([e['compare']['astar_gap']['cost_ratio'] for e in episodes])),
        'n_gt_blocked_eps': int(sum(e['compare']['astar_gap']['n_blocked'] > 0 for e in episodes)),
        # --- ⑦ 위상 ---
        'homotopy_area_median_m2': float(np.median(
            [e['compare']['homotopy_area_m2'] for e in episodes])),
        'n_homotopy_zero': int(sum(e['compare']['homotopy_area_m2'] == 0.0 for e in episodes)),
        # --- 기준선 ⓒ: "A*가 GT 루트를 정확히 골랐을 때 우리 파이프라인이 내는 값" ---
        'floor_chamfer_m': float(np.median([e['compare']['floor']['chamfer_m'] for e in episodes])),
        'floor_frechet_m': float(np.median([e['compare']['floor']['frechet_m'] for e in episodes])),
        'floor_turn_ratio': float(np.median([e['compare']['floor']['turn_ratio'] for e in episodes])),
        'floor_clr_diff_m': float(np.median(
            [e['compare']['floor']['clearance']['median_diff_m'] for e in episodes])),
        'floor_heading_deg': float(np.median(
            [e['compare']['floor']['heading']['median_deg'] for e in episodes])),
        'chamfer_over_floor': float(np.median([e['compare']['chamfer_over_floor'] for e in episodes])),
        'frechet_over_floor': float(np.median([e['compare']['frechet_over_floor'] for e in episodes])),
        # 기준선 ⓑ(refine 없음) — 하한 중 격자/스무딩 기여분만
        'floor_no_refine_chamfer_m': float(np.median(
            [e['compare']['floor_no_refine']['chamfer_m'] for e in episodes])),
        # 원인 분해 (진단용)
        'f2_chamfer_m': float(np.median([e['compare']['floors_f2f3']['f2_chamfer_m'] for e in episodes])),
        'f3_chamfer_m': float(np.median([e['compare']['floors_f2f3']['f3_chamfer_m'] for e in episodes])),
    }
    stats['refine_improves'] = stats['refined_clearance_median_m'] > stats['astar_clearance_median_m']
    # ①만 게이트로 쓴다 — A* 격자 꺾임이 다시 새는 회귀를 잡기 위한 것이다(실측 1.04/1.45×).
    # ②③은 진단용이라 측정만 한다(GT가 컨트롤러 궤적이라 목표값을 못 박는다).
    stats['smooth_ok'] = bool(TURN_RATIO_MIN <= stats['turn_ratio'] <= TURN_RATIO_MAX)
    # 하드 충돌(clearance == 0)은 관용 없이 0이어야 한다 — 질점 로봇도 못 지나는 구간이므로
    # 맵이나 플래너의 버그다. 몸통 침범(< r_b)은 GT도 위반하므로 게이트가 아니다.
    stats['no_collision'] = bool(stats['hard_collision_n'] == 0)
    # 기준선 대비 배수는 진단이다 — 게이트로 걸지 않는다(씬 난이도에 따라 달라진다)
    stats['at_floor'] = bool(stats['chamfer_over_floor'] <= FLOOR_RATIO_AT_LIMIT)
    stats['passed'] = bool(stats['refined_ok'] and stats['refine_improves'] and stats['no_collision']
                           and stats['smooth_ok'] and len(episodes) == n_total)

    # --- 저장 ---
    paths_dir = Path(args.out_dir) / 'paths'
    paths_dir.mkdir(parents=True, exist_ok=True)
    out = {
        'scene_id': args.scene, 'mode': args.mode, 'smooth': args.smooth,
        'params': {'r_b': args.r_b, 'refine_radius': args.refine_radius, 'cell_m': args.cell_m,
                   'astar_cell_m': args.cell_m * args.downsample_factor, 'h_nav': args.h_nav,
                   'refine_mode': args.refine_mode, 'downsample_mode': args.downsample_mode,
                   'smooth_step': args.smooth_step, 'floor_z': floor_z,
                   'h_nav_ratio': args.h_nav_ratio,
                   'waypoint_spacing_m': args.waypoint_spacing_m},
        'stats': stats, 'passed': stats['passed'],
        'episodes': [{
            'episode_id': e['episode'], 'mode': args.mode, 'smooth': args.smooth,
            'h_b': e['planned']['h_b'], 'pitch_deg': e['pitch_deg'], 'floor_z': floor_z,
            'start': e['planned']['start'], 'goal': e['planned']['goal'],
            'waypoints_astar': e['planned']['waypoints_astar'].tolist(),
            'waypoints_refined': e['planned']['waypoints_refined'].tolist(),
            'waypoints_thinned': e['planned']['waypoints_thinned'].tolist(),
            'trajectory': e['planned']['trajectory'].tolist(),
            'path_length_m': e['planned']['check_trajectory']['length_m'],
            'check_refined': e['planned']['check_refined'],
            'check_trajectory': e['planned']['check_trajectory'],
            'gt_compare': e['compare'],
        } for e in episodes],
    }
    json_path = paths_dir / f'{args.scene}.json'
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'  paths -> {json_path}')

    # --- 리포트 ---
    log_dir = Path(args.log_dir) / SCRIPT_NAME / args.scene
    navigable = truncate_navigable(episodes[0]['planned']['esdf'], args.r_b) & compute_scan_coverage_mask(occupancy)
    states = render_states(navigable, origin, args.cell_m, episodes, log_dir)
    worst = render_worst_episodes(navigable, origin, args.cell_m, episodes, log_dir)
    body = (f'<h3>GT vs 기준선 ⓒ vs 우리 경로 ({len(episodes)} 에피소드, 같은 grid)</h3>'
            '<p class="note">‹ ›로 넘긴다. <b>노랑 = GT, 하늘 = 기준선 ⓒ, 초록 = 우리.</b> '
            '③에서 하늘색이 노랑에서 벌어지는 폭이 <b>루트가 완벽해도 남는 오차</b>다 — ④에서 초록이 '
            '그 폭 안에 있으면 더 줄일 여지가 없다. ⑤는 GT와 우리만 겹친 것으로, 포개지면 같은 루트다. '
            '배경 navigable은 ep0의 h_b 기준이므로 다른 에피소드의 경로가 살짝 벗어나 보일 수 있다.</p>'
            + blink_widget_html('paths', states, title=f'{args.scene} {args.mode}/{args.smooth}')
            + reference_button_html('참고 — navigable 배경', [('navigable (ep0 h_b)', states[0][1])])
            + '<h3>기준선 대비 배수가 큰 에피소드 — 어디서 벌어지는가</h3>'
            '<p class="note">겹쳐보기로는 개별 이탈이 안 보인다. 배수 상위 3개를 따로 본다.</p>'
            + ''.join(
                f'<h4>ep {ep} — chamfer {c["chamfer_m"]:.3f} / 기준선 {c["floor"]["chamfer_m"]:.3f} '
                f'= {c["chamfer_over_floor"]:.2f}배 '
                f'({"같은 루트" if c["same_route"] else "<b>다른 루트</b>"}, '
                f'위상 {c["homotopy_area_m2"]:.3f} m², cost비 {c["astar_gap"]["cost_ratio"]:.3f})</h4>'
                + blink_widget_html(f'worst{ep}', st, title=f'ep {ep}')
                for ep, c, st in worst))
    report = save_gallery(log_dir, 'report.html',
                          f'03_sample_gt_paths — {args.scene} ({args.mode}/{args.smooth})',
                          build_summary(args, stats, episodes), body)
    print(f'  report html -> {report}')
    print(f'  => {"PASS" if stats["passed"] else "FAIL"} '
          f'(성공 {stats["n_ok"]}/{n_total}, 같은루트 {stats["n_same_route"]}/{stats["n_ok"]}, '
          f'chamfer {stats["chamfer_median_m"]:.3f} Fréchet {stats["frechet_median_m"]:.3f} m)')
    return 0 if stats['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
