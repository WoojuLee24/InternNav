"""#04 — **우리 지도가 habitat 지도와 같은 판정을 내리나, 그리고 그 위에서 정답이 성립하나.**

## 왜 이걸 하나
학습 중에 정답 경로를 즉석에서 다시 그리려면 **habitat 없이 도는 지도**가 필요하다(Isaac/VLN-PE도
같은 지도를 써야 한다). #03이 **데이터셋이 쓴 지도**를 기준으로 세웠으니, 이제 우리가 3D 스캔으로 만든
지도가 그 기준과 같은 판정을 내리는지 본다. 그리고 그 지도 위에 정답 경로를 올려 여전히 성립하는지 본다.

## 우리 지도는 habitat과 **같은 조건**으로 만든다 — 자유 파라미터 0개
| 파라미터 | 값 | 출처 (배포 navmesh 직독) |
|---|---|---|
| `h_nav` 밴드 하단 | **0.20** | `agent_max_climb` — habitat은 이 높이까지 밟고 넘어간다 |
| `h_obs` 밴드 상단 | **1.50** | `agent_height` |
| `cell` 격자 | **0.05** | `cell_size` |
| `r_b` 로봇 반경 | **0.10** | `agent_radius` |

밴드 높이 스윕은 **없앴다**: 8조합(h_nav 0.10~0.25 × h_obs 1.25/1.50)의 일치율 차이가 **0.08~0.23%p**뿐
이어서 판별력이 없음이 이미 증명됐다. 값을 고르는 대신 navmesh에서 읽는다.

## 두 경로 — 우리 지도 위에 올린다
| | 경로 | 색 | 무엇인가 |
|---|---|---|---|
| ⓐ | **원본 정답** | 노랑 | 저장된 프레임별 위치. 학습이 먹는 라벨 |
| ⓑ | **pathfollower로 이은 경로** | 초록 | waypoint 사이를 **0.25 m 전진 / 15° 회전 이산 액션**으로 |

ⓑ가 핵심이다 — VLN-CE가 저장한 것이 말 그대로 *"a pre-computed shortest path following the waypoints
via low-level actions"*이므로 **pathfollower가 곧 데이터셋 생성기 그 자체**다. #03이 쓴 `find_path`
폴리라인은 측지 최단선이라 median 12.9 / 56.4 cm 어긋났다.

## 게이트 6개
| 게이트 | 묻는 것 | 기준 |
|---|---|---|
| A 좌표 정합 | 우리 격자와 habitat 좌표가 맞물리나 | GT가 래스터 마스크 안 ≥0.98. **실패하면 이후 수치가 전부 무의미** |
| B follower 도달 | follower가 다음 waypoint에 닿나 | leg 끝점 ≤0.5 m **100%** (VLN-CE 논문 자신의 기준) |
| C 불일치 분해 | 밴드 맵의 거짓 승인이 원인으로 설명되나 | `F + R`이 `ours & ~ref`의 **≥90%** |
| G1 | recast v2에서 F(바닥 없음)가 사라졌나 | F < FA의 5% **그리고** < 200셀 |
| G2 | recast v2에서 GT가 살아남나 | 이탈 깊이 ≤ 1칸 (0.05 m — navmesh 래스터 자신이 통과한 기준) |
| G3 | recast v2 일치율 ≥ 밴드 기준선 | 두 씬 모두 |

## 왜 다른가 → 어떻게 맞추나
`derive_obstacle_2d`는 **"바닥 위 밴드에 장애물이 없나"만 묻고 "발 디딜 바닥이 있나"를 묻지 않는다.**
그래서 다층 집에서 다른 층 허공이 통행 가능이 된다 — 거짓 승인을 **F 바닥없음 / R 경계 1칸 / S 나머지**로
배타 분류하면 s8은 F가 66.6%, 17DRP는 R(격자 반올림, 불가침)이 80%다.
→ 해법은 **질문을 바꾸는 것**: `recast_like.py`가 recast의 span 논리를 흉내 내
칸마다 바닥을 찾는다(v2 = +despike +시작점 연결성). F 판정 두께는 `FLOOR_DIAG_M`(0.20)으로 **고정** —
recast 창(0.5)과 묶으면 설명률이 튜닝에 흔들려 지표가 못 된다.
(이전 판의 `& floor_exists` 후보·두께 스윕·게이트 D는 v2로 대체돼 삭제 — 기록은 `reports.md` #04.)

## 좌표 규약
우리 격자는 mesh Z-up `arr[iy, ix]`, navmesh topdown은 habitat `tv[iz, ix]`이고
**mesh y = −habitat z라 축 방향이 반대다.** 두 지도를 반드시 **같은 높이 `floor_y`**에서 자른다
(`reference_path` y의 중앙값 = mesh z == habitat y). 전에 `floor_z + r_b`에서 자르는 실수를 했다.

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/04_calibrate_map.py --scenes 17DRP5sb8fy,s8pcmisQ38h --episodes 14 --out_dir logs/embodiment_augment/s2_mapcal

self-check(씬·habitat 불필요): /usr/bin/python scripts/dataset_converters/3dloader_vlnce/04_calibrate_map.py --selfcheck
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))
from esdf_utils import compute_esdf_2d, compute_scan_coverage_mask, derive_obstacle_2d  # noqa: E402
from esdf_utils import resample_by_arclength, truncate_navigable  # noqa: E402
from viz_utils import GT_COLOR, OURS_COLOR, floorplan_canvas, save_gallery  # noqa: E402
from corridor_utils import dist_to_path_field  # noqa: E402
from pixel_goal_utils import rig_height_m  # noqa: E402
from vlnce_align import build_rawdata_index  # noqa: E402
from waypoint_spine import habitat_to_mesh  # noqa: E402
import recast_like  # noqa: E402
_vng = importlib.import_module('03_verify_navmesh_gt')  # noqa: E402
load_poses, sf2mesh, topdown = _vng.load_poses, _vng.sf2mesh, _vng.topdown  # noqa: E402
match_paths = _vng.match_paths  # noqa: E402
_p03 = importlib.import_module('03_sample_gt_paths')  # noqa: E402

# 전부 배포 navmesh 직독값이다 — 우리가 고른 값이 하나도 없다 (docstring 표 참고).
BASELINE_R_B, H_NAV_M, H_OBS_M = 0.10, 0.20, 1.50
COORD_TOL = 0.98          # 게이트 A: 알려진 navigable 점의 마스크 일치율 하한
EXPLAIN_TOL = 0.90        # 게이트 C: F+R이 거짓 승인의 이 비율 이상을 설명해야 한다
LEG_TOL_M = 0.5           # 게이트 B: VLN-CE 논문의 navigability 기준
BAND_M = 2.0              # 비교 반경 — 집 전체로 넓히면 다층·미스캔 영역이 숫자를 흐린다
FLOOR_DIAG_M = 0.20       # F 버킷 **진단** 두께 = `agent_max_climb`. recast 창(0.5)과 분리해 고정한다 —
                          # 같이 움직이면 "설명률"이 튜닝에 따라 흔들려 지표가 못 된다.
FWD_M, TURN_DEG = 0.25, 15.0     # VLN-CE 이산 액션. habitat 기본값은 15°가 아니라 10°다
GOAL_RADIUS_M = 0.25      # follower 도달 반경. 데이터셋 생성값은 알 수 없다(한계에 명시)
SPLIT_TOL_M = 0.20        # 이보다 멀면 "경로가 갈라졌다"로 본다 — 재현 오차 분포가 두 덩어리라 median이
                          # 실체를 가린다(#03에서 길이비에 속은 것과 같은 패턴)
WP_COLOR, F_COLOR = (0, 220, 255), (255, 0, 255)


def scene_grid(scene, esdf_dir):
    """occ npz 로드. -> dict(occ, origin, cell, coverage)

    `EmbodimentAugmenter._scene_ctx`의 격자 부분만 떼온 것이다 — #04는 렌더러가 필요 없다.
    """
    a = _p03.build_argparser().parse_args([])
    a.scene, a.dataset, a.esdf_dir = scene, 'vln_n1', esdf_dir
    got = _p03.load_esdf(a)
    assert got is not None, f'{esdf_dir}/{scene}.npz 없음 — gs_vlnpe/02_build_freemap_esdf.py를 먼저 돌려라'
    occ, origin, _floor_z = got
    return dict(occ=occ, origin=np.asarray(origin, dtype=np.float64),
                cell=float(a.cell_m), coverage=compute_scan_coverage_mask(occ))


def navmesh_to_our_grid(pf, height, origin, cell, shape, mpp=None):
    """r_b navmesh를 **우리 격자**에 래스터화. -> (Ny,Nx) bool

    `tv[iz, ix]`(habitat)를 우리 셀 중심으로 샘플링한다. mesh y = −habitat z라 y축이 뒤집힌다 —
    인덱스를 직접 계산해 그 뒤집힘을 명시적으로 처리한다(암묵적 `[::-1]`은 off-by-one을 숨긴다).
    """
    mpp = float(cell if mpp is None else mpp)
    tv, org = topdown(pf, mpp, height)
    ny, nx = shape
    mx = origin[0] + (np.arange(nx) + 0.5) * cell          # 우리 셀 중심 (mesh x)
    my = origin[1] + (np.arange(ny) + 0.5) * cell          # 우리 셀 중심 (mesh y)
    jx = np.floor((mx - org[0]) / mpp).astype(int)          # habitat x
    jz = np.floor((-my - org[1]) / mpp).astype(int)         # habitat z = −mesh y
    okx, okz = (jx >= 0) & (jx < tv.shape[1]), (jz >= 0) & (jz < tv.shape[0])
    out = np.zeros(shape, dtype=bool)
    out[np.ix_(okz, okx)] = tv[np.ix_(jz[okz], jx[okx])]
    return out


def mask_of(kind, grid, floor_y, r_b, pf=None):
    """지도 종류별 `r_b` navigable 마스크. -> (Ny,Nx) bool

    `floor_y`는 **mesh z == habitat y**라 두 분기가 같은 값을 쓴다 — navmesh는 반드시 밴드와 **같은
    높이**에서 잘라야 한다(전에 `floor_z + r_b`로 자르는 실수를 해서 identity에 실패했다).
    """
    if kind == 'band':
        obstacle = derive_obstacle_2d(grid['occ'], grid['origin'], floor_y, H_OBS_M,
                                      grid['cell'], H_NAV_M)
        return truncate_navigable(compute_esdf_2d(obstacle, grid['cell']), r_b) & grid['coverage']
    elif kind == 'navmesh':
        return navmesh_to_our_grid(pf, floor_y, grid['origin'], grid['cell'],
                                   grid['coverage'].shape)
    elif kind == 'recast':
        return recast_like.recast_mask(grid, floor_y, r_b)
    else:
        assert False, f'unreachable kind={kind!r}'


def floor_exists_mask(grid, floor_y, below_m=FLOOR_DIAG_M):
    """`[floor_y − below_m, floor_y + h_nav]`에 점유가 있나 -> (Ny,Nx) bool = **이 높이에 바닥이 있다**.

    `derive_obstacle_2d`와 같은 인덱스 산식·축 규약을 쓴다(밴드만 아래로 옮긴 것).
    """
    occ, origin, cell = grid['occ'], grid['origin'], grid['cell']
    k_lo = max(int(np.ceil((floor_y - below_m - origin[2]) / cell)), 0)
    k_hi = min(int(np.floor((floor_y + H_NAV_M - origin[2]) / cell)), occ.shape[2] - 1)
    if k_lo > k_hi:
        return np.zeros(occ.shape[1::-1], dtype=bool)
    return occ[:, :, k_lo:k_hi + 1].any(axis=2).T


def path_verdict(mask, xy, origin, cell):
    """경로 점마다 "이 마스크 안인가". -> (N,) bool. 셀 1/5 간격으로 촘촘히 찍는다.

    격자 밖 점은 False(=밖)로 본다 — 판정을 못 하는 것을 통과로 세면 안 된다.
    """
    d = resample_by_arclength(np.asarray(xy, dtype=np.float64)[:, :2], step_m=cell / 5.0)
    ij = np.floor((d - np.asarray(origin)[:2]) / cell).astype(int)
    ny, nx = mask.shape
    ins = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    good = np.zeros(len(d), dtype=bool)
    good[ins] = mask[ij[ins][:, 1], ij[ins][:, 0]]
    return good


def path_outside(mask, xy, origin, cell):
    """경로가 마스크 밖으로 나간 점 수 + 최대 깊이[m]. -> dict(out, n, depth_max)

    깊이가 있어야 등급이 매겨진다 — 1셀이면 래스터 양자화지만 여러 셀이면 진짜로 못 지나간다.
    """
    from scipy.ndimage import distance_transform_edt
    g = path_verdict(mask, xy, origin, cell)
    if g.all():
        return dict(out=0, n=int(len(g)), depth_max=0.0)
    d = resample_by_arclength(np.asarray(xy, dtype=np.float64)[:, :2], step_m=cell / 5.0)
    ij = np.floor((d - np.asarray(origin)[:2]) / cell).astype(int)
    ny, nx = mask.shape
    ins = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    field = distance_transform_edt(~mask) * cell
    v = np.zeros(len(d))
    v[ins] = field[ij[ins][:, 1], ij[ins][:, 0]]
    return dict(out=int((~g).sum()), n=int(len(g)), depth_max=float(v[~g].max()))


def band_of(gt_xy, origin, cell, shape, band_m=BAND_M):
    """GT 주변 `band_m` 안의 셀. -> (Ny,Nx) bool. 전역 비교는 다층·미스캔 영역 때문에 흐려진다."""
    field = dist_to_path_field(np.asarray(gt_xy)[:, :2], origin, cell, shape)
    return np.isfinite(field) & (field <= band_m)


def compare(ours, ref, band):
    """두 지도의 점 단위 비교. -> dict(agree, iou, ours_free_only, ref_free_only, band)

    이름 그대로 읽어라: `ours_free_only` = 우리만 **갈 수 있음**(거짓 승인),
    `ref_free_only` = navmesh만 갈 수 있음 = **우리만 막음**(거짓 기각). 예전에 이 둘을 뒤집어 출력해
    "우리 맵이 과대 장애물"이라고 반대로 읽었다.
    """
    a, b = ours & band, ref & band
    inter, union, n = int((a & b).sum()), int((a | b).sum()), int(band.sum())
    return dict(agree=float(((a == b) & band).sum()) / n if n else float('nan'),
                iou=inter / union if union else float('nan'),
                ours_free_only=int((a & ~b).sum()), ref_free_only=int((~a & b).sum()),
                band=n)


def decompose(ours, ref, band, floor_ok):
    """거짓 승인(`ours & ~ref`)을 원인별로 쪼갠다. -> (dict, label (Ny,Nx) uint8)

    F 바닥없음 → R 경계 1칸 → S 나머지 순서로 **배타** 분류하므로 세 개의 합이 전체다.
    """
    from scipy.ndimage import distance_transform_cdt
    fa = ours & ~ref & band
    depth = distance_transform_cdt(~ref, metric='chessboard')
    F = fa & ~floor_ok
    R = fa & ~F & (depth == 1)
    S = fa & ~F & ~R
    lab = np.zeros(ours.shape, dtype=np.uint8)
    lab[F], lab[R], lab[S] = 1, 2, 3
    n = int(fa.sum())
    nf, nr, ns = int(F.sum()), int(R.sum()), int(S.sum())
    return dict(total=n, F=nf, R=nr, S=ns,
                explained=(nf + nr) / n if n else 1.0), lab


def follower_path(pf, raw, goal_radius=GOAL_RADIUS_M, fwd_m=FWD_M, turn_deg=TURN_DEG):
    """waypoint를 **이산 액션**으로 이은 경로. -> (poses (N,3) habitat, reach (L,))

    ⚠️ `move_filter_fn`을 넣지 않으면 롤아웃이 **벽을 통과한다**(실측 5점) — `Simulator`가 있을 때만
    habitat이 자동으로 채워준다. `SceneGraph`는 follower가 사는 동안 살려둬야 한다.
    """
    import habitat_sim
    from habitat_sim.agent import Agent, AgentConfiguration, AgentState
    spec, act = habitat_sim.ActionSpec, habitat_sim.ActuationSpec
    sg = habitat_sim.SceneGraph()
    agent = Agent(sg.get_root_node().create_child(), AgentConfiguration(action_space={
        0: spec('stop'), 1: spec('move_forward', act(amount=fwd_m)),
        2: spec('turn_left', act(amount=turn_deg)), 3: spec('turn_right', act(amount=turn_deg))}))
    agent.controls.move_filter_fn = pf.try_step
    fol = habitat_sim.GreedyGeodesicFollower(pf, agent, goal_radius, stop_key=0,
                                             forward_key=1, left_key=2, right_key=3)
    agent.set_state(AgentState(position=np.asarray(raw['start_position'], dtype=np.float32),
                               rotation=np.asarray(raw['start_rotation'], dtype=np.float32)))
    poses, reach = [np.asarray(agent.state.position, dtype=np.float64)], []
    for goal in np.asarray(raw['reference_path'], dtype=np.float64)[1:]:
        try:
            acts = fol.find_path(goal)
        except Exception:
            reach.append(float('inf')); continue      # 실패한 leg는 이어붙이지 않는다
        for a in acts:
            if not a:
                continue
            agent.act(a)
            poses.append(np.asarray(agent.state.position, dtype=np.float64))
        reach.append(float(np.linalg.norm(np.asarray(agent.state.position) - goal)))
    del fol, agent, sg
    return np.stack(poses), np.array(reach)


def map_panel(mask, origin, cell, gt_xy, fol_xy, wps_xy, out_dir, name):
    """지도 한 장에 ⓐ노랑·ⓑ초록·waypoint를 겹친다. ⓑ를 먼저 깔고 ⓐ로 덮는다."""
    base, draw, emit = floorplan_canvas(mask, origin, cell, out_dir)
    img = base.copy()
    if len(fol_xy):
        draw(img, resample_by_arclength(fol_xy, step_m=cell), OURS_COLOR, 2)
    draw(img, resample_by_arclength(gt_xy, step_m=cell), GT_COLOR, 1)
    draw(img, wps_xy, WP_COLOR, 4)
    return emit(img, name)


def diff_panel(ours, ref, lab, band, origin, cell, gt_xy, out_dir, name):
    """일치=초록 · 우리만 통행가능=파랑 · 우리만 막힘=빨강 · **F 바닥없음=자홍**. GT를 노랑으로 겹친다.

    ⚠️ `band` **밖은 어둡게 깐다** — 표의 숫자가 band 안에서만 세어지므로, 전체를 똑같이 칠하면
    그림과 숫자가 어긋나 보인다.
    """
    import cv2
    from geometry_utils import save_jpg
    h, w = ours.shape
    img = np.full((h, w, 3), 38, dtype=np.uint8)
    img[(ours & ref)] = (60, 150, 60)
    img[(~ours & ref)] = (200, 80, 40)       # navmesh는 갈 수 있는데 우리는 막음 = 거짓 기각
    img[(ours & ~ref)] = (40, 80, 220)       # 우리만 갈 수 있음 = 거짓 승인
    img[lab == 1] = F_COLOR                  # 그중 "이 높이에 바닥이 없다"로 설명되는 것
    img[~band] = (img[~band] * 0.30).astype(np.uint8)
    # 화면은 y가 아래로 증가. `[::-1]`은 음수 stride 뷰라 cv2.circle이 거부하므로 연속 배열로 복사한다.
    img = np.ascontiguousarray(img[::-1])
    _b, draw, _e = floorplan_canvas(ours, origin, cell, out_dir)
    draw(img, resample_by_arclength(np.asarray(gt_xy)[:, :2], step_m=cell), GT_COLOR, 1)
    f = max(1, int(np.ceil(700 / max(h, w))))
    return save_jpg(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST),
                    Path(out_dir) / name)


def collect_episodes(args, scene, raw_idx, h_rig):
    """에피소드별 GT 궤적(mesh xy) + floor_y + raw. 계단은 제외. -> (list, skip dict)"""
    tasks = [json.loads(l) for l in open(Path(args.data_root) / scene / 'meta' / 'episodes.jsonl')]
    eps, skip = [], {'raw': 0, 'stairs': 0}
    for ep in range(min(args.episodes, len(tasks))):
        raw = raw_idx.get((scene, tasks[ep]['tasks'][0].strip()))
        if raw is None:
            skip['raw'] += 1; continue
        T = sf2mesh(raw)
        gt = np.stack([(T @ p)[:3, 3] for p in load_poses(args.data_root, scene, args.rig, ep)])
        spread = float(gt[:, 2].max() - gt[:, 2].min())
        if spread > args.max_z_spread:
            skip['stairs'] += 1
            print(f'[S2] ep{ep}: skip (높이 변동 {spread:.2f} > {args.max_z_spread} = 계단)'); continue
        wps = np.asarray(raw['reference_path'], dtype=np.float64)
        # mesh z == habitat y라 `reference_path` y를 그대로 바닥 높이로 쓴다 (rig 높이 역산보다 직접적).
        eps.append(dict(ep=ep, gt_xy=gt[:, :2], floor_y=float(np.median(wps[:, 1])), raw=raw,
                        wps_xy=habitat_to_mesh(wps)[:, :2]))
    return eps, skip


def run_scene(args, scene, raw_idx, h_rig, out):
    """씬 하나를 처리한다. -> dict(에피소드 행 + 씬 합계 + 게이트)."""
    from habitat_sim.nav import PathFinder
    grid = scene_grid(scene, args.esdf_dir)
    f = Path(args.navmesh_cache) / f'{scene}_rb{BASELINE_R_B:.2f}.navmesh'
    assert f.exists(), f'{f} 없음 — #03(03_verify_navmesh_gt.py)을 먼저 돌려 캐시를 만들어라'
    pf = PathFinder(); pf.load_nav_mesh(str(f))
    eps, skip = collect_episodes(args, scene, raw_idx, h_rig)
    print(f'\n[S2] === {scene}: 격자 {grid["coverage"].shape} · cell {grid["cell"]:.4f} · '
          f'에피소드 {len(eps)}개 (계단 {skip["stairs"]} · raw {skip["raw"]} 제외)')

    acc = dict(cmp=[], dec=[], gt=[], fol=[], gt_ref=[], fol_ref=[], mm=[],
               cmp3=[], dec3=[], gt3=[], fol3=[], abl=[], reach_ok=0, reach_n=0)
    coord_hit, rows = float('nan'), []
    for i, e in enumerate(eps):
        fy = e['floor_y']
        ours = mask_of('band', grid, fy, BASELINE_R_B)
        ref = mask_of('navmesh', grid, fy, BASELINE_R_B, pf)
        # 진단용 두께는 고정(FLOOR_DIAG_M), 수정용 두께는 스윕 대상 — 섞으면 설명률이 튜닝에 흔들린다.
        floor_diag = floor_exists_mask(grid, fy, FLOOR_DIAG_M)
        band = band_of(e['gt_xy'], grid['origin'], grid['cell'], ours.shape, args.band_m)

        # 게이트 A: #03이 navmesh 위에 있음을 증명한 GT가 래스터 마스크 안인가. 첫 에피소드로 판정.
        if i == 0:
            o = path_outside(ref, e['gt_xy'], grid['origin'], grid['cell'])
            coord_hit = 1.0 - o['out'] / max(o['n'], 1)
            print(f'[S2] A 좌표 정합 {coord_hit:.4f} (밖 {o["out"]}/{o["n"]})')

        c = compare(ours, ref, band)
        dec, lab = decompose(ours, ref, band, floor_diag)
        # recast-like 후보: 칸마다 바닥 + despike + 시작점 연결성 (근거·단계는 recast_like.py docstring)
        rec = recast_like.recast_mask(grid, fy, BASELINE_R_B, start_xy=e['gt_xy'][0])
        c3 = compare(rec, ref, band)
        dec3, lab3 = decompose(rec, ref, band, floor_diag)
        g3 = path_outside(rec, e['gt_xy'], grid['origin'], grid['cell'])
        abl = {anm: compare(recast_like.recast_mask(grid, fy, BASELINE_R_B, ledge=ld, region=rg,
                                                    despike=ds), ref, band)
               for anm, ld, rg, ds in (('① 바닥+머리공간만', False, False, False),
                                       ('② +단차(ledge)+섬 제거 (v1)', True, True, False))}
        abl['③ +despike+시작점 연결성 (v2)'] = c3

        fol_hab, reach = follower_path(pf, e['raw'], args.goal_radius_m)
        fol_xy = habitat_to_mesh(fol_hab)[:, :2]
        f3 = path_outside(rec, fol_xy, grid['origin'], grid['cell'])   # 승격 기준용 (follower도 살아남나)
        g_ours = path_outside(ours, e['gt_xy'], grid['origin'], grid['cell'])
        f_ours = path_outside(ours, fol_xy, grid['origin'], grid['cell'])
        g_ref = path_outside(ref, e['gt_xy'], grid['origin'], grid['cell'])
        f_ref = path_outside(ref, fol_xy, grid['origin'], grid['cell'])
        m = match_paths(fol_xy, e['gt_xy'])

        figs = None
        if i < args.figs:
            nm = f'{scene}_ep{e["ep"]:03d}'
            map_panel(ours, grid['origin'], grid['cell'], e['gt_xy'], fol_xy, e['wps_xy'],
                      out, f'{nm}_ours.jpg')
            map_panel(ref, grid['origin'], grid['cell'], e['gt_xy'], fol_xy, e['wps_xy'],
                      out, f'{nm}_ref.jpg')
            diff_panel(ours, ref, lab, band, grid['origin'], grid['cell'], e['gt_xy'], out,
                       f'{nm}_diff.jpg')
            diff_panel(rec, ref, lab3, band, grid['origin'], grid['cell'], e['gt_xy'], out,
                       f'{nm}_recast_diff.jpg')
            figs = nm

        acc['cmp'].append(c); acc['dec'].append(dec)
        acc['cmp3'].append(c3); acc['dec3'].append(dec3); acc['gt3'].append(g3)
        acc['fol3'].append(f3)
        acc['abl'].append(abl)
        acc['gt'].append(g_ours); acc['fol'].append(f_ours)
        acc['gt_ref'].append(g_ref); acc['fol_ref'].append(f_ref)
        acc['reach_ok'] += int((reach <= LEG_TOL_M).sum()); acc['reach_n'] += len(reach)
        if m is not None:
            acc['mm'].append(m)
        rows.append(dict(ep=e['ep'], fy=fy, c=c, c3=c3, dec=dec, dec3=dec3, g=g_ours,
                         f=f_ours, m=m, reach=reach, fig=figs))
        print(f'[S2] ep{e["ep"]}: 밴드 {100 * c["agree"]:.2f}% FA {c["ours_free_only"]} '
              f'(F {dec["F"]} R {dec["R"]} S {dec["S"]}) → recast {100 * c3["agree"]:.2f}% '
              f'FA {c3["ours_free_only"]} FR {c3["ref_free_only"]} GT밖 {g3["out"]} · '
              f'ⓐ밖 {g_ours["out"]}/{g_ours["n"]} ⓑ밖 {f_ours["out"]} · '
              + (f'재현 {m["mean"] * 100:.1f} cm' if m else '재현 실패')
              + f' · reach max {reach.max():.3f}')

    agg = dict(
        agree=float(np.mean([c['agree'] for c in acc['cmp']])),
        iou=float(np.mean([c['iou'] for c in acc['cmp']])),
        fa=sum(c['ours_free_only'] for c in acc['cmp']),
        fr=sum(c['ref_free_only'] for c in acc['cmp']),
        F=sum(d['F'] for d in acc['dec']), R=sum(d['R'] for d in acc['dec']),
        S=sum(d['S'] for d in acc['dec']), tot=sum(d['total'] for d in acc['dec']))
    agg['explained'] = (agg['F'] + agg['R']) / agg['tot'] if agg['tot'] else 1.0
    agg.update(agree3=float(np.mean([c['agree'] for c in acc['cmp3']])),
               iou3=float(np.mean([c['iou'] for c in acc['cmp3']])),
               fa3=sum(c['ours_free_only'] for c in acc['cmp3']),
               fr3=sum(c['ref_free_only'] for c in acc['cmp3']),
               F3=sum(d['F'] for d in acc['dec3']), R3=sum(d['R'] for d in acc['dec3']),
               S3=sum(d['S'] for d in acc['dec3']), tot3=sum(d['total'] for d in acc['dec3']),
               g3_out=sum(x['out'] for x in acc['gt3']), g3_n=sum(x['n'] for x in acc['gt3']),
               g3_depth=max((x['depth_max'] for x in acc['gt3']), default=0.0),
               f3_out=sum(x['out'] for x in acc['fol3']),
               f3_depth=max((x['depth_max'] for x in acc['fol3']), default=0.0))
    agg['abl'] = {anm: dict(agree=float(np.mean([a[anm]['agree'] for a in acc['abl']])),
                            fa=sum(a[anm]['ours_free_only'] for a in acc['abl']),
                            fr=sum(a[anm]['ref_free_only'] for a in acc['abl']))
                  for anm in acc['abl'][0]} if acc['abl'] else {}
    # recast-like 게이트(G1~G3)는 기준을 데이터 보기 전에 고정했다 — 계획서.
    gate = dict(A=coord_hit >= COORD_TOL,
                B=acc['reach_ok'] == acc['reach_n'] and acc['reach_n'] > 0,
                C=agg['explained'] >= EXPLAIN_TOL,
                G1=agg['F3'] < 0.05 * max(agg['tot3'], 1) and agg['F3'] < 200,
                G2=agg['g3_depth'] <= 0.051,
                G3=agg['agree3'] >= agg['agree'])
    print(f'[S2] 합계 밴드 {100 * agg["agree"]:.2f}% FA {agg["fa"]} FR {agg["fr"]} '
          f'(F {agg["F"]} R {agg["R"]} S {agg["S"]} 설명 {100 * agg["explained"]:.1f}%) · '
          f'B leg {acc["reach_ok"]}/{acc["reach_n"]}')
    print(f'[S2] recast v2 {100 * agg["agree3"]:.2f}% FA {agg["fa3"]} '
          f'(F {agg["F3"]} R {agg["R3"]} S {agg["S3"]}) FR {agg["fr3"]} · '
          f'GT밖 {agg["g3_out"]}/{agg["g3_n"]} 깊이 {agg["g3_depth"]:.3f} m · '
          f'follower밖 {agg["f3_out"]} 깊이 {agg["f3_depth"]:.3f} m')
    for anm, s in agg['abl'].items():
        print(f'[S2]   {anm}: 일치 {100 * s["agree"]:.2f}% · '
              f'거짓승인 {s["fa"]} · 거짓기각 {s["fr"]}')
    return dict(scene=scene, grid=grid, eps=eps, rows=rows, acc=acc, agg=agg, skip=skip,
                coord_hit=coord_hit, gate=gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='17DRP5sb8fy,s8pcmisQ38h')
    ap.add_argument('--rig', default='125cm_0deg')
    ap.add_argument('--episodes', type=int, default=14, help='씬당 검증할 에피소드 수')
    ap.add_argument('--figs', type=int, default=6, help='씬당 그림을 남길 에피소드 수')
    ap.add_argument('--band_m', type=float, default=BAND_M, help='두 지도를 비교할 GT 주변 반경[m]')
    ap.add_argument('--goal_radius_m', type=float, default=GOAL_RADIUS_M,
                    help='follower 도달 반경[m] — **데이터셋 생성값은 알 수 없다**(한계 참고)')
    ap.add_argument('--max_z_spread', type=float, default=0.30,
                    help='GT 높이 변동이 이보다 크면 계단으로 보고 제외 — #03/#05와 같은 값')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--navmesh_cache', default='data/embodiment_aug/navmesh',
                    help='#03이 저장한 r_b별 기준 navmesh')
    ap.add_argument('--esdf_dir', default='scripts/dataset_converters/gs_vlnpe/logs/esdf')
    ap.add_argument('--out_dir', default='logs/embodiment_augment/s2_mapcal')
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args()
    if args.selfcheck:
        return _selfcheck()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    h_rig = rig_height_m(args.rig)
    raw_idx = build_rawdata_index(Path(args.raw_root))
    print(f'[S2] raw index {len(raw_idx)} 에피소드 · rig {args.rig} · '
          f'r_b={BASELINE_R_B} · 밴드 {H_NAV_M}~{H_OBS_M} m (전부 navmesh 직독값)')

    rows = [run_scene(args, s, raw_idx, h_rig, out) for s in args.scenes.split(',') if s]
    summary, body = summary_html(rows, args), body_html(rows)
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    # body.html을 쓰면 발행기가 **본문이 참조한 그림만** 인라인한다(갤러리 모드는 옛 jpg까지 긁어온다).
    (out / 'body.html').write_text(body, encoding='utf-8')
    save_gallery(out, 'report.html', '#04 — 우리 지도가 habitat 지도와 같은 판정을 내리나',
                 summary, body, eyebrow='embodiment augmentation · #04')
    print(f'\n[S2] report -> {out}/report.html')
    print('[S2] 게이트: ' + ' · '.join(
        r['scene'] + ' ' + ''.join(f'{k}{"o" if v else "X"}' for k, v in r['gate'].items())
        for r in rows))
    return 0 if all(all(r['gate'].values()) for r in rows) else 1


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------
def summary_html(rows, args):
    scope = ' · '.join(f'<b>{r["scene"]}</b> {len(r["eps"])}개'
                       + (f'(계단 {r["skip"]["stairs"]} · raw {r["skip"]["raw"]} 제외)'
                          if any(r['skip'].values()) else '') for r in rows)
    h = [f'<p>씬·에피소드: {scope} · rig {args.rig} · <b>r_b = {BASELINE_R_B} · 밴드 '
         f'{H_NAV_M}~{H_OBS_M} m 고정</b>(전부 배포 navmesh 직독값 — 자유 파라미터 0개) · '
         f'두 지도를 <b>같은 높이 floor_y</b>에서 자르고 <b>GT 주변 {args.band_m} m 안</b>에서만 '
         '비교한다. #03이 데이터셋의 지도를 기준으로 세웠고, #05가 이 지도 위에서 r_b를 흔든다.</p>']

    h.append('<h3>게이트</h3><table border=1 cellpadding=6><tr><th>게이트</th><th>묻는 것</th>'
             + ''.join(f'<th>{r["scene"]}</th>' for r in rows) + '</tr>')
    G = [('A', f'우리 격자와 habitat 좌표가 맞물리나 (≥{COORD_TOL})',
          lambda r: f'{r["coord_hit"]:.4f}'),
         ('B', f'follower가 다음 waypoint ≤{LEG_TOL_M} m에 닿나 (100%)',
          lambda r: f'{r["acc"]["reach_ok"]}/{r["acc"]["reach_n"]}'),
         ('C', f'밴드 맵의 거짓 승인이 F+R로 설명되나 (≥{100 * EXPLAIN_TOL:.0f}%)',
          lambda r: f'{100 * r["agg"]["explained"]:.1f}%'),
         ('G1', 'recast v2에서 F(바닥 없음)가 사라졌나 (<5% 그리고 <200셀)',
          lambda r: f'F {r["agg"]["F3"]} / {r["agg"]["tot3"]}'),
         ('G2', 'recast v2에서 GT 이탈 깊이 ≤ 1칸 (0.05 m)',
          lambda r: f'{r["agg"]["g3_out"]}점 · 깊이 {r["agg"]["g3_depth"]:.3f} m'),
         ('G3', 'recast v2 일치율 ≥ 밴드 기준선',
          lambda r: f'{100 * r["agg"]["agree3"]:.2f}% vs {100 * r["agg"]["agree"]:.2f}%')]
    for k, q, fn in G:
        h.append(f'<tr><td><b>{k}</b></td><td>{q}</td>'
                 + ''.join(f'<td>{fn(r)} — <b>{"PASS" if r["gate"][k] else "FAIL"}</b></td>'
                           for r in rows) + '</tr>')
    h.append('</table><p><b>A가 실패하면 이후 수치는 전부 무의미하다.</b> 게이트 기준은 데이터를 보기 '
             '전에 정했고, 실패를 통과로 만들지 않았다.</p>')

    h.append('<h3>실험결과 1 — 두 지도가 같은 판정을 내리나</h3>'
             '<table border=1 cellpadding=6><tr><th>지도</th><th>씬</th><th>habitat 일치율</th>'
             '<th>IoU</th><th>우리만 통행가능<br><small>거짓 승인 — 위험</small></th>'
             '<th>우리만 막힘<br><small>거짓 기각</small></th></tr>')
    for key, label, why in (('', '<b>밴드 투영</b>(기존 방식)', '"이 높이에 장애물이 없나"만 물음'),
                            ('3', '<b>recast-like v2</b>',
                             '칸마다 바닥 + despike + 시작점 연결성')):
        for r in rows:
            a = r['agg']
            h.append(f'<tr><td>{label}<br><small>{why}</small></td><td>{r["scene"]}</td>'
                     f'<td><b>{100 * a["agree" + key]:.2f}%</b></td>'
                     f'<td>{a["iou" + key]:.4f}</td><td><b>{a["fa" + key]}</b></td>'
                     f'<td>{a["fr" + key]}</td></tr>')
    h.append('</table>')
    h.append('<p><b>방향이 중요하다.</b> 거짓 승인은 <b>못 갈 길을 정답으로 가르치는 것</b>이라 되돌릴 수 '
             '없고, 거짓 기각은 샘플을 버리는 것뿐이다. <b>navmesh 래스터화</b>(현 default)는 같은 원본을 '
             '찍어내므로 정의상 100% — 상한으로만 참고하고 표에서 뺐다.</p>')

    h.append('<h3>실험결과 2 — 왜 다른가 (밴드 맵 거짓 승인의 원인 분해)</h3>'
             '<table border=1 cellpadding=6><tr><th>씬</th><th>거짓 승인</th>'
             '<th>F 바닥 없음</th><th>R 경계 1칸</th><th>S 나머지</th><th>F+R 설명률</th></tr>')
    for r in rows:
        a = r['agg']
        h.append(f'<tr><td><b>{r["scene"]}</b></td><td>{a["tot"]}</td>'
                 f'<td><b>{a["F"]}</b> ({100 * a["F"] / max(a["tot"], 1):.0f}%)</td>'
                 f'<td>{a["R"]} ({100 * a["R"] / max(a["tot"], 1):.0f}%)</td><td>{a["S"]}</td>'
                 f'<td><b>{100 * a["explained"]:.1f}%</b></td></tr>')
    h.append('</table>')
    h.append('<p>밴드 맵은 "이 높이 띠에 장애물이 없나"만 묻고 <b>"발 디딜 바닥이 있나"를 묻지 않는다</b>. '
             '그래서 다층 집(s8)에서는 <b>다른 층 허공(F)</b>이 거짓 승인의 3분의 2를 차지하고, 단층 집'
             '(17DRP)에서는 <b>경계 1칸(R = 격자 반올림, 어떤 방법으로도 못 없앰)</b>이 대부분이다. '
             'F 판정 두께는 <code>agent_max_climb</code> 0.20으로 <b>고정</b>했다 — recast 창과 같이 '
             '움직이면 설명률이 튜닝에 흔들려 지표가 못 된다. 바닥이 <code>floor_y</code>보다 조금 아래에 '
             '있는 이유 셋: habitat 래스터의 수직 허용치 0.5 m · 한 층 안 방별 바닥 높이 차(s8 16.5 cm, '
             '<code>.house</code> 실측) · 층 안의 계단.</p>')

    h.append('<h3>실험결과 3 — 해법: 칸마다 바닥을 찾는 recast-like v2</h3>'
             '<p>habitat(recast)의 질문을 그대로 흉내 낸다 — 배포 navmesh 표면 y를 실측하면 한 층 '
             '안에서도 2.13~3.95 m로 울퉁불퉁하고 <b>계단 위에도 navmesh가 있다</b>(칸마다 바닥이 다르고 '
             '0.2 m 단차로 연결된다는 직접 증거). <code>recast_like.py</code>, <b>habitat 불필요</b>:</p>'
             '<p>칸 (x,y)마다 — ① <code>[floor_y−0.5, +0.2]</code> 창에서 실제 <b>바닥 voxel</b>(없으면 '
             '차단) ② 머리 공간 1.5 m를 <b>자기 바닥에 앵커</b> ③ 이웃 단차 >0.2 m면 절벽 경계 '
             '④ 장애물에서만 침식 ⑤ 1 m² 미만 섬 삭제 ⑥ <b>despike</b>(바닥 voxel이 빠져 아래 면으로 샌 '
             '칸을 이웃 중앙값으로 보간) ⑦ <b>시작점 연결성</b>(시작점이 속한 성분만 — 도달 가능성 근사).'
             '</p>'
             '<table border=1 cellpadding=6><tr><th>씬</th><th>단계 (ablation)</th>'
             '<th>habitat 일치율</th><th>거짓 승인</th><th>거짓 기각</th></tr>')
    for r in rows:
        base = r['agg']
        items = list(base['abl'].items())
        for j, (anm, s) in enumerate(items):
            h.append(f'<tr><td>{r["scene"] if j == 0 else ""}</td><td>{anm}</td>'
                     f'<td>{100 * s["agree"]:.2f}%</td><td>{s["fa"]}</td><td>{s["fr"]}</td></tr>')
        h.append(f'<tr><td></td><td><small>비교: 밴드 기준선</small></td>'
                 f'<td><small>{100 * base["agree"]:.2f}%</small></td>'
                 f'<td><small>{base["fa"]}</small></td><td><small>{base["fr"]}</small></td></tr>')
    h.append('</table>')
    h.append('<p>⑥⑦이 v2다 — v1은 다층 씬에서 실패했었다(스캔 구멍의 kf가 아래 면으로 새어 <b>가짜 단차 '
             'seam이 GT를 1,183점 차단</b> + <b>navmesh가 걸을 수 없다고 판정한 "바닥처럼 생긴 면"</b>을 '
             '승인). 그 진단이 곧 수정이 됐다: despike가 seam을 지우고(s8 거짓 기각 11,233→895 · GT 차단 '
             '→0), 시작점 연결성이 그 면을 떨어뜨린다(F 14,559→745 · 일치율 →92.30%). '
             '<b>G1은 그래도 실패다</b> — s8 잔여 F 745가 사전 등록 기준(200 미만)을 넘는다.</p>'
             '<p>→ <b>default 맵은 navmesh 래스터화 유지</b>(정의상 100% 캐시가 이미 있다). '
             'recast v2의 가치는 <b>navmesh가 없는 씬</b>(Isaac 신규 씬 등)에서 habitat 없이 만들 수 있는 '
             '지도 중 최선이라는 것.</p>')

    h.append('<h3>실험결과 4 — pathfollower가 원본 정답을 재현하나</h3>'
             '<table border=1 cellpadding=6><tr><th>씬</th><th>에피소드</th>'
             '<th>ⓑ follower<br>대응점 거리 median</th><th>max</th><th>길이비</th>'
             f'<th>≤{100 * SPLIT_TOL_M:.0f} cm<br><small>거의 일치</small></th>'
             f'<th>&gt;{100 * SPLIT_TOL_M:.0f} cm<br><small>경로가 갈라짐</small></th>'
             '<th>#03 <code>find_path</code><br>(같은 지표)</th></tr>')
    prev = {'17DRP5sb8fy': '12.9 cm', 's8pcmisQ38h': '56.4 cm'}
    for r in rows:
        mm = r['acc']['mm']
        near = [m for m in mm if m['mean'] <= SPLIT_TOL_M]
        far = [m for m in mm if m['mean'] > SPLIT_TOL_M]
        h.append(f'<tr><td><b>{r["scene"]}</b></td><td>{len(mm)}</td>'
                 + (f'<td><b>{np.median([m["mean"] for m in mm]) * 100:.1f} cm</b></td>'
                    f'<td>{np.max([m["max"] for m in mm]) * 100:.1f} cm</td>'
                    f'<td>{np.mean([m["len_ratio"] for m in mm]):.3f}</td>'
                    f'<td><b>{len(near)}</b>개'
                    + (f' <small>({np.median([m["mean"] for m in near]) * 100:.1f} cm)</small>'
                       if near else '') + '</td>'
                    f'<td><b>{len(far)}</b>개'
                    + (f' <small>({np.median([m["mean"] for m in far]) * 100:.1f} cm)</small>'
                       if far else '') + '</td>'
                    if mm else '<td>—</td>' * 5)
                 + f'<td>{prev.get(r["scene"], "—")}</td></tr>')
    h.append('</table>')
    h.append('<p>VLN-CE 정답은 <i>"waypoints via low-level actions"</i>로 저장됐으므로 <b>pathfollower가 '
             f'곧 데이터셋 생성기</b>다(전진 {FWD_M} m · 회전 {TURN_DEG}°). #03의 측지 최단선 대비 '
             '<b>코너 처리는 크게 개선</b>(s8 56.4→8.8 cm)되지만, <b>median만 보면 안 된다</b> — '
             '분포가 두 덩어리이고 s8 9개 중 4개는 <b>경로가 장애물 반대편으로 갈라진다</b>(이산 액션도 '
             '못 고침). → 회랑 중심선은 여전히 <b>GT 서브궤적이어야 한다</b>.</p>')

    h.append('<h3>실험결과 5 — 두 경로가 두 지도에서 살아남나</h3>'
             '<table border=1 cellpadding=6><tr><th>경로</th><th>씬</th>'
             '<th>우리 지도(밴드) 밖 / 점 수</th><th>깊이 max</th>'
             '<th>habitat 지도 밖 / 점 수</th><th>깊이 max</th></tr>')
    for key, rkey, label in (('gt', 'gt_ref', '<b>ⓐ 원본 정답</b>(노랑)'),
                             ('fol', 'fol_ref', '<b>ⓑ follower</b>(초록)')):
        for r in rows:
            o = r['acc'][key]; q = r['acc'][rkey]
            h.append(f'<tr><td>{label}</td><td>{r["scene"]}</td>'
                     f'<td><b>{sum(x["out"] for x in o)}</b> / {sum(x["n"] for x in o)}</td>'
                     f'<td>{max(x["depth_max"] for x in o):.3f} m</td>'
                     f'<td>{sum(x["out"] for x in q)} / {sum(x["n"] for x in q)}</td>'
                     f'<td>{max(x["depth_max"] for x in q):.3f} m</td></tr>')
    h.append('</table><p><b>결과가 예상과 반대다</b> — GT가 우리 지도를 벗어난 점은 두 씬 모두 <b>0</b>이고 '
             '오히려 navmesh 래스터화 지도가 더 자른다(격자 반올림 + 단일 높이 절단, #03이 술어로 확인). '
             '<b>"정답이 우리 지도를 뚫는다"는 원 증상은 재현되지 않았다.</b> recast v2에서도 GT 이탈 '
             + ' · '.join(f'{r["scene"]} {r["agg"]["g3_out"]}점' for r in rows)
             + ', follower 이탈 '
             + ' · '.join(f'{r["agg"]["f3_out"]}점' for r in rows)
             + ' (전부 깊이 ≤1칸).</p>')

    h.append('<h3>범례</h3>'
             f'<p><b>지도 그림</b>: <span style="color:rgb{GT_COLOR}">■</span> ⓐ 원본 정답 · '
             f'<span style="color:rgb{OURS_COLOR}">■</span> ⓑ follower · '
             f'<span style="color:rgb{WP_COLOR}">■</span> waypoint. 초록 배경 = 통행 가능.</p>'
             '<p><b>diff 그림</b>: <span style="color:#3c963c">■</span> 양쪽 다 통행 가능 · '
             '<span style="color:#28c">■</span> <b>우리만 통행 가능</b>(거짓 승인) · '
             '<span style="color:#d33">■</span> 우리만 막힘(거짓 기각) · '
             f'<span style="color:rgb{F_COLOR}">■</span> <b>그중 F = 이 높이에 바닥이 없다</b> · '
             f'어두움 = 비교 범위(GT ±{args.band_m} m) 밖. '
             '③(밴드) → ④(recast v2)에서 <b>파랑·자홍이 사라지는 것</b>이 이 리포트의 핵심 그림이다.</p>')

    h.append('<h3>결론</h3><ul>'
             '<li><b>두 지도가 다른 이유는 질문의 구조다</b> — 밴드는 "장애물이 없나", recast는 "걸을 수 '
             '있는 바닥이 있나". 다층 씬 거짓 승인의 66.6%가 "바닥 없음"(F)이었다.</li>'
             '<li><b>recast-like v2가 밴드 계열의 답이다</b> — s8 75.83→<b>92.30%</b>(모든 후보 중 최고), '
             '17DRP 96.23%, GT 이탈 <b>0/0</b>, follower 이탈 ≤1칸. 단 s8 잔여 F 745로 G1 실패 → '
             '<b>default는 navmesh 래스터화 유지</b>, v2는 navmesh 없는 씬용.</li>'
             '<li><b>pathfollower는 코너는 고치지만 경로 갈림은 못 고친다</b> — 회랑 중심선은 GT '
             '서브궤적이어야 한다(#03과 일치).</li>'
             '<li><b>"정답이 우리 지도를 뚫는다"는 원 증상은 재현되지 않았다.</b></li></ul>')

    h.append('<h3>한계</h3><ul>'
             f'<li>씬 2채 · <b>계단 에피소드 제외</b>'
             f'({" · ".join(f"{r['scene']} {r['skip']['stairs']}개" for r in rows)}) — 단일 높이 절단의 '
             '한계는 어떤 후보도 못 고친다.</li>'
             '<li><b>게이트 C 실패</b>(86.1%/84.4% &lt; 90%) — S 버킷 13~15%(recast의 '
             '<code>region_min_size</code>·ledge·도달불가 island)를 더 쪼개지 않았다. '
             '<b>G1 실패</b>(s8 F 745) — 남은 축은 <code>.house</code> region별 바닥높이.</li>'
             f'<li>비교는 <b>GT 주변 {args.band_m} m 안</b>에서만 — 먼 곳의 불일치는 재지 않았다.</li>'
             f'<li>follower <code>goal_radius</code>({args.goal_radius_m} m)는 <b>데이터셋 생성값을 알 수 '
             '없다</b>(스윕 안 함).</li>'
             '<li><b>판정 기준이 여전히 habitat</b>이다 — 상한이 habitat이라는 뜻.</li>'
             '<li>이전 판의 <code>& floor_exists</code> 후보와 두께 스윕은 <b>v2로 대체돼 삭제</b>했다'
             '(기록은 reports.md #04 · 게이트 D도 함께 삭제).</li></ul>')
    return ''.join(h)


def body_html(rows):
    h = []
    for r in rows:
        h.append(f'<h2>{r["scene"]}</h2>')
        h.append('<h3>에피소드별</h3><table border=1 cellpadding=6><tr><th>ep</th><th>floor y</th>'
                 '<th>밴드 일치</th><th>recast v2 일치</th><th>거짓승인 (밴드→v2)</th>'
                 '<th>v2 거짓기각</th><th>v2 GT밖</th><th>ⓑ 재현 mean</th><th>reach max</th></tr>')
        for w in r['rows']:
            m = w['m']
            h.append(f'<tr><td>{w["ep"]}</td><td>{w["fy"]:.3f}</td>'
                     f'<td>{100 * w["c"]["agree"]:.2f}%</td>'
                     f'<td><b>{100 * w["c3"]["agree"]:.2f}%</b></td>'
                     f'<td>{w["c"]["ours_free_only"]} → <b>{w["c3"]["ours_free_only"]}</b></td>'
                     f'<td>{w["c3"]["ref_free_only"]}</td>'
                     f'<td>{w["g"]["out"]}/{w["g"]["n"]}</td>'
                     + (f'<td>{m["mean"] * 100:.1f} cm</td>' if m else '<td>—</td>')
                     + f'<td>{w["reach"].max():.3f} m</td></tr>')
        h.append('</table>')
        h.append('<h3>에피소드 그림</h3>'
                 '<p>에피소드마다 위에서 아래로 네 장: ① 우리 지도(밴드) → ② habitat 지도 → '
                 '③ diff(밴드) → ④ diff(recast v2). <b>③→④에서 파랑(거짓 승인)·자홍(바닥 없음)이 '
                 '사라지는 것</b>을 보면 된다.</p>')
        for w in r['rows']:
            if not w['fig']:
                continue
            h.append(f'<h4>ep{w["ep"]} · 밴드 {100 * w["c"]["agree"]:.2f}% → recast v2 '
                     f'{100 * w["c3"]["agree"]:.2f}%</h4>')
            for suffix, cap in (('ours', '① 우리 지도 (밴드 투영) — 노랑 ⓐ GT · 초록 ⓑ follower'),
                                ('ref', '② habitat 지도 (같은 높이에서 자름)'),
                                ('diff', '③ diff (밴드) — 파랑=거짓 승인 · 자홍=그중 바닥 없음'),
                                ('recast_diff', '④ diff (recast v2) — 파랑·자홍이 사라졌으면 성공')):
                h.append(f'<figure style="margin:0 0 10px"><img src="{w["fig"]}_{suffix}.jpg" '
                         f'style="width:100%"><figcaption>{cap}</figcaption></figure>')
    return ''.join(h)


# ---------------------------------------------------------------------------
# self-check — 씬·habitat 불필요. 합성 격자로 좌표 규약과 분해 로직을 검사한다.
# ---------------------------------------------------------------------------
def _selfcheck():
    # 우리 격자가 가짜 navmesh 경계 **안**에 완전히 들어가게 잡는다 — 밖이면 False로 채워져 검사가 공허해진다.
    ny, nx, cell = 40, 60, 0.05
    origin = np.array([0.5, -2.5, 0.0])

    class FakePF:
        def get_bounds(self):
            return ([0, 0, 0], [4, 2, 3])

        def get_topdown_view(self, mpp, _h):
            nz, nxx = int(3 / mpp), int(4 / mpp)
            tv = np.zeros((nz, nxx), dtype=bool)
            tv[:int(1.5 / mpp)] = True          # habitat z < 1.5만 통행 가능
            return tv

    m = navmesh_to_our_grid(FakePF(), 0.0, origin, cell, (ny, nx))
    # 우리 my = origin[1] + (iy+0.5)*cell = −2.5..−0.5 → habitat z = −my = 0.5..2.5.
    # z<1.5가 통행가능이므로 my > −1.5, 즉 iy >= 20이 True여야 한다(y 뒤집힘 확인).
    assert m[:20].sum() == 0, f'y뒤집힘/상단차단 실패: {m[:20].sum()}'
    assert m[20:].all(), 'y뒤집힘/하단통과 실패'

    gt_in = np.array([[1.0, -1.0], [2.0, -1.0], [3.0, -1.0]])
    o = path_outside(m, gt_in, origin, cell)
    assert o['out'] == 0 and o['n'] > 20, f'GT안 실패: {o}'
    o2 = path_outside(m, np.array([[1.0, -2.4], [3.0, -2.4]]), origin, cell)
    assert o2['out'] > 0 and o2['depth_max'] > 0, f'GT밖 실패: {o2}'

    band = band_of(gt_in, origin, cell, (ny, nx), band_m=0.3)
    assert 0 < band.sum() < ny * nx, f'band범위 실패: {band.sum()}'
    c = compare(m, m, band)
    assert abs(c['iou'] - 1.0) < 1e-9 and c['ours_free_only'] == 0, f'자기IoU 실패: {c}'
    c2 = compare(m, ~m, band)
    assert c2['iou'] == 0.0 and c2['ours_free_only'] > 0, f'반대IoU 실패: {c2}'

    # 분해: 우리는 전부 통행가능, navmesh는 아래 절반만 → 위 절반이 거짓 승인.
    ours = np.ones((ny, nx), dtype=bool)
    ref = np.zeros((ny, nx), dtype=bool); ref[:20] = True
    full = np.ones((ny, nx), dtype=bool)
    floor_ok = np.ones((ny, nx), dtype=bool); floor_ok[30:] = False     # 위쪽 10줄은 바닥 없음
    d, lab = decompose(ours, ref, full, floor_ok)
    assert d['total'] == 20 * nx, f'분해 합계 실패: {d}'
    assert d['F'] == 10 * nx, f'F 버킷 실패: {d}'
    assert d['R'] == nx, f'R 버킷(경계 1칸) 실패: {d}'          # ref 경계 바로 위 한 줄
    assert d['F'] + d['R'] + d['S'] == d['total'], f'버킷이 배타/완전하지 않다: {d}'
    assert (lab == 1).sum() == d['F'] and (lab == 2).sum() == d['R'], 'label 불일치'

    # floor_exists: 바닥 밴드에 점유가 있는 열만 True.
    occ = np.zeros((nx, ny, 40), dtype=bool)
    occ[5:10, 5:10, 2] = True                  # z = origin[2] + 2*cell = 0.10 m
    g = dict(occ=occ, origin=origin, cell=cell, coverage=np.ones((ny, nx), dtype=bool))
    fe = floor_exists_mask(g, 0.0, below_m=0.15)
    assert fe.shape == (ny, nx), f'floor_exists 축 규약 실패: {fe.shape}'
    assert fe[5:10, 5:10].all() and fe.sum() == 25, f'floor_exists 실패: {fe.sum()}'
    fe2 = floor_exists_mask(g, 3.0, below_m=0.15)     # 3 m 위에는 바닥이 없다
    assert fe2.sum() == 0, f'floor_exists 다른 높이 실패: {fe2.sum()}'

    print('[selfcheck] calibrate_map 12/12 통과 (y뒤집힘·상단차단·하단통과·GT안·GT밖·band범위·'
          '자기IoU·반대IoU·분해합계·F버킷·R버킷·floor_exists)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
