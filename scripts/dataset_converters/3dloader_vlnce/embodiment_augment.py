"""P-B1 — EmbodimentAugmenter (v1: r_b/h_nav 연속, 장애물 없음, depth+BEV, RGB 미변경).

vln_ce 에피소드 + embodiment `e=(r_b, h_nav)` → **e에 맞춰 GT path 재계획 + 새 path의 depth 렌더 +
BEV**. r_b는 GT path(어떤 경로가 로봇에 안전한가)를 바꾸고, 그 새 경로를 따라 관측(depth→BEV)이 바뀐다.

전부 기존 코드 재사용(3dloader_vlnce 정책: 새 로직만 여기, 나머지 import):
  vlnce_align.align_episode  → T_sf2mesh(start-relative→mesh 절대)
  03_sample_gt_paths.plan_episode/load_esdf → occ→2D→A*+refine+spline 재계획 (G2/G3 검증)
  geometry_utils.synthesize_action_poses/action_to_c2w → 경로 xy→카메라 c2w
  04_render_obs.build_renderer/set_camera_pose → Open3D 렌더러 (G3: **depth-only + 타일 상주**)
  01_build_scene_geo.geo_path → 씬 지오메트리 전용 ply 경로 (타일링 폐기 2026-08-20)
  depth_rgb_to_bev_torch.depth_to_bev_occ_ros2 → depth→BEV

검증 완료 후 internnav/dataset/internvla_n1_lerobot_dataset.py에 이식(P-B2). 지금은 기존 코드 무수정.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_GS = _HERE.parents[0] / 'gs_vlnpe'
_ROOT = _HERE.parents[2]  # repo root (scripts/dataset_converters/3dloader_vlnce -> repo)
sys.path.insert(0, str(_GS)); sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_ROOT))
from geometry_utils import action_to_c2w, synthesize_action_poses  # noqa: E402
from esdf_utils import (  # noqa: E402
    astar, cell_to_world, check_path_navigable, compute_esdf_2d, compute_scan_coverage_mask,
    derive_obstacle_2d, refine_min_move, smooth_cubic_spline, thin_waypoints,
    truncate_navigable, world_to_cell,
)
from corridor_utils import corridor_mask, detour_amount_m  # noqa: E402
from waypoint_spine import LEG_TOL_M, frame_flags  # noqa: E402
from vlnce_align import align_episode, build_rawdata_index  # noqa: E402
_geo_mod = importlib.import_module('01_build_scene_geo')  # noqa: E402
from internnav.model.utils.depth_rgb_to_bev_torch import depth_to_bev_occ_ros2  # noqa: E402
_p03 = importlib.import_module('03_sample_gt_paths')
_render = importlib.import_module('04_render_obs')
_geo = importlib.import_module('geometry_utils')

RENDER_RES = 224
CAM_HEIGHT_M = 1.25   # 125cm 리그 (v1 고정; cam_height 축은 v2)
CAM_PITCH_DEG = 0.0   # pitch_2=0 리그
# 새 r_b로 만든 경로의 **끝점이 원본 GT 끝점과 달라지면 기각**한다(그 에피소드는 원본 r_b 경로를 쓴다).
# 이유: r_b dilation이 start/goal 셀을 지우면 `_snap_navigable`이 가장 가까운 navigable로 **옮기는데**,
# 그러면 도착 지점이 바뀌어 instruction("...화장실에서 멈춰라")이 거짓 라벨이 된다.
# 실측(17DRP ep0 r_b=0.5): goal이 37.8 cm 이동. 반면 정상 계획의 끝점 오차는 격자 스냅 수준(2~3 cm).
GOAL_TOL_M = 0.10          # 격자(5 cm) 2칸 — 양자화 오차는 허용, 실제 goal 이동은 기각
# 계획용 inflation: `r_b + PLAN_MARGIN_M`으로 계획하고 **검증은 실제 `r_b`로** 한다.
# 왜 필요한가: fine 격자 A*는 `clearance >= r_b`를 보장하지만 그 뒤 `smooth_cubic_spline`이 코너를 자르며
# clearance를 떨어뜨린다(실측 17DRP ep0: r_b=0.3에서 A* 보장 0.30 → 스플라인 후 0.250).
# 격자 1칸(0.05 m)이 두 씬 모두에서 수확량 최대였다 — margin 0/0.05/0.10/0.15에서 통과 슬롯
# 17DRP 4/**7**/5/5, s8 8/**7**/4/3 (합 12/**14**/9/8). 더 키우면 위반은 0이 되지만 로봇을 더 뚱뚱하게
# 계획해 통과 자체가 줄어든다. 남은 위반은 `_path_ok`가 기각하므로(→ 원본 GT 폴백) 안전하다.
# ponytail: 근본 원인은 스무딩이다. 위반을 0으로 만들려면 스무딩 후 clearance 투영(재-refine)이 필요.
PLAN_MARGIN_M = 0.05
# leg를 'detour'로 부를 최소 이탈량. **파이프라인 노이즈보다 위여야 한다** — 그렇지 않으면
# "같은 루트를 더 뚱뚱한 몸으로 지나간 것"이 우회로 집계된다. 노이즈 성분:
#   waypoint→프레임 앵커 오차 mean 0.094 m (p90 0.227) · 격자 셀 0.05 · PLAN_MARGIN_M 0.05
#   · 반경 증가분 자체(r_b − 0.10; 같은 벽을 그만큼 더 떨어져 지나면 자동으로 생긴다)
# 실측 이탈량 분포(s8, corridor 모드): median **0.000 m** 전 r_b 공통, p90 0.05~0.15,
# 진짜 우회는 0.791 m 한 건. 임계 0.10이면 11~19%가 노이즈로 오분류되고, **0.20이면 r_b≥0.15에서 0%**,
# r_b=0.1에서 8%(진짜 우회만)다. 반경 인식 임계는 불필요 — 이탈량이 r_b와 함께 커지지 않는다
# (r_b=0.3에서 max 0.05 m. 애매한 leg는 blocked로 걸러지고 살아남은 건 원래 넓은 곳이라서).
DETOUR_THRESH_M = 0.20
# "기존 r_b" — 원본 GT가 만들어진 반경. vln_ce는 habitat navmesh 위 ShortestPathFollower로 수집됐고,
# `scripts/eval/configs/vln_r2r_mini.yaml`에 agent radius를 지정하지 않으므로 **habitat 기본값 0.1 m**다
# (확인: `AgentConfig().radius == 0.1`). 이 값에서 파이프라인은 원본을 재현해야 한다(identity gate).
BASELINE_R_B = 0.10
# vln_ce 밴드 기본값을 **habitat 설정 직독값**에 맞춘다(사용자 결정 2026-08-20).
# 근거: #04에서 h_nav 0.10~0.25 × h_obs 1.25/1.50 8조합의 일치율 차이가 0.08~0.23%p뿐이라
# 실측 최적점을 고를 이유가 약했다. 그러면 **원리 있는 값**을 쓰는 게 낫다(#04는 스윕을 폐기했다).
H_NAV_HABITAT_M = 0.20        # habitat `agent_max_climb` — 이 아래는 밟고 넘어간다
H_OBS_HABITAT_M = 1.50        # habitat `agent_height`
MAP_SOURCES = ('occ', 'navmesh')   # occ = 3D 스캔 밴드 투영 · navmesh = VLN-CE navmesh 래스터화


def densify(traj, step_m):
    """점 사이를 `step_m` 간격으로 채운다 (GT는 프레임 단위라 듬성듬성하다)."""
    p = np.asarray(traj, dtype=np.float64)[:, :2]
    if len(p) < 2:
        return p
    out = [p[:1]]
    for x, y in zip(p[:-1], p[1:]):
        k = max(2, int(np.linalg.norm(y - x) / step_m) + 1)
        out.append(x + (y - x) * np.linspace(0, 1, k)[:, None])
    return np.vstack(out)



def band_heights(h_nav_ratio, h_b, h_nav_m=H_NAV_HABITAT_M, h_obs_m=H_OBS_HABITAT_M):
    """밴드 상·하단 높이[m]를 정한다. -> (h_nav, h_obs)

    기본은 **habitat 정렬 절대값**(0.20 / 1.50). `None`을 주면 예전 비율 방식
    (`h_nav_ratio * h_b`, `h_obs = h_b`)으로 돌아간다 — 하위 호환용.
    """
    h_nav = float(h_nav_ratio) * float(h_b) if h_nav_m is None else float(h_nav_m)
    h_obs = float(h_b) if h_obs_m is None else float(h_obs_m)
    return h_nav, h_obs


def snap_to_grid(mask, origin, cell_m, xy, max_radius_m=1.5):
    """`xy`가 `mask`(bool 2D)에서 False면 **가장 가까운 True 셀 중심**으로 스냅. 없으면 None."""
    ij = np.floor((np.asarray(xy, dtype=np.float64)[:2] - np.asarray(origin)[:2]) / cell_m).astype(int)
    h, w = mask.shape
    if 0 <= ij[0] < w and 0 <= ij[1] < h and mask[ij[1], ij[0]]:
        return np.asarray(xy, dtype=np.float64)[:2]
    rad = int(np.ceil(max_radius_m / cell_m))
    best, bd = None, np.inf
    y0, y1 = max(0, ij[1] - rad), min(h, ij[1] + rad + 1)
    x0, x1 = max(0, ij[0] - rad), min(w, ij[0] + rad + 1)
    sub = np.argwhere(mask[y0:y1, x0:x1])
    for dy, dx in sub:
        yy, xx = y0 + dy, x0 + dx
        d = (xx - ij[0]) ** 2 + (yy - ij[1]) ** 2
        if d < bd:
            bd, best = d, np.array([origin[0] + (xx + 0.5) * cell_m, origin[1] + (yy + 0.5) * cell_m])
    return best


def dilate_bev_occupancy(bev, r_b, bev_range=5.0):
    """BEV 점유셀을 로봇 반경 `r_b`만큼 팽창(configuration-space 변환).

    **모드 (B)에서 `e`를 관측에 넣는 방법.** 현 BEV 파이프라인(`depth_rgb_to_bev_torch`)에는
    robot-radius dilation이 **전혀 없다** — 논문 C1이 지적한 빠진 조각이 정확히 이것이다.
    큰 로봇에겐 좁은 틈이 '막힌 것'으로 보여야 하므로, 장애물을 r_b만큼 부풀린 BEV를 준다.

    bev: (B,H,W) 또는 (H,W). 0=unknown, 0.5=free, 1=occupied. 해상도 = 2*bev_range/size (기본 4.46 cm/px).
    팽창된 셀은 occupied(1.0)로 덮어쓴다(free였던 곳도 로봇이 못 가므로).
    """
    from scipy.ndimage import binary_dilation
    single = (bev.ndim == 2)
    arr = bev[None] if single else bev
    size = arr.shape[-1]
    rad_px = float(r_b) / (2.0 * bev_range / size)
    if rad_px < 0.5:
        return bev
    k = int(np.ceil(rad_px))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = (yy ** 2 + xx ** 2) <= rad_px ** 2          # 원형 구조요소 = 로봇 footprint
    out = arr.copy()
    for i in range(len(arr)):
        occ = binary_dilation(arr[i] >= 1.0, structure=disk)
        out[i] = np.where(occ, 1.0, arr[i])
    return out[0] if single else out


def k_scaled(res):
    """640×480 하드코딩 K(388.19)를 정사각 res로 스케일 (학습 depth 규약과 동일)."""
    return dict(fx=388.19 * res / 640, fy=388.19 * res / 480,
                cx=319.5 * res / 640, cy=239.5 * res / 480)


class EmbodimentAugmenter:
    def __init__(self, data_root, raw_root, mesh_root, esdf_dir, geo_dir, res=RENDER_RES,
                 map_source='occ', navmesh_dir=None):
        # 계획 격자를 만드는 두 방식을 **config로 고른다**. `occ`=3D 스캔 밴드 투영(habitat 파라미터 정렬),
        # `navmesh`=VLN-CE navmesh 래스터화(habitat 판정과 100% 일치, 대신 habitat 의존).
        assert map_source in MAP_SOURCES, f'map_source={map_source!r} not in {MAP_SOURCES}'
        self.map_source, self.navmesh_dir = map_source, navmesh_dir
        self.data_root, self.mesh_root = Path(data_root), Path(mesh_root)
        # 타일링 폐기(2026-08-20): 씬 전체 **지오메트리 전용 ply**가 타일 하나보다 작다
        # (7.4 MB vs 6.0 MB · 로드 0.09 s · RSS 0.08 GB). 무거운 건 지오메트리가 아니라 텍스처였고
        # v1은 depth만 렌더하니 텍스처가 전부 낭비다 → `01_build_scene_geo.py` docstring 참고.
        self.esdf_dir, self.geo_dir, self.res = esdf_dir, Path(geo_dir), res
        # raw index(전 split json.gz, 13k+ 에피소드)는 **align에만** 필요하다 → lazy.
        # dataloader worker는 미리 계산된 T_sf2mesh만 쓰므로 이 비용을 내면 안 된다(worker마다 수십 초).
        self._raw_root, self._raw = Path(raw_root), None
        self.K = k_scaled(res)
        import open3d as o3d
        self._o3d = o3d
        Kmat = np.array([[self.K['fx'], 0, self.K['cx']], [0, self.K['fy'], self.K['cy']], [0, 0, 1]])
        self.renderer = _render.build_renderer(res, res, Kmat)  # worker당 1회 (G3)
        self._scene_cache, self._loaded_geo, self._model_cache = {}, None, {}
        self._mesh_cache = {}

    # --- per-scene 컨텍스트 (occ + 타일 인덱스) LRU ---
    def _scene_ctx(self, scene):
        if scene not in self._scene_cache:
            a = _p03.build_argparser().parse_args([])
            a.scene, a.dataset, a.esdf_dir, a.h_nav_ratio = scene, 'vln_n1', self.esdf_dir, 0.12
            occ, origin, floor_z = _p03.load_esdf(a)  # a.cell_m/downsample_factor 패치
            # A*를 **fine 격자에서** 돈다(다운샘플 안 함). 기본값 factor=4 + mode='any'는 20 cm coarse 셀을
            # "16개 fine 셀 중 하나만 navigable이면 통과 가능"으로 보고, 웨이포인트를 그 **셀 중심**에 놓는다
            # → 셀 중심이 벽 안이면 경로가 장애물을 관통한다. 실측(s8 7 에피소드, r_b=0.1):
            #   factor=4 any      : 계획성공 7/7, **하드충돌 4**, 몸통침범 5, 9 ms
            #   factor=4 majority : 7/7, 하드 1, 몸통 1, 10 ms
            #   factor=4 all      : 3/7(좁은 문이 막힘), 0, 0, 8 ms
            #   factor=1          : 7/7, **하드 0**, 몸통 1, 17 ms   ← 채택
            # sample 예산은 Open3D 렌더 350 ms가 지배하므로(R2) +10 ms는 무시할 수준이다.
            a.downsample_factor = 1
            self._scene_cache[scene] = dict(occ=occ, origin=origin, floor_z=floor_z, args=a,
                                            coverage=compute_scan_coverage_mask(occ))
        return self._scene_cache[scene]

    def _scene_model(self, scene):
        """씬 지오메트리 전용 모델을 렌더러에 상주시킨다(씬 전환 시에만 `add_model`).

        타일링을 없앤 뒤로 **씬당 모델 하나**다 — 프레임마다 타일을 조회·교체할 필요가 없고,
        타일 경계 손실도 원리적으로 불가능하다. 7~10 MB / 0.1 s / 0.08 GB(실측).
        """
        if scene != self._loaded_geo:
            model = self._model_cache.get(scene)
            if model is None:
                f = _geo_mod.geo_path(self.geo_dir, scene)
                assert f.exists(), f'{f} 없음 — 01_build_scene_geo.py 를 먼저 돌려라'
                model = self._o3d.io.read_triangle_model(str(f))
                if len(self._model_cache) >= 4:          # 간단한 상한(메모리 방지)
                    self._model_cache.pop(next(iter(self._model_cache)))
                self._model_cache[scene] = model
            self.renderer.scene.clear_geometry(); self.renderer.scene.add_model('scene', model)
            self._loaded_geo = scene

    def _render_depth(self, scene, c2w):
        """단일 c2w에서 depth만 렌더 (RGB 생략 — G3). 커버 타일 상주 후."""
        self._scene_model(scene)
        _render.set_camera_pose(self.renderer, c2w)
        d = np.asarray(self.renderer.render_to_depth_image(z_in_view_space=True))
        return np.where(np.isfinite(d), d, 0.0).astype(np.float32)

    # --- 공개 API ---
    def align(self, scene, episode, frames, instruction):
        """에피소드를 mesh 절대좌표로 정합. -> dict(T, resid, raw_ep).

        **비용 주의(실측 14.8 s/에피소드)**: 대부분이 씬 mesh(199 MB obj) 로드다 → 씬 mesh를 캐시한다.
        그래도 정합 자체가 무거우므로 **학습 경로에서는 offline 배치로 미리 계산해 캐시**하고
        worker는 `T_sf2mesh`만 읽어야 한다(sample당 0 ms).
        """
        if self._raw is None:
            self._raw = build_rawdata_index(self._raw_root)
        raw_ep = self._raw[(scene, instruction.strip())]
        if scene not in self._mesh_cache:
            self._mesh_cache[scene] = _geo.load_scene_mesh(self.mesh_root, scene)
        T, resid, _ = align_episode(frames, self._mesh_cache[scene], raw_ep)
        return dict(T=T, resid=resid, raw_ep=raw_ep)

    def _path_ok(self, ctx, traj, esdf, cell, r_b, goal_xy, goal_tol_m=GOAL_TOL_M, why=False):
        """생성된 경로가 **학습 라벨로 쓸 수 있는지** 검사. 하나라도 어기면 기각(→ 원본 GT 폴백).

        **원본 GT에는 쓰지 않는다** — 생성 경로 전용이다(2번 조건이 원본 GT에선 성립하지 않는다).
        네 조건 모두 "그 경로를 정답이라고 가르치면 거짓을 가르치는 것"에 해당한다:
          1. **하드 충돌** `clearance == 0` — 장애물 voxel 내부. 질점 로봇도 통과 불가.
          2. **몸통 침범** `clearance < r_b` — 로봇 몸통이 장애물과 겹친다. A*는 fine 격자에서
             `clearance >= r_b`를 이미 보장했으므로, 이걸 어기는 건 refine/스무딩이 낸 오차다.
             ⚠️ 2026-08-19 정정: "우리 맵이 recast보다 보수적"이라고 적었던 것은 **방향이 반대**였다.
             #04 실측 — 우리 밴드 맵은 recast보다 일관되게 **더 관대**하다(s8 거짓 승인 44,219셀 vs
             거짓 기각 164셀). 그래서 default 맵을 navmesh 래스터화로 바꿨다(`reports.md` #04).
          3. **미관측(unknown) 통과** — 스캔 안 된 공간은 장애물이 없다고 볼 근거가 없으므로 ESDF가
             크게 나온다(장애물 검사로는 안 잡힌다). `compute_scan_coverage_mask`로 따로 막는다.
             실측으로는 0/7이었지만 검사가 없으면 보장이 없다.
          4. **target 도달** — 끝점이 요청된 goal에서 `goal_tol_m` 안이어야 한다. 도착지가 밀리면
             instruction이 거짓 라벨이 된다. 끄는 우회로를 두지 않는다(`None` 미지원).
             ⚠️ 비교 대상은 **요청된 goal**이다. dilation으로 goal 셀이 지워져 스냅한 경우
             (`snap_to_grid`는 최대 1.5 m까지 옮긴다) 스냅된 goal과 비교하면 목표가 밀려도 통과한다.

        `why=True`면 통과/실패 대신 **사유 문자열**을 준다(`'ok'`/`'target'`/`'hard'`/`'body'`/`'unknown'`).
        진단용이다 — 리포트가 기준을 따로 구현하면 두 곳이 어긋나므로 여기 한 곳에서만 판정한다.
        """
        traj = np.asarray(traj, dtype=np.float64)
        if np.linalg.norm(traj[-1] - np.asarray(goal_xy, dtype=np.float64)[:2]) > float(goal_tol_m):
            return 'target' if why else False
        chk = check_path_navigable(traj, esdf, ctx['origin'], cell, float(r_b))
        if not chk['hard_ok']:
            return 'hard' if why else False
        if not chk['segments_ok']:
            return 'body' if why else False
        cov = ctx['coverage']
        step = cell / 5.0
        pts = traj
        dense = [pts[0][None, :]]
        for x, y in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.linalg.norm(y - x) / step) + 1)
            dense.append(x + (y - x) * np.linspace(0, 1, n)[:, None])
        ij = np.floor((np.vstack(dense) - np.asarray(ctx['origin'])[:2]) / cell).astype(int)
        h_, w_ = cov.shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < w_) & (ij[:, 1] >= 0) & (ij[:, 1] < h_)
        return bool(inside.all() and cov[ij[:, 1], ij[:, 0]].all())

    def deform_gt(self, scene, gt_xy, r_b, h_nav_ratio=0.12, h_b=CAM_HEIGHT_M, floor_z=None,
                  refine_radius_m=0.30, goal_tol_m=GOAL_TOL_M, plan_margin_m=PLAN_MARGIN_M):
        """**원본 GT를 기준으로 유지**하며 embodiment(`r_b`)에 맞게 변형한다. -> trajectory or None.

        `replan`(start→goal 순수 A*)과의 차이가 핵심이다. A*는 **instruction을 모른다** — R2R의 GT는
        사람이 주석한 `reference_path`(특정 방·물체를 경유하도록 instruction에 맞춘 것)를 따라간
        궤적인데, A*는 기하적 최단만 찾으므로 다른 방으로 돌아버릴 수 있다. 그러면 baseline `r_b`에서도
        경로가 GT와 달라지고 **instruction이 거짓 라벨**이 된다.

        그래서 여기서는 GT 자체를 seed로 주고 `refine_min_move`로 **필요한 만큼만** 밀어낸다:
        clearance가 이미 `r_b` 이상인 점은 그대로 두고, 미달인 점만 창 안에서 `esdf >= r_b`인 가장
        가까운 셀로 옮긴다. 즉 "GT를 따라가되 큰 로봇이 못 지나가는 구간에서만 국소 우회"다.

        `refine_min_move`의 docstring은 이 방식이 실제 생성에서 `argmax`보다 나쁘다고 기록하는데,
        그 근거는 "그 이점은 **GT 루트를 모르는** 실제 생성에서 쓸 수 없다"였다. 우리 목표는 반대로
        **GT 루트를 알고 그것을 유지하는** 것이므로 그 기각 사유가 적용되지 않는다.
        """
        ctx = self._scene_ctx(scene)
        a = ctx['args']
        fz = ctx['floor_z'] if floor_z is None else float(floor_z)
        cell = float(a.cell_m)
        obstacle = derive_obstacle_2d(ctx['occ'], ctx['origin'], fz, h_b, cell, h_nav_ratio * h_b)
        esdf = compute_esdf_2d(obstacle, cell)
        # 계획은 inflation 반경으로, 검증은 실제 r_b로 (PLAN_MARGIN_M 주석 참고)
        wp = refine_min_move(densify(gt_xy, cell), esdf, ctx['origin'], cell, refine_radius_m,
                             fix_endpoints=True, r_b=float(r_b) + float(plan_margin_m))
        traj = np.asarray(smooth_cubic_spline(thin_waypoints(wp, a.waypoint_spacing_m), a.smooth_step))
        if not self._path_ok(ctx, traj, esdf, cell, r_b, np.asarray(gt_xy)[-1], goal_tol_m):
            return None
        return traj

    # --- leg 단위 국소 우회 -------------------------------------------------
    def _leg_grids(self, scene, r_b, h_nav_ratio, h_b, floor_z, plan_margin_m,
                   h_nav_m=H_NAV_HABITAT_M, h_obs_m=H_OBS_HABITAT_M):
        """leg 계획에 쓰는 격자들을 한 번만 만든다. -> (ctx, cell, esdf, navigable)

        `navigable`은 **inflation 반경**(`r_b + plan_margin_m`)으로 자른다 — 스플라인이 코너를 자르며
        clearance를 떨어뜨리기 때문(PLAN_MARGIN_M 주석). 검증(`_path_ok`)은 실제 `r_b`로 한다.

        `navmesh_dir`가 있으면 **VLN-CE navmesh 래스터화**로 위임한다(#04 결론: 밴드 투영은 GT 주변
        2 m 안에서 s8 75.8%만 일치하고 방향이 과소 장애물이다 — `reports.md` #04). `floor_z`가 절단 높이.
        """
        ctx = self._scene_ctx(scene)
        fz = ctx['floor_z'] if floor_z is None else float(floor_z)
        src = getattr(self, 'map_source', 'occ')
        if src == 'navmesh':
            import navmesh_grid
            nd = self.navmesh_dir or navmesh_grid.DEFAULT_NAVMESH_DIR
            return navmesh_grid.leg_grids(ctx, nd, scene, r_b, fz, plan_margin_m)
        elif src == 'occ':
            pass                      # 아래 기존 밴드 투영
        else:
            assert False, f'unreachable map_source={src!r}'
        a = ctx['args']
        cell = float(a.cell_m)
        h_nav, h_obs = band_heights(h_nav_ratio, h_b, h_nav_m, h_obs_m)
        obstacle = derive_obstacle_2d(ctx['occ'], ctx['origin'], fz, h_obs, cell, h_nav)
        esdf = compute_esdf_2d(obstacle, cell)
        navigable = truncate_navigable(esdf, float(r_b) + float(plan_margin_m)) & ctx['coverage']
        return ctx, cell, esdf, navigable

    def _leg_verbatim(self, ctx, cell, esdf, gt, r_b, leg_tol_m):
        """GT 서브궤적이 `r_b`로 **이미 통과 가능하면 그대로** 쓴다. -> (gt or None, 0.0, None)

        refine·thin·spline을 전부 생략하므로 GT 이탈이 **정의상 0**이다. `_path_ok`의 5번째 위치인자가
        이미 반경이라 새 인자가 필요 없다 — 4조건(하드 충돌/몸통 침범/미관측/target)이 한 곳에 유지된다.
        """
        if not self._path_ok(ctx, gt, esdf, cell, float(r_b), gt[-1], leg_tol_m):
            return None, float('nan'), None
        return gt, 0.0, None

    def _plan_leg(self, ctx, cell, esdf, navigable, leg_gt_xy, r_b, mode, corridor_m, leg_tol_m,
                  gtc_cap_m=None, gtc_fixed_m=None):
        """leg 하나를 `mode`로 계획. -> (path_xy or None, detour_max_m, detour_point or None)

        `leg_gt_xy`는 그 leg의 **GT 서브궤적**(직선이 아니라 실제 궤적 — 실측으로 직선은 최악 4.36 m
        어긋난다). 성공 판정은 **VLN-CE 자신의 기준**: 끝점이 다음 waypoint의 `leg_tol_m`(0.5 m) 안.

        mode:
          `none`     — GT를 **그대로**. 통과 가능하면 손대지 않는다(ladder의 밑단). 이탈 0.
          `nudge`    — `refine_min_move` 점별 국소 밀어내기. 창(`refine_radius_m`) 밖으로 못 나가
                       **재라우팅 불가**하지만 GT에 가장 충실하고 가장 싸다.
          `corridor` — GT 서브궤적 주변 회랑 안에서만 A*. 가구 우회 O, 다른 방 X.
          `free`     — 전체 navigable에서 A*. waypoint 순서만 제약 → 루트 이탈 위험.
        """
        gt = np.asarray(leg_gt_xy, dtype=np.float64)[:, :2]
        # `none`은 **맨 앞에서** 빠진다(guard clause) — GT 폴리라인 자신이라 이탈량 0이 측정값이 아니라
        # 정의이고, `ctx['args']`·거리장(EDT 2회)도 필요 없다.
        if mode == 'none':
            return self._leg_verbatim(ctx, cell, esdf, gt, r_b, leg_tol_m)
        a = ctx['args']
        target = gt[-1]
        r_plan = float(r_b) + PLAN_MARGIN_M

        # GT 서브궤적까지의 거리장은 **mode와 무관하게** 만든다. 이전 판은 `corridor`에서만 만들고
        # `nudge`/`free`에서는 `field=None`으로 뒀는데, 그러면 이탈량이 **항상 0으로 보고**된다.
        # ladder 기본값이 nudge 먼저라 결과적으로 detour가 거의 안 잡히는 구조였다.
        corr, field = corridor_mask(gt, ctx['origin'], cell, navigable.shape, corridor_m)
        if mode == 'nudge':
            wp = refine_min_move(densify(gt, cell), esdf, ctx['origin'], cell, 0.30,
                                 fix_endpoints=True, r_b=r_plan)
        elif mode in ('corridor', 'corridor_gtc', 'free'):
            mask = navigable if mode == 'free' else (navigable & corr)
            # 끝점이 마스크 밖이면 A*가 즉시 None을 낸다 → 마스크 위로 스냅(1.5 m 상한)
            s = snap_to_grid(mask, ctx['origin'], cell, gt[0])
            g = snap_to_grid(mask, ctx['origin'], cell, target)
            if s is None or g is None:
                return None, float('nan'), None
            # **시작점은 순간이동시킬 수 없다.** 로봇은 물리적으로 거기 있다. `snap_to_grid`는 최대 1.5 m
            # 옮기는데 끝점만 검사하면(`_path_ok`) 시작점이 밀린 경로가 통과한다 — 실측: r_b=0.2에서
            # 시작 0.79 m 이동 후 A* 실패, r_b=0.45에서 1.68 m 이동 후 **성공**해 결과가 비단조가 됐다.
            # 시작점이 이 embodiment로 점유 불가면 그 leg는 진짜 blocked다.
            if np.linalg.norm(s - gt[0]) > GOAL_TOL_M:
                return None, float('nan'), None
            ij = world_to_cell(np.array([s, g]), ctx['origin'], cell)
            path_ij = astar(mask, ij[0], ij[1])
            if path_ij is None:
                return None, float('nan'), None
            wp = cell_to_world(path_ij, ctx['origin'], cell)
            wp[0], wp[-1] = s, g
            if mode == 'corridor_gtc':
                # GT 여유 매칭 refine — 벽 여유를 GT 스타일로 (근거·기제는 gt_clearance_refine.py)
                from gt_clearance_refine import CAP_M, refine_match_gt
                wp = refine_match_gt(wp, esdf, ctx['origin'], cell, gt, r_plan, corr=corr,
                                     cap_m=CAP_M if gtc_cap_m is None else gtc_cap_m,
                                     fixed_m=gtc_fixed_m)
            else:
                # A*는 fine 격자 계단이라 그대로 쓰면 거칠다 → 최소 이동 refine으로 다듬는다
                wp = refine_min_move(wp, esdf, ctx['origin'], cell, 0.30, fix_endpoints=True,
                                     r_b=r_plan)
        else:
            assert False, f'unreachable mode={mode!r}'

        traj = np.asarray(smooth_cubic_spline(thin_waypoints(wp, a.waypoint_spacing_m), a.smooth_step))
        # **라벨 유효성 4조건**을 leg에도 그대로 건다 — 하드 충돌 / 몸통 침범 / 미관측 통과 / target 도달.
        # 끝점 거리만 보면 안 된다: `nudge`(refine_min_move)는 실패를 반환하지 않고 점을 조금 밀기만
        # 하므로 끝점은 거의 항상 GT 끝점이고 leg_tol(0.5 m)을 통과한다 → 벽을 지나는 경로가 전부
        # '성공'으로 집계된다(실제로 r_b=0.45에서도 blocked가 0으로 나왔다).
        # target 도달 허용오차는 GOAL_TOL_M(0.10) 대신 **VLN-CE의 leg 기준 0.5 m**를 쓴다.
        if not self._path_ok(ctx, traj, esdf, cell, r_b, target, leg_tol_m):
            return None, float('nan'), None
        det, pt = detour_amount_m(traj, field, ctx['origin'], cell, with_point=True)
        return traj, det, pt

    def follow_waypoints(self, scene, spine, gt_xy, r_b, h_nav_ratio=0.12, h_b=CAM_HEIGHT_M,
                         floor_z=None, ladder=('none', 'nudge', 'corridor'), corridor_m=2.0,
                         leg_tol_m=LEG_TOL_M, detour_thresh_m=DETOUR_THRESH_M, plan_margin_m=PLAN_MARGIN_M,
                         goal_tol_m=GOAL_TOL_M, h_nav_m=H_NAV_HABITAT_M, h_obs_m=H_OBS_HABITAT_M,
                         gtc_cap_m=None, gtc_fixed_m=None):
        """**사람 주석 waypoint를 이어서 따라가되 못 지나가는 leg에서만 국소 우회.** -> dict.

        VLN-CE가 데이터셋을 만든 절차와 같다 — *"We run this algorithm between each waypoint in a
        trajectory to the next … navigable if … within 0.5 m of the next waypoint"*. 우리는 거기서
        `r_b`만 바꿔 재적용하고, **실패를 leg 단위로 국소화**한다.

        왜 leg 단위인가: 전 구간이 통과 가능한 경로만 쓰면 샘플이 너무 적다. leg 하나가 막혀도
        **앞쪽 leg의 프레임은 전부 살아남는다**.

        `ladder`는 자유도 낮은 순으로 시도하고 **첫 성공에서 멈춘다**. 밑단이 `none`이다 — **통과 가능한
        leg는 GT를 그대로 쓰고**(이탈 0), 밀어내기·회랑은 정말 못 지나갈 때만 쓴다. rung을 위에 얹는 것은
        커버리지를 낮추지 않는다(`none` 실패 시 아래 rung들이 이전과 동일 경로로 돈다).

        반환: `legs`(leg별 status/우회량/사용 mode), `path_xy`(성공 구간 이어붙임),
        `reach_ok`(프레임별 0 ok / 1 detour / 2 blocked), `first_blocked_frame`, `n_ok/n_detour/n_blocked`.
        `reach_ok`가 로더가 실제로 쓰는 것이다(제외 모드 / 도달불가 학습 모드 둘 다 이걸로 판정).
        """
        ctx, cell, esdf, navigable = self._leg_grids(scene, r_b, h_nav_ratio, h_b, floor_z,
                                                     plan_margin_m, h_nav_m, h_obs_m)
        gt = np.asarray(gt_xy, dtype=np.float64)
        legs_out, pieces, status = [], [], []

        for i, (lo, hi) in enumerate(spine['legs']):
            leg_gt = gt[lo:hi + 1, :2]
            traj, det, pt, used = None, float('nan'), None, None
            if len(leg_gt) >= 2:
                for mode in ladder:
                    traj, det, pt = self._plan_leg(ctx, cell, esdf, navigable, leg_gt, r_b,
                                                   mode, corridor_m, leg_tol_m,
                                                   gtc_cap_m=gtc_cap_m, gtc_fixed_m=gtc_fixed_m)
                    if traj is not None:
                        used = mode
                        break
            if traj is None:
                st = 'blocked'
            elif np.isfinite(det) and det > detour_thresh_m:
                st = 'detour'
            else:
                st = 'ok'
            legs_out.append(dict(i=i, frames=[int(lo), int(hi)], status=st, mode=used,
                                 detour_max_m=None if not np.isfinite(det) else round(float(det), 4),
                                 detour_point=None if (st != 'detour' or pt is None) else pt.tolist(),
                                 n_pts=0 if traj is None else int(len(traj))))
            status.append(st)
            if st == 'blocked':
                break                      # 여기를 못 지나가면 뒤쪽도 도달 불가
            pieces.append(traj)

        n_frames = len(gt)
        flags = frame_flags([tuple(l['frames']) for l in legs_out], status, n_frames)
        path = np.vstack(pieces) if pieces else np.zeros((0, 2))
        blocked = np.where(flags == 2)[0]
        # `path_xy`가 덮는 **GT 프레임 구간**. blocked에서 끊기므로 GT와 비교할 때 반드시 이 구간으로
        # 잘라야 한다 — 부분 경로를 전체 GT와 비교하면 거리가 터진다(실측 225 cm / 409 cm의 원인).
        n_good = len(pieces)
        covered = ([int(spine['legs'][0][0]), int(spine['legs'][n_good - 1][1])]
                   if n_good else None)
        return dict(legs=legs_out, path_xy=path, reach_ok=flags, frames_covered=covered,
                    first_blocked_frame=int(blocked[0]) if len(blocked) else None,
                    n_ok=status.count('ok'), n_detour=status.count('detour'),
                    n_blocked=status.count('blocked'),
                    frames_usable=int((flags != 2).sum()), n_frames=int(n_frames))

    def replan(self, scene, start_xy, goal_xy, r_b, h_nav_ratio=0.12, h_b=CAM_HEIGHT_M, floor_z=None,
               goal_tol_m=GOAL_TOL_M, plan_margin_m=PLAN_MARGIN_M):
        """e에 맞춰 GT path 재계획. -> trajectory(mesh world xy, (N,2)) 또는 None.

        None인 경우: (a) 계획 실패, (b) **라벨 유효성 4종 중 하나라도 어겨 기각**(`_path_ok`).
        (b)는 dilation이 goal 셀을 지워 스냅으로 도착지가 밀린 경우다 → 그 에피소드는 기존 r_b 경로를
        써야 한다(`replan_or_fallback`).

        `floor_z`는 **에피소드 층**을 넘겨야 한다(다층 씬: occ npz의 global floor_z는 다른 층일 수 있음).
        정합 결과에서 `cam_z - cam_height`로 구한다.
        """
        ctx = self._scene_ctx(scene)
        fz = ctx['floor_z'] if floor_z is None else float(floor_z)
        a = ctx['args']
        # 계획은 inflation 반경으로, 검증은 실제 r_b로 (PLAN_MARGIN_M 주석 참고)
        r_plan = float(r_b) + float(plan_margin_m)
        a.r_b, a.h_nav_ratio = r_plan, float(h_nav_ratio)
        # e-navigable로 start/goal 스냅(큰 로봇은 벽에서 더 안쪽에서 출발) — 안 하면 벽 근처 GT 끝점이
        # dilation으로 지워져 계획 불가. 논문상 embodiment 투영으로 정당.
        s = self._snap_navigable(ctx, fz, h_b, r_plan, float(h_nav_ratio), np.asarray(start_xy))
        g = self._snap_navigable(ctx, fz, h_b, r_plan, float(h_nav_ratio), np.asarray(goal_xy))
        if s is None or g is None:
            return None
        res = _p03.plan_episode(ctx['occ'], ctx['origin'], fz, h_b, s, g, a)
        if res.get('status') != 'ok':
            return None
        traj = np.asarray(res['trajectory'])
        # 라벨 유효성 4종(하드 충돌 / 몸통 침범 / unknown 통과 / target 도달) — 상세는 `_path_ok`.
        if not self._path_ok(ctx, traj, res['esdf'], float(a.cell_m), r_b, goal_xy, goal_tol_m):
            return None
        return traj

    def replan_or_fallback(self, scene, start_xy, goal_xy, r_b, h_nav_ratio=0.12, h_b=CAM_HEIGHT_M,
                           floor_z=None, baseline_r_b=BASELINE_R_B, goal_tol_m=GOAL_TOL_M):
        """`r_b`로 재계획하되, **끝점이 원본 goal과 달라지면 기각하고 기존 r_b 경로로 되돌린다.**

        -> (trajectory or None, used_r_b, status)
           status: 'ok' | 'fallback' (기각·계획실패로 baseline 사용) | 'failed' (baseline도 실패)
        """
        traj = self.replan(scene, start_xy, goal_xy, r_b, h_nav_ratio, h_b, floor_z, goal_tol_m)
        if traj is not None:
            return traj, float(r_b), 'ok'
        base = self.replan(scene, start_xy, goal_xy, baseline_r_b, h_nav_ratio, h_b, floor_z, goal_tol_m)
        if base is not None:
            return base, float(baseline_r_b), 'fallback'
        return None, None, 'failed'

    def fov_mask(self, ctx, cam_pose_mesh, floor_z, rig='125cm_30deg', margin_px=0, near_radius_m=1.0):
        """현재 카메라의 **수평 시야 안**에 있는 격자 셀만 True인 마스크.

        경로가 화면 **좌우로 벗어나는 것을 원천 차단**하기 위해 A* 입력 navigable에 AND한다.
        (사후 기각이 아니라 계획 단계에서 막는다 → 좌우 이탈 0%가 구조적으로 보장된다.)

        세로는 제한하지 않는다: 룩다운 카메라는 **발밑 근거리가 이미지 아래로 빠지는데**(원본 GT도 31%),
        이는 데이터 고유 성질이고 로봇 자신의 발밑이라 문제되지 않는다. 막는 것은 **좌우**뿐이다.
        """
        from pixel_goal_utils import intrinsics_for_rig
        fx, _fy, cx, _cy = intrinsics_for_rig(rig)
        occ = ctx['occ']; origin = ctx['origin']; cell = float(ctx['args'].cell_m)
        ny, nx = occ.shape[1], occ.shape[0]          # occupancy는 (Nx,Ny,Nz)
        xs = origin[0] + (np.arange(nx) + 0.5) * cell
        ys = origin[1] + (np.arange(ny) + 0.5) * cell
        X, Y = np.meshgrid(xs, ys)                    # (ny,nx) — esdf/navigable과 같은 배열 규약
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, floor_z), np.ones(X.size)])
        Pc = np.linalg.inv(np.asarray(cam_pose_mesh, dtype=np.float64)) @ pts
        Z = Pc[2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = cx + fx * Pc[0] / Z
        ok = (Z > 1e-6) & np.isfinite(u) & (u >= -margin_px) & (u < 640 + margin_px)
        # **근거리 예외**: 카메라 원점(=로봇 자신)은 Z≈0이라 마스크에서 빠진다 → 출발 셀이 막혀
        # 계획이 전부 실패한다(실측: 경로 12→1). 로봇 주변 `near_radius_m`는 항상 허용한다.
        cam_xy = np.asarray(cam_pose_mesh, dtype=np.float64)[:2, 3]
        near = ((pts[0] - cam_xy[0]) ** 2 + (pts[1] - cam_xy[1]) ** 2) <= near_radius_m ** 2
        return (ok | near).reshape(X.shape)

    def replan_in_fov(self, scene, start_xy, goal_xy, r_b, cam_pose_mesh, rig='125cm_30deg',
                      h_nav_ratio=0.12, h_b=CAM_HEIGHT_M, floor_z=None, goal_tol_m=GOAL_TOL_M,
                      plan_margin_m=PLAN_MARGIN_M):
        """`replan`과 같되 **수평 시야 밖 셀을 A*가 못 밟게** 막는다. -> trajectory or None.

        `plan_episode`(기존 코드)는 내부에서 navigable을 만들어 마스크를 끼울 수 없으므로,
        같은 체인을 `esdf_utils` 순수 함수로 조립하고 그 사이에 FOV 마스크를 AND한다(재구현 아님, 조립).
        """
        from esdf_utils import (astar, cell_to_world, check_path_navigable, compute_scan_coverage_mask,
                                downsample_navigable, greedy_refine, smooth_cubic_spline, thin_waypoints,
                                world_to_cell)
        ctx = self._scene_ctx(scene)
        a = ctx['args']
        fz = ctx['floor_z'] if floor_z is None else float(floor_z)
        cell = float(a.cell_m)
        obstacle = derive_obstacle_2d(ctx['occ'], ctx['origin'], fz, h_b, cell, h_nav_ratio * h_b)
        esdf = compute_esdf_2d(obstacle, cell)
        # 계획은 inflation 반경으로, 검증은 실제 r_b로 (PLAN_MARGIN_M 주석 참고)
        navigable = truncate_navigable(esdf, float(r_b) + float(plan_margin_m)) \
            & compute_scan_coverage_mask(ctx['occ'])
        navigable = navigable & self.fov_mask(ctx, cam_pose_mesh, fz, rig)     # ← 좌우 이탈 차단
        nav_coarse = downsample_navigable(navigable, a.downsample_factor, a.downsample_mode)

        # start/goal이 dilation으로 지워졌으면 **마스크된 navigable 위로 스냅**한다.
        # (`replan`은 `_snap_navigable`로 이미 하고 있다. 이걸 빼먹으면 계획이 거의 전부 실패한다 —
        #  실측: 스냅 없이는 12개 중 1개만 성공.)
        s_snap = snap_to_grid(navigable, ctx['origin'], cell, start_xy)
        g_snap = snap_to_grid(navigable, ctx['origin'], cell, goal_xy)
        if s_snap is None or g_snap is None:
            return None
        sg = np.array([s_snap, g_snap], dtype=np.float64)

        coarse = cell * a.downsample_factor
        ij = world_to_cell(sg, ctx['origin'], coarse)
        hc, wc = nav_coarse.shape
        ij[:, 0] = np.clip(ij[:, 0], 0, wc - 1); ij[:, 1] = np.clip(ij[:, 1], 0, hc - 1)
        if not nav_coarse[ij[0, 1], ij[0, 0]] or not nav_coarse[ij[1, 1], ij[1, 0]]:
            return None
        path_ij = astar(nav_coarse, ij[0], ij[1])
        if path_ij is None:
            return None
        wp = cell_to_world(path_ij, ctx['origin'], coarse)
        wp[0], wp[-1] = sg[0], sg[1]
        wp = greedy_refine(wp, esdf, ctx['origin'], cell, a.refine_radius, fix_endpoints=True)
        traj = smooth_cubic_spline(thin_waypoints(wp, a.waypoint_spacing_m), a.smooth_step)
        traj = np.asarray(traj)
        # `replan`과 같은 라벨 유효성 4종. 이전 판은 (a) `check_path_navigable`을 import만 해두고
        # 호출하지 않았고, (b) 도달 여부를 **스냅된** goal(`sg[1]`)과 비교해 목표가 밀려도 통과했다.
        # 비교 대상은 요청된 `goal_xy`여야 한다.
        if not self._path_ok(ctx, traj, esdf, cell, r_b, goal_xy, goal_tol_m):
            return None
        return traj

    def _snap_navigable(self, ctx, floor_z, h_b, r_b, h_nav_ratio, xy):
        """xy가 e-navigable이 아니면 가장 가까운 navigable 셀 중심으로 스냅. navigable 없으면 None."""
        cell = float(ctx['args'].cell_m)
        obs = derive_obstacle_2d(ctx['occ'], ctx['origin'], floor_z, h_b, cell, h_nav_ratio * h_b)
        nav = truncate_navigable(compute_esdf_2d(obs, cell), r_b) & compute_scan_coverage_mask(ctx['occ'])
        ij = world_to_cell(xy[None], ctx['origin'], cell)[0]
        h, w = nav.shape
        if 0 <= ij[1] < h and 0 <= ij[0] < w and nav[ij[1], ij[0]]:
            return xy
        cells = np.argwhere(nav)  # (M,2) [iy,ix]
        if len(cells) == 0:
            return None
        best = cells[np.argmin((cells[:, 1] - ij[0]) ** 2 + (cells[:, 0] - ij[1]) ** 2)]
        return cell_to_world(np.array([[best[1], best[0]]]), ctx['origin'], cell)[0]

    def render_bev_along(self, scene, traj_xy, n_frames=6, h_b=CAM_HEIGHT_M, pitch=CAM_PITCH_DEG,
                         floor_z=None, with_bev=True):
        """새 path를 따라 n_frames에서 depth 렌더 (+선택적 BEV).

        **학습 경로에서는 `with_bev=False`로 쓴다. 이유는 "느려서"가 아니라 "중복이라서"다.**
        - S1 BEV는 모델 하류(`internvla_n1_unified_provider`)가 `traj_depths`에서 **GPU로** 계산한다.
        - 경로 계획도 BEV를 쓰지 않는다 — 캐시된 **3D occupancy**(mesh 유래)에서 `derive_obstacle_2d`로
          2D를 뽑는다. 즉 이 loader에는 BEV가 **필요한 곳이 없다**. (BEV를 planning 격자로 쓰는 것은
          2dloader 쪽 설계다.)
        ⚠️ 예전 주석의 "CPU BEV가 165 ms/frame로 병목"은 **오측이었다** — torch 스레드가 코어 수(24)와
        같아 생긴 경합 현상이고, 스레드 1개로 제한하면 **3 ms/frame**이다(60배). BEV는 애초에 병목이 아니다.
        시각화 검증용으로만 True.

        -> (poses_xy(n,2), depths[n,H,W], bevs[n,H,W] or None)
        """
        ctx = self._scene_ctx(scene)
        fz = ctx['floor_z'] if floor_z is None else float(floor_z)
        idxs = np.unique(np.linspace(0, len(traj_xy) - 1, min(n_frames, len(traj_xy))).astype(int))
        sub = traj_xy[idxs]
        # 인자 순서 주의: (xy_world, floor_z, h_b, pitch_down_deg) — geometry_utils.py:253
        poses = synthesize_action_poses(traj_xy, fz, h_b, pitch)[idxs]
        return self._render_poses(scene, poses, sub, h_b, pitch, with_bev)

    def render_at_poses(self, scene, poses_abs_c2w, h_b=CAM_HEIGHT_M, pitch=CAM_PITCH_DEG, with_bev=True,
                        r_b=None):
        """**모드 (B)**: 주어진(원본 에피소드) 카메라 pose에서 렌더. 관측 위치는 그대로 두고,
        `r_b`를 주면 BEV에 dilation을 적용해 **`e`를 관측에 명시적으로 주입**한다.

        -> (xy(n,2), depths[n,H,W], bevs[n,H,W] or None)
        """
        xy = np.stack([c[:3, 3][:2] for c in poses_abs_c2w])
        depths = np.stack([self._render_depth(scene, c) for c in poses_abs_c2w])
        if not with_bev:
            return xy, depths, None
        bev = depth_to_bev_occ_ros2(torch.from_numpy(depths), cam_height=h_b, cam_pitch_deg=pitch,
                                    fx=self.K['fx'], fy=self.K['fy'], cx=self.K['cx'], cy=self.K['cy'],
                                    bev_range=5.0, bev_size=self.res).numpy()
        if r_b is not None:
            bev = dilate_bev_occupancy(bev, r_b)
        return xy, depths, bev

    def _render_poses(self, scene, poses, sub, h_b, pitch, with_bev):
        depths = np.stack([self._render_depth(scene, action_to_c2w(p, 'cam2world_gl')) for p in poses])
        if not with_bev:
            return sub, depths, None
        bev = depth_to_bev_occ_ros2(torch.from_numpy(depths), cam_height=h_b, cam_pitch_deg=pitch,
                                    fx=self.K['fx'], fy=self.K['fy'], cx=self.K['cx'], cy=self.K['cy'],
                                    bev_range=5.0, bev_size=self.res).numpy()
        return sub, depths, bev


# ---------------------------------------------------------------------------
# self-check — `_path_ok`의 4개 기각 조건이 각각 실제로 발동하는지 확인한다.
# 씬/렌더러 없이 합성 격자로 돌린다: /usr/bin/python .../embodiment_augment.py
# ---------------------------------------------------------------------------
def _selfcheck():
    cell, n = 0.05, 60
    origin = np.zeros(3)
    # 세로 벽(x=1.0 m 근방)에 폭 40 cm 문이 있는 방. occupancy는 안 쓰고 esdf/coverage만 합성한다.
    obstacle = np.zeros((n, n), dtype=bool)
    wx = int(1.0 / cell)
    obstacle[:, wx] = True
    door = slice(int(1.0 / cell), int(1.4 / cell))          # y 1.0~1.4 m
    obstacle[door, wx] = False
    esdf = compute_esdf_2d(obstacle, cell)
    coverage = np.ones((n, n), dtype=bool)
    coverage[:, int(2.0 / cell):] = False                    # x > 2.0 m 는 미관측
    ctx = {'origin': origin, 'coverage': coverage}
    aug = EmbodimentAugmenter.__new__(EmbodimentAugmenter)   # __init__(렌더러) 없이 메서드만 쓴다

    def line(a, b, step=0.01):
        a, b = np.array(a, float), np.array(b, float)
        k = max(2, int(np.linalg.norm(b - a) / step) + 1)
        return a + (b - a) * np.linspace(0, 1, k)[:, None]

    goal = (1.6, 1.2)
    door_y = 1.2
    # (1) 문을 통과하는 정상 경로 — 작은 로봇이면 전부 통과해야 한다.
    ok = line((0.5, door_y), goal)
    assert aug._path_ok(ctx, ok, esdf, cell, 0.05, goal), '정상 경로가 기각됐다'
    # (2) 벽을 관통 — 하드 충돌
    thru = line((0.5, 0.3), (1.6, 0.3))
    assert not aug._path_ok(ctx, thru, esdf, cell, 0.05, (1.6, 0.3)), '하드 충돌이 안 잡혔다'
    # (3) 문 폭 40 cm → 반경 0.25 로봇은 몸통이 겹친다(문 중앙 clearance 0.2 m)
    assert not aug._path_ok(ctx, ok, esdf, cell, 0.25, goal), '몸통 침범이 안 잡혔다'
    # (4) 미관측(x > 2.0 m)으로 진입
    unk = line((0.5, door_y), (2.5, door_y))
    assert not aug._path_ok(ctx, unk, esdf, cell, 0.05, (2.5, door_y)), '미관측 통과가 안 잡혔다'
    # (5) target 미달 — 끝점이 goal에서 GOAL_TOL_M 초과
    short = line((0.5, door_y), (1.2, door_y))
    assert not aug._path_ok(ctx, short, esdf, cell, 0.05, goal), 'target 미달이 안 잡혔다'
    # (6) 허용오차 안이면 통과 (격자 양자화 오차는 허용해야 한다)
    near = line((0.5, door_y), (goal[0] - GOAL_TOL_M * 0.5, door_y))
    assert aug._path_ok(ctx, near, esdf, cell, 0.05, goal), '허용오차 내 도달이 기각됐다'
    print('[selfcheck] _path_ok 6/6 통과 (정상·하드충돌·몸통침범·미관측·target미달·허용오차)')

    # --- `none` rung: 통과 가능하면 GT를 **그대로** 쓴다 ---
    tol = LEG_TOL_M
    # (1) identity — 배열이 바뀌지 않고 이탈량이 정확히 0
    out, det, pt = aug._leg_verbatim(ctx, cell, esdf, ok, 0.05, tol)
    assert out is not None and np.array_equal(out, ok), 'identity 실패: GT가 그대로 반환되지 않았다'
    assert det == 0.0 and pt is None, f'이탈량이 0이 아니다 {det}'
    # (2) 벽 관통은 반경과 무관하게 기각 (hard_ok)
    assert aug._leg_verbatim(ctx, cell, esdf, thru, 0.05, tol)[0] is None, '벽 관통이 통과했다'
    # (3) 문 중앙 clearance 0.20 → r_b=0.25는 기각 → ladder가 nudge로 내려간다
    assert aug._leg_verbatim(ctx, cell, esdf, ok, 0.25, tol)[0] is None, '몸통 침범이 통과했다'
    # (4) 미관측 통과는 GT라도 기각
    assert aug._leg_verbatim(ctx, cell, esdf, unk, 0.05, tol)[0] is None, '미관측 통과가 승인됐다'
    # (5) `_plan_leg`이 mode='none'을 `_leg_verbatim`으로 위임하나 (navigable/corridor는 안 쓴다)
    p, d, _ = aug._plan_leg(ctx, cell, esdf, np.ones((n, n), bool), ok, 0.05, 'none', 0.75, tol)
    assert p is not None and np.array_equal(p, ok) and d == 0.0, 'mode=none 위임이 안 된다'
    # (6) ladder 밑단이 'none'인가 — 순서가 뒤집히면 통과 가능한 leg가 전부 밀려난다
    import inspect
    assert inspect.signature(EmbodimentAugmenter.follow_waypoints).parameters['ladder'].default[0] \
        == 'none', 'ladder 기본값의 밑단이 none이 아니다'
    print('[selfcheck] _leg_verbatim 6/6 통과 (identity·하드충돌·몸통침범·미관측·위임·ladder밑단)')


if __name__ == '__main__':
    _selfcheck()
