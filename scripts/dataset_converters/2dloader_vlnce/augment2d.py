"""on-the-fly embodiment augmentation — depth 한 장 + `e` -> 새 GT path & pixel goal.

한 프레임 안에서 닫힌다 (씬 mesh·3D occupancy·렌더러 불필요):

    depth ─(장애물 합성)→ depth' ─(BEV)→ local map ─(e: r_b/h_nav/h_b)→ ESDF
                                                          │
      원본 goal ─(adjust_goal)→ 안전 goal ─(A*+refine+spline)→ 새 GT path
                                                          │
                                              robot→world→projection → 새 pixel goal

`e`가 관측(depth/BEV)과 GT(path/goal)를 **동시에** 바꾼다 — 논문 §제안 차별점 (나).

## 끝점 보호 (3dloader에서 승계)
새 경로의 끝점이 조정된 goal에서 `goal_tol_m`(10 cm) 넘게 벗어나면 **기각**하고 baseline `r_b`로
되돌린다. 그렇지 않으면 도착지가 밀려 instruction("…에서 멈춰라")이 거짓 라벨이 된다.

self-check: `/usr/bin/python scripts/dataset_converters/2dloader_vlnce/augment2d.py`
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_CONV = _HERE.parent
for _p in (str(_CONV / 'gs_vlnpe'), str(_CONV / '3dloader_vlnce'), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esdf_utils import check_path_navigable, sample_esdf_at  # noqa: E402
from pixel_goal_utils import (  # noqa: E402
    check_visible, goal_world_from_poses, intrinsics_for_rig, project_to_pixel, rig_height_m,
)

from episode_io import IMG_H, IMG_W, to_224_depth  # noqa: E402
from local_map import (  # noqa: E402
    BEV_RANGE_M, BEV_SIZE, DOWNSAMPLE_FACTOR, build_local_map, limit_torch_threads, plan_path,
    robot_to_plan, robot_to_world, robot_xy_to_bev_ij, world_to_robot,
)
from obstacle_synth import Cam, ObstacleCfg, composite, sample_obstacle  # noqa: E402

BASELINE_R_B = 0.10       # habitat AgentConfig().radius — 원본 GT가 수집된 반경 (항등 게이트 기준)
GOAL_TOL_M = 0.10


@dataclass
class Embodiment:
    """`e`의 연속 축. `h_nav`/`h_b`는 BEV slab, `r_b`는 ESDF 임계로 들어간다."""
    r_b: float = BASELINE_R_B
    h_nav: float = 0.15
    h_b: float = 1.25

    def tag(self) -> str:
        return f'rb{self.r_b:.2f}_hn{self.h_nav:.2f}_hb{self.h_b:.2f}'


@dataclass
class Aug2DCfg:
    unknown: str = 'nontraversable'
    bev_range: float = BEV_RANGE_M
    bev_size: int = BEV_SIZE
    # 'retreat'(2D 권장) | 'shift' | 'nearest'. 실측(17DRP ep0, 6프레임×4 r_b):
    #   retreat  기각 0 · adjusted 8 · V2 단조 True   <- 채택
    #   shift    기각 4 · adjusted 4 · V2 단조 True(단, 거리 고정이라 자명)
    #   nearest  기각 0 · adjusted 8 · V2 단조 **False**
    # 3dloader(3D mesh)에서는 shift가 최선이었는데 2D에서 뒤집힌다 — shift는 거리를 유지한 채 옆으로
    # 트는데, 관측 부채꼴이 좁아 그 방위가 **미관측 영역**으로 나가 안전점을 못 찾기 때문이다.
    goal_adjust: str = 'retreat'
    goal_tol_m: float = GOAL_TOL_M
    baseline_r_b: float = BASELINE_R_B
    # 후퇴의 하한. 이게 없으면 목표가 로봇 자기 자리(0 m)까지 당겨져 라벨이 무의미해진다(실측 f23).
    min_goal_m: float = 0.50
    obstacle: bool = True
    obstacle_cfg: ObstacleCfg = field(default_factory=ObstacleCfg)
    downsample_factor: int = DOWNSAMPLE_FACTOR
    refine_radius_m: float = 0.30
    # 03 기본값(cubic·0.8 m)은 최종 경로가 장애물을 통과한다 — local_map.plan_path의 실측표 참고
    spacing_m: float = 0.4
    refine_mode: str = 'argmax'      # esdf_utils.REFINERS — 'argmax' | 'min_move'
    smooth_mode: str = 'bezier'      # esdf_utils.SMOOTHERS — 볼록껍질을 벗어나지 않는다
    downsample_mode: str = 'any'
    max_unfree: float = 0.02         # 최종 경로가 관측 free 밖에 있어도 되는 비율 상한
    visible_check: bool = True


class Augmenter2D:
    """rig 한 쌍(FPV pitch_1 / 룩다운 pitch_2)에 묶인 augmenter. 프레임마다 `augment()`."""

    def __init__(self, rig_fpv: str, rig_ld: str, cfg: Aug2DCfg = None):
        limit_torch_threads()      # worker 스코프. 없으면 BEV 한 장에 427 ms (실측)
        self.rig_fpv, self.rig_ld = rig_fpv, rig_ld
        self.cfg = cfg or Aug2DCfg()
        self.pitch_fpv = float(rig_fpv.split('_')[1].replace('deg', ''))
        self.pitch_ld = float(rig_ld.split('_')[1].replace('deg', ''))
        self.cam_height = rig_height_m(rig_ld)
        self.cam_fpv = Cam(self.pitch_fpv, rig_height_m(rig_fpv), *intrinsics_for_rig(rig_fpv), IMG_W, IMG_H)
        self.cam_ld = Cam(self.pitch_ld, rig_height_m(rig_ld), *intrinsics_for_rig(rig_ld), IMG_W, IMG_H)

    # -- 기하 헬퍼 ---------------------------------------------------------
    def gt_path_robot(self, poses_ld: np.ndarray, frame_idx: int, goal_idx: int) -> np.ndarray:
        """원본 GT 경로 (robot XY). W0에서 학습 `traj_poses` 프레임과 3.6e-8 m 일치 확인됨."""
        fut = poses_ld[frame_idx:goal_idx + 1, :3, 3]
        return world_to_robot(fut, poses_ld[frame_idx], self.pitch_ld, self.cam_height)[:, :2]

    def goal_robot(self, poses_ld: np.ndarray, frame_idx: int, goal_idx: int) -> np.ndarray:
        G = goal_world_from_poses(poses_ld[frame_idx:goal_idx + 1], self.rig_ld)
        return world_to_robot(G[None], poses_ld[frame_idx], self.pitch_ld, self.cam_height)[0]

    def pixel_goal(self, goal_xy_robot: np.ndarray, pose_ld: np.ndarray, depth_m=None):
        """robot XY(바닥점) -> 룩다운 이미지 (u, v, Z) + 가시성. depth를 주면 가림까지 본다."""
        p = np.array([goal_xy_robot[0], goal_xy_robot[1], 0.0])
        G = robot_to_world(p[None], pose_ld, self.pitch_ld, self.cam_height)[0]
        u, v, Z = project_to_pixel(G, pose_ld, self.rig_ld)
        ok, why = check_visible(u, v, Z, depth_m if self.cfg.visible_check else None)
        return u, v, Z, ok, why

    # -- 본체 -------------------------------------------------------------
    def augment(self, frame, poses_ld: np.ndarray, e: Embodiment,
                rng: np.random.Generator = None, with_obstacle: Optional[bool] = None) -> dict:
        """한 프레임 augmentation. 반환 dict의 `status`는
        'ok' | 'fallback' | 'rejected:<why>' 중 하나다."""
        cfg = self.cfg
        rng = rng if rng is not None else np.random.default_rng(0)
        want_obs = cfg.obstacle if with_obstacle is None else with_obstacle
        gi = frame.idx + frame.rel_goal + 1
        gt = self.gt_path_robot(poses_ld, frame.idx, gi)
        goal0 = self.goal_robot(poses_ld, frame.idx, gi)[:2]

        # ① 장애물 합성 — RGB(두 rig)·depth를 한 3D 물체로 동시에 바꾼다.
        # `gap_m` 배치는 **넣기 전 지도의 여유**를 알아야 하므로 map을 먼저 한 번 만든다
        # (장애물을 안 쓰면 이 비용은 발생하지 않는다).
        obs = None
        if want_obs:
            # 이 지도는 **한 점의 clearance(esdf)** 를 읽으려는 것뿐이다. 바닥 carving은 `free`/
            # `esdf_nav`만 바꾸고 `esdf`에는 영향이 없으므로 끈다 (프레임당 3.3 ms 절약).
            m0 = build_local_map(to_224_depth(frame.depth_m), self.rig_ld, self.pitch_ld,
                                 e.r_b, e.h_nav, e.h_b, cfg.bev_range, cfg.bev_size,
                                 unknown=cfg.unknown, free_from_floor=False)
            obs = sample_obstacle(gt, rng, self.cam_ld, cfg.obstacle_cfg, m=m0)
        rgb_ld, depth, rgb_fpv = frame.rgb_ld, frame.depth_m, frame.rgb_fpv
        if obs is not None:
            rgb_ld, depth, _ = composite(frame.rgb_ld, frame.depth_m, obs, self.cam_ld)
            rgb_fpv, _, _ = composite(frame.rgb_fpv, frame.depth_fpv_m, obs, self.cam_fpv)

        # ② 관측 -> local map (e의 높이축이 여기서 들어간다)
        depth224 = to_224_depth(depth)
        m = build_local_map(depth224, self.rig_ld, self.pitch_ld, e.r_b, e.h_nav, e.h_b,
                            cfg.bev_range, cfg.bev_size, unknown=cfg.unknown)
        out = dict(obstacle=obs, rgb_ld=rgb_ld, rgb_fpv=rgb_fpv, depth=depth, depth224=depth224,
                   map=m, gt=gt, goal0=goal0, e=e)

        # 원본 GT가 이 e에서 통행 가능한가 (논문의 "e가 경로를 바꾼다"의 반대 증거)
        out['gt_check'] = check_path_navigable(robot_to_plan(gt), m['esdf'], m['origin'],
                                               m['cell_m'], e.r_b)

        # ③④ goal 판정 + 재계획 (아래 `plan_and_label` 참고 — 둘을 함께 정해야 단조성이 성립한다)
        out.update(self.plan_and_label(m, goal0, gt, e))
        if out['status'].startswith('rejected'):
            return out

        # ⑤ 새 pixel goal — **라벨용 goal**로 투영한다
        # 가림 판정은 **640×480 원본 해상도** depth로 한다 (u,v가 그 좌표계다)
        u, v, Z, ok, why = self.pixel_goal(out['label_goal'], poses_ld[frame.idx], depth)
        out.update(pixel_goal=(u, v), pixel_Z=Z, pixel_ok=ok, pixel_why=why)
        if not ok:
            out['status'] = f'rejected:{why}'
        return out

    def decide_goal(self, m: dict, goal0: np.ndarray, gt: np.ndarray, e: Embodiment) -> dict:
        """goal 셀 자체를 이 `e`로 밟을 수 있는지만 본다 (계획은 `plan_and_label`이 한다).

        ⚠️ **미관측**과 **통행 불가**를 섞으면 baseline에서도 라벨이 튄다. 실측(s8pcmisQ38h ep0 f11):
        원본 goal이 BEV 안에 있지만 관측된 free 셀이 아니어서, 실제 장애물까지 clearance가 0.34 m로
        충분한데도 `esdf_nav`가 미관측을 0으로 눌러 라벨이 **371 px** 끌려갔다(baseline r_b=0.10인데도).
        """
        S = m['bev_size']
        gij = robot_xy_to_bev_ij(goal0[None])[0]
        in_map = 0 <= gij[0] < S and 0 <= gij[1] < S
        observed = bool(in_map and m['free'][gij[0], gij[1]])
        clear0 = float(sample_esdf_at(m['esdf'], robot_to_plan(goal0[None]),
                                      m['origin'], m['cell_m'])[0])
        return dict(goal_observed=observed, goal_clearance=clear0,
                    goal_tight=bool(observed and clear0 < e.r_b))

    def _plan_to_farthest(self, m: dict, goal: np.ndarray, gt: np.ndarray, e: Embodiment):
        """`goal`로 계획하고, 안 되면 **GT 경로를 따라 뒤로 물러나며 가장 먼 도달 지점**을 찾는다.

        반환 `(trajectory, goal_used, status, nav_coarse)` — `nav_coarse`는 **A*가 실제로 쓴 격자**로,
        시각화가 "경로를 고른 근거 지도"를 그리는 데 쓴다.

        왜 clearance만으로 물리면 안 되는가 (실측, 17DRP ep0 f29, 장애물 있음):
        `r_b`=0.10/0.20은 goal(3.18 m)의 clearance가 충분해 그대로 두고 계획했다가 실패(`rejected`),
        0.35/0.50은 clearance 부족으로 goal이 0.24 m까지 당겨져 **쉽게 성공**했다. 즉 큰 로봇이
        작은 로봇보다 잘 되는 뒤집힌 결과가 나온다. **도달 가능성으로 물려야** 단조가 성립한다:
        작은 로봇은 더 멀리, 큰 로봇은 더 일찍 멈춘다.
        """
        cfg = self.cfg
        cands = [np.asarray(goal, dtype=np.float64)]
        p = np.asarray(gt, dtype=np.float64)[:, :2]
        if len(p) > 1:                                   # 먼 쪽부터, 최대 8지점만 시도
            idx = np.unique(np.linspace(len(p) - 1, 0, 8).astype(int))
            cands += [p[k] for k in idx if np.linalg.norm(p[k]) >= cfg.min_goal_m]
        last, first_nav = 'astar_failed', None
        for c in cands:
            pl = plan_path(m, c, cfg.downsample_factor, cfg.refine_radius_m, cfg.spacing_m,
                           cfg.refine_mode, cfg.smooth_mode, cfg.downsample_mode, cfg.max_unfree)
            first_nav = first_nav if first_nav is not None else pl.get('nav_coarse')
            if pl['status'] != 'ok':
                last = pl['status']; continue
            if float(np.linalg.norm(pl['trajectory'][-1] - c)) <= cfg.goal_tol_m:
                return pl['trajectory'], c, 'ok', pl['nav_coarse']
            last = 'endpoint_off'
        # 실패해도 **A*에게 준 격자**는 돌려준다 — "이 지도에서 못 찾았다"를 보여야 하므로
        return None, None, last, first_nav

    def plan_and_label(self, m: dict, goal0: np.ndarray, gt: np.ndarray, e: Embodiment) -> dict:
        """계획 목표와 pixel goal **라벨**을 함께 정한다.

        라벨 규칙 — 실제로 도달한 지점을 따른다:
        - 도달점이 원본 goal에서 `goal_tol_m` 이내 -> `unchanged` (**라벨 그대로**)
        - 벗어났고 원본 goal이 관측된 칸이었다 -> `adjusted` (**라벨 갱신**: 이 로봇은 거기까지 못 간다)
        - 벗어났고 원본 goal이 미관측이었다 -> `unobserved` (**라벨 그대로**: 못 본 곳을 위험하다 할 근거 없음)

        baseline 항등이 깨지지 않는 이유: baseline `r_b`에서 원본 goal에 도달하지 못하면 그 샘플은
        아예 기각되므로(아래 fallback 분기), 라벨이 틀린 채로 남는 경우가 없다.
        """
        cfg = self.cfg
        info = self.decide_goal(m, goal0, gt, e)
        traj, used, st, nav_coarse = self._plan_to_farthest(m, goal0, gt, e)
        out = dict(info, nav_coarse=nav_coarse)
        if traj is not None:
            moved = float(np.linalg.norm(used - goal0))
            gstat = ('unchanged' if moved <= cfg.goal_tol_m
                     else ('adjusted' if info['goal_observed'] else 'unobserved'))
            out.update(status='ok', trajectory=traj, goal=used, plan_status='ok',
                       endpoint_err=moved,
                       label_goal=(used if gstat == 'adjusted' else goal0), goal_status=gstat)
            return out

        # 이 e로는 `min_goal_m` 이상 나아갈 수 없다 -> 기각.
        # (3dloader의 baseline fallback은 여기선 쓰지 않는다: 라벨을 원본으로 되돌리면 "큰 로봇이
        #  작은 로봇보다 멀리 간다"는 비단조가 생긴다 — 실측 17DRP f23. 기각이 정직하고 단조롭다.)
        out.update(status=f'rejected:{st}', trajectory=None, goal=None, label_goal=goal0,
                   goal_status='no_safe_point', plan_status=st, endpoint_err=float('nan'))
        return out


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    from episode_io import load_episode

    DR = os.environ.get('VLNCE_ROOT', 'data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ep = load_episode(DR, '17DRP5sb8fy', 0, '125cm_0_30', n_frames=4)
    aug = Augmenter2D(ep.rig_fpv, ep.rig_ld)
    f = ep.frames[0]

    # 1) baseline e + 장애물 없음 -> 관측이 원본과 **완전히 동일**해야 한다 (항등)
    r = aug.augment(f, ep.poses_ld, Embodiment(), with_obstacle=False)
    assert np.array_equal(r['rgb_ld'], f.rgb_ld) and np.array_equal(r['depth'], f.depth_m), '항등 깨짐'
    assert r['goal_status'] == 'unchanged', (r['goal_status'], r.get('endpoint_err'))
    print(f"[aug] baseline: status={r['status']} goal={r['goal_status']} "
          f"endpoint_err={r.get('endpoint_err', float('nan')):.3f}m "
          f"GT feasible={r['gt_check']['points_ok']} min_clear={r['gt_check']['points_min_m']:.2f}m")
    # 경로가 점유·미관측 칸을 지나지 않아야 한다 (planner 기본값의 근거)
    from local_map import path_violations
    v = path_violations(r['map'], r['trajectory'])
    assert v['occ'] == 0.0 and v['unfree'] <= 0.02, v
    print(f"[aug] 경로 위반 점유 {100*v['occ']:.0f}% / 미관측 {100*v['unfree']:.0f}% (표본 {v['n']})")

    # 2) r_b를 키우면 원본 GT가 통행 불가로 바뀌어야 한다
    feas = [aug.augment(f, ep.poses_ld, Embodiment(r_b=rb), with_obstacle=False)['gt_check']['points_ok']
            for rb in (0.10, 0.20, 0.35, 0.50)]
    print(f'[aug] 원본 GT feasibility r_b=0.10/0.20/0.35/0.50 -> {feas}')
    assert feas[0] and not feas[-1], f'r_b가 GT 통행성을 안 바꾼다: {feas}'
    assert all(feas[i] >= feas[i + 1] for i in range(len(feas) - 1)), f'단조성 위반 {feas}'

    # 3) 장애물을 넣으면 관측이 바뀌고 계획이 여전히 돌아간다
    rng = np.random.default_rng(0)
    r3 = aug.augment(f, ep.poses_ld, Embodiment(r_b=0.20), rng=rng, with_obstacle=True)
    assert r3['obstacle'] is not None and not np.array_equal(r3['depth'], f.depth_m)
    print(f"[aug] 장애물 있음: status={r3['status']} plan={r3['plan_status']} "
          f"traj={None if r3.get('trajectory') is None else len(r3['trajectory'])}pt")
    print('[aug] PASS')
