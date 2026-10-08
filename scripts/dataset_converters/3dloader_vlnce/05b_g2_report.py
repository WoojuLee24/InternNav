"""G2 — `e`(로봇 반경 `r_b`)가 GT path를 바꾸는지 검증. **vln_ce 기준.**

## 왜 다시 썼나
초판은 `03_sample_gt_paths.py`를 썼는데 그건 `--dataset {vln_n1, vln_pe}`만 지원한다. 그런데 이 작업이
학습에 쓰는 데이터셋은 **vln_ce**다(G1·P-A·R3·P-B1·pixel-goal 전부 vln_ce). 데이터셋이 다르면
baseline `r_b`도 다르다 — vln_n1은 0.25(`esdf_utils.ROBOT_RADIUS_M`, NavDP), **vln_ce는 0.1**
(habitat 기본 agent radius; `vln_r2r_mini.yaml`에 미지정). 그래서 vln_ce로 다시 측정한다.

## 무엇을 재나
에피소드마다 **원본 GT 궤적**(vln_ce에 저장된 카메라 궤적을 mesh 좌표로 정합한 것)을 기준으로,
같은 start/goal에서 `r_b`별로 재계획한 경로를 비교한다.
- `gt_dist` : 원본 GT와의 평균 거리 → **baseline(0.1)에서 최소여야** 파이프라인이 원본을 재현한다는 뜻
- `shift`   : 최소 r_b 경로 대비 이동량 → `e`가 경로를 바꾸는 정도
- `feasible`: 계획 성공 수 → 큰 로봇이 못 지나가는 물리적 사실

재사용: `vlnce_align`(정합), `EmbodimentAugmenter.replan`(재계획),
`viz_utils.floorplan_canvas`(배경 평면도는 에피소드 계획 격자로 직접 만든다).

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/05b_g2_report.py --scene 17DRP5sb8fy --r_bs 0.10,0.25,0.40,0.55
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_GS = _HERE.parents[0] / 'gs_vlnpe'
sys.path.insert(0, str(_GS)); sys.path.insert(0, str(_HERE))
from viz_utils import floorplan_canvas, save_gallery  # noqa: E402
from embodiment_augment import EmbodimentAugmenter, _p03  # noqa: E402
from esdf_utils import (  # noqa: E402
    compute_esdf_2d, compute_scan_coverage_mask, derive_obstacle_2d, truncate_navigable,
)
from pixel_goal_utils import clearance_at, rig_height_m  # noqa: E402
_m_verify_pose_mesh = importlib.import_module('00_verify_pose_mesh')  # noqa: E402
load_episode = _m_verify_pose_mesh.load_episode  # noqa: E402

RB_COLORS = [(90, 190, 255), (255, 210, 60), (255, 90, 90), (0, 255, 90), (200, 120, 255)]
GT_COLOR, START_COLOR, GOAL_COLOR = (255, 255, 255), (255, 200, 0), (255, 0, 255)
# vln_ce GT의 **명목** 반경. 근거는 config가 아니라 배포된 navmesh 직독:
#   habitat_sim.nav.PathFinder().load_nav_mesh('<scan>.navmesh').nav_mesh_settings
#   → agent_radius=0.1, cell_size=0.05, agent_height=1.5.
# GT(`reference_path`)가 이 navmesh 위를 걸어 생성됐다. 데이터셋 json에는 radius 필드가 없다.
# 주의: habitat-sim은 config 설정이 로드된 navmesh와 다르면 **런타임에 navmesh를 다시 굽는다**
#   (Simulator.cpp:215-224, habitat_simulator.py:361, default_agent_navmesh=True).
#   r2r에서 배포본이 쓰이는 건 AgentConfig 기본값(0.1/1.5/0.2/45)이 배포 navmesh와 **정확히 같기 때문**이다.
BASELINE_R_B = 0.10


def densify(traj, step_m):
    """점 사이를 `step_m` 간격으로 채운다.

    `floorplan_canvas.draw`는 **점마다 원을 찍고 선을 잇지 않는다**. 원본 GT는 12프레임만 샘플링하므로
    그냥 넘기면 선이 아니라 흰 점 12개로 보인다(실제로 그렇게 나갔다). 격자 간격으로 채워서 연속선으로 만든다.
    """
    p = np.asarray(traj, dtype=np.float64)[:, :2]
    if len(p) < 2:
        return p
    out = [p[:1]]
    for x, y in zip(p[:-1], p[1:]):
        k = max(2, int(np.linalg.norm(y - x) / step_m) + 1)
        out.append(x + (y - x) * np.linspace(0, 1, k)[:, None])
    return np.vstack(out)


def resample(traj, n=120):
    traj = np.asarray(traj, dtype=np.float64)[:, :2]
    if len(traj) < 2:
        return np.repeat(traj, n, axis=0)[:n]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.repeat(traj[:1], n, axis=0)
    u = np.linspace(0, d[-1], n)
    return np.stack([np.interp(u, d, traj[:, 0]), np.interp(u, d, traj[:, 1])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--rig', default='125cm_0deg')
    ap.add_argument('--r_bs', default='0.10,0.25,0.40,0.55')
    ap.add_argument('--episodes', type=int, default=6)
    ap.add_argument('--max_resid', type=float, default=0.02,
                    help='정합 residual 상한[m]. G1이 확립한 신뢰 구간(<0.01~0.02) 밖 에피소드는 제외')
    ap.add_argument('--max_z_spread', type=float, default=0.30,
                    help='카메라 높이 변동 상한[m]. 초과 = 계단/층간 이동 → 단일 floor_z로 2D 투영 불가')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_n1')
    ap.add_argument('--esdf_dir', default='scripts/dataset_converters/gs_vlnpe/logs/esdf')
    ap.add_argument('--geo_dir', default='data/embodiment_aug/scene_geo',
                    help='씬 지오메트리 전용 ply 폴더 (01_build_scene_geo.py 산출물)')
    ap.add_argument('--out_dir', default='logs/embodiment_augment/g2')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    r_bs = [float(x) for x in args.r_bs.split(',')]
    h_rig = rig_height_m(args.rig)

    aug = EmbodimentAugmenter(args.data_root, args.raw_root, args.mesh_root, args.esdf_dir, args.geo_dir)
    tasks = [json.loads(l) for l in open(Path(args.data_root) / args.scene / 'meta' / 'episodes.jsonl')]
    ctx = aug._scene_ctx(args.scene)
    cell = float(ctx['args'].cell_m)

    body, per_ep, gt_clear = [], [], []
    gt_dist = {}                 # ep -> {r_b: 원본 GT와의 평균거리}
    feasible = {rb: 0 for rb in r_bs}
    a_feasible = {rb: 0 for rb in r_bs}
    a_dist = {}
    n_ep = 0

    for ep in range(min(args.episodes, len(tasks))):
        try:
            frames = load_episode(args.data_root, args.scene, args.rig, ep, 12)
            al = aug.align(args.scene, ep, frames, tasks[ep]['tasks'][0])
        except Exception as e:
            print(f'[G2] ep{ep}: skip ({type(e).__name__})')
            continue
        if al['resid'] > args.max_resid:
            # 정합이 나쁘면 원본 GT를 mesh에 잘못 놓은 것이라 비교 자체가 무의미하다 → 제외.
            print(f'[G2] ep{ep}: skip (정합 resid {al["resid"]:.4f} > {args.max_resid})')
            continue
        T = al['T']
        # 원본 GT 궤적 = vln_ce에 저장된 카메라 궤적을 mesh 좌표로 정합한 것
        gt = np.stack([(T @ p)[:3, 3] for _f, p, _d, _v in frames])
        gt_xy, fz = gt[:, :2], float(np.median(gt[:, 2]) - h_rig)
        z_spread = float(gt[:, 2].max() - gt[:, 2].min())
        if z_spread > args.max_z_spread:
            # 계단/층간 이동: 단일 floor_z로 3D occ를 2D로 눌러 담을 수 없다(계획 체계의 알려진 한계).
            print(f'[G2] ep{ep}: skip (카메라 높이 변동 {z_spread:.2f} > {args.max_z_spread} = 계단)')
            continue
        s_xy, g_xy = gt_xy[0], gt_xy[-1]
        n_ep += 1

        # 원본 GT가 **우리 맵**에서 실제로 가지는 여유거리 = 우리 플래너의 캘리브레이션 r_b*.
        # 명목 반경(navmesh 직독 0.1)과 다른 양이다 — recast의 추가 보수성 + 우리 obstacle 유도 방식
        # (height slab)의 차이가 섞여 있어 어떤 config에도 없고 측정해야만 알 수 있다.
        a = ctx['args']; a.r_b = BASELINE_R_B
        _res = _p03.plan_episode(ctx['occ'], ctx['origin'], fz, h_rig, s_xy, g_xy, a)
        gt_clear.append(np.array([clearance_at(_res['esdf'], ctx['origin'], cell, q) for q in gt_xy]))

        trajs, dists, adists = {}, {}, {}
        for rb in r_bs:
            # 시야(FOV) 제약을 걸지 않는다: G2의 질문은 "r_b가 GT path를 바꾸나"이고, 원본 GT 자체가
            # 12프레임 동안 회전하며 이동하므로 한 프레임의 시야 콘에 갇힐 이유가 없다.
            # (FOV 제약은 pixel-goal 국소 재계획(≤3.25m, 카메라 고정)의 조건 — replan_in_fov.)
            # 주 방식: **GT를 기준으로 유지하며 r_b에 맞게 국소 변형**.
            # 순수 A*(replan)는 instruction을 모르므로 baseline r_b에서도 다른 방으로 돌 수 있고,
            # 그러면 instruction이 거짓 라벨이 된다 → 비교용으로만 함께 측정한다.
            ta = aug.replan(args.scene, s_xy, g_xy, rb, floor_z=fz, goal_tol_m=0.10)
            if ta is not None:
                adists[rb] = float(np.mean(np.linalg.norm(resample(ta) - resample(gt_xy), axis=1)))
                a_feasible[rb] += 1
            tr = aug.deform_gt(args.scene, gt_xy, rb, floor_z=fz, goal_tol_m=0.10)
            if tr is None:
                continue
            trajs[rb] = tr
            feasible[rb] += 1
            dists[rb] = float(np.mean(np.linalg.norm(resample(tr) - resample(gt_xy), axis=1)))
        gt_dist[ep] = dists
        a_dist[ep] = adists

        # 배경은 **이 에피소드의 계획 격자**로 그린다. 이전 판은 npz의 `nav_mask_ref`를 썼는데 그건
        # npz 빌드 시점의 전역 floor_z(-0.00)·h_nav=0.10·h_obs=ref_h_b=0.926·r_b=0.25 기준이라,
        # 다층 씬에서 **다른 층 평면도**가 깔렸다(s8 ep0: 배경 vs 실제 계획 격자 IoU=0.241,
        # 배경에선 통행 가능한데 실제론 장애물인 셀이 49,986개). 경로가 벽을 통과하는 것처럼 보였다.
        _obs = derive_obstacle_2d(ctx['occ'], ctx['origin'], fz, h_rig, cell, a.h_nav_ratio * h_rig)
        _nav = truncate_navigable(compute_esdf_2d(_obs, cell), BASELINE_R_B) & compute_scan_coverage_mask(ctx['occ'])
        base, draw, emit = floorplan_canvas(_nav, ctx['origin'], cell, out)
        img = base.copy()
        # 원본 GT = 굵은 흰 선. **먼저·두껍게** 깔아서 위에 겹치는 얇은 색 선의 테두리로 보이게 한다
        # (반대로 하면 기준선이 색 선에 완전히 가려 조각만 보인다).
        draw(img, densify(gt_xy, cell), GT_COLOR, 4)
        for rb, tr in trajs.items():
            draw(img, tr, RB_COLORS[r_bs.index(rb) % len(RB_COLORS)], 1)
        draw(img, s_xy[None], START_COLOR, 5)
        draw(img, g_xy[None], GOAL_COLOR, 5)
        emit(img, f'g2_ep{ep:03d}.jpg')

        shift = (float(np.mean(np.linalg.norm(resample(trajs[max(trajs)]) - resample(trajs[min(trajs)]), axis=1)))
                 if len(trajs) >= 2 else float('nan'))
        per_ep.append((ep, shift))
        st = ' · '.join(f'r_b={rb}: ' + (f'GT거리 {dists[rb]*100:.0f}cm' if rb in trajs else '계획불가')
                        for rb in r_bs)
        body.append(f'<h3>episode {ep}</h3><p>정합 resid {al["resid"]:.4f} m · '
                    f'경로 이동(최소→최대 r_b) {shift*100:.0f} cm<br>{st}</p>'
                    f'<img src="g2_ep{ep:03d}.jpg" style="width:70%">')
        print(f'[G2] ep{ep}: resid={al["resid"]:.4f} shift={shift*100:.0f}cm | ' +
              ' '.join(f'{rb}:' + (f'{dists[rb]*100:.0f}cm' if rb in trajs else 'X') for rb in r_bs))

    # 공통 집합 = 모든 r_b에서 계획이 성공한 에피소드. r_b마다 다른 집합의 평균을 비교하면
    # 생존 편향(큰 r_b에서 살아남는 건 원래 쉬운 에피소드)이 생겨 비교가 무의미하다.
    common = [e for e, d in gt_dist.items() if len(d) == len(r_bs)]
    cmean = {rb: float(np.mean([gt_dist[e][rb] for e in common])) for rb in r_bs} if common else {}
    allmean = {rb: float(np.mean([d[rb] for d in gt_dist.values() if rb in d]))
               for rb in r_bs if any(rb in d for d in gt_dist.values())}
    # argmin 집계: 에피소드별로 원본 GT에 가장 가까운 r_b를 세어 다수결로 본다(집합 편향 없음)
    argmin = {rb: 0 for rb in r_bs}
    for d in gt_dist.values():
        if d:
            argmin[min(d, key=d.get)] += 1

    gt_rows = ''.join(
        f'<tr><td>r_b={rb}{" <b>(baseline)</b>" if abs(rb - BASELINE_R_B) < 1e-9 else ""}</td>'
        f'<td>{allmean[rb]*100:.1f} cm</td><td>{feasible[rb]}/{n_ep}</td>'
        f'<td>' + (f'{cmean[rb]*100:.1f} cm' if rb in cmean else '—') + f'</td><td>{argmin[rb]}</td></tr>'
        for rb in r_bs if rb in allmean)
    best = min(cmean, key=cmean.get) if cmean else (min(allmean, key=allmean.get) if allmean else None)
    # 우리 플래너의 캘리브레이션 r_b* = 에피소드별 GT 최소 clearance의 median.
    # 주의: 이건 **vln_ce의 속성이 아니다**(명목은 navmesh 직독 0.1). 우리 occupancy 맵이 recast와
    # 다르게 장애물을 만들기 때문에 생기는 우리 쪽 값이다 → 어디서도 읽을 수 없고 측정해야 한다.
    cl_min = np.array([c.min() for c in gt_clear]) if gt_clear else np.array([np.nan])
    r_eff = float(np.median(cl_min))
    r_ref = min(r_bs, key=lambda rb: abs(rb - r_eff))   # sweep 값 중 실효값에 가장 가까운 것
    shifts = [s_ for _e, s_ in per_ep if np.isfinite(s_)]
    med_shift = float(np.median(shifts)) if shifts else 0.0
    # 판정: "우리 파이프라인은 명목 반경(0.1)보다 캘리브레이션 r_b*에서 원본 GT를 더 잘 재현한다"
    # — 공통 집합 평균으로 비교하는 falsifiable한 진술. 단일 argmin은 n이 작아 신뢰할 수 없으므로
    # 곡선 전체를 함께 보고한다.
    # (이전 판은 "계획 성공률의 정점이 r_b*"를 판정으로 썼는데 그건 **coarse 다운샘플 artifact**였다:
    #  downsample_factor=1로 고치면 성공률은 r_b에 단조 감소한다 — 물리적으로도 그게 맞다.)
    identity_ok = (r_ref in cmean and BASELINE_R_B in cmean and cmean[r_ref] < cmean[BASELINE_R_B])
    changed = med_shift > 0.02 or len(set(feasible.values())) > 1

    summary = (
        f'<p><b>{args.scene}</b> · <b>vln_ce</b> · rig {args.rig} · 정합 resid ≤ {args.max_resid} m인 에피소드 <b>{n_ep}개</b> · r_b {r_bs}</p>'
        f'<table border=1 cellpadding=6><tr><th>게이트</th><th>결과</th></tr>'
        f'<tr><td><b>baseline 재현</b> — 캘리브레이션 <b>r_b*={r_ref}</b>이 명목 <b>{BASELINE_R_B}</b>보다 '
        f'원본 GT를 더 잘 재현하나<br><small>공통 {len(common)}개 에피소드 평균 비교</small></td>'
        f'<td>' + (f'{cmean[r_ref]*100:.1f} cm(r_b*) vs {cmean[BASELINE_R_B]*100:.1f} cm(명목) '
                   if (r_ref in cmean and BASELINE_R_B in cmean) else '표본 부족 ')
        + f'→ {"PASS" if identity_ok else "CHECK"}<br>'
        f'<small>보조: GT 최단거리 r_b={best}. n이 작아 단일 argmin은 신뢰하지 말고 아래 곡선을 볼 것</small></td></tr>'
        f'<tr><td><b>e가 경로를 바꾸나</b> — 경로 이동 median / feasibility 변화</td>'
        f'<td>{med_shift*100:.1f} cm · 계획성공 {min(feasible.values())}~{max(feasible.values())}/{n_ep} '
        f'→ {"PASS" if changed else "FAIL"}</td></tr></table>'
        f'<h3>왜 GT를 변형하는가 — 순수 A*와의 비교</h3>'
        f'<p>이전 판은 같은 start/goal에서 <b>순수 A*</b>로 다시 계획했다. 그런데 A*는 '
        f'<b>instruction을 모른다</b> — R2R의 GT는 사람이 주석한 <code>reference_path</code>(특정 방·물체를 '
        f'경유하도록 instruction에 맞춘 것)를 따라간 궤적인데, A*는 기하적 최단만 찾으므로 다른 방으로 '
        f'돌아버릴 수 있다. 그러면 <b>baseline r_b에서도 경로가 GT와 달라지고 instruction이 거짓 라벨이 '
        f'된다</b>. 그래서 주 방식을 <b>GT 변형</b>(<code>deform_gt</code>)으로 바꿨다: GT 자체를 seed로 주고 '
        f'<code>refine_min_move</code>로 <b>필요한 만큼만</b> 밀어낸다 — clearance가 이미 r_b 이상인 점은 '
        f'그대로 두고, 미달인 점만 가장 가까운 안전 셀로 옮긴다. 즉 "GT를 따라가되 큰 로봇이 못 지나가는 '
        f'구간에서만 국소 우회"다.</p>'
        f'<table border=1 cellpadding=6><tr><th>r_b</th>'
        f'<th>GT 변형 — GT와 거리</th><th>GT 변형 — 성공</th>'
        f'<th>순수 A* — GT와 거리</th><th>순수 A* — 성공</th></tr>'
        + ''.join(
            f'<tr><td>{rb}{" <b>(명목)</b>" if abs(rb-BASELINE_R_B)<1e-9 else ""}</td>'
            + (f'<td><b>{np.mean([d[rb] for d in gt_dist.values() if rb in d])*100:.1f} cm</b></td>'
               if any(rb in d for d in gt_dist.values()) else '<td>—</td>')
            + f'<td>{feasible[rb]}/{n_ep}</td>'
            + (f'<td>{np.mean([d[rb] for d in a_dist.values() if rb in d])*100:.1f} cm</td>'
               if any(rb in d for d in a_dist.values()) else '<td>—</td>')
            + f'<td>{a_feasible[rb]}/{n_ep}</td></tr>' for rb in r_bs)
        + '</table>'
        f'<p><b>GT 변형이 실패하면 그 샘플은 기각한다</b>(원본 GT 폴백). 국소 변형으로 못 뚫리는 구간은 '
        f'"이 embodiment로는 이 instruction의 루트를 따라갈 수 없다"는 <b>물리적 사실</b>이므로, A*처럼 '
        f'다른 루트를 찾아 억지로 성공시키면 instruction이 깨진다.</p>'
        f'<h3>r_b별 — 원본 GT와의 거리, 계획 성공 (GT 변형)</h3>'
        f'<table border=1 cellpadding=6><tr><th>r_b</th>'
        f'<th>GT 거리<br><small>성공한 것만 평균</small></th><th>계획 성공</th>'
        f'<th>GT 거리<br><small>공통 {len(common)}개만</small></th>'
        f'<th>이 r_b가 GT에<br>가장 가까운 에피소드</th></tr>'
        f'{gt_rows}</table>'
        f'<p><b>왜 두 개의 평균인가</b>: "성공한 것만 평균"은 r_b마다 에피소드 집합이 달라 비교할 수 없다 '
        f'(큰 r_b에서 살아남는 건 원래 쉬운 에피소드 — 생존 편향). 게이트 판정은 <b>공통 집합</b> 열과 '
        f'에피소드별 argmin 집계(마지막 열)로 한다.</p>'
        f'<h3>두 개의 다른 r_b를 구분해야 한다</h3>'
        f'<p><b>(1) vln_ce GT의 명목 반경 = {BASELINE_R_B} m — 측정할 필요 없이 데이터에서 직접 읽힌다.</b> '
        f'배포된 <code>&lt;scan&gt;.navmesh</code>를 <code>PathFinder.load_nav_mesh</code>로 열면 '
        f'<code>agent_radius=0.1, cell_size=0.05, agent_height=1.5</code>다. recast는 로봇을 <b>수직 '
        f'원기둥</b>(반지름 r·높이 h)으로 보고 걷기 가능 표면을 <code>ceil(r/cell_size)</code>셀 깎아 '
        f'"원기둥 중심축을 놓을 수 있는 영역"만 남긴다(0.1/0.05 = 정확히 2셀). GT(<code>reference_path</code>)가 '
        f'이 navmesh 위를 걸어 생성됐고, 데이터셋 json에는 radius 필드가 <b>없다</b>. '
        f'(<code>vln_r2r_mini.yaml</code>은 <b>eval용</b>이고 main에도 없는 파일이라 GT의 근거가 아니다.)</p>'
        f'<p><b>다만 config의 radius는 무해하지 않다</b> — habitat-sim은 config 설정이 로드된 navmesh와 '
        f'다르면 <b>런타임에 navmesh를 다시 굽는다</b>(<code>Simulator.cpp:215-224</code>, '
        f'<code>habitat_simulator.py:361</code>, <code>default_agent_navmesh=True</code>). r2r에서 배포본이 '
        f'쓰이는 건 <code>AgentConfig</code> 기본값(0.1/1.5/0.2/45)이 배포 navmesh와 <b>정확히 같기 때문</b>이다. '
        f'→ <b>eval에서 r_b를 바꾸려면 yaml에 <code>radius:</code> 한 줄</b>이면 되고 새 코드가 필요 없다.</p>'
        f'<p><b>(2) 우리 플래너의 캘리브레이션 r_b* = {r_eff:.3f} m — 이건 측정해야 한다.</b> '
        f'원본 GT가 <b>우리 occupancy 맵</b>에서 실제로 확보한 여유거리(ESDF)다. 명목 0.1과 벌어지는 이유는 '
        f'둘이 섞여 있다 — (a) recast 자체의 추가 보수성(<code>edge_max_error</code>=1.3 voxel=6.5 cm 폴리곤 '
        f'단순화, <code>region_merge_size</code>=20), (b) <b>우리 맵이 recast와 다르게 장애물을 만든다</b>'
        f'(height slab 투영 vs recast의 span walkability). 즉 이건 <b>vln_ce의 속성이 아니라 우리 쪽 값</b>이라 '
        f'어떤 config에도 없고 측정 외에는 알 방법이 없다.</p>'
        f'<table border=1 cellpadding=6><tr><th>지표</th><th>값</th><th>성격</th></tr>'
        f'<tr><td>navmesh <code>agent_radius</code> (직독)</td><td><b>{BASELINE_R_B:.3f} m</b></td>'
        f'<td>vln_ce 명목 — 측정 불필요</td></tr>'
        f'<tr><td>에피소드별 GT <b>최소</b> clearance의 median = <b>r_b*</b></td><td><b>{r_eff:.3f} m</b></td>'
        f'<td>우리 플래너 캘리브레이션</td></tr>'
        f'<tr><td>전 프레임 clearance median</td><td>{np.median(np.concatenate(gt_clear)):.3f} m</td>'
        f'<td>참고</td></tr></table>'
        f'<p>게이트는 <b>r_b*</b>로 판정한다 — 우리 플래너가 원본 GT를 재현하는지 묻는 것이므로 우리 쪽 '
        f'기준이 맞다. augment 범위 하단도 0.1이 아니라 r_b*로 잡아야 baseline 샘플이 원본과 일치한다. '
        f'<b>단, 지금 r_b*는 씬 1개·{n_ep}에피소드·각 앞 12프레임만의 잠정치</b>이므로 논문에 쓰기 전 '
        f'전 씬으로 재측정해야 한다. '
        f'(참고: <code>vln_n1</code>은 <code>esdf_utils.ROBOT_RADIUS_M</code>=<b>0.25</b>다.)</p>'
        f'<p>재계획은 원본과 <b>같은 start/goal</b>에서 하고, 끝점이 원본 goal에서 10 cm 넘게 벗어나면 '
        f'(dilation이 goal 셀을 지워 도착지가 밀린 경우) <b>계획 실패</b>로 본다. 여기엔 <b>시야 제약을 걸지 '
        f'않는다</b> — 원본 GT 자체가 12프레임 동안 회전하며 이동하므로 한 프레임의 시야 콘에 갇힐 이유가 없다. '
        f'(시야 제약은 카메라를 고정하는 pixel-goal 국소 재계획의 조건이다.)</p>'
        f'<p><b>제외한 에피소드 2종</b> — (1) 정합 residual > {args.max_resid} m: 원본 GT를 mesh에 잘못 '
        f'놓으면 비교가 무의미하다(#00이 확립한 신뢰 구간). (2) 카메라 높이 변동 > {args.max_z_spread} m: '
        f'계단·층간 이동이라 3D occ를 <b>단일 floor_z</b>로 2D에 눌러 담을 수 없다 — 실제로 이런 '
        f'에피소드는 GT clearance가 0(= 우리 맵에서 벽 속)으로 나왔다. 이는 slab projection의 알려진 '
        f'한계이고 논문 한계 항목에 이미 있다.</p>')
    trouble = (
        f'<h3>생성 경로가 장애물을 통과하던 문제 — 원인과 조치</h3>'
        f'<p>이전 판에서 생성 경로가 <b>장애물 voxel 내부를 통과</b>했다(r_b=0.1에서 4/7. 원본 GT는 0/7이라 '
        f'데이터가 아니라 우리 플래너 문제). 원인을 <code>points_min</code>/<code>segments_min</code>을 나눠 '
        f'특정했다 — <b>A* 웨이포인트 자체가 장애물 안</b>이었다(선분이 코너를 자른 것도, 스무딩도 아니다). '
        f'기본값 <code>downsample_factor=4, mode=\'any\'</code>가 20 cm coarse 셀을 "16개 fine 셀 중 하나만 '
        f'navigable이면 통과 가능"으로 보고 웨이포인트를 <b>셀 중심</b>에 놓기 때문이다. '
        f'<b>작은 r_b에서 더 심한 것은 로봇이 작아서가 아니다</b> — r_b가 작으면 fine navigable 마스크가 벽에 '
        f'붙어 커지므로, 거의 벽인 coarse 셀도 자격을 얻는다. r_b가 크면 마스크가 벽에서 물러나 그런 셀이 '
        f'애초에 안 생긴다.</p>'
        f'<table border=1 cellpadding=6><tr><th>설정</th><th>r_b=0.10</th><th>r_b=0.18</th><th>시간</th></tr>'
        f'<tr><td>factor=4 <code>any</code> (기존 기본값)</td><td>7/7 · <b>하드 4</b> · 몸통 5</td>'
        f'<td>7/7 · 하드 0 · 몸통 3</td><td>9 ms</td></tr>'
        f'<tr><td>factor=4 <code>majority</code></td><td>7/7 · 하드 1 · 몸통 1</td><td>3/7 · 0 · 0</td>'
        f'<td>10 ms</td></tr>'
        f'<tr><td>factor=4 <code>all</code></td><td>3/7 · 0 · 0</td><td>3/7 · 0 · 0</td><td>8 ms</td></tr>'
        f'<tr><td><b>factor=1 (채택)</b></td><td><b>7/7 · 하드 0</b> · 몸통 1</td><td>4/7 · <b>0 · 0</b></td>'
        f'<td>17–24 ms</td></tr></table>'
        f'<p><b>조치</b>: A*를 fine 격자에서 돈다(<code>downsample_factor=1</code>). 비용 +10 ms는 sample 예산 '
        f'393 ms(Open3D 렌더 350 ms 지배, R2)에서 무시할 수준이다. 그 위에 라벨 유효성 <b>3중 기각</b>을 '
        f'건다 — (1) 하드 충돌 <code>clearance==0</code>, (2) 몸통 침범 <code>clearance &lt; r_b</code>, '
        f'(3) <b>미관측(unknown) 통과</b>. (3)은 장애물 검사로 안 잡힌다(스캔 안 된 곳은 ESDF가 크게 나옴) — '
        f'실측 0/7이었지만 검사가 없으면 보장이 없어 <code>compute_scan_coverage_mask</code>로 따로 막는다. '
        f'(2)는 <b>생성 경로에만</b> 건다: 원본 GT는 우리 맵에서 이걸 위반하는데(r_b=0.2에서 4/7) 우리 맵이 '
        f'recast보다 보수적이기 때문이고, 생성 경로에 대해서는 A*가 fine 격자에서 이미 보장한 것을 '
        f'refine·스무딩이 깨지 않았는지 확인하는 자기 정합성 검사다.</p>'
        f'<p><b>부수 정정</b>: 가드 직후 "계획 성공률이 r_b*에서 정점"이라고 봤는데 그것도 <b>coarse 다운샘플 '
        f'artifact</b>였다. factor=1에서는 성공률이 r_b에 <b>단조 감소</b>한다 — 물리적으로 그게 맞다.</p>'
        f'<p><b>추가로 정정한 것</b>: 가드 직후 "계획 성공률이 r_b*에서 정점"이라고 봤는데 그것도 '
        f'coarse 다운샘플 artifact였다. factor=1에서는 성공률이 r_b에 <b>단조 감소</b>한다 — 물리적으로 '
        f'그게 맞다. 또 배경 평면도로 npz의 <code>nav_mask_ref</code>를 깔았는데 그건 전역 floor_z·다른 '
        f'h_nav/h_obs 기준이라 <b>다층 씬에서 다른 층 평면도</b>가 깔렸다(IoU 0.241) → 에피소드 계획 '
        f'격자로 교체. 계산에는 안 쓰였지만 그림으로 검증할 수 없게 만들고 있었다.</p>')
    (out / 'trouble.html').write_text(trouble, encoding='utf-8')
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'G2 — r_b가 GT path를 바꾸는가 (vln_ce)', summary, ''.join(body))
    print(f'[G2] 캘리브레이션 r_b*={r_eff:.3f}m(→{r_ref}) · 명목(navmesh)={BASELINE_R_B} · GT최소거리 r_b={best} · '
          f'shift median {med_shift*100:.1f}cm · feasible {feasible}')
    print(f'[G2] report -> {out}/report.html')


if __name__ == '__main__':
    main()
