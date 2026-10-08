"""pixel-goal 보존형 재계획 검증 (V1~V5).

목표(사용자 재정의):
1. **기본 path·원본 이미지·기존 pixel goal 유지** — 현재 프레임 기준으로 `r_b`가 바뀌어도
   **pixel goal에 도달하는 경로**를 생성한다.
2. `r_b`가 커져 **pixel goal 지점이 충돌**하면 **goal 위치를 조정**한다(`retreat` / `nearest`).
   조정 후 화면 밖·가림이면 **샘플 기각 → 기존 r_b 사용**.

산출: 룩다운 이미지에 원본/조정 goal 오버레이 + floorplan 경로 + r_b별 통계표 → report + Artifact.

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/07_validate_pixel_goal.py --scene 17DRP5sb8fy --episode 0 --r_bs 0.15,0.25,0.35,0.50 --goal_adjust retreat
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

_HERE = Path(__file__).resolve().parent
_GS = _HERE.parents[0] / 'gs_vlnpe'
sys.path.insert(0, str(_GS)); sys.path.insert(0, str(_HERE))
from esdf_utils import compute_esdf_2d, derive_obstacle_2d  # noqa: E402
from geometry_utils import save_jpg  # noqa: E402
from viz_utils import floorplan_canvas, save_gallery  # noqa: E402
from embodiment_augment import EmbodimentAugmenter  # noqa: E402
from pixel_goal_utils import (  # noqa: E402
    _heading_at_end, adjust_goal, check_visible, clearance_at, goal_world_from_poses,
    project_to_pixel, rig_height_m,
)
_m_validate_augment = importlib.import_module('06_validate_augment')  # noqa: E402
RB_COLORS = _m_validate_augment.RB_COLORS  # noqa: E402
chip = _m_validate_augment.chip  # noqa: E402
_p03 = importlib.import_module('03_sample_gt_paths')

ORIG_COLOR = (255, 60, 60)      # 원본 goal — **빈 원**(테두리만)
# 조정된 goal은 **r_b별 RB_COLORS로 채워진 원**으로 그린다(색=어느 r_b인지). 'unchanged'는 아무것도
# 안 그린다(원본과 같은 자리이므로). 범례는 반드시 실제로 그린 것만 표시할 것.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--rig', default='125cm_0deg', help='pixel goal이 정의된 룩다운 rig')
    ap.add_argument('--r_bs', default='0.15,0.25,0.35,0.50')
    ap.add_argument('--goal_adjust', default='retreat', choices=['retreat', 'shift', 'nearest'])
    ap.add_argument('--baseline_r_b', type=float, default=0.10,
                    help='원본 GT가 만들어진 반경(habitat 기본 agent radius 0.1) — V0 identity 기준')
    ap.add_argument('--max_samples', type=int, default=6, help='검사할 (프레임) 샘플 수')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_n1')
    ap.add_argument('--esdf_dir', default='scripts/dataset_converters/gs_vlnpe/logs/esdf')
    ap.add_argument('--geo_dir', default='data/embodiment_aug/scene_geo',
                    help='씬 지오메트리 전용 ply 폴더 (01_build_scene_geo.py 산출물)')
    ap.add_argument('--out_dir', default='logs/embodiment_augment/pixelgoal')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    r_bs = [float(x) for x in args.r_bs.split(',')]
    h_rig = rig_height_m(args.rig)

    scene_dir = Path(args.data_root) / args.scene
    t = pq.read_table(scene_dir / 'data' / 'chunk-000' / f'episode_{args.episode:06d}.parquet')
    poses = [np.asarray(p).reshape(4, 4) for p in t[f'pose.{args.rig}'].to_pylist()]
    goals = np.array(t[f'goal.{args.rig}'].to_pylist())
    rels = np.array(t[f'relative_goal_frame_id.{args.rig}'].to_pylist())
    rgb_dir = scene_dir / 'videos' / 'chunk-000' / f'observation.images.rgb.{args.rig}'
    dep_dir = scene_dir / 'videos' / 'chunk-000' / f'observation.images.depth.{args.rig}'

    # 정합(mesh 좌표) + occupancy — clearance/재계획용
    aug = EmbodimentAugmenter(args.data_root, args.raw_root, args.mesh_root, args.esdf_dir, args.geo_dir)
    tasks = [json.loads(l) for l in open(scene_dir / 'meta' / 'episodes.jsonl')]
    frames_for_align = [(i, poses[i], None, None) for i in
                        np.unique(np.linspace(0, len(poses) - 1, 12).astype(int))]
    # align은 depth가 필요하므로 verify_pose_mesh.load_episode 경로를 그대로 쓴다
    load_episode = importlib.import_module('00_verify_pose_mesh').load_episode
    al_frames = load_episode(args.data_root, args.scene, args.rig, args.episode, 12)
    al = aug.align(args.scene, args.episode, al_frames, tasks[args.episode]['tasks'][0])
    T = al['T']
    ctx = aug._scene_ctx(args.scene)
    cell = float(ctx['args'].cell_m)
    cam_mesh_all = np.stack([(T @ p)[:3, 3] for p in poses])
    floor_z_mesh = float(np.median(cam_mesh_all[:, 2]) - h_rig)
    print(f'[PG] {args.scene} ep{args.episode} rig={args.rig} resid={al["resid"]:.4f} '
          f'floor_z(mesh)={floor_z_mesh:.2f} mode={args.goal_adjust}')

    # r_b별 ESDF(mesh 좌표) 준비
    esdfs = {}
    for r_b in r_bs:
        obs = derive_obstacle_2d(ctx['occ'], ctx['origin'], floor_z_mesh, h_rig, cell, 0.12 * h_rig)
        esdfs[r_b] = compute_esdf_2d(obs, cell)

    cand = [i for i in range(len(poses)) if rels[i] >= 0 and i + int(rels[i]) + 1 < len(poses)]
    sel = cand[:args.max_samples]
    rows, body, stats = [], [], {r: dict(unchanged=0, adjusted=0, retreated=0, rejected=0) for r in r_bs}
    v1_err, v2_mono, v3_reach = [], [], []

    # --- V0 identity: baseline r_b에서 **원본 pixel goal이 그대로 재현**되는가 ---
    # vln_ce GT는 habitat 기본 agent radius(0.1 m)로 수집됐다 → 그 값에선 goal이 안 움직여야 정상.
    esdf_base = None
    v0 = dict(n=0, unchanged=0, px_max=0.0, worst=None)
    for i in cand:
        j = i + int(rels[i]) + 1
        G = goal_world_from_poses(poses[i:j + 1], args.rig)
        if esdf_base is None:
            obs_b = derive_obstacle_2d(ctx['occ'], ctx['origin'], floor_z_mesh, h_rig, cell, 0.12 * h_rig)
            esdf_base = compute_esdf_2d(obs_b, cell)
        Gm = (T @ np.array([G[0], G[1], G[2], 1.0]))[:3]
        pm = np.stack([(T @ p)[:3, 3] for p in poses[i:j + 1]])[:, :2]
        g2, st = adjust_goal(Gm[:2], args.baseline_r_b, esdf_base, ctx['origin'], cell,
                             mode=args.goal_adjust, path_xy_mesh=pm, robot_xy_mesh=pm[0])
        v0['n'] += 1
        if st == 'unchanged':
            v0['unchanged'] += 1
        # **goal이 안 움직이면 원본 저장 라벨을 그대로 쓴다**(재투영하지 않는다).
        # 재투영하면 정수 절삭·hfov 상수 차이로 1~2 px 잔차가 생기는데, 그건 embodiment 효과가 아니라
        # 우리 투영식의 재현 오차다. 이미 정답이 있는데 다시 유도할 이유가 없다 → baseline 변화 = 0 px 보장.
        if st == 'unchanged':
            dpx = 0.0
        elif g2 is not None:
            Gb = np.linalg.inv(T) @ np.array([g2[0], g2[1], Gm[2], 1.0])
            ub, vb, _z = project_to_pixel(Gb[:3], poses[i], args.rig)
            dpx = max(abs(ub - goals[i][0]), abs(vb - goals[i][1]))
        else:
            dpx = 0.0
        if dpx > v0['px_max']:
            v0['px_max'], v0['worst'] = float(dpx), i
    print(f"[PG] V0 identity @ r_b={args.baseline_r_b}: unchanged {v0['unchanged']}/{v0['n']}, "
          f"픽셀 최대 변화 {v0['px_max']:.2f}px (worst frame {v0['worst']})")

    for i in sel:
        j = i + int(rels[i]) + 1
        G = goal_world_from_poses(poses[i:j + 1], args.rig)              # start-relative
        u_chk, v_chk, _z_chk = project_to_pixel(G, poses[i], args.rig)
        v1_err.append(max(abs(u_chk - goals[i][0]), abs(v_chk - goals[i][1])))

        img = np.asarray(Image.open(rgb_dir / f'episode_{args.episode:06d}_{i}.jpg').convert('RGB')).copy()
        depth = np.asarray(Image.open(dep_dir / f'episode_{args.episode:06d}_{i}.png')).astype(np.float32) / 1000.0
        cv2.circle(img, (int(goals[i][0]), int(goals[i][1])), 9, ORIG_COLOR, 2)   # 원본 goal(저장값)

        G_mesh = (T @ np.array([G[0], G[1], G[2], 1.0]))[:3]
        P_mesh = (T @ poses[i])[:3, 3]
        path_mesh = np.stack([(T @ p)[:3, 3] for p in poses[i:j + 1]])[:, :2]
        # floorplan: 시야 마스크(밝은 영역) 위에 원본 경로 + r_b별 재계획 경로 + goal 이동을 겹친다
        fovm = aug.fov_mask(ctx, T @ poses[i], floor_z_mesh, args.rig)
        nav_vis = np.load(Path(args.esdf_dir) / f'{args.scene}.npz')['nav_mask_ref']
        # 3색으로 구분: 시야 안 navigable(밝은 초록) / 시야 밖 navigable(어두운 초록) / 장애물(검정)
        fp_base, fp_draw, fp_emit = floorplan_canvas(nav_vis, ctx['origin'], cell, out)
        fp = fp_base.copy()
        h_fp, w_fp = nav_vis.shape
        out_fov = (~fovm[:h_fp, :w_fp]) & nav_vis
        sc = fp.shape[0] // h_fp                       # floorplan_canvas가 확대한 배율
        of = np.kron(out_fov[::-1], np.ones((sc, sc), bool))[:fp.shape[0], :fp.shape[1]]
        fp[of] = (28, 44, 28)                          # 시야 밖 navigable은 어둡게
        fp_draw(fp, path_mesh, (255, 255, 255), 1)                  # 원본 GT 경로 = 흰 선
        per_rb, prev_dist = [], None
        for r_b in r_bs:
            esdf = esdfs[r_b]
            clr = clearance_at(esdf, ctx['origin'], cell, G_mesh[:2])
            g_new, st = adjust_goal(G_mesh[:2], r_b, esdf, ctx['origin'], cell,
                                    mode=args.goal_adjust, path_xy_mesh=path_mesh,
                                    robot_xy_mesh=P_mesh[:2])
            if g_new is None:
                per_rb.append((r_b, 'rejected', clr, None, None)); stats[r_b]['rejected'] += 1; continue
            # mesh -> start-relative 로 되돌려 재투영
            G2 = np.linalg.inv(T) @ np.array([g_new[0], g_new[1], G_mesh[2], 1.0])
            u, v, Z = project_to_pixel(G2[:3], poses[i], args.rig)
            ok, why = check_visible(u, v, Z, depth)
            if not ok:
                per_rb.append((r_b, f'rejected({why})', clr, None, None)); stats[r_b]['rejected'] += 1; continue
            # r_b 경로 재계획: 현재 위치 -> (조정된) goal
            # 시야 제약 재계획: 좌우로 화면을 벗어나는 경로를 A* 단계에서 원천 차단
            traj = aug.replan_in_fov(args.scene, P_mesh[:2], g_new, r_b, T @ poses[i],
                                     rig=args.rig, floor_z=floor_z_mesh, goal_tol_m=0.10)
            reach = None if traj is None else float(np.linalg.norm(traj[-1] - g_new))
            if reach is not None:
                v3_reach.append(reach)
            # V2 단조성은 **진행 방향 전진량**(heading 투영)으로 잰다. 직선거리로 재면 lateral shift가
            # 거리를 소폭 늘려 단조성이 깨진 것처럼 보인다(실제로는 앞으로 더 간 게 아니다).
            _h = _heading_at_end(path_mesh)
            d = float(np.dot(g_new - P_mesh[:2], _h)) if _h is not None else float(np.linalg.norm(g_new - P_mesh[:2]))
            if prev_dist is not None:
                v2_mono.append(d <= prev_dist + 0.02)      # 2 cm 여유(격자 양자화)
            prev_dist = d
            stats[r_b]['unchanged' if st == 'unchanged'
                       else ('retreated' if st == 'adjusted_retreat' else 'adjusted')] += 1
            per_rb.append((r_b, st, clr, (u, v), reach))
            col_rb = RB_COLORS[r_bs.index(r_b) % len(RB_COLORS)]
            if traj is not None:
                fp_draw(fp, traj, col_rb, 1)                        # r_b별 재계획 경로
            if st != 'unchanged':
                # lateral 통과 = 채운 원 / retreat 폴백 = 테두리 원(구분해서 보이게)
                thick = -1 if st == 'adjusted' else 2
                cv2.circle(img, (int(round(u)), int(round(v))), 7, col_rb, thick)
                fp_draw(fp, g_new[None], col_rb, 4)                 # 옮겨진 goal
        name = f'pg_f{i:03d}.jpg'
        save_jpg(img, out / name)
        fp_draw(fp, G_mesh[None, :2], ORIG_COLOR, 5)                # 원본 goal
        fp_draw(fp, P_mesh[None, :2], (255, 140, 0), 5)             # 로봇 현재 위치
        fp_name = f'pg_fp{i:03d}.jpg'
        fp_emit(fp, fp_name)
        # 색 규약은 summary에 **한 번만** 쓴다(프레임마다 반복하면 읽기 피곤하다).
        # 여기서는 그 프레임의 **데이터**만 짧게 남긴다.
        status = ' · '.join(f'r_b={r}: {s}' + (f' ({rc*100:.0f}cm)' if rc is not None else '')
                            for r, s, _c, _p, rc in per_rb)
        body.append(f'<h3>frame {i}</h3>'
                    f'<p>goal_len={rels[i]} · 원본 goal=({goals[i][0]},{goals[i][1]}) · '
                    f'원본 goal의 clearance={per_rb[0][2]:.2f} m<br>{status}</p>'
                    f'<img src="{name}" style="width:62%">'
                    f'<img src="{fp_name}" style="width:62%">')
        rows.append((i, rels[i], per_rb))
        print(f'[PG] frame {i}: ' + ' | '.join(f'r_b={r}:{s}' for r, s, _c, _p, _rc in per_rb))

    v1 = float(np.max(v1_err)) if v1_err else float('nan')
    v2 = (all(v2_mono), len(v2_mono))
    v3 = float(np.max(v3_reach)) if v3_reach else float('nan')
    tbl = ''.join(f'<tr><td>{r}</td><td>{s["unchanged"]}</td><td>{s["adjusted"]}</td>'
                  f'<td>{s["retreated"]}</td><td>{s["rejected"]}</td></tr>' for r, s in stats.items())
    summary = (
        f'<p><b>{args.scene}</b> ep{args.episode} · rig {args.rig} · 조정 모드 <b>{args.goal_adjust}</b> · '
        f'정합 resid {al["resid"]:.4f} m · 검사 프레임 {len(sel)}개</p>'
        f'<table border=1 cellpadding=6><tr><th>게이트</th><th>결과</th></tr>'
        f'<tr><td><b>V0 identity</b> — baseline r_b={args.baseline_r_b}에서 원본 pixel goal 재현</td>'
        f'<td>goal 유지 <b>{v0["unchanged"]}/{v0["n"]}</b>, 픽셀 최대 변화 <b>{v0["px_max"]:.2f} px</b> '
        f'{"PASS" if (v0["unchanged"] == v0["n"] and v0["px_max"] < 3) else "CHECK"}</td></tr>'
        f'<tr><td>V1 투영식 재현(원본 goal 대비 최대 오차)</td><td>{v1:.2f} px {"PASS" if v1 < 3 else "CHECK"}</td></tr>'
        f'<tr><td>V2 단조성(r_b↑ → <b>진행 방향 전진량</b> 비증가)</td><td>{"PASS" if v2[0] else "FAIL"} ({v2[1]}쌍)</td></tr>'
        f'<tr><td>V3 도달성(새 경로 끝점 vs goal)</td><td>{v3*100:.1f} cm {"PASS" if v3 < 0.10 else "CHECK"}</td></tr>'
        f'</table>'
        f'<h3>V5 — r_b별 goal 처리 결과</h3>'
        f'<table border=1 cellpadding=6><tr><th>r_b</th><th>그대로(안전)</th>'
        f'<th>{"lateral 통과" if args.goal_adjust == "shift" else "조정됨"}</th>'
        f'<th>retreat 폴백</th><th>기각(→기존 r_b)</th></tr>{tbl}</table>'
        f'<p><b>파이프라인</b>: 원본 이미지·pixel goal 유지 → 현재 프레임에서 goal까지 <b>r_b 경로 재계획</b>. '
        f'goal 지점의 clearance가 r_b보다 작으면 <b>{args.goal_adjust}</b> 기준으로 goal을 옮기고 '
        f'<b>픽셀 재투영</b>. 화면 밖·가림이면 <b>기각 → 기존 r_b 사용</b>.</p>'
        f'<p><b>baseline에서 픽셀 변화가 0인 이유</b>: goal이 안 움직이면(<code>unchanged</code>) '
        f'<b>원본 저장 라벨을 그대로 쓴다</b> — 재투영하지 않는다. 재투영하면 저장값이 정수로 절삭돼 있고 '
        f'hfov 상수도 미세하게 달라 1~2 px 잔차(V1={v1:.2f}px)가 생기는데, 그건 embodiment 효과가 아니라 '
        f'우리 투영식의 재현 오차다. 이미 정답이 있으니 다시 유도하지 않는다.</p>'
        f'<p><b>V0가 중요한 이유</b>: vln_ce GT는 habitat 기본 agent radius <b>0.1 m</b>로 수집됐다'
        f'(`vln_r2r_mini.yaml`에 radius 미지정 → `AgentConfig().radius=0.1`). 그 값에서는 파이프라인이 '
        f'원본 pixel goal을 <b>그대로 재현</b>해야 하며, 그래야 r_b를 키웠을 때의 변화가 embodiment 효과라고 말할 수 있다.</p>'
        f'<h3>이 리포트의 그림 색 규약 (모든 프레임 공통 — 한 번만 설명)</h3>'
        f'<p>프레임마다 <b>두 장</b>이 짝지어 나온다: 위 = <b>룩다운 사진</b>, 아래 = <b>floorplan</b>(위에서 본 그림).</p>'
        f'<p><b>룩다운 사진</b> — {chip(ORIG_COLOR, "원본 goal = 빈 원(테두리만)")} · '
        f'조정된 goal = <b>같은 r_b 색으로 채운 원</b>(lateral 통과) 또는 <b>테두리 원</b>(retreat 폴백). '
        f'<code>unchanged</code>인 r_b는 원본과 같은 자리라 그리지 않는다.</p>'
        f'<p><b>floorplan</b> — {chip((60, 90, 60), "시야 안 통행가능")}'
        f'{chip((28, 44, 28), "시야 밖 = A* 금지")}{chip((38, 38, 38), "장애물")} · '
        f'{chip((255, 255, 255), "원본 GT 경로")}{chip(ORIG_COLOR, "원본 goal")}'
        f'{chip((255, 140, 0), "로봇 현재 위치")} · r_b별 색 <b>선</b> = 재계획 경로, 같은 색 <b>점</b> = 옮겨진 goal.</p>'
        f'<p><b>r_b 색</b> — '
        + ''.join(chip(RB_COLORS[k % len(RB_COLORS)], f'r_b={rb}') for k, rb in enumerate(r_bs))
        + f'</p>'
        + (f'<p><b>shift의 핵심</b>: goal 지점의 <b>진행 방향(heading)에 수직인 lateral</b>으로만 옮겨 '
           f'좁은 틈의 <b>중앙에 정렬</b>한다 → <b>원래 목적지를 유지한 채 통과</b>. 로봇→goal 방위를 회전시키면 '
           f'통과가 아니라 <b>옆으로 돌아가</b> 목적지가 바뀐다(초기 구현의 오류). lateral로도 못 비키면 '
           f'통과 자체가 불가하므로 <b>경로를 따라 뒤로</b> 물린다(retreat 폴백, 테두리 원으로 표시).</p>'
           if args.goal_adjust == 'shift' else
           f'<p><b>retreat의 핵심</b>: goal이 충돌하면 <b>원본 경로를 따라 뒤로</b> 물려 clearance≥r_b인 '
           f'가장 먼 점을 고른다(<b>전진량↓·방위 유지</b>). 의미는 "큰 로봇은 덜 간다"이며, '
           f'lateral 정렬로 통과를 시도하는 <code>shift</code>와 대비된다.</p>')
        + f'<p><b>조정 기준 3종</b> — <code>retreat</code>: 경로를 따라 뒤로 물려 clearance≥r_b인 가장 먼 점'
          f'(<b>전진량↓·방위 유지</b>). <code>shift</code>: goal 지점의 <b>진행 방향에 수직(lateral)</b>으로만 '
          f'최소 이동해 틈 중앙에 정렬(<b>목적지 유지·통과</b>), 불가하면 retreat 폴백. '
          f'<code>nearest</code>: 최소 변위 안전점(방향 무관, 단조성 없음).</p>')
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')  # 발행 시 그림+설명 같이 실림
    save_gallery(out, 'report.html', 'pixel-goal 보존형 재계획 검증', summary, ''.join(body))
    print(f'[PG] V1 max={v1:.2f}px  V2 monotonic={v2[0]}({v2[1]})  V3 reach max={v3*100:.1f}cm')
    print(f'[PG] report -> {out}/report.html')


if __name__ == '__main__':
    main()
