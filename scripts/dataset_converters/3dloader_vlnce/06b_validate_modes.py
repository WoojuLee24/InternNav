"""모드 (A)/(B) 비교 검증 — `e`가 관측·정답을 바꾸는 두 가지 semantics.

사용자 질문에서 출발: "robot radius만 바꿨는데 왜 depth/BEV가 바뀌나?"
답: 현재(A)는 r_b가 **경로**를 바꾸고 카메라가 그 경로를 따라가므로 관측 위치 자체가 달라진다.

- **(A) follow**: 카메라가 새 경로를 따라감. 관측+정답 둘 다 변함.
  → 데이터 다양성엔 좋지만, paired counterfactual(C3)에선 "다른 위치 × 다른 정답"이라
    모델이 e를 무시하고 위치로 설명해버릴 수 있다.
- **(B) fixed**: 카메라는 **원본 에피소드 위치 고정**, GT path만 r_b로 재계획.
  `e`는 **BEV robot-radius dilation**으로 관측에 명시적으로 주입한다(논문 C1의 빠진 조각).
  → "같은 장면 × 다른 e → 다른 정답" = C3 gradient 논리가 성립.

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/06b_validate_modes.py --scene 17DRP5sb8fy --episode 0 --r_bs 0.15,0.30,0.45
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
from geometry_utils import colorize_depth, save_jpg  # noqa: E402
from viz_utils import floorplan_canvas, save_gallery  # noqa: E402
_m_verify_pose_mesh = importlib.import_module('00_verify_pose_mesh')  # noqa: E402
load_episode = _m_verify_pose_mesh.load_episode  # noqa: E402
from embodiment_augment import EmbodimentAugmenter  # noqa: E402
_m_validate_augment = importlib.import_module('06_validate_augment')  # noqa: E402
RB_COLORS = _m_validate_augment.RB_COLORS  # noqa: E402
START_COLOR = _m_validate_augment.START_COLOR  # noqa: E402
GOAL_COLOR = _m_validate_augment.GOAL_COLOR  # noqa: E402
bev_to_rgb = _m_validate_augment.bev_to_rgb  # noqa: E402
chip = _m_validate_augment.chip  # noqa: E402
_plen = _m_validate_augment._plen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--rig', default='125cm_0deg')
    ap.add_argument('--r_bs', default='0.15,0.30,0.45')
    ap.add_argument('--frame', type=int, default=4, help='(B)에서 관측을 고정할 원본 프레임 인덱스')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_n1')
    ap.add_argument('--esdf_dir', default='scripts/dataset_converters/gs_vlnpe/logs/esdf')
    ap.add_argument('--geo_dir', default='data/embodiment_aug/scene_geo',
                    help='씬 지오메트리 전용 ply 폴더 (01_build_scene_geo.py 산출물)')
    ap.add_argument('--out_dir', default='logs/embodiment_augment/modes')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    r_bs = [float(x) for x in args.r_bs.split(',')]

    aug = EmbodimentAugmenter(args.data_root, args.raw_root, args.mesh_root, args.esdf_dir, args.geo_dir)
    tasks = [json.loads(l) for l in open(Path(args.data_root) / args.scene / 'meta' / 'episodes.jsonl')]
    frames = load_episode(args.data_root, args.scene, args.rig, args.episode, 12)
    al = aug.align(args.scene, args.episode, frames, tasks[args.episode]['tasks'][0])
    T = al['T']
    poses_abs = [T @ p for _f, p, _d, _v in frames]
    cam = np.stack([p[:3, 3] for p in poses_abs])
    start_xy, goal_xy = cam[0, :2], cam[-1, :2]
    fz = float(np.median(cam[:, 2]) - 1.25)
    fi = min(args.frame, len(poses_abs) - 1)
    print(f'[modes] {args.scene} ep{args.episode} resid={al["resid"]:.4f} 고정프레임={fi} '
          f'cam={np.round(cam[fi][:2],2)}')

    npz = np.load(Path(args.esdf_dir) / f'{args.scene}.npz')
    base, draw, emit = floorplan_canvas(npz['nav_mask_ref'], npz['origin'], float(npz['voxel_size']), out)
    ov_a, ov_b = base.copy(), base.copy()

    body, rows, drawn = [], [], []
    for i, r_b in enumerate(r_bs):
        col = RB_COLORS[i % len(RB_COLORS)]
        traj, used_rb, status = aug.replan_or_fallback(args.scene, start_xy, goal_xy, r_b, floor_z=fz)
        if traj is None:
            rows.append((r_b, None, 'infeasible', 'infeasible')); print(f'[modes] r_b={r_b}: infeasible'); continue
        if status == 'fallback':
            rows.append((r_b, None, f'기각 → 기존 r_b={used_rb}', '—'))
            print(f'[modes] r_b={r_b}: 기각 → 기존 r_b={used_rb} 경로'); continue
        drawn.append((r_b, col))
        # (A) 카메라가 새 경로를 따라감
        sub_a, d_a, bev_a = aug.render_bev_along(args.scene, traj, 6, floor_z=fz, with_bev=True)
        mid = len(sub_a) // 2
        draw(ov_a, traj, col, 2)
        # (B) 카메라 고정(원본 프레임) + BEV dilation으로 e 주입
        _xy, d_b, bev_b = aug.render_at_poses(args.scene, [poses_abs[fi]], with_bev=True, r_b=r_b)
        draw(ov_b, traj, col, 2)

        sa = np.hstack([colorize_depth(d_a[mid], d_a[mid] > 0.05), bev_to_rgb(bev_a[mid])])
        sb = np.hstack([colorize_depth(d_b[0], d_b[0] > 0.05), bev_to_rgb(bev_b[0])])
        tag = str(r_b).replace('.', '')
        save_jpg(sa, out / f'modeA_rb{tag}.jpg'); save_jpg(sb, out / f'modeB_rb{tag}.jpg')
        rows.append((r_b, col, f'{_plen(traj):.2f} m', f'cam {np.round(sub_a[mid],2)}'))
        body.append(
            f'<h3>{chip(col, f"r_b={r_b}")} — 경로 {_plen(traj):.2f} m</h3>'
            f'<p><b>(A) 카메라가 새 경로를 따라감</b> — 관측 위치 {np.round(sub_a[mid],2)} (r_b마다 다름). depth | BEV</p>'
            f'<img src="modeA_rb{tag}.jpg" style="width:55%">'
            f'<p><b>(B) 카메라 고정</b>(원본 프레임 {fi}, 위치 {np.round(cam[fi][:2],2)} — r_b 무관) '
            f'+ BEV에 r_b dilation. depth(동일) | BEV(<b>r_b만큼 팽창</b>)</p>'
            f'<img src="modeB_rb{tag}.jpg" style="width:55%">')
        print(f'[modes] r_b={r_b}: (A) cam={np.round(sub_a[mid],2)} | (B) cam={np.round(cam[fi][:2],2)} 고정, '
              f'BEV occupied {100*(bev_b[0]>=1).mean():.1f}%')

    draw(ov_a, start_xy[None], START_COLOR, 5); draw(ov_a, goal_xy[None], GOAL_COLOR, 5)
    draw(ov_b, start_xy[None], START_COLOR, 5); draw(ov_b, goal_xy[None], GOAL_COLOR, 5)
    draw(ov_b, cam[fi][None, :2], (255, 140, 0), 6)      # 고정 관측 위치
    emit(ov_a, 'modes_paths.jpg'); emit(ov_b, 'modes_fixedcam.jpg')

    legend = ''.join(chip(c, f'r_b={rb}') for rb, c in drawn) + chip(START_COLOR, 'start') + chip(GOAL_COLOR, 'goal')
    body.insert(0, f'<h3>GT path (두 모드 공통 — r_b로 재계획)</h3><p>{legend}</p>'
                   f'<img src="modes_paths.jpg" style="width:60%">'
                   f'<p>(B)의 고정 관측 위치 = {chip((255,140,0), "주황")}</p>'
                   f'<img src="modes_fixedcam.jpg" style="width:60%">')
    summary = (
        f'<p><b>{args.scene}</b> ep{args.episode} · 정합 resid {al["resid"]:.4f} m</p><p>{legend}</p>'
        f'<table border=1 cellpadding=6><tr><th>r_b</th><th>색</th><th>경로 길이</th>'
        f'<th>(A) 관측 위치</th></tr>'
        + ''.join(f'<tr><td>{rb}</td><td>{chip(c,"") if c else "—"}</td><td>{a}</td><td>{b}</td></tr>'
                  for rb, c, a, b in rows) + '</table>'
        f'<h3>왜 (A)에서 depth가 바뀌나</h3>'
        f'<p>r_b → dilation → navigable 축소 → <b>A* 경로 변경</b> → 카메라를 그 경로 위에 재배치 → '
        f'다른 위치에서 렌더 → depth·BEV 변경. occupancy dilation이 depth를 직접 바꾸는 게 아니라 '
        f'<b>경로를 통해 간접적으로</b> 바꾼다.</p>'
        f'<h3>(A) vs (B)</h3>'
        f'<table border=1 cellpadding=6><tr><th></th><th>(A) follow</th><th>(B) fixed</th></tr>'
        f'<tr><td>관측</td><td>r_b마다 다름(위치 이동)</td><td><b>동일</b>(원본 프레임 고정)</td></tr>'
        f'<tr><td>GT path</td><td>다름</td><td>다름</td></tr>'
        f'<tr><td>e를 관측에 넣는 법</td><td>위치 변화로 간접</td><td><b>BEV robot-radius dilation</b>(논문 C1)</td></tr>'
        f'<tr><td>C3 paired counterfactual</td><td>약함 — 모델이 위치로 설명 가능</td>'
        f'<td><b>성립</b> — 같은 장면에 정답 차이를 설명할 변수가 e뿐</td></tr>'
        f'<tr><td>비용</td><td>경로마다 재렌더</td><td>원본 depth 재사용 가능(재렌더 불필요)</td></tr></table>')
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    save_gallery(out, 'report.html', '모드 (A) follow vs (B) fixed — e가 관측·정답을 바꾸는 방식', summary, ''.join(body))
    print(f'[modes] report -> {out}/report.html')


if __name__ == '__main__':
    main()
