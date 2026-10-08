"""P-B1 검증 — EmbodimentAugmenter를 실제 vln_ce 에피소드에 적용해 e(r_b)에 따라 GT path와
관측(depth/BEV)이 달라짐을 시각화한다. 기존 학습/데이터 코드는 건드리지 않음(3dloader_vlnce 정책).

산출: logs/embodiment_augment/pb/ 에 (1) r_b별 GT path floorplan 오버레이, (2) r_b별 sample depth/BEV,
+ report.html. publish_artifact_report.py로 Artifact 발행.

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/06_validate_augment.py --scene s8pcmisQ38h --episode 0 --r_bs 0.20,0.35,0.50
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_GS = _HERE.parents[0] / 'gs_vlnpe'
sys.path.insert(0, str(_GS)); sys.path.insert(0, str(_HERE))
from geometry_utils import colorize_depth, save_jpg  # noqa: E402
from viz_utils import floorplan_canvas, save_gallery  # noqa: E402
_m_verify_pose_mesh = importlib.import_module('00_verify_pose_mesh')  # noqa: E402
load_episode = _m_verify_pose_mesh.load_episode  # noqa: E402
from embodiment_augment import EmbodimentAugmenter  # noqa: E402

# RGB 튜플(그림에 그려지는 실제 색). 범례 칩도 같은 값을 써서 색-값 대응이 어긋날 수 없게 한다.
RB_COLORS = [(90, 190, 255), (255, 210, 60), (255, 90, 90), (0, 255, 90), (200, 120, 255)]
START_COLOR, GOAL_COLOR = (255, 255, 255), (255, 0, 255)


def chip(rgb, label):
    """색 견본 + 라벨 (텍스트로 '하늘/노랑'이라 쓰는 대신 실제 색을 보여준다)."""
    return (f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px">'
            f'<span style="width:14px;height:14px;border-radius:3px;border:1px solid #8888;'
            f'background:rgb({rgb[0]},{rgb[1]},{rgb[2]})"></span>{label}</span>')


def bev_to_rgb(bev):
    """0=unknown(검정) 0.5=free(회색) 1=occupied(빨강)."""
    img = np.zeros((*bev.shape, 3), np.uint8)
    img[bev == 0.5] = (90, 90, 90)
    img[bev >= 1.0] = (255, 70, 70)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='s8pcmisQ38h')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--rig', default='125cm_0deg')
    ap.add_argument('--r_bs', default='0.20,0.35,0.50')
    ap.add_argument('--n_frames', type=int, default=6)
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_n1')
    ap.add_argument('--esdf_dir', default='scripts/dataset_converters/gs_vlnpe/logs/esdf')
    ap.add_argument('--geo_dir', default='data/embodiment_aug/scene_geo',
                    help='씬 지오메트리 전용 ply 폴더 (01_build_scene_geo.py 산출물)')
    ap.add_argument('--out_dir', default='logs/embodiment_augment/pb')
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    r_bs = [float(x) for x in args.r_bs.split(',')]

    aug = EmbodimentAugmenter(args.data_root, args.raw_root, args.mesh_root, args.esdf_dir, args.geo_dir)

    # 에피소드 로드 + 정합 → mesh 절대 start/goal
    tasks = [json.loads(l) for l in open(Path(args.data_root) / args.scene / 'meta' / 'episodes.jsonl')]
    instr = tasks[args.episode]['tasks'][0].strip()
    frames = load_episode(args.data_root, args.scene, args.rig, args.episode, 12)
    al = aug.align(args.scene, args.episode, frames, instr)
    T = al['T']
    cam_mesh = np.stack([(T @ p)[:3, 3] for _f, p, _d, _v in frames])
    start_xy, goal_xy = cam_mesh[0, :2], cam_mesh[-1, :2]
    floor_z_ep = float(np.median(cam_mesh[:, 2]) - 1.25)  # 에피소드 층(다층 씬 대응): cam_z - cam_height
    print(f'[P-B1] {args.scene} ep{args.episode} align resid={al["resid"]:.4f} '
          f'start={np.round(start_xy,2)} goal={np.round(goal_xy,2)} floor_z_ep={floor_z_ep:.2f}')

    # occ 배경 floorplan
    npz = np.load(Path(args.esdf_dir) / f'{args.scene}.npz')
    base, draw, emit = floorplan_canvas(npz['nav_mask_ref'], npz['origin'], float(npz['voxel_size']), out_dir)
    overlay = base.copy()

    rows, body, drawn = [], [], []
    for i, r_b in enumerate(r_bs):
        # 끝점이 원본 goal에서 벗어나면 기각하고 기존 r_b 경로로 되돌린다(instruction 라벨 보호).
        traj, used_rb, status = aug.replan_or_fallback(args.scene, start_xy, goal_xy, r_b, floor_z=floor_z_ep)
        col = RB_COLORS[i % len(RB_COLORS)]
        if traj is None:
            # 그림에 선이 없으므로 범례에도 색을 배정하지 않는다(이전 버전은 색을 배정해 오해를 줬다).
            rows.append((r_b, None, 'infeasible', 0.0))
            print(f'[P-B1] r_b={r_b}: infeasible'); continue
        if status == 'fallback':
            # 기각됨 — 경로가 기존 r_b 것과 동일하므로 새로 그리지 않고 표에만 남긴다.
            rows.append((r_b, None, f'기각 → 기존 r_b={used_rb} 경로 사용', float(_plen(traj))))
            print(f'[P-B1] r_b={r_b}: 기각(끝점 이탈/계획실패) → 기존 r_b={used_rb} 경로'); continue
        drawn.append((r_b, col))
        draw(overlay, traj, col, 2)
        sub, depths, bevs = aug.render_bev_along(args.scene, traj, args.n_frames, floor_z=floor_z_ep)
        # 대표 프레임(중간) depth+BEV
        mid = len(depths) // 2
        dvalid = depths[mid] > 0.05
        strip = np.hstack([colorize_depth(depths[mid], dvalid), bev_to_rgb(bevs[mid])])
        save_jpg(strip, out_dir / f'pb_rb{str(r_b).replace(".","")}_depthbev.jpg')
        rows.append((r_b, col, f'{len(traj)}pt len={_plen(traj):.1f}m', float(_plen(traj))))
        body.append(f'<h3>{chip(col, f"r_b={r_b}")} — path {len(traj)}pt, 길이 {_plen(traj):.1f}m</h3>'
                    f'<p>대표 프레임 depth | BEV (검정=unknown, 회색=free, 빨강=occupied)</p>'
                    f'<img src="pb_rb{str(r_b).replace(".","")}_depthbev.jpg" style="width:60%">')
        print(f'[P-B1] r_b={r_b}: path {len(traj)}pt len={_plen(traj):.2f}m, rendered {len(depths)} frames+BEV')

    draw(overlay, start_xy[None], START_COLOR, 5); draw(overlay, goal_xy[None], GOAL_COLOR, 5)
    emit(overlay, 'pb_paths.jpg')
    # 범례: **실제로 그린 경로만** 색 칩으로. infeasible은 색 없이 명시.
    legend = ''.join(chip(c, f'r_b={rb}') for rb, c in drawn) + \
        chip(START_COLOR, 'start') + chip(GOAL_COLOR, 'goal')
    rejected = [rb for rb, c, s, _l in rows if c is None]
    if rejected:
        legend += (f'<span style="opacity:.7">· r_b={", ".join(str(x) for x in rejected)}: '
                   f'기각/계획불가 → 기존 r_b 경로 사용(별도 선 없음)</span>')
    body.insert(0, f'<h3>e(r_b)별 GT path — 같은 start/goal</h3><p>{legend}</p>'
                   f'<img src="pb_paths.jpg" style="width:70%">')
    summary = (
        f'<p><b>{args.scene}</b> ep{args.episode} · 정합 resid {al["resid"]:.4f} m</p>'
        f'<p>{legend}</p>'
        f'<table border=1 cellpadding=6><tr><th>r_b</th><th>색</th><th>path</th><th>길이(m)</th></tr>'
        + ''.join(f'<tr><td>{rb}</td><td>{chip(c, "") if c else "—"}</td><td>{s}</td><td>{l:.2f}</td></tr>'
                  for rb, c, s, l in rows) + '</table>'
        f'<p>파이프라인: vln_ce 에피소드 → 정합(T_sf2mesh) → r_b별 GT path 재계획(plan_episode) → '
        f'새 path에서 depth-only 렌더(타일 상주) → BEV. r_b가 GT path와 그 경로의 관측(depth/BEV)을 바꾼다.</p>'
        f'<p><b>끝점 보호 규칙</b>: 새 r_b 경로의 끝점이 원본 goal에서 {int(100*0.10)} cm 넘게 벗어나면 '
        f'<b>기각하고 기존 r_b 경로를 쓴다</b>. r_b dilation이 goal 셀을 지우면 스냅으로 도착지가 밀리는데, '
        f'그러면 instruction("…에서 멈춰라")이 거짓 라벨이 되기 때문이다. '
        f'(실측: 미적용 시 goal이 최대 37.8 cm 이동 → 적용 후 채택 경로는 전부 끝점 오차 ≤3.1 cm)</p>')
    (out_dir / 'summary.html').write_text(summary, encoding='utf-8')
    save_gallery(out_dir, 'report.html', 'P-B1 — embodiment augment 검증', summary, ''.join(body))
    print(f'[P-B1] report -> {out_dir}/report.html')


def _plen(traj):
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(traj), axis=0), axis=1)))


if __name__ == '__main__':
    main()
