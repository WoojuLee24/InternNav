"""W1 — depth 1장 격자 vs 3D mesh 오라클. **논문 "구현 3안" 비교표의 근거.**

같은 프레임에서 두 장애물 맵을 만들어 겹친다.
- **ours(2D)**: 그 프레임 depth 한 장 -> `depth_to_bev_occ_ros2` (시야각 내만 관측)
- **oracle(3D)**: 씬 mesh 3D occupancy -> `derive_obstacle_2d` (전역, 3dloader가 쓰는 것과 동일)

정합은 3dloader의 `vlnce_align.align_episode`(해석적 `T_sf2mesh`)를 그대로 재사용한다.

측정 (전부 ±`bev_range` 정사각 안에서):
| 지표 | 뜻 |
|---|---|
| IoU(관측영역) | 우리가 **본** 곳에서 장애물 판정이 오라클과 얼마나 일치하는가 |
| false-free(관측영역) | 오라클은 장애물인데 우리는 free — **관측 품질**의 손실 |
| false-occ(관측영역) | 오라클은 비었는데 우리는 장애물 (depth 노이즈/클립) |
| unseen-obstacle | 오라클 장애물 중 우리가 **아예 못 본**(unknown) 비율 — **시야각 한계의 비용** |

실행: /usr/bin/python scripts/dataset_converters/2dloader_vlnce/validate_w1_oracle.py --scene 17DRP5sb8fy --episode 0 --preset 125cm_0_30 --n_frames 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CONV = _HERE.parent
for _p in (str(_CONV / 'gs_vlnpe'), str(_CONV / '3dloader_vlnce'), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esdf_utils import derive_obstacle_2d, world_to_cell  # noqa: E402
from geometry_utils import load_scene_mesh, save_jpg  # noqa: E402
from viz_utils import save_gallery  # noqa: E402
from pixel_goal_utils import rig_height_m  # noqa: E402
from vlnce_align import align_episode, build_rawdata_index  # noqa: E402
from verify_pose_mesh import load_episode as load_episode_3d  # noqa: E402  (정합용 프레임 로더)

from episode_io import load_episode, to_224_depth  # noqa: E402
from local_map import (  # noqa: E402
    limit_torch_threads,
    bev_ij_to_plan_xy, build_local_map, plan_to_robot, robot_to_world, robot_xy_to_bev_ij, world_to_robot,
)
from report_common import (  # noqa: E402
    bev_rows, gate_note, legend_table, param_glossary, path_rows, pipeline_note,
)
from viz2d import GT_COLOR, START_COLOR, bev_rgb, draw_pts, hstrip, label  # noqa: E402

AGREE_TP = (0, 200, 90)      # 둘 다 장애물
AGREE_FP = (255, 90, 90)     # 우리만 장애물
AGREE_FN = (90, 150, 255)    # 오라클만 장애물 (관측했는데 놓침)
AGREE_UNSEEN = (200, 120, 255)   # 오라클 장애물인데 미관측


def match_raw_episode(idx, scene: str, meta_task: str):
    """meta의 tasks[0] 문자열로 raw_data 에피소드를 찾는다 (전체 -> 분리된 각 지시문 순)."""
    cands = [meta_task.strip()] + [s.strip() for s in meta_task.split('<INSTRUCTION_SEP>')]
    for it in cands:
        if (scene, it) in idx:
            return idx[(scene, it)]
    raise KeyError(f'raw_data에서 못 찾음: {scene} / {cands[0][:60]}...')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='17DRP5sb8fy')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--preset', default='125cm_0_30')
    ap.add_argument('--n_frames', type=int, default=6)
    ap.add_argument('--h_nav', type=float, default=0.15)
    ap.add_argument('--h_b', type=float, default=1.25)
    ap.add_argument('--r_b', type=float, default=0.10)
    ap.add_argument('--unknown', default='nontraversable',
                    choices=('nontraversable', 'free', 'block'))
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_n1')
    ap.add_argument('--esdf_dir', default='scripts/dataset_converters/gs_vlnpe/logs/esdf')
    ap.add_argument('--out_dir', default='logs/embodiment_augment2d/w1')
    args = ap.parse_args()
    limit_torch_threads()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ep = load_episode(args.data_root, args.scene, args.episode, args.preset, n_frames=args.n_frames)
    pitch_2 = float(ep.rig_ld.split('_')[1].replace('deg', ''))
    cam_h = rig_height_m(ep.rig_ld)

    # --- 3D 오라클 준비: 정합 + 씬 occupancy
    meta = [json.loads(l) for l in open(Path(args.data_root) / args.scene / 'meta' / 'episodes.jsonl')]
    task = next(m['tasks'][0] for m in meta if m['episode_index'] == args.episode)
    raw_ep = match_raw_episode(build_rawdata_index(args.raw_root), args.scene, task)
    mesh = load_scene_mesh(Path(args.mesh_root), args.scene)
    # 정합은 **FPV(pitch_1) rig**로 한다. 룩다운 rig는 바닥을 스치듯 보아 mesh 거리 잔차가 커진다
    # (17DRP ep0 실측: 125cm_0deg 0.0033 m vs 125cm_30deg 0.0156 m). T는 rig와 무관하므로 더 좋은 쪽을 쓴다.
    frames_3d = load_episode_3d(args.data_root, args.scene, ep.rig_fpv, args.episode, args.n_frames)
    T_sf2mesh, resid, _ = align_episode(frames_3d, mesh, raw_ep)
    npz = np.load(Path(args.esdf_dir) / f'{args.scene}.npz')
    occ, occ_origin, cell_occ = npz['occupancy'], npz['origin'], float(npz['voxel_size'])
    print(f'[W1] {args.scene} ep{args.episode} align resid={resid:.4f} m · occ{occ.shape} cell={cell_occ}')
    assert resid < 0.05, f'정합 residual이 너무 크다({resid:.3f} m) — W1 비교가 무의미하다'

    rows, body = [], []
    for f in ep.frames:
        d224 = to_224_depth(f.depth_m)
        m = build_local_map(d224, ep.rig_ld, pitch_2, args.r_b, args.h_nav, args.h_b, unknown=args.unknown)
        S = m['bev_size']

        # BEV 셀 중심 -> robot -> world(start-rel) -> mesh -> 오라클 2D 장애물
        ij = np.stack(np.meshgrid(np.arange(S), np.arange(S), indexing='ij'), -1).reshape(-1, 2)
        r_xy = plan_to_robot(bev_ij_to_plan_xy(ij))
        r_xyz = np.concatenate([r_xy, np.zeros((len(r_xy), 1))], axis=1)     # robot 바닥 평면
        w = robot_to_world(r_xyz, f.pose_ld, pitch_2, cam_h)
        mesh_xyz = (T_sf2mesh @ np.concatenate([w, np.ones((len(w), 1))], 1).T).T[:, :3]
        floor_z = float((T_sf2mesh @ f.pose_ld)[2, 3] - cam_h)              # 이 프레임의 바닥(다층 대응)
        obs_oracle = derive_obstacle_2d(occ, occ_origin, floor_z, args.h_b, cell_occ, args.h_nav)
        cij = world_to_cell(mesh_xyz[:, :2], occ_origin, cell_occ)
        inside = ((cij[:, 0] >= 0) & (cij[:, 0] < obs_oracle.shape[1]) &
                  (cij[:, 1] >= 0) & (cij[:, 1] < obs_oracle.shape[0]))
        oracle = np.zeros(len(ij), bool)
        oracle[inside] = obs_oracle[cij[inside, 1], cij[inside, 0]]
        oracle = oracle.reshape(S, S)
        inside = inside.reshape(S, S)

        ours, seen = m['occupied'], m['observed']
        sel = seen & inside                      # 관측했고 오라클 격자 안
        tp = int((ours & oracle & sel).sum()); fp = int((ours & ~oracle & sel).sum())
        fn = int((~ours & oracle & sel).sum())
        iou = tp / max(tp + fp + fn, 1)
        n_or_in = int((oracle & inside).sum())
        unseen = int((oracle & ~seen & inside).sum())
        rows.append(dict(idx=f.idx, iou=iou, ff=fn / max(tp + fn, 1), fo=fp / max(tp + fp, 1),
                         unseen=unseen / max(n_or_in, 1), seen=float(seen.mean()),
                         n_or=n_or_in, tp=tp, fp=fp, fn=fn))

        # 그림: ours | oracle | agreement
        agree = np.zeros((S, S, 3), np.uint8)
        agree[oracle & ~seen & inside] = AGREE_UNSEEN
        agree[(~ours & oracle) & sel] = AGREE_FN
        agree[(ours & ~oracle) & sel] = AGREE_FP
        agree[(ours & oracle) & sel] = AGREE_TP
        orc_img = np.zeros((S, S, 3), np.uint8)
        orc_img[inside] = (90, 90, 90); orc_img[oracle & inside] = (255, 70, 70)
        j = f.idx + f.rel_goal + 1
        gt_ij = robot_xy_to_bev_ij(world_to_robot(ep.poses_ld[f.idx:j + 1, :3, 3],
                                                  f.pose_ld, pitch_2, cam_h)[:, :2])
        ours_img = bev_rgb(m['bev'])
        for im in (ours_img, orc_img, agree):
            draw_pts(im, gt_ij, GT_COLOR, 1)
            draw_pts(im, np.array([[S // 2, S // 2]]), START_COLOR, 3)
        # cv2.putText는 한글을 못 그린다 — 이미지 안 라벨은 ASCII로 (설명은 HTML 쪽에)
        save_jpg(hstrip([label(ours_img, 'ours (single depth)'), label(orc_img, 'oracle (3D mesh)'),
                         label(agree, 'agreement')], height=448), out / f'w1_f{f.idx:04d}.jpg')
        body.append(f'<h3>frame {f.idx}</h3>'
                    f'<p>IoU(관측영역) <b>{iou:.3f}</b> · false-free {100*rows[-1]["ff"]:.1f}% · '
                    f'false-occ {100*rows[-1]["fo"]:.1f}% · '
                    f'오라클 장애물 중 미관측 <b>{100*rows[-1]["unseen"]:.1f}%</b> · '
                    f'관측 면적 {100*rows[-1]["seen"]:.1f}%</p>'
                    f'<img src="w1_f{f.idx:04d}.jpg" style="width:100%">')
        print(f'[W1] f{f.idx}: IoU={iou:.3f} false-free={100*rows[-1]["ff"]:.1f}% '
              f'false-occ={100*rows[-1]["fo"]:.1f}% unseen={100*rows[-1]["unseen"]:.1f}% '
              f'seen={100*seen.mean():.1f}%')

    agg = {k: float(np.mean([r[k] for r in rows])) for k in ('iou', 'ff', 'fo', 'unseen', 'seen')}
    legend = legend_table(
        [(AGREE_TP, 'box', 'TP — 둘 다 장애물 (초록)', '우리도 오라클도 장애물이라 본 칸'),
         (AGREE_FP, 'box', 'FP — 우리만 장애물 (빨강)', '오라클은 빈 곳인데 우리가 막았다 (depth 노이즈·5 m clip 벽)'),
         (AGREE_FN, 'box', 'FN — 오라클만 (파랑)', '봤는데 놓친 장애물 = 관측 품질의 손실'),
         (AGREE_UNSEEN, 'box', '미관측 장애물 (보라)', '오라클엔 있는데 우리는 아예 못 본 칸 = 시야각의 비용')]
        + path_rows(with_gt=True) + bev_rows(),
        title='범례 — agreement 그림(3번째 패널)의 4색 + 공통')
    summary = (
        f'<p><b>{args.scene}</b> ep{args.episode} · preset {args.preset} · '
        f'mesh 정합 residual {resid:.4f} m</p>'
        + gate_note(
            what='같은 프레임에서 <b>depth 한 장으로 만든 장애물 맵</b>과 <b>씬 mesh 전체에서 만든 '
                 '오라클</b>을 겹쳐, 본 곳의 판정 일치도(IoU)와 <b>아예 못 본 장애물의 비율</b>을 잰다.',
            why='논문 §해결방법3의 세 구현안 중 "시야각 내 이미지·depth만"이 "3D occ 전부"에 비해 '
                '<b>무엇을 잃는지</b>를 수치로 만드는 유일한 게이트다. 문서 L158의 한계'
                '("멀리 돌아가는 gt_path는 만들 수 없다")가 여기서 숫자가 된다.',
            criterion='사전 임계 없음 — 측정·보고가 목적이다. 다만 정합 residual이 &lt; 0.05 m여야 '
                      '비교 자체가 성립한다.')
        + '<p><b>이 게이트만 mesh를 쓴다.</b> 나머지 W0·W2~W6은 mesh도 3D occupancy도 정합도 쓰지 않고 '
        'depth 한 장으로만 돌아간다 — 여기서 mesh는 <i>비교 대상 정답</i>일 뿐 파이프라인의 입력이 아니다.</p>'
        + pipeline_note()
        + param_glossary(preset=args.preset, r_b=args.r_b, h_nav=args.h_nav, h_b=args.h_b,
                         unknown=args.unknown)
        + '<h4>세 패널의 뜻</h4><p><b>ours (single depth)</b> = 이 프레임 depth 한 장으로 만든 BEV · '
        '<b>oracle (3D mesh)</b> = 씬 mesh 전체 occupancy를 같은 높이 밴드로 자른 정답(3dloader가 쓰는 것) · '
        '<b>agreement</b> = 둘을 겹쳐 칠한 것. 두 맵은 <code>vlnce_align</code>의 해석적 정합 '
        f'(residual {resid:.4f} m)으로 같은 좌표에 놓았다.</p>'
        + legend
        + f'<h4>지표</h4><table border=1 cellpadding=6><tr><th>지표(프레임 평균)</th><th>값</th><th>뜻</th></tr>'
        f'<tr><td>IoU (관측영역)</td><td><b>{agg["iou"]:.3f}</b></td>'
        f'<td>본 곳에서의 장애물 판정 일치도</td></tr>'
        f'<tr><td>false-free</td><td>{100*agg["ff"]:.1f}%</td>'
        f'<td>봤는데 장애물을 놓친 비율 (depth 노이즈·slab 밖)</td></tr>'
        f'<tr><td>false-occ</td><td>{100*agg["fo"]:.1f}%</td>'
        f'<td>빈 곳을 장애물로 본 비율 (5 m clip 벽 포함)</td></tr>'
        f'<tr><td><b>미관측 장애물</b></td><td><b>{100*agg["unseen"]:.1f}%</b></td>'
        f'<td><b>시야각 한계의 비용</b> — 3D 오라클 대비 못 보는 장애물</td></tr>'
        f'<tr><td>관측 면적</td><td>{100*agg["seen"]:.1f}%</td>'
        f'<td>±5 m 정사각형 중 관측된 비율 (HFOV 79°·5 m 부채꼴 ≈ 17%)</td></tr></table>'
        f'<table border=1 cellpadding=6><tr><th>frame</th><th>IoU</th><th>false-free</th>'
        f'<th>false-occ</th><th>미관측</th><th>관측면적</th><th>TP/FP/FN</th></tr>'
        + ''.join(f'<tr><td>{r["idx"]}</td><td>{r["iou"]:.3f}</td><td>{100*r["ff"]:.1f}%</td>'
                  f'<td>{100*r["fo"]:.1f}%</td><td>{100*r["unseen"]:.1f}%</td>'
                  f'<td>{100*r["seen"]:.1f}%</td>'
                  f'<td>{r["tp"]}/{r["fp"]}/{r["fn"]}</td></tr>' for r in rows) + '</table>'
        f'<p><b>읽는 법</b> — 이 표가 논문 §해결방법3의 세 구현안 중 "시야각 내 이미지·depth만"이 '
        f'"3D occ 전부"에 비해 무엇을 잃는지다. IoU는 <i>본 곳의 품질</i>, 미관측 비율은 '
        f'<i>못 보는 범위</i>. 후자가 문서 L158의 한계("멀리 돌아가는 gt_path는 만들 수 없다")를 '
        f'수치로 만든 것이다.</p>')
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'W1 — 2D(depth 1장) vs 3D(mesh) 오라클', summary, ''.join(body))
    print(f'[W1] 평균 IoU={agg["iou"]:.3f} 미관측={100*agg["unseen"]:.1f}% -> {out}/report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
