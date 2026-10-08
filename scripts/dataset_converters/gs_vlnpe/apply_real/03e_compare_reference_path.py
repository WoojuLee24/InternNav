"""M1.3 부속 — vln_pe 재현 경로를 "계획 GT"(raw_data reference_path)와 재평가.

vln_pe의 traj_data는 물리 로봇 롤아웃이라 배회/진동이 섞여 있어(씬2 GT 회전 500~1000deg/m)
03의 "같은루트" 비교가 롤아웃 기준으로는 왜곡된다. R2R의 진짜 계획 경로는
`raw_data/r2r/<split>.json.gz`의 `reference_path`(navmesh 웨이포인트)에 있다 — 이 스크립트는
traj 에피소드를 instruction 텍스트로 raw 에피소드와 매칭하고(실측 10/10 매칭), Habitat
y-up 좌표를 z-up으로 변환해(`(x,y,z)_hab -> (x,-z,y)`, start 오차 0.01~0.17m로 검증),
03이 계획한 경로 vs reference_path를 비교한다.

실행:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/03e_compare_reference_path.py \\
        --scene 17DRP5sb8fy --out_dir scripts/dataset_converters/gs_vlnpe/logs
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_utils  # noqa: E402
from geometry_utils import save_jpg  # noqa: E402
from viz_utils import blink_widget_html, save_gallery  # noqa: E402

SCRIPT_NAME = '03e_compare_reference_path'
DEFAULT_LOG_DIR = 'logs/gs-vlnpe/apply_real'
RAW_SPLITS = ('train', 'val_seen', 'val_unseen')
# reference_path는 navmesh 웨이포인트(간격 수 m)라 밀집 궤적과의 chamfer가 과대평가된다 —
# 비교 전에 선형 보간으로 촘촘하게 만든다.
REF_RESAMPLE_M = 0.1
# "같은 루트" 판정: reference_path 보간점의 이 비율 이상이 planned 경로 0.5m 안에 있으면 같은 루트.
COVER_RADIUS_M = 0.5
COVER_FRAC_MIN = 0.8


def habitat_to_zup(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def load_raw_episodes(raw_root: Path, scene: str) -> list:
    eps = []
    for split in RAW_SPLITS:
        gz = raw_root / split / f'{split}.json.gz'
        if not gz.is_file():
            continue
        for e in json.loads(gzip.open(gz).read())['episodes']:
            if scene in e['scene_id']:
                eps.append(e)
    return eps


def resample_polyline(xy: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return xy[:1]
    t = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(t, s, xy[:, i]) for i in range(xy.shape[1])], axis=1)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    d_ab = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float((d_ab.min(1).mean() + d_ab.min(0).mean()) / 2)


def render_overlay(nav_mask, origin, cell, ref_xy, planned_xy, rollout_xy, path: Path):
    h, w = nav_mask.shape
    img = np.full((h, w, 3), 30, np.uint8)
    img[nav_mask] = (70, 110, 70)

    def draw(xy, color):
        ij = np.round((xy - origin[:2]) / cell).astype(int)
        pts = np.clip(ij, 0, [w - 1, h - 1])
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, 1)

    draw(rollout_xy, (120, 120, 200))   # 롤아웃(참고): 붉은 계열
    draw(ref_xy, (60, 230, 255))        # reference_path: 노랑
    draw(planned_xy, (255, 200, 60))    # planned: 하늘
    img = cv2.flip(img, 0)
    scale = max(1, int(np.ceil(700 / max(h, w))))
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    return save_jpg(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), path)


def compare_noeun_generated_gt(args) -> int:
    """03 A* waypoint, 최종 path GT, 04 camera pose를 기존 03e 리포트 형식으로 비교한다."""
    root = Path(args.out_dir)
    planned_eps = json.load(open(root / 'paths' / f'{args.scene}_random.json'))['episodes']
    npz = np.load(root / 'esdf' / f'{args.scene}.npz')
    nav_mask, origin, cell = npz['nav_mask_ref'], npz['origin'], float(npz['voxel_size'])
    obs_root = root / 'obs' / f'{args.scene}_random_isaac_d455_nominal'
    log_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.scene}_random'

    rows, body, n_same = [], '', 0
    for pe in planned_eps:
        ep_i = int(pe['episode_id'])
        ref_xy = np.asarray(pe['trajectory'], dtype=np.float64)
        planned_xy = resample_polyline(np.asarray(pe['waypoints_astar'], dtype=np.float64), REF_RESAMPLE_M)
        pose_files = sorted((obs_root / f'episode_{ep_i:06d}' / 'extrinsic').glob('*.npy'))
        rollout_xy = np.stack([np.load(p)[:2, 3] for p in pose_files])
        ch_plan, ch_roll = chamfer(planned_xy, ref_xy), chamfer(rollout_xy, ref_xy)
        d_ref_to_roll = np.sqrt(((ref_xy[:, None, :] - rollout_xy[None, :, :]) ** 2).sum(-1)).min(1)
        cover = float((d_ref_to_roll < COVER_RADIUS_M).mean())
        same = cover >= COVER_FRAC_MIN
        n_same += int(same)
        reason = '-' if same else '04 pose가 03 path GT에서 벗어남'
        rows.append((ep_i, ch_plan, ch_roll, cover, same, reason))
        overlay = render_overlay(nav_mask, origin, cell, ref_xy, planned_xy, rollout_xy,
                                 log_dir / f'ep{ep_i}_overlay.jpg')
        body += (f'<h4>ep {ep_i} — chamfer(A*↔path GT) {ch_plan:.3f} m, '
                 f'04 pose cover {cover*100:.0f}% {"✓ same-route" if same else "✗ different"}</h4>'
                 + blink_widget_html(
                     f'ov{ep_i}',
                     [('overlay — 노랑=03 path GT, 하늘=A* 원경로, 빨강=04 camera pose', overlay)]))

    ch_all = [r[1] for r in rows]
    summary = f'''
<div class="stat-row">
  <div class="stat"><b>04 pose 같은 루트</b><span class="pill good">{n_same}/{len(rows)}</span></div>
  <div class="stat"><b>chamfer(A*↔path GT) median</b><span class="pill">{np.median(ch_all):.3f} m</span></div>
</div>
<p>노은역 생성 GT의 단계 간 일관성을 비교한다. 노랑은 03 최종 경로 GT, 하늘은 refine/smoothing 전
A* waypoint 경로, 빨강은 04가 실제 렌더링하며 저장한 camera pose이다.</p>
<table><tr><th>ep</th><th>chamfer(A*,path GT) m</th><th>chamfer(04 pose,path GT) m</th>
<th>04 pose cover</th><th>판정</th><th>원인</th></tr>
{''.join(f"<tr><td>{r[0]}</td><td>{r[1]:.3f}</td><td>{r[2]:.3f}</td><td>{r[3]*100:.0f}%</td>"
         f"<td>{'same' if r[4] else 'different'}</td><td>{r[5]}</td></tr>" for r in rows)}
</table>
'''
    report = save_gallery(log_dir, 'report.html',
                          f'{SCRIPT_NAME} — {args.scene} (노은역 생성GT 재평가)', summary, body)
    print(f'  report html -> {report}')
    print(f'  => 04 pose 같은루트 {n_same}/{len(rows)}, A*↔path GT chamfer median {np.median(ch_all):.3f} m')
    return 0 if n_same == len(rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default='17DRP5sb8fy')
    parser.add_argument('--dataset', default='vln_pe', choices=['vln_pe', 'noeun_generated'],
                        help='reference_path는 vln_pe(raw_data/r2r) 전용')
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_pe/raw_data/r2r')
    parser.add_argument('--out_dir', default='scripts/dataset_converters/gs_vlnpe/apply_real')
    parser.add_argument('--log_dir', default=DEFAULT_LOG_DIR)
    args = parser.parse_args()
    if args.dataset == 'noeun_generated':
        return compare_noeun_generated_gt(args)
    if args.data_root is None:
        args.data_root = dataset_utils.default_data_root(args.dataset)
    print(f'[{SCRIPT_NAME}] scene={args.scene}')

    paths_json = Path(args.out_dir) / 'paths' / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}.json'
    if not paths_json.is_file():
        print(f'  [ERROR] {paths_json} 없음 — 03을 --dataset vln_pe로 먼저 돌릴 것')
        return 2
    planned_eps = json.load(open(paths_json))['episodes']

    esdf_npz = Path(args.out_dir) / 'esdf' / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}.npz'
    npz = np.load(esdf_npz)
    nav_mask, origin, cell = npz['nav_mask_ref'], npz['origin'], float(npz['voxel_size'])

    traj_meta = {m['episode_index']: m for m in
                 (json.loads(l) for l in (Path(args.data_root) / args.scene / 'meta' /
                                          'episodes.jsonl').read_text().strip().split('\n'))}
    raw_eps = load_raw_episodes(Path(args.raw_root), args.scene)
    print(f'  raw 에피소드 {len(raw_eps)}개, planned {len(planned_eps)}개')

    log_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.scene}{dataset_utils.dataset_tag(args.dataset)}'
    rows, body, n_same = [], '', 0
    for pe in planned_eps:
        ep_i = pe['episode_id']
        instr = traj_meta[ep_i]['tasks'][0].strip()
        cands = [e for e in raw_eps if e['instruction']['instruction_text'].strip() == instr]
        if not cands:
            print(f'    ep {ep_i:>3}: instruction 매칭 실패 — 건너뜀')
            continue
        gt = dataset_utils.load_gt_episode(args.data_root, args.scene, ep_i, args.dataset)
        # 같은 instruction이 여러 raw 에피소드에 있을 수 있어 start가 가장 가까운 것을 고른다.
        starts = np.stack([habitat_to_zup(np.array(c['start_position'])) for c in cands])
        best = int(np.argmin(np.linalg.norm(starts[:, :2] - gt['body_xyz'][0][:2], axis=1)))
        ref_full = habitat_to_zup(np.array(cands[best]['reference_path']))
        ref_xy = resample_polyline(ref_full[:, :2], REF_RESAMPLE_M)
        planned_xy = np.asarray(pe['trajectory'], dtype=np.float64)
        rollout_xy = gt['body_xyz'][:, :2]

        ch_plan = chamfer(planned_xy, ref_xy)
        ch_roll = chamfer(rollout_xy, ref_xy)
        d_ref_to_plan = np.sqrt(((ref_xy[:, None, :] - planned_xy[None, :, :]) ** 2).sum(-1)).min(1)
        cover = float((d_ref_to_plan < COVER_RADIUS_M).mean())
        same = cover >= COVER_FRAC_MIN
        n_same += same
        # 다른 루트가 나온 원인 분류 — planned의 start/goal은 "롤아웃"의 양 끝점이다(03의
        # reproduce 정의). 롤아웃이 목표 미도달로 끝났으면 planned는 애초에 ref의 목표로 가지
        # 않는다. 끝점이 맞는데도 다르면 A*가 다른 corridor를 고른 것(최단경로 우선이거나 GT
        # corridor 일부가 H1 밴드 맵에서 차단돼 우회).
        d_goal = float(np.linalg.norm(rollout_xy[-1] - ref_xy[-1]))
        if same:
            reason = '-'
        elif d_goal > 1.0:
            reason = f'롤아웃 목표 미도달({d_goal:.1f}m) → planned가 잘못된 끝점 기준'
        else:
            reason = 'A*가 다른 corridor 선택(최단경로/차단 우회)'
        rows.append((ep_i, ch_plan, ch_roll, cover, same, reason))
        print(f'    ep {ep_i:>3}: chamfer(planned,ref)={ch_plan:.3f}m  chamfer(rollout,ref)={ch_roll:.3f}m '
              f'cover={cover*100:.0f}% {"same-route" if same else "DIFFERENT — " + reason}')
        overlay = render_overlay(nav_mask, origin, cell, ref_xy, planned_xy, rollout_xy,
                                 log_dir / f'ep{ep_i}_overlay.jpg')
        body += (f'<h4>ep {ep_i} — chamfer(planned↔ref) {ch_plan:.3f} m, '
                 f'cover {cover*100:.0f}% {"✓ same-route" if same else "✗ different"}</h4>'
                 + blink_widget_html(f'ov{ep_i}', [('overlay — 노랑=reference(계획GT), 하늘=planned(A*), 빨강=rollout(실주행)',
                                                    overlay)]))

    if not rows:
        print('  [ERROR] 비교 가능한 에피소드 없음')
        return 2
    ch_all = [r[1] for r in rows]
    legend = ('<p><b>overlay 범례</b>: '
              '<span style="color:#ffe63c">━ 노랑 = reference_path(계획 GT, R2R navmesh 웨이포인트)</span> · '
              '<span style="color:#3cc8ff">━ 하늘 = planned(03의 A* 재현 경로 — 롤아웃 양 끝점 기준)</span> · '
              '<span style="color:#c87878">━ 빨강 = rollout(traj_data의 실제 로봇 주행 기록)</span> · '
              '초록 배경 = navigable 영역</p>')
    summary = f'''
<div class="stat-row">
  <div class="stat"><b>같은 루트(vs reference)</b><span class="pill good">{n_same}/{len(rows)}</span></div>
  <div class="stat"><b>chamfer(planned↔ref) median</b><span class="pill">{np.median(ch_all):.3f} m</span></div>
</div>
<p>03이 계획한 경로를 <b>계획 GT</b>(raw_data reference_path, navmesh 웨이포인트)와 비교 —
롤아웃(traj_data)은 배회가 섞여 있어 루트 판정 기준으로 부적합하다. 같은 루트 판정:
reference 보간점의 {COVER_FRAC_MIN*100:.0f}% 이상이 planned {COVER_RADIUS_M} m 안.</p>
{legend}
<table><tr><th>ep</th><th>chamfer(planned,ref) m</th><th>chamfer(rollout,ref) m</th><th>cover</th><th>판정</th><th>다른 루트 원인</th></tr>
{''.join(f"<tr><td>{r[0]}</td><td>{r[1]:.3f}</td><td>{r[2]:.3f}</td><td>{r[3]*100:.0f}%</td>"
         f"<td>{'same' if r[4] else 'different'}</td><td>{r[5]}</td></tr>" for r in rows)}
</table>
'''
    report = save_gallery(log_dir, 'report.html',
                          f'{SCRIPT_NAME} — {args.scene} (계획GT 재평가)', summary, body)
    print(f'  report html -> {report}')
    print(f'  => 같은루트 {n_same}/{len(rows)}, chamfer median {np.median(ch_all):.3f} m')
    return 0


if __name__ == '__main__':
    sys.exit(main())
