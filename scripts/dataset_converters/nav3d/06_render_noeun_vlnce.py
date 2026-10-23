"""06 — 노은역 경로(04 paths json)를 Isaac(GS RGB + 충돌 mesh depth)으로 렌더해 vln_ce 포맷으로 내보낸다.

    04 noeun_station_paths.json (nav3d 좌표, 점마다 바닥 z)
      -> vlnce_discretize (전진 0.25 m / 회전 15°, xy)            : 이산 action · 몸 위치 · yaw
      -> 몸 바닥 z = 04 경로 z 를 몸 위치에 투영 (계단을 따라 오르내림)
      -> 카메라 = 몸 + 전방 0.365 m · 높이 0.584 m · 아래로 14.76° (RBQ10 D455 장착값)
      -> Isaac render_along (apply_real/04_render_obs_isaac)      : rgb (NuRec GS) · depth (충돌 mesh)
      -> apply_real/05_export_vlnce.assemble                      : pixel goal · parquet · meta (vln_ce 21 컬럼)

카메라 칸: vln_ce 스키마의 **`60cm_15deg`** (2026-10-16 사용자 결정) — 640×480, hfov 68.7°(이 칸의 vln_ce 규약),
높이·pitch 만 D455 장착값(0.584 m · 14.76°). 다른 rig 칸은 writer 가 센티널로 채운다.
depth 는 구멍을 채우지 않는다(2026-10-15 결정). 지시문은 placeholder — labeling 단계에서 채운다.

## 용어
- **몸 위치**: 로봇 base 의 바닥 투영. 경로(04)는 몸이 지나는 선이다.
- **카메라 위치**: 몸 + 진행 방향 0.365 m 앞 (`--forward_m`, sensor_configuration.yaml 의 camera_link 를 base_link 로 접은 값).

사용 (isaac 컨테이너, 저장소 루트):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/06_render_noeun_vlnce.py
"""

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GS = HERE.parent / 'gs_vlnpe'
for p in (str(GS), str(GS / 'apply_real')):
    if p not in sys.path:
        sys.path.insert(0, p)
from render_poses import body_floor_z, episode_poses, start_yaw_for  # noqa: E402,F401


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths_json', default='scripts/dataset_converters/nav3d/logs/04_export_paths/noeun_station/noeun_station_paths.json')
    ap.add_argument('--usd_path', default='data/GS_USDZ/Subway/noeun_station_collision.usdz', help='역 전체 usdz (GS + 충돌 mesh)')
    ap.add_argument('--out_root', default='data/InternData-N1-v0.5-mini/vln_ce_noeun_nav3d')
    ap.add_argument('--out_scene', default='noeunstation', help="'_' 금지 — internnav loader 가 key.split('_') 로 자른다")
    ap.add_argument('--rig', default='60cm_15deg', help='vln_ce 스키마 칸')
    ap.add_argument('--mount_height_m', type=float, default=0.584, help='D455 높이 = base_height 0.55 + camera_link z 0.034')
    ap.add_argument('--pitch_down_deg', type=float, default=14.76, help='D455 pitch (camera_link quat)')
    ap.add_argument('--forward_m', type=float, default=0.365, help='D455 가 몸 중심보다 앞선 거리')
    ap.add_argument('--waypoint_spacing_m', type=float, default=0.5)
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--schema_ref', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/17DRP5sb8fy/data/chunk-000/episode_000000.parquet')
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=10.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--log_dir', default='scripts/dataset_converters/nav3d/logs/06_render_noeun')
    ap.add_argument('--start_heading', default='path', choices=['path', 'r2r_train'],
                    help='path = 첫 구간 방향으로 출발 (이전 동작) / r2r_train = R2R train 시작 각 분포에서 뽑아 제자리 회전부터 (episode 의 start_yaw 가 있으면 그것)')
    ap.add_argument('--start_seed', type=int, default=0)
    ap.add_argument('--r2r_raw', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/train/train.json.gz')
    return ap.parse_args()


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def main() -> int:
    args = parse()
    ex = _load('ex05', GS / 'apply_real' / '05_export_vlnce.py')
    cp = ex.load_camera_profiles()
    import cv2
    import vlnce_discretize as vd
    from geometry_utils import save_jpg

    base = cp.get(args.rig)
    prof = dataclasses.replace(base, mount_height_m=args.mount_height_m, pitch_down_deg=args.pitch_down_deg)
    log = Path(args.log_dir)
    sent = log / 'sentinels'
    sent.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_root) / 'traj_data' / 'r2r'
    O = ex.scene_paths(out_root, args.out_scene)
    rgb_dir, dep_dir = ex.stream_dir(O['videos_dir'], 'rgb', args.rig), ex.stream_dir(O['videos_dir'], 'depth', args.rig)
    for d in (rgb_dir, dep_dir):
        d.mkdir(parents=True, exist_ok=True)
    eps = json.loads(Path(args.paths_json).read_text())['episodes']
    if args.max_episodes > 0:
        eps = eps[:args.max_episodes]
    print(f'[06] {len(eps)} 에피소드 · rig {args.rig} {prof.width}x{prof.height} fx {prof.k[0, 0]:.2f} · 높이 {prof.mount_height_m} m · '
          f'pitch {prof.pitch_down_deg}° · 전방 {args.forward_m} m · usd {args.usd_path}', flush=True)

    r4 = ex.load_renderer_module()
    cas = r4.build_renderer(prof.width, prof.height, prof.k, Path(args.usd_path), args.light, None, args.rtx_ambient,
                            args.film_iso, prof.render_near_m, prof.render_far_m)
    t0, total, store, meta = time.time(), 0, {}, []
    for ei, ep in enumerate(eps):
        traj = np.asarray(ep['trajectory'], dtype=np.float64)
        y0 = start_yaw_for(ep, traj, args.start_heading, args.start_seed, args.r2r_raw)
        disc, zb, poses_cv = episode_poses(traj, prof, args.forward_m, args.waypoint_spacing_m, initial_yaw=y0)
        n = len(disc['action'])
        rgb, depth_m = r4.render_along(cas, poses_cv)
        for i in range(n):
            save_jpg(rgb[i], rgb_dir / f'episode_{ei:06d}_{i}.jpg')
            valid = np.isfinite(depth_m[i]) & (depth_m[i] > 0) & (depth_m[i] < prof.render_far_m * 0.99)
            cv2.imwrite(str(dep_dir / f'episode_{ei:06d}_{i}.png'), ex.depth_m_to_vlnce_png(depth_m[i], valid))
        store[f'pose_{ei}'] = vd.to_start_relative(poses_cv, floor_z=float(zb[0])).astype(np.float32)
        store[f'action_{ei}'] = disc['action'].astype(np.int32)
        total += n
        meta.append({'episode': ei, 'src_episode_id': ep['episode_id'], 'kind': ep['kind'], 'frames': n, 'start_yaw': y0,
                     'n_forward': int(disc['n_forward']), 'n_turn': int(disc['n_turn']), 'end_gap_m': float(disc['end_gap_m']),
                     'body_xy': disc['xy'].round(4).tolist(), 'body_z': zb.round(4).tolist(), 'yaw': disc['yaw'].round(5).tolist(),
                     'cam_c2w_cv': poses_cv.round(5).tolist()})
        print(f'[06] ep {ei:>3} {ep["kind"]:10s} {n:>4} f (전진 {disc["n_forward"]} 회전 {disc["n_turn"]}) '
              f'z {zb.min():+.2f}~{zb.max():+.2f} 종점 {disc["end_gap_m"]:.2f} m', flush=True)
    np.savez_compressed(sent / f'pose__{args.rig}.npz', **store)
    (log / 'frames_meta.json').write_text(json.dumps({'args': vars(args), 'profile': prof.to_dict(), 'episodes': meta}))
    print(f'[06] 렌더 {len(eps)} ep · {total} 프레임 · {time.time() - t0:.0f} s', flush=True)

    asm_args = argparse.Namespace(out_root=args.out_root, out_scene=args.out_scene, schema_ref=args.schema_ref,
                                  goal_report_rig=args.rig, paths_json=args.paths_json, scene='noeun_station',
                                  light=args.light, rtx_ambient=args.rtx_ambient, film_iso=args.film_iso)
    res = ex.assemble(asm_args, [args.rig], sent)
    print(f'[06] 조립 {res}', flush=True)
    return 0


if __name__ == '__main__':
    import os
    import traceback
    try:
        code = main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 3
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)          # simulation_app.close() 가 오래 멈춘다 (gs_vlnpe 실측)
