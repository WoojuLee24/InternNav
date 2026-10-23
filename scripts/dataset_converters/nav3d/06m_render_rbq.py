"""06m — RBQ 가 물리로 걸은 자세(06k 결과)로 D455 이미지를 렌더해 vln_ce / vln_pe 포맷으로 저장한다 (Isaac).

06 / 06p 는 경로 위에 카메라를 바로 놓는다(몸 흔들림 없음). 여기서는 **06k 가 기록한 trunk 자세(action 마다)** 에
D455 를 붙인다 — 계단에서 몸이 기울고 흔들리는 것이 그대로 들어간다. 렌더러·포맷 코드는 06 / 06p 와 같은 것을 쓴다.

D455 장착 (sim_docker `sensor_configuration.yaml`, base_link = trunk):
  base_link → os_sensor  xyz (0.3, 0, 0.05)
  os_sensor → camera_link xyz (0.065151, 0.002019, −0.015822) · quat_xyzw (−0.000391, 0.128428, 0.001329, 0.991718) (아래로 14.76°)
  camera_link 는 ROS 규약 (x 앞 · y 왼 · z 위) → OpenCV 카메라 (x 오른 · y 아래 · z 앞)
성공한 ep 만 저장한다 (06k success). action 규약: vln_ce = 06 과 같음 (action[0] = −1), vln_pe = 06p 와 같음.

사용 (isaac 컨테이너, 저장소 루트):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/06m_render_rbq.py --walk_json <06k walk json> --format vlnce
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/06m_render_rbq.py --walk_json <06k walk json> --format vlnpe
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
GS = HERE.parent / 'gs_vlnpe'
for p in (str(GS), str(GS / 'apply_real')):
    if p not in sys.path:
        sys.path.insert(0, p)
from render_poses import T_BASE_OS, T_OS_CAM, Q_OS_CAM_XYZW, quat_wxyz_to_R, trunk_to_cam_cv, frames_to_poses  # noqa: E402,F401
from render_poses import c2w_to_vlnpe_quat, ce_to_pe_actions  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walk_json', required=True)
    ap.add_argument('--format', required=True, choices=['vlnce', 'vlnpe'])
    ap.add_argument('--usd_path', default=None, help='None = walk json 의 background (노은역) · MP3D 는 --mesh_root/--scene')
    ap.add_argument('--mesh_root', default=None)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--out_root', default=None, help='기본: data/.../{vln_ce|vln_pe}_noeun_nav3d_rbq')
    ap.add_argument('--out_scene', default='noeunstation')
    ap.add_argument('--rig', default='60cm_15deg', help='vln_ce 칸 (06 과 같음)')
    ap.add_argument('--schema_ref', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/17DRP5sb8fy/data/chunk-000/episode_000000.parquet')
    ap.add_argument('--pe_schema_ref', default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/17DRP5sb8fy/data/chunk-000/episode_000000.parquet')
    ap.add_argument('--render_far_m', type=float, default=None)
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=10.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--log_dir', default='scripts/dataset_converters/nav3d/logs/06m_render_rbq')
    return ap.parse_args()


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def main() -> int:
    args = parse()
    W = json.loads(Path(args.walk_json).read_text())
    ok = [r for r in W['results'] if r['success']]
    if args.max_episodes:
        ok = ok[:args.max_episodes]
    fmt = args.format
    out_root = Path(args.out_root or f"data/InternData-N1-v0.5-mini/{'vln_ce' if fmt == 'vlnce' else 'vln_pe'}_noeun_nav3d_rbq")
    log = Path(args.log_dir) / fmt
    sent = log / 'sentinels'
    sent.mkdir(parents=True, exist_ok=True)
    ex = _load('ex05', GS / 'apply_real' / '05_export_vlnce.py')
    r4 = ex.load_renderer_module()
    usd = r4.load_scene_model(args.mesh_root, args.scene) if args.mesh_root else Path(args.usd_path or W['background'])
    far = {} if args.render_far_m is None else {'render_far_m': args.render_far_m}
    if fmt == 'vlnce':
        cp = ex.load_camera_profiles()
        prof = cp.get(args.rig)
        cas = r4.build_renderer(prof.width, prof.height, prof.k, usd, args.light, None, args.rtx_ambient, args.film_iso,
                                prof.render_near_m, far.get('render_far_m', prof.render_far_m))
    else:
        du = _load('du_06m', GS / 'dataset_utils.py')
        cas = r4.build_renderer(*du.VLNPE_IMG_WH, du.VLNPE_K, usd, args.light, None, args.rtx_ambient, args.film_iso, **far)
    print(f'[06m {fmt}] 성공 ep {len(ok)}/{len(W["results"])} · usd {usd}', flush=True)
    t0, total, meta = time.time(), 0, []
    if fmt == 'vlnce':
        import cv2
        import vlnce_discretize as vd
        from geometry_utils import save_jpg
        O = ex.scene_paths(Path(out_root) / 'traj_data' / 'r2r', args.out_scene)
        rgb_dir, dep_dir = ex.stream_dir(O['videos_dir'], 'rgb', args.rig), ex.stream_dir(O['videos_dir'], 'depth', args.rig)
        for d in (rgb_dir, dep_dir):
            d.mkdir(parents=True, exist_ok=True)
        store = {}
        for ei, r in enumerate(ok):
            P = frames_to_poses(r['frames'])
            act = np.array([f['action'] for f in r['frames']], np.int32)
            rgb, depth_m = r4.render_along(cas, P)
            for i in range(len(P)):
                save_jpg(rgb[i], rgb_dir / f'episode_{ei:06d}_{i}.jpg')
                valid = np.isfinite(depth_m[i]) & (depth_m[i] > 0) & (depth_m[i] < prof.render_far_m * 0.99)
                cv2.imwrite(str(dep_dir / f'episode_{ei:06d}_{i}.png'), ex.depth_m_to_vlnce_png(depth_m[i], valid))
            floor0 = float(r['frames'][0]['pos'][2]) - 0.55                 # 스폰 직후 trunk − base_height
            store[f'pose_{ei}'] = vd.to_start_relative(P, floor_z=floor0).astype(np.float32)
            store[f'action_{ei}'] = act
            total += len(P)
            meta.append({'episode': ei, 'key': r['key'], 'frames': len(P)})
            print(f'[06m vlnce] ep {ei} {r["key"]} {len(P)} f ({time.time() - t0:.0f}s)', flush=True)
        np.savez_compressed(sent / f'pose__{args.rig}.npz', **store)
        asm = argparse.Namespace(out_root=str(out_root), out_scene=args.out_scene, schema_ref=args.schema_ref, goal_report_rig=args.rig,
                                 paths_json=args.walk_json, scene='noeun_station', light=args.light, rtx_ambient=args.rtx_ambient, film_iso=args.film_iso)
        print(f'[06m vlnce] 조립 {ex.assemble(asm, [args.rig], sent)}', flush=True)
    else:
        vw = _load('vw_06m', GS / 'vlnpe_writer.py')
        e05 = _load('ex05p_06m', GS / '05_export_vlnpe.py')
        schema, stats_fn = vw.load_schema(args.pe_schema_ref), vw.load_stats_fn()
        root = Path(out_root) / 'traj_data' / 'r2r'
        assert not (root / args.out_scene / 'meta' / 'episodes.jsonl').exists(), f'{root / args.out_scene} 에 이미 meta 가 있다'
        ph = [{'text': 'placeholder instruction (to be labeled).', 'tokens': [0] * 200}] * 3
        for ei, r in enumerate(ok):
            P = frames_to_poses(r['frames'])
            a_pe = ce_to_pe_actions(np.array([f['action'] for f in r['frames']]))
            rgb, depth_m = r4.render_along(cas, P)
            xy = np.array([f['pos'][:2] for f in r['frames']])
            L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            buf = []
            for i, f in enumerate(r['frames']):
                Rt = quat_wxyz_to_R(f['quat_wxyz'])
                fw = P[i, :3, 2]
                buf.append({'camera_position': P[i, :3, 3], 'camera_orientation': c2w_to_vlnpe_quat(P[i, :3, :3]),
                            'camera_yaw': float(np.arctan2(fw[1], fw[0])), 'robot_position': np.asarray(f['pos']),
                            'robot_orientation': list(f['quat_wxyz']), 'robot_yaw': float(np.arctan2(Rt[1, 0], Rt[0, 0])),
                            'progress': float(L[i] / max(L[-1], 1e-9)), 'action': int(a_pe[i]),
                            'rgb': np.asarray(rgb[i], np.uint8), 'depth': e05.depth_m_to_vlnpe(depth_m[i])})
            T = vw.save_episode_vlnpe(root, args.out_scene, ei, buf, ph, schema=schema, stats_fn=stats_fn)
            total += T
            meta.append({'episode': ei, 'key': r['key'], 'frames': T})
            vw.finalize_info_json_vlnpe(root, {args.out_scene: {'episode_count': ei + 1, 'frame_count': total}}, robot_type='rbq10_d455')
            print(f'[06m vlnpe] ep {ei} {r["key"]} {T} f ({time.time() - t0:.0f}s)', flush=True)
    (log / 'render_meta.json').write_text(json.dumps({'args': vars(args), 'episodes': meta, 'frames_total': total}))
    print(f'[06m {fmt}] 렌더 {len(ok)} ep · {total} 프레임 · {time.time() - t0:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    import traceback
    code = 0
    try:
        code = main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 3
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
