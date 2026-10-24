"""06p — 노은역 경로를 **vln_pe 포맷**으로 렌더·저장한다 (Isaac). 자세 계산은 06 과 같은 함수(`episode_poses`).

    04(b) paths json -> 06.episode_poses (이산화 0.25 m / 15° · 몸 바닥 z · D455 자세)
      -> Isaac render_along (vln_pe 카메라: 256×256 · hfov 90° · fx 128 — vln_pe 로더 규약)
      -> gs_vlnpe/vlnpe_writer.save_episode_vlnpe (릴리스와 같은 parquet 스키마 · npy · meta)

vln_ce(06) 와 다른 점
| | vln_ce (06) | vln_pe (06p) |
|---|---|---|
| 이미지 | 640×480 · `60cm_15deg` 칸 (hfov 68.7°) · jpg/png | **256×256 · hfov 90°** · npy (rgb uint8 · depth float32 0<v≤1, m = v×10) |
| 자세 | `pose.<rig>` 시작 기준 상대 | **절대 좌표** camera_position · camera_orientation(wxyz) |
| action | `action[0] = −1`, `action[t]` = t 로 온 action | **`action[i]` = i 에서 한 action**, 마지막 0 (STOP) |
| 지시문 | placeholder | placeholder 3 개 (labeling 뒤 채움) |

카메라 자세(높이 0.584 · pitch 14.76° · 전방 0.365 m)는 06 과 같다(D455). robot_position = 몸 xy · 바닥 z + `--robot_base_h`.
progress = 경로 길이 비율(0→1, 연속). 릴리스는 0.2 단위 계단 값이라 다르다 — 로더가 쓰는지 확인 필요.

사용 (isaac 컨테이너, 저장소 루트):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/vlnpe/54_render_camera.py
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
GS = HERE.parent / 'gs_vlnpe'
for p in (str(GS), str(GS / 'apply_real')):
    if p not in sys.path:
        sys.path.insert(0, p)
from render_poses import c2w_to_vlnpe_quat, ce_to_pe_actions, episode_poses, start_yaw_for  # noqa: E402,F401


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paths_json', default='scripts/dataset_converters/nav3d/logs/noeun/paths/noeun_station_paths_meshfloor.json')
    ap.add_argument('--usd_path', default='data/GS_USDZ/Subway/noeun_station_collision.usdz')
    ap.add_argument('--out_root', default='data/InternData-N1-v0.5-mini/vln_pe_noeun_nav3d/traj_data/r2r')
    ap.add_argument('--out_scene', default='noeunstation')
    ap.add_argument('--mount_height_m', type=float, default=0.584)
    ap.add_argument('--pitch_down_deg', type=float, default=14.76)
    ap.add_argument('--forward_m', type=float, default=0.365)
    ap.add_argument('--robot_base_h', type=float, default=0.55, help='robot_position z = 바닥 + 이 값 (RBQ base_link 높이)')
    ap.add_argument('--waypoint_spacing_m', type=float, default=0.5)
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--schema_ref', default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/17DRP5sb8fy/data/chunk-000/episode_000000.parquet')
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=10.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--log_dir', default='scripts/dataset_converters/nav3d/logs/vlnpe/54_render_camera')
    ap.add_argument('--mesh_root', default=None, help='MP3D 씬일 때: data/scene_data/mp3d_pe (텍스처 심링크까지 준비하는 load_scene_model 로 연다; --usd_path 무시)')
    ap.add_argument('--scene', default=None, help='--mesh_root 와 함께: MP3D 씬 이름')
    ap.add_argument('--start_heading', default='path', choices=['path', 'r2r_train'], help='06 과 같음')
    ap.add_argument('--start_seed', type=int, default=0)
    ap.add_argument('--r2r_raw', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/train/train.json.gz')
    ap.add_argument('--render_far_m', type=float, default=None, help='None = 렌더러 기본 10 m. MP3D mesh 는 그 너머가 검게 나온다 (GS 는 영향 없음)')
    return ap.parse_args()


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def yaw_quat(yaw: float) -> list:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def main() -> int:
    args = parse()
    du = _load('du_06p', GS / 'dataset_utils.py')
    vw = _load('vw_06p', GS / 'vlnpe_writer.py')
    ex = _load('ex05p', GS / '05_export_vlnpe.py')
    prof = SimpleNamespace(mount_height_m=args.mount_height_m, pitch_down_deg=args.pitch_down_deg)
    eps = json.loads(Path(args.paths_json).read_text())['episodes']
    if args.max_episodes > 0:
        eps = eps[:args.max_episodes]
    W, H = du.VLNPE_IMG_WH
    schema, stats_fn = vw.load_schema(args.schema_ref), vw.load_stats_fn()
    out_scene_dir = Path(args.out_root) / args.out_scene
    assert not (out_scene_dir / 'meta' / 'episodes.jsonl').exists(), f'{out_scene_dir} 에 이미 meta 가 있다 (append 라 섞인다) — 지우고 다시'
    r4 = _load('r4_06p', GS / 'apply_real' / '04_render_obs_isaac.py')
    usd = r4.load_scene_model(args.mesh_root, args.scene) if args.mesh_root else Path(args.usd_path)
    far = {} if args.render_far_m is None else {'render_far_m': args.render_far_m}
    cas = r4.build_renderer(W, H, du.VLNPE_K, usd, args.light, None, args.rtx_ambient, args.film_iso, **far)
    log = Path(args.log_dir)
    log.mkdir(parents=True, exist_ok=True)
    print(f'[06p] {len(eps)} ep · {W}x{H} fx {du.VLNPE_K[0, 0]} · 높이 {prof.mount_height_m} · pitch {prof.pitch_down_deg} · 전방 {args.forward_m} · {args.paths_json}', flush=True)
    t0, total, meta = time.time(), 0, []
    placeholder = [{'text': 'placeholder instruction (to be labeled).', 'tokens': [0] * 200}] * 3
    for ei, ep in enumerate(eps):
        traj = np.asarray(ep['trajectory'], dtype=np.float64)
        y0 = start_yaw_for(ep, traj, args.start_heading, args.start_seed, args.r2r_raw)
        disc, zb, P = episode_poses(traj, prof, args.forward_m, args.waypoint_spacing_m, initial_yaw=y0)
        rgb, depth_m = r4.render_along(cas, P)
        a_pe = ce_to_pe_actions(disc['action'])
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(disc['xy'], axis=0), axis=1))]
        prog = L / max(L[-1], 1e-9)
        buf = []
        for i in range(len(P)):
            f = P[i, :3, 2]
            buf.append({'camera_position': P[i, :3, 3], 'camera_orientation': c2w_to_vlnpe_quat(P[i, :3, :3]),
                        'camera_yaw': float(np.arctan2(f[1], f[0])),
                        'robot_position': np.r_[disc['xy'][i], zb[i] + args.robot_base_h], 'robot_orientation': yaw_quat(disc['yaw'][i]),
                        'robot_yaw': float(disc['yaw'][i]), 'progress': float(prog[i]), 'action': int(a_pe[i]),
                        'rgb': np.asarray(rgb[i], np.uint8), 'depth': ex.depth_m_to_vlnpe(depth_m[i])})
        T = vw.save_episode_vlnpe(args.out_root, args.out_scene, ei, buf, placeholder, schema=schema, stats_fn=stats_fn)
        total += T
        meta.append({'episode': ei, 'src_episode_id': ep['episode_id'], 'kind': ep['kind'], 'frames': T, 'start_yaw': y0})
        print(f'[06p] ep {ei:>3} {ep["kind"]:10s} {T:>4} f ({total} f, {time.time() - t0:.0f}s)', flush=True)
    vw.finalize_info_json_vlnpe(args.out_root, {args.out_scene: {'episode_count': len(eps), 'frame_count': total}}, robot_type='rbq10_d455')
    (out_scene_dir / 'meta' / 'render_provenance.json').write_text(json.dumps({
        'produced_by': 'scripts/dataset_converters/nav3d/vlnpe/54_render_camera.py', 'args': vars(args), 'images_from': 'isaac_rtx_render (GS rgb + collision mesh depth)',
        'intrinsic_K': du.VLNPE_K.tolist(), 'episodes': meta, 'frames_total': total}, indent=1))
    print(f'[06p] 렌더 {len(eps)} ep · {total} 프레임 · {time.time() - t0:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    import traceback
    try:
        code = main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 3
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
