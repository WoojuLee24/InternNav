"""06m2 — 06m 으로 만든 데이터셋에서 **바뀐 ep 만** 다시 렌더해 새 데이터셋으로 낸다 (나머지 ep 는 이전 렌더를 그대로 쓴다).

왜: v5 → v6 에서 걸음이 바뀐 것은 23 ep (출발점을 옮긴 22 + 계단을 편 ep 86) 뿐이다. 200 ep 전체 렌더는 ~6 h, 23 ep 는 ~1 h.

방법 (이전 데이터셋은 건드리지 않는다)
1. 이전 출력을 새 out_root 로 복사: 이미지·npy(큰 파일)는 **하드링크**, parquet·meta 는 복사.
2. 바뀐 ep 의 파일을 새 쪽에서 먼저 **지운 뒤**(하드링크 끊기) 06m 과 같은 코드로 렌더해 쓴다.
3. vln_ce: 이전 pose 센티널(npz)에서 바뀐 ep 만 갈아 끼우고 05 assemble 로 parquet·meta 를 전부 다시 조립 (pixel goal 도 다시 계산).
   vln_pe: ep 마다 parquet·npy 를 쓰고, meta 3 파일에서 바뀐 ep 줄만 새 줄로 바꾼 뒤 info.json 다시.
ep 번호는 walk json 의 성공 ep 순서 (06m 과 같음). 이전 렌더의 render_meta.json 과 순서·키가 같은지 먼저 확인한다.

사용 (isaac 컨테이너, 저장소 루트):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/tools/rerender_rbq_subset.py --format vlnce --walk_json <새 walk> --prev_root <이전 out_root> --prev_log <이전 06m log_dir> --out_root <새 out_root>
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
GS = HERE.parent / 'gs_vlnpe'
for p in (str(GS), str(GS / 'apply_real')):
    if p not in sys.path:
        sys.path.insert(0, p)
from render_poses import c2w_to_vlnpe_quat, ce_to_pe_actions, frames_to_poses, quat_wxyz_to_R  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walk_json', required=True)
    ap.add_argument('--format', required=True, choices=['vlnce', 'vlnpe'])
    ap.add_argument('--prev_root', required=True, help='이전 06m out_root (예: data/.../vln_ce_noeun_nav3d_rbq_v5)')
    ap.add_argument('--prev_log', required=True, help='이전 06m --log_dir (render_meta.json · vln_ce 센티널)')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--keys', default='', help='쉼표로. 비우면 walk json 의 merge.changed')
    ap.add_argument('--out_scene', default='noeunstation')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--schema_ref', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/17DRP5sb8fy/data/chunk-000/episode_000000.parquet')
    ap.add_argument('--pe_schema_ref', default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r/17DRP5sb8fy/data/chunk-000/episode_000000.parquet')
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=10.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--log_dir', default='scripts/dataset_converters/nav3d/logs/noeun/51_render_rbq/v6')
    return ap.parse_args()


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def clone_tree(src: Path, dst: Path, link_suffixes=('.jpg', '.png', '.npy')):
    """src → dst. 큰 이미지 파일은 하드링크, 나머지는 복사."""
    assert not dst.exists(), f'{dst} 가 이미 있다'
    for root, _, files in os.walk(src):
        r = Path(root)
        o = dst / r.relative_to(src)
        o.mkdir(parents=True, exist_ok=True)
        for f in files:
            (os.link if f.endswith(link_suffixes) else shutil.copy2)(r / f, o / f)


def main() -> int:
    args = parse()
    W = json.loads(Path(args.walk_json).read_text())
    ok = [r for r in W['results'] if r['success']]
    keys = [k for k in args.keys.split(',') if k] or W.get('merge', {}).get('changed', [])
    assert keys, '다시 렌더할 ep 가 없다'
    fmt = args.format
    prev_meta = json.loads((Path(args.prev_log) / fmt / 'render_meta.json').read_text())['episodes']
    assert [m['key'] for m in prev_meta] == [r['key'] for r in ok], '이전 렌더와 ep 순서·키가 다르다'
    idx = {r['key']: i for i, r in enumerate(ok)}
    sel = sorted(idx[k] for k in keys)
    log = Path(args.log_dir) / fmt
    sent = log / 'sentinels'
    sent.mkdir(parents=True, exist_ok=True)
    src_scene = Path(args.prev_root) / 'traj_data' / 'r2r' / args.out_scene
    dst_scene = Path(args.out_root) / 'traj_data' / 'r2r' / args.out_scene
    clone_tree(src_scene, dst_scene)
    print(f'[06m2 {fmt}] {args.prev_root} → {args.out_root} 복제 · 다시 렌더 ep {sel}', flush=True)
    ex = _load('ex05', GS / 'apply_real' / '05_export_vlnce.py')
    r4 = ex.load_renderer_module()
    usd = Path(W['background'])
    t0, meta = time.time(), list(prev_meta)
    if fmt == 'vlnce':
        import cv2
        import vlnce_discretize as vd
        from geometry_utils import save_jpg
        prof = ex.load_camera_profiles().get(args.rig)
        cas = r4.build_renderer(prof.width, prof.height, prof.k, usd, args.light, None, args.rtx_ambient, args.film_iso,
                                prof.render_near_m, prof.render_far_m)
        O = ex.scene_paths(Path(args.out_root) / 'traj_data' / 'r2r', args.out_scene)
        rgb_dir, dep_dir = ex.stream_dir(O['videos_dir'], 'rgb', args.rig), ex.stream_dir(O['videos_dir'], 'depth', args.rig)
        store = dict(np.load(Path(args.prev_log) / fmt / 'sentinels' / f'pose__{args.rig}.npz'))
        for ei in sel:
            r = ok[ei]
            for p in list(rgb_dir.glob(f'episode_{ei:06d}_*.jpg')) + list(dep_dir.glob(f'episode_{ei:06d}_*.png')):
                p.unlink()                                                     # 하드링크 끊기 (이전 데이터셋은 그대로)
            P = frames_to_poses(r['frames'])
            rgb, depth_m = r4.render_along(cas, P)
            for i in range(len(P)):
                save_jpg(rgb[i], rgb_dir / f'episode_{ei:06d}_{i}.jpg')
                valid = np.isfinite(depth_m[i]) & (depth_m[i] > 0) & (depth_m[i] < prof.render_far_m * 0.99)
                cv2.imwrite(str(dep_dir / f'episode_{ei:06d}_{i}.png'), ex.depth_m_to_vlnce_png(depth_m[i], valid))
            store[f'pose_{ei}'] = vd.to_start_relative(P, floor_z=float(r['frames'][0]['pos'][2]) - 0.55).astype(np.float32)
            store[f'action_{ei}'] = np.array([f['action'] for f in r['frames']], np.int32)
            meta[ei] = {'episode': ei, 'key': r['key'], 'frames': len(P)}
            print(f'[06m2 vlnce] ep {ei} {r["key"]} {len(P)} f ({time.time() - t0:.0f}s)', flush=True)
        np.savez_compressed(sent / f'pose__{args.rig}.npz', **store)
        asm = argparse.Namespace(out_root=str(args.out_root), out_scene=args.out_scene, schema_ref=args.schema_ref, goal_report_rig=args.rig,
                                 paths_json=args.walk_json, scene='noeun_station', light=args.light, rtx_ambient=args.rtx_ambient, film_iso=args.film_iso)
        print(f'[06m2 vlnce] 조립 {ex.assemble(asm, [args.rig], sent)}', flush=True)
    else:
        du = _load('du_06m2', GS / 'dataset_utils.py')
        vw = _load('vw_06m2', GS / 'vlnpe_writer.py')
        e05 = _load('ex05p_06m2', GS / '05_export_vlnpe.py')
        cas = r4.build_renderer(*du.VLNPE_IMG_WH, du.VLNPE_K, usd, args.light, None, args.rtx_ambient, args.film_iso)
        schema, stats_fn = vw.load_schema(args.pe_schema_ref), vw.load_stats_fn()
        root = Path(args.out_root) / 'traj_data' / 'r2r'
        meta_dir = dst_scene / 'meta'
        old = {n: [json.loads(l) for l in (meta_dir / n).read_text().splitlines() if l.strip()] for n in ('episodes.jsonl', 'tasks.jsonl', 'episodes_stats.jsonl')}
        for n in old:
            (meta_dir / n).unlink()                                            # writer 가 append 하므로 비우고 새 줄만 받는다
        ph = [{'text': 'placeholder instruction (to be labeled).', 'tokens': [0] * 200}] * 3
        for ei in sel:
            r = ok[ei]
            for p in (dst_scene / 'videos' / 'chunk-000').glob(f'*/episode_{ei:06d}.npy'):
                p.unlink()
            (dst_scene / 'data' / 'chunk-000' / f'episode_{ei:06d}.parquet').unlink(missing_ok=True)
            P = frames_to_poses(r['frames'])
            a_pe = ce_to_pe_actions(np.array([f['action'] for f in r['frames']]))
            rgb, depth_m = r4.render_along(cas, P)
            xy = np.array([f['pos'][:2] for f in r['frames']])
            Lc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            buf = []
            for i, f in enumerate(r['frames']):
                Rt = quat_wxyz_to_R(f['quat_wxyz'])
                fw = P[i, :3, 2]
                buf.append({'camera_position': P[i, :3, 3], 'camera_orientation': c2w_to_vlnpe_quat(P[i, :3, :3]),
                            'camera_yaw': float(np.arctan2(fw[1], fw[0])), 'robot_position': np.asarray(f['pos']),
                            'robot_orientation': list(f['quat_wxyz']), 'robot_yaw': float(np.arctan2(Rt[1, 0], Rt[0, 0])),
                            'progress': float(Lc[i] / max(Lc[-1], 1e-9)), 'action': int(a_pe[i]),
                            'rgb': np.asarray(rgb[i], np.uint8), 'depth': e05.depth_m_to_vlnpe(depth_m[i])})
            vw.save_episode_vlnpe(root, args.out_scene, ei, buf, ph, schema=schema, stats_fn=stats_fn)
            meta[ei] = {'episode': ei, 'key': r['key'], 'frames': len(P)}
            print(f'[06m2 vlnpe] ep {ei} {r["key"]} {len(P)} f ({time.time() - t0:.0f}s)', flush=True)
        new = {n: {d['episode_index']: d for d in (json.loads(l) for l in (meta_dir / n).read_text().splitlines() if l.strip())} if 'task' not in n else
               [json.loads(l) for l in (meta_dir / n).read_text().splitlines() if l.strip()] for n in old}
        # episodes / episodes_stats: episode_index 로 갈아 끼움. tasks: 이전 그대로 (지시문 placeholder 동일)
        for n in ('episodes.jsonl', 'episodes_stats.jsonl'):
            rows = [new[n].get(d['episode_index'], d) for d in old[n]]
            (meta_dir / n).write_text(''.join(json.dumps(d) + '\n' for d in rows))
        (meta_dir / 'tasks.jsonl').write_text(''.join(json.dumps(d) + '\n' for d in old['tasks.jsonl']))
        total = sum(m['frames'] for m in meta)
        vw.finalize_info_json_vlnpe(root, {args.out_scene: {'episode_count': len(ok), 'frame_count': total}}, robot_type='rbq10_d455')
    (log / 'render_meta.json').write_text(json.dumps({'args': vars(args), 'rerendered': sel, 'episodes': meta,
                                                      'frames_total': sum(m['frames'] for m in meta)}))
    print(f'[06m2 {fmt}] 다시 렌더 {len(sel)} ep · 전체 {sum(m["frames"] for m in meta)} 프레임 · {time.time() - t0:.0f} s', flush=True)
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
