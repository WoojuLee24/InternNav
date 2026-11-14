"""noeun 51b — 51_render_rbq 의 vln_ce 렌더를 여러 GPU · 여러 서버로 나눠 찍고, 마지막에 한 번 조립한다.

51 은 한 프로세스가 전부 찍고 끝에서 pose npz 저장 + 조립(parquet · meta)까지 한다 → 나눠 찍을 수 없다.
51b 는 51 을 고치지 않고 (기존 파일 수정 금지) 같은 렌더러·포맷 코드를 쓰되:
  · `--ep_start/--ep_end` : 성공 ep 목록(51 과 같은 순서)의 일부만 찍는다. 파일 번호 episode_{ei:06d} 는 **전체 번호** 그대로.
  · 조각마다 pose 를 `<log_dir>/vlnce/sentinels/pose__<rig>.part_<start>_<end>.npz` 로 저장 (조립 안 함).
  · `--skip_existing` : 그 ep 의 마지막 프레임 jpg·png 가 이미 있으면 렌더를 건너뛰고 pose 만 계산 (이어 찍기).
  · `--assemble` : 모든 part npz 를 합쳐 51 과 똑같이 ex.assemble 한 번 (모든 ep 가 다 있는지 먼저 확인).
결과는 51 을 한 번에 돌린 것과 같은 폴더 구조 · 같은 ep 번호.

사용 (isaac 컨테이너, 저장소 루트):
    python.sh 51b_render_rbq_shard.py --walk_json sel_train.json --out_root <OUT> --log_dir <LOG> --ep_start 0 --ep_end 200
    python.sh 51b_render_rbq_shard.py --walk_json sel_train.json --out_root <OUT> --log_dir <LOG> --assemble   (habitat 컨테이너도 됨)
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'utils'))
GS = HERE.parent / 'gs_vlnpe'
for p in (str(GS), str(GS / 'apply_real')):
    if p not in sys.path:
        sys.path.insert(0, p)
from render_poses import frames_to_poses  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walk_json', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--log_dir', required=True, help='조각들이 같은 곳을 써야 --assemble 이 모은다')
    ap.add_argument('--ep_start', type=int, default=0)
    ap.add_argument('--ep_end', type=int, default=0, help='0 = 끝까지')
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--assemble', action='store_true', help='렌더 안 하고 part npz 합쳐 조립만')
    ap.add_argument('--max_episodes', type=int, default=0, help='0 = 전부 (51 과 같음 · 시험용)')
    ap.add_argument('--usd_path', default=None)
    ap.add_argument('--out_scene', default='noeunstation')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--schema_ref', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/17DRP5sb8fy/data/chunk-000/episode_000000.parquet')
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=10.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    return ap.parse_args()


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def start_relative(r, P):
    import vlnce_discretize as vd
    floor0 = float(r['frames'][0]['pos'][2]) - 0.55                 # 51 과 같음: 스폰 직후 trunk − base_height
    return vd.to_start_relative(P, floor_z=floor0).astype(np.float32)


def assemble(args, ok, ex, sent):
    store = {}
    for f in sorted(sent.glob(f'pose__{args.rig}.part_*.npz')):
        z = np.load(f)
        for k in z.files:
            assert k not in store, f'{k} 가 두 조각에 있다 ({f.name})'
            store[k] = z[k]
    have = {int(k.split('_')[1]) for k in store if k.startswith('pose_')}
    miss = sorted(set(range(len(ok))) - have)
    assert not miss, f'pose 없는 ep {len(miss)} 개: {miss[:20]}'
    np.savez_compressed(sent / f'pose__{args.rig}.npz', **store)
    asm = argparse.Namespace(out_root=args.out_root, out_scene=args.out_scene, schema_ref=args.schema_ref, goal_report_rig=args.rig,
                             paths_json=args.walk_json, scene='noeun_station', light=args.light, rtx_ambient=args.rtx_ambient, film_iso=args.film_iso)
    print(f'[51b] 조립 {ex.assemble(asm, [args.rig], sent)}', flush=True)
    meta = [{'episode': i, 'key': r['key'], 'frames': len(r['frames'])} for i, r in enumerate(ok)]
    (sent.parent / 'render_meta.json').write_text(json.dumps({'args': vars(args), 'episodes': meta,
                                                              'frames_total': sum(m['frames'] for m in meta)}))
    print(f'[51b] 조립 끝 {len(ok)} ep · {sum(m["frames"] for m in meta)} 프레임', flush=True)


def main() -> int:
    args = parse()
    W = json.loads(Path(args.walk_json).read_text())
    ok = [r for r in W['results'] if r['success']]                  # 51 과 같은 순서 = 같은 ep 번호
    if args.max_episodes:
        ok = ok[:args.max_episodes]
    ex = _load('ex05', GS / 'apply_real' / '05_export_vlnce.py')
    sent = Path(args.log_dir) / 'vlnce' / 'sentinels'
    sent.mkdir(parents=True, exist_ok=True)
    if args.assemble:
        assemble(args, ok, ex, sent)
        return 0
    import cv2
    from geometry_utils import save_jpg
    e1 = args.ep_end or len(ok)
    O = ex.scene_paths(Path(args.out_root) / 'traj_data' / 'r2r', args.out_scene)
    rgb_dir, dep_dir = ex.stream_dir(O['videos_dir'], 'rgb', args.rig), ex.stream_dir(O['videos_dir'], 'depth', args.rig)
    for d in (rgb_dir, dep_dir):
        d.mkdir(parents=True, exist_ok=True)
    cp = ex.load_camera_profiles()
    prof = cp.get(args.rig)
    cas = None
    store, t0, nr, total = {}, time.time(), 0, 0
    print(f'[51b] ep {args.ep_start}..{e1} / 성공 {len(ok)}', flush=True)
    for ei in range(args.ep_start, e1):
        r = ok[ei]
        P = frames_to_poses(r['frames'])
        n = len(P)
        done = args.skip_existing and (rgb_dir / f'episode_{ei:06d}_{n - 1}.jpg').is_file() and (dep_dir / f'episode_{ei:06d}_{n - 1}.png').is_file()
        if not done:
            if cas is None:
                r4 = ex.load_renderer_module()
                cas = r4.build_renderer(prof.width, prof.height, prof.k, Path(args.usd_path or W['background']), args.light, None,
                                        args.rtx_ambient, args.film_iso, prof.render_near_m, prof.render_far_m)
            rgb, depth_m = r4.render_along(cas, P)
            for i in range(n):
                save_jpg(rgb[i], rgb_dir / f'episode_{ei:06d}_{i}.jpg')
                valid = np.isfinite(depth_m[i]) & (depth_m[i] > 0) & (depth_m[i] < prof.render_far_m * 0.99)
                cv2.imwrite(str(dep_dir / f'episode_{ei:06d}_{i}.png'), ex.depth_m_to_vlnce_png(depth_m[i], valid))
            nr += 1
        store[f'pose_{ei}'] = start_relative(r, P)
        store[f'action_{ei}'] = np.array([f['action'] for f in r['frames']], np.int32)
        total += n
        print(f'[51b] ep {ei} {r["key"]} {n} f {"skip" if done else ""} ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(sent / f'pose__{args.rig}.part_{args.ep_start:05d}_{e1:05d}.npz', **store)
    print(f'[51b] 렌더 {e1 - args.ep_start} ep (새로 {nr}) · {total} 프레임 · {time.time() - t0:.0f} s', flush=True)
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
