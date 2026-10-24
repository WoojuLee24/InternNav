"""06v 렌더 (Isaac) — 52_camera_values.py --stage prepare 가 만든 자세(jobs.npz)를 **06 과 같은 렌더러**로 그린다.

같은 세션·같은 설정으로 (a) GT 카메라 자세 (기준선) (b) 06 방식 자세 (navmesh z · mesh z, gt·wp 경로) 를 모두 그린다
— 렌더러 rgb 가 세션마다 미세하게 달라지므로(gs_vlnpe 실측) 기준선도 이 자리에서 다시 그린다.
렌더러 = gs_vlnpe/apply_real/04_render_obs_isaac (06 이 쓰는 것), 설정 = vln_pe_measured 와 같은 값(ambient_only · 10 · iso 70).
씬당 프로세스 하나 (한 프로세스에서 build_renderer 두 번이면 죽는다).

사용 (isaac 컨테이너, 저장소 루트):
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/vlnpe/camera_values_worker.py
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
GS = HERE.parent / 'gs_vlnpe'
ISAAC_PY = '/workspace/isaaclab/_isaac_sim/python.sh'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log_dir', default='scripts/dataset_converters/nav3d/logs/vlnpe/52_camera_values')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_pe', help='MP3D Isaac USD')
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=10.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--timeout_s', type=int, default=3600, help='씬 하나 상한')
    ap.add_argument('--_worker_scene', default='')
    return ap.parse_args()


def worker(args) -> int:
    s = args._worker_scene
    log = Path(args.log_dir)
    pr = json.load(open(log / 'prepare.json'))
    J = np.load(log / 'jobs.npz')
    keys = [k for k in J.files if k.startswith(f'{s}|')]
    sp = importlib.util.spec_from_file_location('r4_06v', GS / 'apply_real' / '04_render_obs_isaac.py')
    r4 = importlib.util.module_from_spec(sp)
    sys.modules['r4_06v'] = r4
    sp.loader.exec_module(r4)
    W, H = pr['rig']['wh']
    usd = r4.load_scene_model(args.mesh_root, s)
    cas = r4.build_renderer(W, H, np.asarray(pr['rig']['K']), usd, args.light, None, args.rtx_ambient, args.film_iso)
    out = log / 'render'
    out.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    for k in keys:
        _, kind, ep = k.split('|')
        rgb, depth = r4.render_along(cas, J[k])
        np.savez_compressed(out / f'{s}__{kind}__{ep}.npz', rgb=np.asarray(rgb, np.uint8), depth=np.asarray(depth, np.float16))
        n += len(J[k])
        print(f'[06v render {s}] {kind} ep{ep} {len(J[k])} f ({n} f, {time.time() - t0:.0f}s)', flush=True)
    (out / f'_done_{s}.json').write_text(json.dumps({'scene': s, 'keys': len(keys), 'frames': n, 'sec': time.time() - t0}))
    return 0


def main() -> int:
    args = parse()
    if args._worker_scene:
        return worker(args)
    log = Path(args.log_dir)
    pa = json.load(open(log / 'prepare.json'))['args']
    scenes = pa['scenes'].split(',') + [x for x in pa.get('stairs_scenes', '').split(',') if x]
    for s in scenes:
        done = log / 'render' / f'_done_{s}.json'
        if done.exists():
            print(f'[06v render] {s} 이미 있음', flush=True)
            continue
        cmd = [ISAAC_PY, str(Path(__file__).resolve()), '--_worker_scene', s, '--log_dir', args.log_dir, '--mesh_root', args.mesh_root,
               '--light', args.light, '--rtx_ambient', str(args.rtx_ambient), '--film_iso', str(args.film_iso)]
        p = subprocess.Popen(cmd)
        t0 = time.time()
        while p.poll() is None and not done.exists() and time.time() - t0 < args.timeout_s:
            time.sleep(5)
        if p.poll() is None:            # 산출물이 다 나오면 PID 로 끊는다 (close() 가 오래 멈춘다 — gs_vlnpe 실측)
            p.kill()
            p.wait()
        print(f'[06v render] {s} {"완료" if done.exists() else "실패"} ({time.time() - t0:.0f}s)', flush=True)
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
