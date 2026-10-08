"""vln_ce GT pose를 그대로 재렌더해 **vln_ce와 동일한 구조**로 저장한다.

`05_export_vlnpe.py`의 vln_ce 판이다. 하는 일이 같다 — GT의 카메라 pose를 그대로 재생하고
**이미지만** 우리 렌더로 바꾼다. 궤적·action·픽셀목표·지시문은 GT에서 **바이트 복사**한다.

```
05_export_vlnpe.py   vln_pe GT pose 재생  -> vln_pe 구조   (완료: 61씬 SSIM 0.8295)
05_export_vlnce.py   vln_ce GT pose 재생  -> vln_ce 구조   (이 파일)
```

두 파일이 같은 05인 이유: 둘 다 **export 단계**이고 출력 포맷만 다르다.

vln_pe 판과 다른 점 하나 — 좌표계
-----------------------------------
vln_pe GT는 `observation.camera_position`이 **절대 world 좌표**라 그대로 렌더하면 됐다.
vln_ce GT의 `pose.<rig>`는 **에피소드 시작 기준 상대**다(학습이 상대좌표만 쓰기 때문).
렌더하려면 mesh 절대좌표가 필요하다.

`3dloader_vlnce/vlnce_align.py`가 이 변환을 이미 푼다 — 에피소드 depth를 start-frame 점구름으로
쌓아 mesh에 **강체 정합**(yaw + 평행이동)한다. 실측 잔차 point-to-mesh median **0.0039 m**.
새로 짜지 않고 `align_episode`를 그대로 쓴다.

rig 5개
-------
릴리스와 같은 세팅으로 5개를 전부 만든다 — `125cm_{0,30,45}deg` · `60cm_{15,30}deg`.
**한 프로세스에서 `build_renderer`를 두 번 부르면 죽으므로**(`/World/RenderCamera` 충돌, 실측)
rig 하나당 프로세스 하나로 돈다. 씬 루프도 같은 이유로 프로세스를 나눈다.

커맨드
------
1씬 스모크 (3 에피소드 · rig 5개 ≈ 700 프레임):
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05_export_vlnce.py --scenes 17DRP5sb8fy --max_episodes 3 --out_root data/InternData-N1-v0.5-mini/vln_ce_render --log_dir logs/gs-vlnpe

표본 3씬 × 5 에피소드 (GT 대조용):
timeout --signal=KILL 14400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05_export_vlnce.py --scenes 17DRP5sb8fy,1LXtFkjw3qL,29hnd4uzFmX --max_episodes 5 --out_root data/InternData-N1-v0.5-mini/vln_ce_render --log_dir logs/gs-vlnpe

⚠️ 전수(61씬)는 98만 렌더 · 약 140 GB다. 여유(195 GB)를 거의 다 쓰므로 기본값은 표본이다.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_3D = REPO / 'scripts' / 'dataset_converters' / '3dloader_vlnce'
for p in (str(HERE), str(HERE / 'apply_real'), str(_3D)):
    if p not in sys.path:
        sys.path.insert(0, p)

SCRIPT_NAME = '05_export_vlnce'
ISAAC_PY = '/workspace/isaaclab/_isaac_sim/python.sh'

# vln_ce 이미지 규약 (릴리스 실측)
IMG_W, IMG_H = 640, 480
DEPTH_SCALE_MM = 1000.0        # uint16 밀리미터
DEPTH_INVALID = 0              # 0 = 무효
DEPTH_MAX_STORE_M = 60.0       # uint16 상한(65.535 m)보다 넉넉히 아래로 자른다
META_FILES = ('info.json', 'episodes.jsonl', 'episodes_stats.jsonl', 'tasks.jsonl')


def load_renderer_module():
    """apply_real/04를 모듈로 로드 (import 시점에 SimulationApp 부팅 — pxr보다 먼저)."""
    import importlib.util

    path = HERE / 'apply_real' / '04_render_obs_isaac.py'
    spec = importlib.util.spec_from_file_location('r4_export_vlnce', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['r4_export_vlnce'] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 경로 규약
# ---------------------------------------------------------------------------

def scene_paths(root: Path, scene: str) -> dict:
    base = Path(root) / scene
    return {
        'base': base,
        'parquet_dir': base / 'data' / 'chunk-000',
        'videos_dir': base / 'videos' / 'chunk-000',
        'meta_dir': base / 'meta',
    }


def stream_dir(videos_dir: Path, kind: str, rig: str) -> Path:
    """`observation.images.rgb.125cm_30deg` 같은 스트림 폴더."""
    return videos_dir / f'observation.images.{kind}.{rig}'


def gt_episode_ids(gt_root: Path, scene: str) -> list:
    d = scene_paths(gt_root, scene)['parquet_dir']
    return sorted(int(p.stem.split('_')[1]) for p in d.glob('episode_*.parquet')) if d.is_dir() else []


# ---------------------------------------------------------------------------
# depth 저장 — vln_ce 규약
# ---------------------------------------------------------------------------

def depth_m_to_vlnce_png(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """렌더 depth(m) -> vln_ce 저장형(uint16 **밀리미터**, 0 = 무효).

    ⚠️ 우리 기존 파이프라인은 0.1 mm 단위(`DEPTH_SCALE_RAW_TO_M = 1/10000`)로 저장한다.
    vln_ce는 **mm**다 — 그대로 쓰면 깊이가 10배 틀어진다.
    """
    mm = np.zeros(depth_m.shape, dtype=np.uint16)
    scaled = np.clip(np.round(np.asarray(depth_m, dtype=np.float64) * DEPTH_SCALE_MM),
                     1, int(DEPTH_MAX_STORE_M * DEPTH_SCALE_MM))
    mm[valid] = scaled[valid].astype(np.uint16)
    return mm


# ---------------------------------------------------------------------------
# GT pose(상대) -> 절대
# ---------------------------------------------------------------------------

def read_gt_episode(gt_root: Path, scene: str, ep: int, rigs: list) -> dict:
    import pyarrow.parquet as pq

    P = scene_paths(gt_root, scene)
    cols = ['action'] + [f'pose.{r}' for r in rigs]
    t = pq.read_table(P['parquet_dir'] / f'episode_{ep:06d}.parquet', columns=cols).to_pydict()
    poses = {r: np.stack([np.asarray(p, dtype=np.float64).reshape(4, 4) for p in t[f'pose.{r}']])
             for r in rigs}
    return {'action': np.asarray(t['action'], dtype=np.int32), 'pose_rel': poses}


def episode_instruction(gt_root: Path, scene: str, ep: int) -> str:
    """`meta/episodes.jsonl`에서 그 에피소드의 지시문. 정합 초기값을 찾는 열쇠다.

    `vlnce_align.build_rawdata_index`의 키가 **(씬, 지시문 텍스트)**라서, raw_data의
    `start_position`/`start_rotation`을 얻으려면 지시문으로 맞춰야 한다
    (`episode_index`는 raw_data의 `episode_id`와 다르다).
    """
    f = scene_paths(gt_root, scene)['meta_dir'] / 'episodes.jsonl'
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d['episode_index'] == ep:
            tasks = d.get('tasks') or []
            return tasks[0].strip() if tasks else ''
    return ''


def gt_depth_frames(gt_root: Path, scene: str, ep: int, rig: str, pose_rel: np.ndarray,
                    n_frames: int) -> list:
    """정합용 `[(i, pose_rel, depth_m, valid)]`. GT depth PNG(uint16 mm)를 읽는다."""
    from PIL import Image

    d = stream_dir(scene_paths(gt_root, scene)['videos_dir'], 'depth', rig)
    out = []
    idx = np.unique(np.linspace(0, len(pose_rel) - 1, min(n_frames, len(pose_rel))).astype(int))
    for i in idx:
        f = d / f'episode_{ep:06d}_{int(i)}.png'
        if not f.is_file():
            continue
        dm = np.asarray(Image.open(f), dtype=np.float32) / DEPTH_SCALE_MM
        valid = dm > 0
        if valid.any():
            out.append((int(i), pose_rel[i], dm, valid))
    return out


# ---------------------------------------------------------------------------
# worker — 씬 하나, rig 하나
# ---------------------------------------------------------------------------

def render_worker(args) -> int:
    scene, rig = args._worker_scene, args._worker_rig
    r4 = load_renderer_module()
    import camera_profiles
    import vlnce_align
    from geometry_utils import load_scene_mesh, save_jpg

    import cv2

    gt_root = Path(args.gt_root)
    out_root = Path(args.out_root) / 'traj_data' / 'r2r'
    G, O = scene_paths(gt_root, scene), scene_paths(out_root, scene)
    for key in ('parquet_dir', 'videos_dir', 'meta_dir'):
        O[key].mkdir(parents=True, exist_ok=True)

    prof = camera_profiles.get(rig)
    rig_all = list(camera_profiles.vlnce_rig_names())
    eps = gt_episode_ids(gt_root, scene)
    if args.max_episodes > 0:
        eps = eps[:args.max_episodes]

    usd_path = r4.load_scene_model(args.mesh_root, scene)
    mesh = load_scene_mesh(args.mesh_root, scene)
    raw_idx = vlnce_align.build_rawdata_index(Path(args.raw_root))

    print(f'[{SCRIPT_NAME}:{scene}:{rig}] usd={usd_path.name} 에피소드 {len(eps)}개 '
          f'{prof.width}x{prof.height} hfov->fx {prof.k[0, 0]:.2f}', flush=True)

    cas = r4.build_renderer(prof.width, prof.height, prof.k, usd_path,
                            args.light, None, args.rtx_ambient, args.film_iso,
                            prof.render_near_m, prof.render_far_m)

    rgb_dir = stream_dir(O['videos_dir'], 'rgb', rig)
    dep_dir = stream_dir(O['videos_dir'], 'depth', rig)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    dep_dir.mkdir(parents=True, exist_ok=True)

    t0, total, resids = time.time(), 0, []
    per_ep = []
    for ep in eps:
        E = read_gt_episode(gt_root, scene, ep, rig_all)
        pose_rel = E['pose_rel'][rig]
        n = len(pose_rel)

        # --- start-relative -> mesh 절대 ---
        instr = episode_instruction(gt_root, scene, ep)
        raw = raw_idx.get((scene, instr))
        if raw is None:
            print(f'[{SCRIPT_NAME}:{scene}:{rig}]   ep {ep} 건너뜀 — raw_data에서 지시문 매칭 실패',
                  flush=True)
            continue
        frames = gt_depth_frames(gt_root, scene, ep, rig, pose_rel, args.align_frames)
        if not frames:
            print(f'[{SCRIPT_NAME}:{scene}:{rig}]   ep {ep} 건너뜀 — GT depth 없음', flush=True)
            continue
        T_sf2mesh, resid, _ = vlnce_align.align_episode(frames, mesh, raw, seed=0, refine=args.refine)
        resids.append(float(resid))

        pose_abs = np.einsum('ij,njk->nik', T_sf2mesh, pose_rel)
        rgb, depth_m = r4.render_along(cas, pose_abs)

        for i in range(n):
            save_jpg(rgb[i], rgb_dir / f'episode_{ep:06d}_{i}.jpg')
            valid = (np.isfinite(depth_m[i]) & (depth_m[i] > 0)
                     & (depth_m[i] < prof.render_far_m * 0.99))
            cv2.imwrite(str(dep_dir / f'episode_{ep:06d}_{i}.png'),
                        depth_m_to_vlnce_png(depth_m[i], valid))
        total += n
        per_ep.append({'episode': ep, 'frames': n, 'align_residual_m': float(resid)})
        print(f'[{SCRIPT_NAME}:{scene}:{rig}]   ep {ep:>4} {n:>4} 프레임 · 정합 잔차 {resid:.4f} m',
              flush=True)

    # rig 하나가 끝날 때마다 sentinel — driver가 성공/skip 판정에 쓴다
    sent = Path(args._sentinel_dir) / f'{scene}__{rig}.json'
    sent.parent.mkdir(parents=True, exist_ok=True)
    sent.write_text(json.dumps({
        'scene': scene, 'rig': rig, 'episodes': len(per_ep), 'frames': total,
        'max_episodes': int(args.max_episodes),
        'align_residual_median_m': float(np.median(resids)) if resids else None,
        'align_residual_max_m': float(max(resids)) if resids else None,
        'per_episode': per_ep, 'elapsed_s': round(time.time() - t0, 1),
    }, ensure_ascii=False))
    print(f'[{SCRIPT_NAME}:{scene}:{rig}] 완료 — {len(per_ep)}ep {total}프레임 '
          f'{time.time() - t0:.0f}s', flush=True)
    return 0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def copy_gt_sidecars(gt_root: Path, out_root: Path, scene: str, eps: list, render_cfg: dict) -> None:
    """parquet·meta를 GT에서 **바이트 복사**한다.

    GT pose를 그대로 재생했으므로 궤적·action·픽셀목표·지시문이 전부 같다. 복사하면 구조
    동일성이 보장되고 프레임 단위 대조가 된다 (`05_export_vlnpe`와 같은 판단).
    """
    G, O = scene_paths(gt_root, scene), scene_paths(out_root, scene)
    O['parquet_dir'].mkdir(parents=True, exist_ok=True)
    O['meta_dir'].mkdir(parents=True, exist_ok=True)
    for ep in eps:
        src = G['parquet_dir'] / f'episode_{ep:06d}.parquet'
        if src.is_file():
            shutil.copy2(src, O['parquet_dir'] / src.name)
    for name in META_FILES:
        src = G['meta_dir'] / name
        if src.is_file():
            shutil.copy2(src, O['meta_dir'] / name)
    (O['meta_dir'] / 'render_provenance.json').write_text(json.dumps(render_cfg, indent=2,
                                                                    ensure_ascii=False))


def run_workers(args, scenes: list, rigs: list, sentinel_dir: Path, log_root: Path) -> tuple:
    done, failed = [], []
    jobs = [(s, r) for s in scenes for r in rigs]
    for i, (scene, rig) in enumerate(jobs, 1):
        sent = sentinel_dir / f'{scene}__{rig}.json'
        if args.skip_existing and sent.is_file():
            rec = json.loads(sent.read_text())
            if rec.get('max_episodes') == args.max_episodes:
                done.append(rec)
                print(f'[{SCRIPT_NAME}] {i}/{len(jobs)} {scene}/{rig} — 이미 있음, 건너뜀', flush=True)
                continue
        cmd = [ISAAC_PY, str(Path(__file__).resolve()),
               '--_worker_scene', scene, '--_worker_rig', rig,
               '--_sentinel_dir', str(sentinel_dir),
               '--gt_root', args.gt_root, '--out_root', args.out_root,
               '--mesh_root', args.mesh_root, '--raw_root', args.raw_root,
               '--max_episodes', str(args.max_episodes),
               '--align_frames', str(args.align_frames),
               '--light', args.light, '--rtx_ambient', str(args.rtx_ambient),
               '--film_iso', str(args.film_iso)]
        if args.refine:
            cmd.append('--refine')
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f'{scene}__{rig}.log'
        t0 = time.time()
        with open(log_path, 'w') as fh:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=args.per_job_timeout)
        dt = time.time() - t0
        # 성공 판정은 종료코드가 아니라 sentinel — Isaac은 정상 종료에도 세그폴트를 낸다(실측)
        if sent.is_file():
            rec = json.loads(sent.read_text())
            done.append(rec)
            print(f'[{SCRIPT_NAME}] {i}/{len(jobs)} {scene}/{rig}  '
                  f'{rec["episodes"]}ep {rec["frames"]}프레임 · 정합 '
                  f'{rec["align_residual_median_m"]:.4f} m  ({dt:.0f}s)', flush=True)
        else:
            failed.append(f'{scene}/{rig}')
            print(f'[{SCRIPT_NAME}] {i}/{len(jobs)} {scene}/{rig}  실패 (exit {p.returncode}) '
                  f'-> {log_path}', flush=True)
    return done, failed


def main() -> int:
    import camera_profiles

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r',
                    help='원본 vln_ce. **읽기만 한다**')
    ap.add_argument('--out_root', default='data/InternData-N1-v0.5-mini/vln_ce_render')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_pe')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r',
                    help='정합 초기값(start_position/rotation)이 있는 raw_data')
    ap.add_argument('--scenes', default='17DRP5sb8fy', help='콤마 구분')
    ap.add_argument('--rigs', default='all', help="콤마 구분 또는 all (릴리스 5종)")
    ap.add_argument('--max_episodes', type=int, default=3, help='씬당 에피소드 상한. 0이면 전부')
    ap.add_argument('--align_frames', type=int, default=8, help='정합에 쓸 프레임 수')
    ap.add_argument('--refine', action='store_true', help='정합을 수치 최적화로 다듬는다(느림)')
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=10.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--per_job_timeout', type=int, default=7200)
    ap.add_argument('--no_skip_existing', dest='skip_existing', action='store_false')
    ap.add_argument('--log_dir', default='logs/gs-vlnpe')
    ap.add_argument('--_worker_scene', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--_worker_rig', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--_sentinel_dir', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker_scene:
        return render_worker(args)

    # ---- driver: Isaac을 띄우지 않는다 ----
    gt_root, out_root = Path(args.gt_root), Path(args.out_root)
    if not gt_root.is_dir():
        print(f'[{SCRIPT_NAME}] GT 루트가 없다: {gt_root}', file=sys.stderr)
        return 2
    scenes = [s for s in args.scenes.split(',') if s]
    rigs = (list(camera_profiles.vlnce_rig_names()) if args.rigs == 'all'
            else [r for r in args.rigs.split(',') if r])

    n_ep = sum(len(gt_episode_ids(gt_root, s)[:args.max_episodes] if args.max_episodes > 0
                   else gt_episode_ids(gt_root, s)) for s in scenes)
    print(f'[{SCRIPT_NAME}] 씬 {len(scenes)} × rig {len(rigs)} · 에피소드 {n_ep} '
          f'· light={args.light} ambient={args.rtx_ambient} iso={args.film_iso}', flush=True)
    print(f'[{SCRIPT_NAME}] GT(읽기) {gt_root}', flush=True)
    print(f'[{SCRIPT_NAME}] 출력      {out_root / "traj_data" / "r2r"}', flush=True)

    sentinel_dir = Path(args.log_dir) / SCRIPT_NAME / 'sentinels'
    log_root = Path(args.log_dir) / SCRIPT_NAME / 'logs'
    t0 = time.time()
    done, failed = run_workers(args, scenes, rigs, sentinel_dir, log_root)

    # rig가 전부 끝난 씬에만 parquet/meta를 복사한다
    out_r2r = out_root / 'traj_data' / 'r2r'
    for scene in scenes:
        got = {d['rig'] for d in done if d['scene'] == scene}
        if not set(rigs) <= got:
            print(f'[{SCRIPT_NAME}] {scene}: rig {sorted(set(rigs) - got)} 미완 — sidecar 복사 보류',
                  flush=True)
            continue
        eps = gt_episode_ids(gt_root, scene)
        if args.max_episodes > 0:
            eps = eps[:args.max_episodes]
        rs = [d['align_residual_median_m'] for d in done
              if d['scene'] == scene and d['align_residual_median_m'] is not None]
        copy_gt_sidecars(gt_root, out_r2r, scene, eps, {
            'produced_by': f'scripts/dataset_converters/gs_vlnpe/{SCRIPT_NAME}.py',
            'source_gt_root': str(gt_root), 'scene': scene,
            'poses_from': 'vln_ce GT pose.<rig> (start-relative) -> vlnce_align.align_episode -> mesh 절대',
            'parquet_and_meta': 'copied_from_gt (byte-identical)',
            'images_from': 'isaac_rtx_render',
            'rigs': rigs, 'episodes': eps,
            'align_residual_median_m': float(np.median(rs)) if rs else None,
            'depth_encoding': 'uint16 millimetres, 0 = invalid',
            'render': {'light': args.light, 'rtx_ambient': args.rtx_ambient,
                       'film_iso': args.film_iso, 'width': IMG_W, 'height': IMG_H},
        })
        print(f'[{SCRIPT_NAME}] {scene}: parquet {len(eps)}개 + meta 복사 완료', flush=True)

    tf = sum(d['frames'] for d in done)
    print(f'\n[{SCRIPT_NAME}] 성공 {len(done)}/{len(scenes) * len(rigs)} 작업 · '
          f'{tf:,}프레임 · {(time.time() - t0) / 60:.1f}분', flush=True)
    if failed:
        print(f'[{SCRIPT_NAME}] 실패: {failed}', flush=True)
        print(f'[{SCRIPT_NAME}] 로그: {log_root}', flush=True)
    return 0 if not failed else 1


if __name__ == '__main__':
    import os
    import traceback

    try:
        code = main()
    except BaseException:  # noqa: BLE001  트레이스백을 남기고 나서 끊는다
        traceback.print_exc()
        code = 3
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)   # simulation_app.close()가 수 분 걸리거나 세그폴트를 낸다
