"""GT pose를 그대로 재렌더해 **vln_pe와 동일한 구조**로 저장한다 (Phase A).

무엇을 만드나
-------------
릴리스 `vln_pe/traj_data/r2r/`의 모든 씬·에피소드에 대해, **GT parquet에 적힌 실제 camera pose를
그대로 재생**해 우리 Isaac 렌더러로 rgb/depth를 다시 만든다. 궤적·action·지시문은 GT와 완전히
같고 **이미지만 우리 렌더**다.

```
<out_root>/traj_data/r2r/<scene>/
├── data/chunk-000/episode_%06d.parquet          # GT에서 바이트 복사
├── videos/chunk-000/observation.images.rgb/episode_%06d.npy    # (T,256,256,3) uint8   ← 우리 렌더
├── videos/chunk-000/observation.images.depth/episode_%06d.npy  # (T,256,256) float32   ← 우리 렌더
└── meta/{info,episodes,episodes_stats,tasks}.*  # GT에서 바이트 복사
    + render_provenance.json                     # 신규 사이드카
```

**parquet·meta를 복사하는 것이 핵심 단순화다.** GT pose를 쓰므로 궤적이 동일하고, 따라서
action 이산화·지시문 생성·통계 재계산이 전부 불필요하다. 구조 동일성이 복사로 보장되고
GT와 프레임 단위로 대응돼 SSIM 대조가 정확해진다. (새로 뽑은 경로는 Phase C에서 다룬다 —
그쪽은 parquet을 직접 써야 한다.)

왜 씬당 프로세스 하나인가
-------------------------
한 프로세스에서 `build_renderer`를 두 번 부르면 `/World/RenderCamera`가 이미 있어서 죽고,
씬을 3번 올리면 애노테이터가 떨어진다(`AnnotatorRegistryError`, 둘 다 실측).
`04e_sample_compare.py`·`usd_study/s9_tonemap_batch.py`와 같은 회피책을 쓴다.

커맨드
------
한 씬 스모크 (전수 전에 반드시 먼저):
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05_export_vlnpe.py --dataset vln_pe --scenes 17DRP5sb8fy --out_root data/InternData-N1-v0.5-mini/vln_pe_render --log_dir logs/gs-vlnpe

전수 (61씬 · 2813 에피소드 · 271,252 프레임 · 약 3시간 · 115 GiB):
timeout --signal=KILL 86400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05_export_vlnpe.py --dataset vln_pe --n_scenes 0 --out_root data/InternData-N1-v0.5-mini/vln_pe_render --log_dir logs/gs-vlnpe

에피소드 수를 제한해 먼저 보려면 `--max_episodes 5`.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCRIPT_NAME = '05_export_vlnpe'
ISAAC_PY = '/workspace/isaaclab/_isaac_sim/python.sh'

# vln_pe depth 규약 (실측): float32, 0 < v <= 1, 미터 = v * 10. 0은 릴리스에 존재하지 않고
# 1.0이 ">=10 m 클립 = 무효"를 뜻한다. `dataset_utils.load_depth_frame_m`이 `raw>0 & raw<1.0`을
# 유효로 보므로, 1.0으로 저장한 픽셀은 자동으로 무효 처리된다.
DEPTH_SCALE_M = 10.0
DEPTH_MIN_STORED = 1e-6

# 렌더 설정의 단일 출처는 `04_render_obs_isaac.py`의 `MEASURED_RENDER_PRESET`이다.
# 여기서 값을 복사해 두지 않는다 — worker가 04를 import해 직접 읽는다.
# CLI 인자는 전부 기본값 None(센티널)이고, **주어진 것만** 프리셋을 덮는다.
RENDER_KEYS = ('light', 'rtx_ambient', 'film_iso', 'tonemap_op', 'tonemap_crush')


def resolve_render_settings(r4, args) -> dict:
    """프리셋 위에 사용자가 명시한 인자만 덮어써서 최종 렌더 설정을 만든다.

    `--tonemap_op`/`--tonemap_crush`는 **None이 "설정을 건드리지 않음"이라는 뜻의 유효한 값**이라
    센티널로 구분할 수 없다. vln_n1 프리셋은 op 7을 켜므로, 그걸 끄고 싶으면 `--no_tonemap`을 쓴다.
    """
    cfg = dict(r4.MEASURED_RENDER_PRESET[args.dataset])
    for key in RENDER_KEYS:
        given = getattr(args, key)
        if given is not None:
            cfg[key] = given
    if args.no_tonemap:
        cfg['tonemap_op'] = None
        cfg['tonemap_crush'] = None
    return cfg


def load_renderer_module():
    """04를 모듈로 로드 (import 시점에 SimulationApp 부팅 — pxr보다 먼저 해야 한다)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location('r4_s06', HERE / '04_render_obs_isaac.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['r4_s06'] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 경로 규약 — GT와 산출물이 완전히 같은 모양이어야 한다
# ---------------------------------------------------------------------------

def scene_paths(root: Path, scene: str) -> dict:
    base = Path(root) / scene
    return {
        'base': base,
        'parquet_dir': base / 'data' / 'chunk-000',
        'rgb_dir': base / 'videos' / 'chunk-000' / 'observation.images.rgb',
        'depth_dir': base / 'videos' / 'chunk-000' / 'observation.images.depth',
        'meta_dir': base / 'meta',
    }


def gt_episode_ids(gt_root: Path, scene: str) -> list:
    """그 씬의 GT 에피소드 번호 — parquet 파일명에서 읽는다(연속이 아닐 수도 있으니 파일 기준)."""
    d = scene_paths(gt_root, scene)['parquet_dir']
    return sorted(int(p.stem.split('_')[1]) for p in d.glob('episode_*.parquet'))


def available_scenes(mesh_root: Path, gt_root: Path) -> list:
    """USD와 GT 궤적이 **둘 다** 있는 씬만. (USD 90개, vln_pe 궤적 61개)"""
    have_usd = {p.name for p in Path(mesh_root).glob('*') if (p / 'matterport_mesh').is_dir()}
    have_gt = {p.name for p in Path(gt_root).glob('*') if (p / 'data' / 'chunk-000').is_dir()}
    return sorted(have_usd & have_gt)


def pick_scenes(all_scenes: list, n: int, seed: int) -> list:
    if n <= 0 or n >= len(all_scenes):
        return all_scenes
    import random

    return sorted(random.Random(seed).sample(all_scenes, n))


# ---------------------------------------------------------------------------
# 용량 가드 — 중간에 디스크가 차면 npy가 잘려 저장돼 조용히 깨진다
# ---------------------------------------------------------------------------

def bytes_per_frame() -> int:
    """(256,256,3) uint8 + (256,256) float32 — vln_pe 프레임 하나의 디스크 비용."""
    return 256 * 256 * 3 + 256 * 256 * 4


def npy_frame_count(path: Path) -> int:
    """npy **헤더만** 읽어 첫 축 길이를 얻는다 (17 MB 배열을 로드하지 않는다)."""
    with open(path, 'rb') as fh:
        version = np.lib.format.read_magic(fh)
        shape, _, _ = np.lib.format._read_array_header(fh, version)
    return int(shape[0])


def estimate_bytes(gt_root: Path, scenes: list, max_episodes: int) -> tuple:
    """(프레임 수, 바이트 수) — GT rgb npy 헤더에서 T를 읽어 정확히 센다."""
    total_frames = 0
    for scene in scenes:
        P = scene_paths(gt_root, scene)
        ids = gt_episode_ids(gt_root, scene)
        if max_episodes > 0:
            ids = ids[:max_episodes]
        for ep in ids:
            f = P['rgb_dir'] / f'episode_{ep:06d}.npy'
            if f.is_file():
                total_frames += npy_frame_count(f)
    return total_frames, total_frames * bytes_per_frame()


def check_disk(out_root: Path, need_bytes: int, headroom_gib: float) -> bool:
    out_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_root).free
    need = need_bytes + int(headroom_gib * (1 << 30))
    g = 1 << 30
    print(f'[{SCRIPT_NAME}] 용량: 필요 {need_bytes / g:.1f} GiB + 여유분 {headroom_gib:.0f} GiB '
          f'= {need / g:.1f} GiB · 사용 가능 {free / g:.1f} GiB', flush=True)
    if free < need:
        print(f'[{SCRIPT_NAME}] 용량 부족 — 중단한다. {(need - free) / g:.1f} GiB를 더 비우거나 '
              f'--max_episodes / --n_scenes 로 범위를 줄일 것', file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# worker — 씬 하나
# ---------------------------------------------------------------------------

def depth_m_to_vlnpe(depth_m: np.ndarray) -> np.ndarray:
    """렌더 depth(m, 배경=inf) -> vln_pe 저장형(float32, 0<v<=1, m = v*10).

    무효 픽셀(비유한 또는 >=10 m)은 **1.0**으로 둔다 — 릴리스 데이터의 규약이고,
    `dataset_utils.load_depth_frame_m`의 `raw < 1.0` 조건에서 자동으로 걸러진다.
    """
    v = np.asarray(depth_m, dtype=np.float32) / np.float32(DEPTH_SCALE_M)
    bad = ~np.isfinite(v) | (v >= 1.0)
    v = np.clip(v, DEPTH_MIN_STORED, 1.0)
    v[bad] = 1.0
    return v.astype(np.float32)


def git_sha() -> str:
    try:
        return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def render_worker(args) -> int:
    """씬 하나를 렌더해 vln_pe 레이아웃으로 저장하고, 완료 sentinel을 남긴다."""
    scene = args._worker_scene
    r4 = load_renderer_module()
    import dataset_utils

    gt_root = Path(args.gt_root)
    out_root = Path(args.out_root) / 'traj_data' / 'r2r'
    G, O = scene_paths(gt_root, scene), scene_paths(out_root, scene)
    for key in ('parquet_dir', 'rgb_dir', 'depth_dir', 'meta_dir'):
        O[key].mkdir(parents=True, exist_ok=True)

    ids = gt_episode_ids(gt_root, scene)
    if args.max_episodes > 0:
        ids = ids[:args.max_episodes]

    # 씬 USD — **반드시 load_scene_model.** 텍스처 심링크가 이 안에서 만들어진다.
    # USD를 직접 열면 텍스처 없는 그림이 나온다(실측 SSIM 0.164 vs 0.613).
    usd_path = r4.load_scene_model(args.mesh_root, scene)
    k = dataset_utils.load_gt_episode(args.gt_root, scene, ids[0], args.dataset)['k']
    W, H = dataset_utils.render_wh(args.dataset)
    cfg = resolve_render_settings(r4, args)

    print(f'[{SCRIPT_NAME}:{scene}] usd={usd_path.name} 에피소드 {len(ids)}개 {W}x{H}', flush=True)
    print(f'[{SCRIPT_NAME}:{scene}] 렌더 설정 {cfg}', flush=True)
    # 씬당 딱 한 번 (위 docstring의 프로세스 제약)
    cas = r4.build_renderer(W, H, k, usd_path, cfg['light'], None,
                            cfg['rtx_ambient'], cfg['film_iso'],
                            cfg['tonemap_op'], cfg['tonemap_crush'])

    frames_total, t0 = 0, time.time()
    per_ep = []
    for ep in ids:
        gt = dataset_utils.load_gt_episode(args.gt_root, scene, ep, args.dataset)
        poses = gt['poses_c2w']
        rgb, depth_m = r4.render_along(cas, poses)
        rgb = np.asarray(rgb, dtype=np.uint8)
        depth = depth_m_to_vlnpe(depth_m)

        # 프레임 수는 GT와 정확히 같아야 한다 — parquet 행수가 곧 pose 수다
        assert rgb.shape == (len(poses), H, W, 3), f'{scene} ep{ep} rgb {rgb.shape}'
        assert depth.shape == (len(poses), H, W), f'{scene} ep{ep} depth {depth.shape}'

        np.save(O['rgb_dir'] / f'episode_{ep:06d}.npy', rgb)
        np.save(O['depth_dir'] / f'episode_{ep:06d}.npy', depth)
        shutil.copy2(G['parquet_dir'] / f'episode_{ep:06d}.parquet',
                     O['parquet_dir'] / f'episode_{ep:06d}.parquet')
        frames_total += len(poses)
        per_ep.append({'episode': ep, 'frames': int(len(poses))})
        print(f'[{SCRIPT_NAME}:{scene}]   ep {ep:>4} {len(poses):>4} 프레임', flush=True)

    # meta는 GT에서 바이트 복사 — 지시문·통계·info가 전부 그대로여야 구조가 같다
    for f in sorted(G['meta_dir'].glob('*')):
        if f.is_file():
            shutil.copy2(f, O['meta_dir'] / f.name)

    # 이 데이터셋이 실제 캡처로 오해되지 않게 하는 사이드카
    prov = {
        'produced_by': f'scripts/dataset_converters/gs_vlnpe/{SCRIPT_NAME}.py',
        'git_sha': git_sha(),
        'dataset': args.dataset,
        'scene': scene,
        'source_gt_root': str(gt_root),
        'poses_from': 'gt_parquet (observation.camera_position/orientation)',
        'parquet_and_meta': 'copied_from_gt (byte-identical)',
        'images_from': 'isaac_rtx_render',
        'render': dict(cfg, width=W, height=H,
                       preset_source='04_render_obs_isaac.MEASURED_RENDER_PRESET'),
        'intrinsic_K': np.asarray(k, dtype=float).tolist(),
        'depth_encoding': 'float32, meters = stored * 10; invalid/>=10m stored as exactly 1.0',
        'episodes': per_ep,
        'frames_total': frames_total,
        'elapsed_s': round(time.time() - t0, 1),
    }
    (O['meta_dir'] / 'render_provenance.json').write_text(json.dumps(prov, indent=2, ensure_ascii=False))

    # sentinel — driver가 성공/skip 판정에 쓴다 (반드시 마지막에 쓴다)
    sentinel = Path(args._sentinel_dir) / f'{scene}.json'
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({'scene': scene, 'episodes': len(ids), 'frames': frames_total,
                                    'elapsed_s': prov['elapsed_s'],
                                    # **부분 실행을 완료로 착각하지 않게 범위를 남긴다.**
                                    # 실측 사고: --max_episodes 3 스모크가 sentinel을 남겼고
                                    # 이어진 전수 실행이 그 씬을 건너뛰어 25에피소드 중 3개만 남았다.
                                    'max_episodes': int(args.max_episodes)}, ensure_ascii=False))
    print(f'[{SCRIPT_NAME}:{scene}] 완료 — {len(ids)}에피소드 {frames_total}프레임 '
          f'{prov["elapsed_s"]:.0f}s', flush=True)
    return 0


# ---------------------------------------------------------------------------
# driver — 씬마다 worker 프로세스를 띄운다
# ---------------------------------------------------------------------------

def sentinel_covers(rec: dict, requested_max: int, gt_episodes: int) -> bool:
    """이미 있는 sentinel이 지금 요청한 범위를 **덮는가**.

    이걸 안 보면 부분 실행을 완료로 착각한다 — `--max_episodes 3` 스모크가 남긴 sentinel 때문에
    이어진 전수 실행이 그 씬을 건너뛰어 25에피소드 중 3개만 남는 사고가 있었다(실측).

    판정은 **기록된 범위가 아니라 실제로 만든 에피소드 수**를 기준으로 한다. `max_episodes`
    키가 없는 옛 sentinel도 `episodes`는 있으므로 GT 개수와 비교하면 완료 여부를 알 수 있다.
    (처음엔 옛 sentinel을 무조건 "안 덮는다"로 했더니 이미 완성된 60씬을 다시 렌더했다 — 실측.)
    """
    made = int(rec.get('episodes', 0))
    want = gt_episodes if requested_max == 0 else min(requested_max, gt_episodes)
    return made >= want


def run_workers(args, scenes: list, sentinel_dir: Path, log_root: Path) -> tuple:
    done, failed = [], []
    for i, scene in enumerate(scenes, 1):
        sentinel = sentinel_dir / f'{scene}.json'
        if args.skip_existing and sentinel.is_file():
            rec = json.loads(sentinel.read_text())
            n_gt = len(gt_episode_ids(Path(args.gt_root), scene))
            if sentinel_covers(rec, args.max_episodes, n_gt):
                done.append(rec)
                print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene} — 이미 있음, 건너뜀 '
                      f'({rec["episodes"]}ep)', flush=True)
                continue
            print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene} — 기존 결과가 부족하다 '
                  f'(만든 것 {rec.get("episodes")}ep, GT {n_gt}ep, 요청 {args.max_episodes or "전수"}) '
                  f'-> 다시 렌더', flush=True)
        cmd = [ISAAC_PY, str(Path(__file__).resolve()),
               '--_worker_scene', scene, '--_sentinel_dir', str(sentinel_dir),
               '--dataset', args.dataset, '--gt_root', args.gt_root, '--out_root', args.out_root,
               '--mesh_root', args.mesh_root, '--max_episodes', str(args.max_episodes)]
        # 렌더 설정은 **사용자가 명시한 것만** 넘긴다. 안 넘긴 것은 worker가 04의
        # MEASURED_RENDER_PRESET에서 채운다 (설정의 단일 출처를 04에 둔다).
        for key in RENDER_KEYS:
            val = getattr(args, key)
            if val is not None:
                cmd += [f'--{key}', str(val)]
        if args.no_tonemap:
            cmd.append('--no_tonemap')

        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f'{scene}.log'
        t0 = time.time()
        with open(log_path, 'w') as fh:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=args.per_scene_timeout)
        dt = time.time() - t0
        # **성공 판정은 종료코드가 아니라 sentinel 존재로 한다** — Isaac은 정상 종료에도
        # atexit에서 세그폴트를 내는 일이 있어 returncode를 믿을 수 없다(실측).
        if sentinel.is_file():
            rec = json.loads(sentinel.read_text())
            done.append(rec)
            print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene}  '
                  f'{rec["episodes"]}ep {rec["frames"]}프레임  ({dt:.0f}s)', flush=True)
        else:
            failed.append(scene)
            print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene}  실패 (exit {p.returncode}) '
                  f'-> {log_path}', flush=True)
    return done, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['vln_n1', 'vln_pe'], default='vln_pe')
    ap.add_argument('--gt_root', default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r',
                    help='원본 GT 씬 폴더들이 있는 곳. **읽기만 한다**')
    ap.add_argument('--out_root', default='data/InternData-N1-v0.5-mini/vln_pe_render',
                    help='산출물 뿌리. 아래에 traj_data/r2r/<scene>/ 이 만들어진다')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_pe')
    ap.add_argument('--scenes', default=None, help='씬을 직접 지정 (콤마). 주면 --n_scenes 무시')
    ap.add_argument('--n_scenes', type=int, default=0, help='표본 씬 수. 0이면 전수(기본)')
    ap.add_argument('--seed', type=int, default=20260908, help='표본 추출 시드(재현용)')
    ap.add_argument('--max_episodes', type=int, default=0, help='씬당 에피소드 상한. 0이면 전부')
    # 렌더 설정 5개는 전부 기본 None = "04의 MEASURED_RENDER_PRESET을 따른다".
    # 명시하면 그 값이 프리셋을 덮는다. vln_pe 프리셋은 ambient_only · ambient 10.0 · iso 70 ·
    # 톤맵 손대지 않음(op 6 = Isaac 기본)이다.
    ap.add_argument('--light', default=None, help='생략 시 프리셋 값')
    ap.add_argument('--rtx_ambient', type=float, default=None, help='생략 시 프리셋 값')
    ap.add_argument('--film_iso', type=float, default=None, help='생략 시 프리셋 값')
    ap.add_argument('--tonemap_op', type=int, default=None, help='생략 시 프리셋 값')
    ap.add_argument('--tonemap_crush', type=float, default=None, help='생략 시 프리셋 값')
    ap.add_argument('--no_tonemap', action='store_true',
                    help='프리셋이 톤맵 op을 켜도 건드리지 않는다(vln_n1 프리셋의 op7을 끄는 용도)')
    ap.add_argument('--headroom_gib', type=float, default=20.0, help='남겨둘 디스크 여유분')
    ap.add_argument('--skip_disk_check', action='store_true', help='용량 가드를 끈다(권장 안 함)')
    ap.add_argument('--per_scene_timeout', type=int, default=14400, help='씬당 하위 프로세스 제한(초)')
    ap.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                    help='이미 만들어진 씬도 다시 렌더한다')
    ap.add_argument('--log_dir', default='logs/gs-vlnpe')
    # 내부용 — driver가 씬마다 자기 자신을 부를 때 쓴다
    ap.add_argument('--_worker_scene', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--_sentinel_dir', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker_scene:
        return render_worker(args)

    # ---- driver: Isaac을 띄우지 않는다 (worker와 GPU/prim이 겹친다) ----
    gt_root, out_root = Path(args.gt_root), Path(args.out_root)
    if not gt_root.is_dir():
        print(f'[{SCRIPT_NAME}] GT 루트가 없다: {gt_root}', file=sys.stderr)
        return 2

    all_scenes = available_scenes(Path(args.mesh_root), gt_root)
    scenes = ([s for s in args.scenes.split(',') if s] if args.scenes
              else pick_scenes(all_scenes, args.n_scenes, args.seed))
    unknown = [s for s in scenes if s not in all_scenes]
    if unknown:
        print(f'[{SCRIPT_NAME}] USD 또는 GT가 없는 씬: {unknown}', file=sys.stderr)
        return 2

    n_ep = sum(len(gt_episode_ids(gt_root, s)[:args.max_episodes] if args.max_episodes > 0
                   else gt_episode_ids(gt_root, s)) for s in scenes)
    given = {k: getattr(args, k) for k in RENDER_KEYS if getattr(args, k) is not None}
    print(f'[{SCRIPT_NAME}] dataset={args.dataset} 씬 {len(scenes)}/{len(all_scenes)} · '
          f'에피소드 {n_ep}', flush=True)
    print(f'[{SCRIPT_NAME}] 렌더 설정: 프리셋(04의 MEASURED_RENDER_PRESET) + 지정 {given or "없음"}'
          f'{" + --no_tonemap" if args.no_tonemap else ""} — 확정값은 각 씬 로그에 찍힌다', flush=True)
    print(f'[{SCRIPT_NAME}] GT(읽기) {gt_root}', flush=True)
    print(f'[{SCRIPT_NAME}] 출력      {out_root / "traj_data" / "r2r"}', flush=True)

    frames, need = estimate_bytes(gt_root, scenes, args.max_episodes)
    print(f'[{SCRIPT_NAME}] 예상 프레임 {frames:,}개', flush=True)
    if not args.skip_disk_check and not check_disk(out_root, need, args.headroom_gib):
        return 3

    sentinel_dir = Path(args.log_dir) / SCRIPT_NAME / args.dataset / 'scenes'
    log_root = Path(args.log_dir) / SCRIPT_NAME / args.dataset / 'logs'
    t0 = time.time()
    done, failed = run_workers(args, scenes, sentinel_dir, log_root)

    tf = sum(d['frames'] for d in done)
    te = sum(d['episodes'] for d in done)
    print(f'\n[{SCRIPT_NAME}] 성공 {len(done)}/{len(scenes)}씬 · {te}에피소드 · {tf:,}프레임 · '
          f'{(time.time() - t0) / 60:.1f}분', flush=True)
    if failed:
        print(f'[{SCRIPT_NAME}] 실패 {len(failed)}씬: {failed}', flush=True)
        print(f'[{SCRIPT_NAME}] 로그: {log_root}', flush=True)
    summary = Path(args.log_dir) / SCRIPT_NAME / args.dataset / 'summary.json'
    summary.write_text(json.dumps({'scenes_done': len(done), 'scenes_failed': failed,
                                   'episodes': te, 'frames': tf,
                                   'render_overrides': given, 'no_tonemap': args.no_tonemap,
                                   'render_preset_source': '04_render_obs_isaac.MEASURED_RENDER_PRESET',
                                   'out_root': str(out_root)}, indent=2, ensure_ascii=False))
    print(f'[{SCRIPT_NAME}] summary -> {summary}', flush=True)
    return 0 if not failed else 1


if __name__ == '__main__':
    import os
    import traceback

    try:
        code = main()
    except BaseException:  # noqa: BLE001  트레이스백을 반드시 남기고 나서 끊는다
        traceback.print_exc()
        code = 3
    # `simulation_app.close()`가 수 분 걸리거나 atexit에서 세그폴트가 난다. 산출물은 이 시점에
    # 이미 디스크에 있으므로 즉시 끊는다.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
