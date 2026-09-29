"""여러 씬을 표본으로 뽑아 **GT 이미지와 우리 렌더를 한 리포트에서** 나란히 본다.

왜 필요한가
-----------
전 씬(90개) 이미지를 저장하기 전에, 확정한 렌더 설정이 실제로 GT와 닮았는지 **사람 눈으로**
확인하는 단계다. `04_render_obs_isaac.py --mode gt_replay`가 씬 하나에 대해 같은 일을 하지만
리포트가 씬마다 흩어져 여러 씬을 한눈에 비교할 수 없다. 이 스크립트는 씬 여러 개를 한 페이지에
모아 화살표로 GT ↔ 렌더를 겹쳐 보게 한다.

**04를 수정하지 않는다.** 04를 그대로 모듈로 불러 `build_renderer`/`render_along`을 재사용한다.
설정도 04의 `MEASURED_RENDER_PRESET`을 그대로 읽으므로 값이 두 곳에 갈리지 않는다.

커맨드
------
vln_pe 6씬 표본 (확정 설정):
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04e_sample_compare.py --dataset vln_pe --n_scenes 6 --preset measured --log_dir logs/gs-vlnpe

vln_n1 6씬 표본:
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04e_sample_compare.py --dataset vln_n1 --n_scenes 6 --preset measured --log_dir logs/gs-vlnpe

기존 설정과 나란히 보려면 (대조군 — 프리셋 없이):
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/04e_sample_compare.py --dataset vln_pe --n_scenes 6 --light ambient_only --rtx_ambient 10.0 --film_iso 70 --log_dir logs/gs-vlnpe

씬을 직접 고르려면 `--scenes A,B,C`. `--both_settings`를 주면 한 번의 실행에서
**기존 설정과 확정 설정을 둘 다** 렌더해 같은 프레임에서 비교한다(씬당 렌더 2회).
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCRIPT_NAME = '04e_sample_compare'

# driver는 Isaac을 띄우지 않으므로 04 모듈을 import할 수 없다. ambient 기본값만 여기 둔다
# (04의 MEASURED_RENDER_PRESET과 같은 값 — worker 쪽은 04에서 직접 읽는다).
MEASURED_AMBIENT = {'vln_n1': 6.0, 'vln_pe': 10.0}


def load_renderer_module():
    """04를 모듈로 로드 (import 시점에 SimulationApp 부팅 — pxr보다 먼저 해야 한다)."""
    spec = importlib.util.spec_from_file_location('r4_s05', HERE / '04_render_obs_isaac.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['r4_s05'] = mod
    spec.loader.exec_module(mod)
    return mod


def pick_scenes(mesh_root: Path, dataset: str, n: int, seed: int, data_root: str) -> list:
    """GT 궤적과 씬 USD가 **둘 다 있는** 씬만 후보로 두고 무작위 표본."""
    have_usd = {p.name for p in Path(mesh_root).glob('*') if (p / 'matterport_mesh').is_dir()}
    have_gt = {p.name for p in Path(data_root).glob('*') if p.is_dir()}
    cands = sorted(have_usd & have_gt)
    if not cands:
        raise FileNotFoundError(f'씬 USD와 GT가 둘 다 있는 씬이 없다 (usd={mesh_root}, gt={data_root})')
    if n <= 0 or n >= len(cands):
        return cands
    return sorted(random.Random(seed).sample(cands, n))


def settings_variants(r4, args) -> list:
    """(라벨, 설정 dict) 목록. `--both_settings`면 기존/확정 둘 다."""
    measured = dict(r4.MEASURED_RENDER_PRESET[args.dataset])
    explicit = {'light': args.light, 'rtx_ambient': args.rtx_ambient, 'film_iso': args.film_iso,
                'tonemap_op': args.tonemap_op, 'tonemap_crush': args.tonemap_crush}
    if args.both_settings:
        return [('기존 (iso 70 · op 6)',
                 {'light': 'ambient_only', 'rtx_ambient': measured['rtx_ambient'],
                  'film_iso': 70.0, 'tonemap_op': None, 'tonemap_crush': None}),
                ('확정 (126씬 측정)', measured)]
    elif args.preset == 'measured':
        return [('확정 (126씬 측정)', measured)]
    elif args.preset == 'none':
        return [('지정한 설정', explicit)]
    else:
        assert False, f'unreachable preset={args.preset!r}'


# carb 설정 키 — s8_tonemap_adopt.py와 같은 값(두 곳이 갈리지 않게 이름도 같게 둔다)
TM_OP = '/rtx/post/tonemap/op'
TM_ISO = '/rtx/post/tonemap/filmIso'
TM_CRUSH = '/rtx/post/tonemap/irayReinhard/crushBlacks'
RTX_AMBIENT = '/rtx/sceneDb/ambientLightIntensity'


def apply_settings(cfg: dict, sim, warmup: int) -> None:
    """설정만 갈아끼운다 — **씬/카메라를 다시 만들지 않는다.**

    `build_renderer`를 두 번 부르면 `/World/RenderCamera`가 이미 있어서 죽고, 한 프로세스에서
    씬을 3번 올리면 애노테이터가 떨어진다(둘 다 실측). 그래서 s8과 같이 렌더러는 씬당 한 번만
    만들고 설정은 carb로 바꾼 뒤 몇 스텝 돌려 반영을 기다린다.

    `None`인 항목은 건드리지 않는다 — 04의 `build_renderer`와 같은 규칙이다.
    """
    import carb

    s = carb.settings.get_settings()
    if cfg.get('rtx_ambient') is not None:
        s.set_float(RTX_AMBIENT, float(cfg['rtx_ambient']))
    if cfg.get('film_iso') is not None:
        s.set_float(TM_ISO, float(cfg['film_iso']))
    if cfg.get('tonemap_op') is not None:
        s.set_int(TM_OP, int(cfg['tonemap_op']))
    if cfg.get('tonemap_crush') is not None:
        s.set_float(TM_CRUSH, float(cfg['tonemap_crush']))
    for _ in range(warmup):
        sim.step()


def render_scene(r4, args, scene: str, variants: list) -> dict:
    """한 씬에서 GT 프레임 몇 장과, 각 설정으로 렌더한 같은 프레임을 얻는다."""
    import dataset_utils
    from skimage.metrics import structural_similarity as ssim_metric

    gt = dataset_utils.load_gt_episode(args.data_root, scene, args.episode, args.dataset)
    W, H = dataset_utils.render_wh(args.dataset)
    n = len(gt['poses_c2w'])
    idx = np.unique(np.linspace(0, n - 1, args.n_frames).astype(int)).tolist()
    poses = [gt['poses_c2w'][i] for i in idx]
    # 파이프라인과 **같은 함수**로 씬 USD를 얻는다 — 이 안에서 텍스처 심링크가 만들어진다.
    # 이걸 우회해 USD를 직접 열면 텍스처 없는 그림을 재게 된다(실측: SSIM 0.164 vs 0.613).
    usd_path = r4.load_scene_model(args.mesh_root, scene)

    rgb_dir = Path(args.data_root) / scene / 'videos' / 'chunk-000' / 'observation.images.rgb'
    reals = [dataset_utils.load_rgb_frame(rgb_dir, args.episode, i, args.dataset) for i in idx]

    out = {'scene': scene, 'frames': idx, 'gt': reals, 'variants': []}
    first_label, first_cfg = variants[0]
    cas = r4.build_renderer(W, H, gt['k'], usd_path, first_cfg['light'], None,
                            first_cfg['rtx_ambient'], first_cfg['film_iso'],
                            first_cfg['tonemap_op'], first_cfg['tonemap_crush'])
    sim = cas[1]

    for i, (label, cfg) in enumerate(variants):
        if i > 0:
            apply_settings(cfg, sim, args.warmup)
        rgb, _ = r4.render_along(cas, poses)
        scores = [float(ssim_metric(reals[j], rgb[j], channel_axis=2, data_range=255))
                  for j in range(len(idx))]
        out['variants'].append({'label': label, 'cfg': cfg, 'rgb': rgb, 'ssim': scores})
        print(f'[{SCRIPT_NAME}]   {scene} · {label}: SSIM 중앙값 {float(np.median(scores)):.4f}', flush=True)
    return out


def save_scene_result(res: dict, out_dir: Path, dataset: str) -> Path:
    """씬 하나의 결과를 프레임 jpg + json으로 남긴다 (프로세스가 곧 끝나므로 메모리로 못 넘긴다)."""
    import json

    import cv2

    img_dir = out_dir / 'frames'
    img_dir.mkdir(parents=True, exist_ok=True)
    scene = res['scene']
    meta = {'scene': scene, 'frames': res['frames'], 'gt': [], 'variants': []}
    for j, frame in enumerate(res['frames']):
        gp = img_dir / f'{scene}_f{frame:04d}_gt.jpg'
        cv2.imwrite(str(gp), cv2.cvtColor(np.asarray(res['gt'][j]), cv2.COLOR_RGB2BGR))
        meta['gt'].append(gp.name)
    for vi, v in enumerate(res['variants']):
        names = []
        for j, frame in enumerate(res['frames']):
            vp = img_dir / f'{scene}_f{frame:04d}_v{vi}.jpg'
            cv2.imwrite(str(vp), cv2.cvtColor(v['rgb'][j], cv2.COLOR_RGB2BGR))
            names.append(vp.name)
        meta['variants'].append({'label': v['label'], 'ssim': v['ssim'], 'files': names})
    jp = out_dir / 'scenes' / f'{scene}.json'
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(meta, ensure_ascii=False))
    return jp


def run_workers(args, scenes: list, out_dir: Path) -> list:
    """씬마다 **별도 프로세스**로 이 스크립트를 다시 부른다.

    한 프로세스에서 `build_renderer`를 두 번 부르면 `/World/RenderCamera`가 이미 있어서 죽고,
    씬을 3번 올리면 애노테이터가 떨어진다(둘 다 실측). `s9_tonemap_batch.py`가 쓰는 것과 같은
    회피책 — 씬당 프로세스 하나.
    """
    import json
    import subprocess
    import time

    isaac_py = '/workspace/isaaclab/_isaac_sim/python.sh'
    metas = []
    for i, scene in enumerate(scenes, 1):
        jp = out_dir / 'scenes' / f'{scene}.json'
        if args.skip_existing and jp.is_file():
            print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene} — 이미 있음, 건너뜀', flush=True)
            metas.append(json.loads(jp.read_text()))
            continue
        cmd = [isaac_py, str(Path(__file__).resolve()), '--_worker_scene', scene,
               '--dataset', args.dataset, '--episode', str(args.episode),
               '--n_frames', str(args.n_frames), '--preset', args.preset,
               '--light', args.light, '--rtx_ambient', str(args.rtx_ambient),
               '--film_iso', str(args.film_iso), '--warmup', str(args.warmup),
               '--mesh_root', args.mesh_root, '--data_root', args.data_root,
               '--log_dir', args.log_dir, '--_out_dir', str(out_dir)]
        if args.both_settings:
            cmd.append('--both_settings')
        if args.tonemap_op is not None:
            cmd += ['--tonemap_op', str(args.tonemap_op)]
        if args.tonemap_crush is not None:
            cmd += ['--tonemap_crush', str(args.tonemap_crush)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.per_scene_timeout)
        (out_dir / 'logs').mkdir(parents=True, exist_ok=True)
        (out_dir / 'logs' / f'{scene}.log').write_text(proc.stdout + '\n' + proc.stderr)
        if jp.is_file():
            meta = json.loads(jp.read_text())
            metas.append(meta)
            meds = ' · '.join(f'{v["label"]} {float(np.median(v["ssim"])):.4f}' for v in meta['variants'])
            print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene}  {meds}  ({time.time()-t0:.0f}s)', flush=True)
        else:
            print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene}  실패 (exit {proc.returncode}) '
                  f'-> {out_dir / "logs" / f"{scene}.log"}', flush=True)
    return metas


def build_report(metas: list, args, out_dir: Path) -> Path:
    import viz_utils

    img_dir = out_dir / 'frames'
    body, rows = [], []
    labels = [v['label'] for v in metas[0]['variants']]
    for m in metas:
        scene = m['scene']
        rows.append([scene] + [f'{float(np.median(v["ssim"])):.4f}' for v in m['variants']])
        body.append(f'<h2>{scene}</h2>')
        # **정적 <img>로 나란히 놓는다.** 화살표로 겹쳐 보는 blink 위젯은 자바스크립트가
        # 이미지를 채우는 방식이라, 스크립트를 안 돌리는 뷰어에서는 빈칸으로 보인다(실측).
        # 여기서는 GT/설정들을 한 줄에 같이 띄워 스크립트 없이도 바로 비교되게 한다.
        for j, frame in enumerate(m['frames']):
            cells = [('GT (실제 데이터셋)', img_dir / m['gt'][j], '')]
            for v in m['variants']:
                cells.append((v['label'], img_dir / v['files'][j], f'SSIM {v["ssim"][j]:.3f}'))
            tds = ''.join(
                f'<div style="flex:0 0 auto;text-align:center">'
                f'<div style="font-size:12px;opacity:.75;margin-bottom:4px">{lab}</div>'
                f'<img src="{viz_utils.image_to_data_uri(path)}" alt="{lab}" '
                f'style="width:240px;height:auto;image-rendering:pixelated;'
                f'border:1px solid rgba(255,255,255,.15);border-radius:4px">'
                f'<div style="font-size:12px;font-variant-numeric:tabular-nums;'
                f'margin-top:4px">{note}</div></div>'
                for lab, path, note in cells)
            body.append(f'<div style="font-size:12px;opacity:.6;margin:10px 0 2px">frame {frame}</div>'
                        f'<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start;'
                        f'overflow-x:auto">{tds}</div>')

    header = ['씬'] + labels
    table = ('<table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in header) + '</tr></thead><tbody>'
             + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows)
             + '</tbody></table>')
    overall = []
    for k, lab in enumerate(labels):
        allv = [float(np.median(m['variants'][k]['ssim'])) for m in metas]
        overall.append(f'<b>{lab}</b> SSIM 중앙값 {float(np.median(allv)):.4f} (최악 씬 {min(allv):.4f})')
    summary = (f'<p><b>목적</b>: 전 씬 이미지를 저장하기 전에, 확정한 렌더 설정이 GT와 닮았는지 '
               f'표본 {len(metas)}씬 × {args.n_frames}프레임으로 눈으로 확인한다. '
               f'화살표(&lsaquo; &rsaquo;)로 GT와 렌더를 겹쳐 본다.</p>'
               f'<p>{" · ".join(overall)}</p>{table}')
    return viz_utils.save_gallery(out_dir, 'report.html',
                                  f'{SCRIPT_NAME} — {args.dataset} 표본 {len(metas)}씬',
                                  summary, ''.join(body))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['vln_n1', 'vln_pe'], default='vln_pe')
    ap.add_argument('--scenes', default=None, help='씬을 직접 지정 (콤마). 주면 --n_scenes 무시')
    ap.add_argument('--n_scenes', type=int, default=6, help='표본 씬 수. 0이면 전수')
    ap.add_argument('--seed', type=int, default=20260908, help='표본 추출 시드(재현용)')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--n_frames', type=int, default=4, help='씬당 GT에서 균등 표본할 프레임 수')
    ap.add_argument('--preset', choices=['none', 'measured'], default='measured',
                    help="'measured'=126씬 측정으로 확정한 설정 / 'none'=아래 인자를 그대로 사용")
    ap.add_argument('--both_settings', action='store_true',
                    help='기존 설정과 확정 설정을 둘 다 렌더해 같은 프레임에서 비교(씬당 렌더 2회)')
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=None, help='생략 시 데이터셋 권장값')
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--tonemap_op', type=int, default=None)
    ap.add_argument('--tonemap_crush', type=float, default=None)
    ap.add_argument('--warmup', type=int, default=20, help='설정 변경 후 반영을 기다리는 스텝 수')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_pe')
    ap.add_argument('--data_root', default=None, help='생략 시 --dataset의 기본 경로')
    ap.add_argument('--per_scene_timeout', type=int, default=1800, help='씬당 하위 프로세스 제한 시간(초)')
    ap.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                    help='이미 만들어진 씬 결과도 다시 렌더한다')
    ap.add_argument('--log_dir', default='logs/gs-vlnpe')
    # 아래 두 개는 내부용 — 사용자가 직접 줄 필요 없다. driver가 씬마다 자기 자신을 부를 때 쓴다.
    ap.add_argument('--_worker_scene', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--_out_dir', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.data_root is None:
        import importlib
        sys.path.insert(0, str(HERE))
        args.data_root = importlib.import_module('dataset_utils').default_data_root(args.dataset)

    # ---- worker: 씬 하나만 렌더하고 결과를 디스크에 남기고 끝난다 (Isaac을 여기서만 띄운다) ----
    if args._worker_scene:
        r4 = load_renderer_module()
        if args.rtx_ambient is None:
            args.rtx_ambient = r4.MEASURED_RENDER_PRESET[args.dataset]['rtx_ambient']
        out_dir = Path(args._out_dir)
        res = render_scene(r4, args, args._worker_scene, settings_variants(r4, args))
        save_scene_result(res, out_dir, args.dataset)
        return 0

    # ---- driver: 씬 목록을 정하고 씬마다 worker를 띄운 뒤 리포트 하나로 합친다 ----
    # **driver는 Isaac을 띄우지 않는다** — 띄우면 worker와 GPU/prim이 겹친다.
    if args.rtx_ambient is None:
        args.rtx_ambient = MEASURED_AMBIENT[args.dataset]
    scenes = ([x for x in args.scenes.split(',') if x] if args.scenes
              else pick_scenes(Path(args.mesh_root), args.dataset, args.n_scenes, args.seed, args.data_root))
    out_dir = Path(args.log_dir) / SCRIPT_NAME / f'{args.dataset}_{len(scenes)}scenes'
    n_settings = 2 if args.both_settings else 1
    print(f'[{SCRIPT_NAME}] dataset={args.dataset} 씬 {len(scenes)}개 · 설정 {n_settings}종 '
          f'· 프레임 {args.n_frames}장 · 씬당 프로세스 1개', flush=True)
    print(f'[{SCRIPT_NAME}] 씬: {", ".join(scenes)}', flush=True)

    metas = run_workers(args, scenes, out_dir)
    if not metas:
        print(f'[{SCRIPT_NAME}] 성공한 씬이 없다 — {out_dir / "logs"} 의 로그를 볼 것', file=sys.stderr)
        return 2
    report = build_report(metas, args, out_dir)
    print(f'[{SCRIPT_NAME}] 성공 {len(metas)}/{len(scenes)}씬', flush=True)
    print(f'[{SCRIPT_NAME}] report -> {report}', flush=True)
    return 0


if __name__ == '__main__':
    import os
    import traceback

    try:
        code = main()
    except BaseException:  # noqa: BLE001  트레이스백을 반드시 남기고 나서 끊는다
        traceback.print_exc()
        code = 3
    # `simulation_app.close()`가 수 분 걸리거나 atexit에서 세그폴트가 나므로 즉시 끊는다.
    # report.html은 main() 반환 시점에 이미 저장 완료다.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
