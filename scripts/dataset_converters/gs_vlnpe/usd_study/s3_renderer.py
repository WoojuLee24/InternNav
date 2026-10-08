"""S3 — 렌더러/톤매퍼: Iray 노브가 왜 무반응이었나.

## 왜 이 스크립트가 있는가

`reports.md`에 실측 결론이 이렇게 적혀 있다:

    기각한 후보: Iray `crushBlacks`/`burnHighlights`는 RTX Real-Time에서 **무반응**

원인은 안 적혀 있다. S4의 carb 덤프로 확인한 사실이 후보 하나를 먼저 지웠다:

    /rtx/post/tonemap/op = 6   (raw · AppLauncher 둘 다 동일)
    /rtx/post/tonemap/irayReinhard/crushBlacks = 0.5
    /rtx/post/tonemap/irayReinhard/burnHighlights = 0.7

즉 톤맵 op은 이미 Iray(6)이고 두 부팅 경로에서 같다 — **"톤매퍼가 Iray가 아니라서"는 아니다.**
그런데 같은 톤맵 트리의 `filmIso`는 잘 먹는다(그게 유일하게 듣는 노브였다). 같은 트리 안에서
어떤 키는 먹고 어떤 키는 안 먹는 것이므로, 남은 후보는 이렇다:

  1. `irayReinhard/*` 하위 파라미터를 읽는 op 값이 6이 아닌 다른 값이다
  2. rendermode(RaytracedLighting = Real-Time)가 그 하위 파라미터를 무시하고, Path Tracing에서만 쓴다
  3. 카메라 annotator가 가져오는 이미지가 그 후처리 단계를 안 거친다

이 스크립트는 1과 2를 직접 스윕해서 가른다. **대조군을 반드시 같이 돌린다** — `filmIso`를
바꿔서 실제로 그림이 변하는지 확인하고, 그게 변하지 않으면 "설정 적용 자체가 안 되는 것"이므로
나머지 결론을 낼 수 없다(그 경우 3번이 유력해진다).

## 실험

| 그룹 | 바꾸는 것 |
|---|---|
| `control` | `filmIso` 70 / 140 — **적용 메커니즘이 동작하는지 확인하는 대조군** |
| `crush` | `irayReinhard/crushBlacks` 0.0 / 0.5 / 1.0 |
| `burn` | `irayReinhard/burnHighlights` 0.0 / 0.7 / 2.0 |
| `op` | `tonemap/op` 0..8 × crushBlacks 0.0/1.0 — **어느 op가 하위 노브에 반응하나** |
| `rendermode` | `RaytracedLighting` vs `PathTracing` × crushBlacks 0.0/1.0 |

기존 `04d_tonemap_sweep.py`의 방식을 그대로 따른다 — **Isaac을 한 번만 띄우고** 설정 조합을
순회하며 같은 pose를 다시 렌더한다(`build_renderer` 1회 + `render_along` N회).

## 실행 (한 줄)

    timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s3_renderer.py --scene 17DRP5sb8fy --dataset vln_n1 --n_frames 2 --log_dir logs/gs-vlnpe/usd_study

Path Tracing은 프레임당 시간이 크게 늘 수 있어 프레임 수를 최소로 둔다(`--n_frames 2`).
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from usd_study_utils import (
    DEFAULT_LOG_DIR,
    DEFAULT_SCENE,
    DEFAULT_USD_ROOT,
    TABLE_CSS,
    PreservationGuard,
    esc,
    exit_skipping_isaac_teardown,
    glossary_html,
    glossary_note_html,
    mp3d_usd_variants,
    pill,
    preservation_section,
    stat_row_html,
    table_html,
)

import viz_utils  # noqa: E402

GS_VLNPE = Path(__file__).resolve().parent.parent

# 설정 경로는 `04d_tonemap_sweep.py`와 같은 값을 쓴다 (그 파일이 이미 실측으로 확정한 것).
TM_OP = '/rtx/post/tonemap/op'
TM_ISO = '/rtx/post/tonemap/filmIso'
TM_CRUSH = '/rtx/post/tonemap/irayReinhard/crushBlacks'
TM_BURN = '/rtx/post/tonemap/irayReinhard/burnHighlights'
RTX_AMBIENT = '/rtx/sceneDb/ambientLightIntensity'
RENDERMODE = '/rtx/rendermode'

BRIGHTNESS_EPS = 0.5  # 밝기 차가 이보다 작으면 "무반응"으로 본다


def load_renderer_module():
    """`04_render_obs_isaac.py`를 모듈로 로드 (import 시점에 SimulationApp 부팅).

    **pxr보다 먼저 호출해야 한다** — 부팅 전에 standalone pxr을 잡으면 Omniverse 스키마 확장이
    전부 import 실패한다(S5에서 실측).
    """
    path = GS_VLNPE / '04_render_obs_isaac.py'
    spec = importlib.util.spec_from_file_location('render_obs_isaac_reused_s3', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_configs(args):
    """(그룹, 라벨, 설정 dict) 목록. 첫 항목은 항상 baseline."""
    base = {TM_OP: 6, TM_ISO: args.film_iso, TM_CRUSH: 0.5, TM_BURN: 0.7, RENDERMODE: 'RaytracedLighting'}
    cfgs = [('baseline', 'baseline (op6, iso70, crush0.5, burn0.7, RT)', dict(base))]

    for iso in (140.0,):
        cfgs.append(('control', f'filmIso {iso:g}  <-- 대조군', dict(base, **{TM_ISO: iso})))
    for c in (0.0, 1.0):
        cfgs.append(('crush', f'crushBlacks {c:g}', dict(base, **{TM_CRUSH: c})))
    for b in (0.0, 2.0):
        cfgs.append(('burn', f'burnHighlights {b:g}', dict(base, **{TM_BURN: b})))
    for op in range(0, 9):
        for c in (0.0, 1.0):
            cfgs.append(('op', f'op {op} · crush {c:g}', dict(base, **{TM_OP: op, TM_CRUSH: c})))
    for mode in ('PathTracing',):
        for c in (0.0, 1.0):
            cfgs.append(('rendermode', f'{mode} · crush {c:g}', dict(base, **{RENDERMODE: mode, TM_CRUSH: c})))
    return cfgs


def apply_config(cfg: dict, ambient: float, warmup_steps: int, sim):
    import carb

    s = carb.settings.get_settings()
    s.set_float(RTX_AMBIENT, float(ambient))
    for key, val in cfg.items():
        if key == RENDERMODE:
            s.set_string(key, str(val))
        elif key == TM_OP:
            s.set_int(key, int(val))
        else:
            s.set_float(key, float(val))
    # 설정이 렌더 파이프라인에 반영되도록 여유 스텝을 준다 (rendermode 전환은 특히 느리다).
    for _ in range(warmup_steps):
        sim.step()


def main():
    ap = argparse.ArgumentParser(description='S3 — Iray 노브 무반응의 원인 규명')
    ap.add_argument('--scene', default=DEFAULT_SCENE)
    ap.add_argument('--dataset', choices=['vln_n1', 'vln_pe'], default='vln_n1')
    ap.add_argument('--usd_root', default=str(DEFAULT_USD_ROOT))
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--n_frames', type=int, default=2)
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=6.0)
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--warmup_steps', type=int, default=20)
    ap.add_argument('--groups', default='baseline,control,crush,burn,op,rendermode')
    ap.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR))
    ap.add_argument('--skip_hash', action='store_true')
    args = ap.parse_args()

    variants = dict(mp3d_usd_variants(args.usd_root, args.scene))
    base_usd = variants.get('isaacsim')
    if base_usd is None:
        print('[fail] isaacsim_<hash>.usd 없음', file=sys.stderr)
        return 2

    out_dir = Path(args.log_dir) / 's3_renderer' / f'{args.scene}_{args.dataset}'
    out_dir.mkdir(parents=True, exist_ok=True)

    guard = PreservationGuard([base_usd])
    if not args.skip_hash:
        guard.snapshot_before()

    print('[s3] Isaac 부팅')
    r4 = load_renderer_module()
    import cv2
    import dataset_utils
    from skimage.metrics import structural_similarity as ssim_metric

    data_root = args.data_root or dataset_utils.default_data_root(args.dataset)
    gt = dataset_utils.load_gt_episode(data_root, args.scene, args.episode, args.dataset)
    n = len(gt['poses_c2w'])
    frame_idx = np.unique(np.linspace(0, n - 1, args.n_frames).astype(int)).tolist()
    poses = [gt['poses_c2w'][f] for f in frame_idx]
    rgb_dir = Path(data_root) / args.scene / 'videos' / 'chunk-000' / 'observation.images.rgb'
    reals = [dataset_utils.load_rgb_frame(rgb_dir, args.episode, f, args.dataset) for f in frame_idx]

    print('[s3] 씬 로드 (build_renderer 1회)')
    cas = r4.build_renderer(r4.RENDER_W, r4.RENDER_H, gt['k'], base_usd,
                            args.light, None, args.rtx_ambient, args.film_iso)
    sim = cas[1]

    keep = {g for g in args.groups.split(',') if g}
    cfgs = [(g, lbl, c) for g, lbl, c in build_configs(args) if g in keep or g == 'baseline']

    img_dir = out_dir / 'frames'
    img_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (group, label, cfg) in enumerate(cfgs):
        apply_config(cfg, args.rtx_ambient, args.warmup_steps, sim)
        t0 = time.time()
        rgb, _ = r4.render_along(cas, poses)
        elapsed = time.time() - t0
        ssims = [float(ssim_metric(reals[j], rgb[j], channel_axis=2, data_range=255)) for j in range(len(reals))]
        p = img_dir / f'{i:02d}_{group}.jpg'
        cv2.imwrite(str(p), cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR))
        rec = {'group': group, 'label': label, 'config': {k: str(v) for k, v in cfg.items()},
               'brightness': float(np.mean([x.mean() for x in rgb])),
               'ssim_median': float(np.median(ssims)), 'seconds': round(elapsed, 2),
               'sec_per_frame': round(elapsed / max(1, len(poses)), 2), 'image': str(p)}
        results.append(rec)
        print(f'[s3] {group:11s} {label:34s} 밝기 {rec["brightness"]:7.2f}  SSIM {rec["ssim_median"]:.4f}  '
              f'{rec["sec_per_frame"]:.2f}s/frame', flush=True)

    if not args.skip_hash:
        guard.snapshot_after()

    base = results[0]

    def delta(rec):
        return rec['brightness'] - base['brightness']

    # 판정: 대조군(filmIso)이 움직였는가 -> 설정 적용 메커니즘이 동작하는가
    control = [r for r in results if r['group'] == 'control']
    control_moved = any(abs(delta(r)) > BRIGHTNESS_EPS for r in control)
    crush_moved = any(abs(delta(r)) > BRIGHTNESS_EPS for r in results if r['group'] == 'crush')
    burn_moved = any(abs(delta(r)) > BRIGHTNESS_EPS for r in results if r['group'] == 'burn')
    # op별로 crush 0.0 vs 1.0 밝기차
    op_pairs = {}
    for r in results:
        if r['group'] != 'op':
            continue
        op = int(float(r['config'][TM_OP]))
        op_pairs.setdefault(op, {})[float(r['config'][TM_CRUSH])] = r['brightness']
    op_response = {op: (v.get(1.0, float('nan')) - v.get(0.0, float('nan'))) for op, v in sorted(op_pairs.items())}
    responsive_ops = [op for op, d in op_response.items() if abs(d) > BRIGHTNESS_EPS]
    pt = [r for r in results if r['group'] == 'rendermode']
    pt_response = None
    if len(pt) >= 2:
        by_crush = {float(r['config'][TM_CRUSH]): r['brightness'] for r in pt}
        pt_response = by_crush.get(1.0, float('nan')) - by_crush.get(0.0, float('nan'))

    (out_dir / 's3_renderer.json').write_text(json.dumps(
        {'args': vars(args), 'results': results, 'op_crush_response': op_response,
         'responsive_ops': responsive_ops, 'control_moved': control_moved,
         'crush_moved': crush_moved, 'burn_moved': burn_moved, 'pathtracing_crush_response': pt_response,
         'preservation': guard.rows()}, indent=2, default=str, ensure_ascii=False), encoding='utf-8')

    # ---------------- 리포트 ----------------
    parts = [TABLE_CSS, glossary_note_html()]
    parts.append('<h2>판정</h2>')
    parts.append(table_html(['질문', '답', '근거'], [
        ['설정 적용 메커니즘이 동작하나 (대조군 filmIso)', pill(control_moved, 'YES', 'NO — 아래 결론 무효'),
         f'밝기 변화 {max((abs(delta(r)) for r in control), default=0):.2f}'],
        ['crushBlacks가 반응하나 (op6 · Real-Time)', pill(not crush_moved, 'NO (무반응 재현)', 'YES — 기존 기록과 다름'),
         f'최대 밝기 변화 {max((abs(delta(r)) for r in results if r["group"]=="crush"), default=0):.2f}'],
        ['burnHighlights가 반응하나', pill(not burn_moved, 'NO (무반응 재현)', 'YES'),
         f'최대 밝기 변화 {max((abs(delta(r)) for r in results if r["group"]=="burn"), default=0):.2f}'],
        ['crush에 반응하는 op이 있나', pill(bool(responsive_ops), f'있음: {responsive_ops}', '없음'),
         'op 0..8 스윕'],
        ['Path Tracing에서는 반응하나', (pill(abs(pt_response) > BRIGHTNESS_EPS, 'YES', 'NO') if pt_response is not None and pt_response == pt_response else '—'),
         f'밝기차 {pt_response:.2f}' if pt_response is not None and pt_response == pt_response else '미측정'],
    ]))

    parts.append('<h2>전체 결과</h2>')
    rows, classes = [], []
    for r in results:
        d = delta(r)
        rows.append([r['group'], r['label'], f'{r["brightness"]:.2f}', f'{d:+.2f}',
                     f'{r["ssim_median"]:.4f}', f'{r["sec_per_frame"]:.2f}'])
        classes.append('differ' if abs(d) > BRIGHTNESS_EPS else '')
    parts.append(table_html(['그룹', '설정', '밝기', 'baseline 대비', 'SSIM', 's/frame'], rows, row_classes=classes))
    parts.append(f'<p>밝기 차가 ±{BRIGHTNESS_EPS} 이내면 <b>무반응</b>으로 본다. 강조된 행만 실제로 그림을 바꿨다.</p>')

    parts.append('<h2>op별 crushBlacks 반응 (0.0 -> 1.0 밝기차)</h2>')
    parts.append(table_html(['tonemap op', '밝기차', '반응'],
                            [[op, f'{d:+.2f}', pill(abs(d) > BRIGHTNESS_EPS, '반응', '무반응')]
                             for op, d in op_response.items()]))

    key_imgs = [(r['label'], Path(r['image'])) for r in results if r['group'] in ('baseline', 'control', 'crush', 'rendermode')]
    if len(key_imgs) >= 2:
        parts.append('<h2>같은 프레임 비교</h2>')
        parts.append(viz_utils.blink_widget_html('s3_blink', key_imgs, title='설정별 같은 pose'))

    parts.append(preservation_section(guard))
    parts.append(glossary_html())

    summary = stat_row_html([
        ('설정 조합', len(results)),
        ('대조군 반응', 'YES' if control_moved else 'NO'),
        ('crush 반응', 'YES' if crush_moved else 'NO (무반응 재현)'),
        ('반응하는 op', str(responsive_ops) if responsive_ops else '없음'),
        ('PathTracing s/frame', f'{max((r["sec_per_frame"] for r in pt), default=float("nan")):.2f}' if pt else '—'),
        ('원본 보존', 'ALL UNCHANGED' if guard.all_unchanged else 'MUTATION'),
    ])
    path = viz_utils.save_gallery(out_dir, 'report.html', f'S3 · 렌더러/톤매퍼 노브 반응 — {args.scene}',
                                  summary, ''.join(parts), eyebrow='gs_vlnpe usd_study')
    print(f'[s3] report: {path}')
    ok = guard.all_unchanged or args.skip_hash
    print(f'[s3] {"PASS" if ok else "FAIL"}')
    # Isaac 정리 단계에서 세그폴트가 나므로 건너뛰고 나간다 (이유는 헬퍼 docstring)
    exit_skipping_isaac_teardown(0 if ok else 1)


if __name__ == '__main__':
    sys.exit(main())
