"""S8 — 톤맵 op 7(Iray) 채택 검토. **GT 사진과 나란히 놓고 눈으로 확인한다.**

## 왜

S3에서 발견한 것: `/rtx/post/tonemap/op`을 현재 **6(ACES)**에서 **7(Iray)**로 바꾸면
SSIM이 **0.875 → 0.910**으로 올랐다(씬1 · vln_n1 · 18프레임). `260809` 메모리가
"어떤 렌더 설정을 써도 최대 +0.034, 남는 0.096은 메쉬/텍스처 충실도"라고 적어둔 상한을 넘는다.

기존 스윕이 op 6의 **하위 노브만** 훑고 **op 자체를 바꿔본 적이 없어서** 놓친 것이다.
그런데 씬 하나·데이터셋 하나에서만 봤으므로 아직 채택할 수 없다. 이 스크립트가 채택 판단에
필요한 것을 모은다.

## 채택 전 확인해야 할 4가지

1. **다른 씬·데이터셋에서 재현되나** — `--scene`/`--dataset`으로 각각 돌린다
2. **op 7에서는 `crushBlacks`가 살아난다** — 기본 0.5를 그대로 두면 SSIM이 **떨어진다**(0.705).
   그래서 crush를 같이 스윕한다. **이게 채택의 함정이다**
3. **`film_iso` 재튜닝** — op이 바뀌면 밝기 곡선이 바뀌므로 노출 최적점도 옮겨간다
4. **GT 밝기와의 정합** — SSIM만 보면 밝기가 GT에서 멀어지는 것을 놓친다

## 왜 op 8은 안 쓰나

톤맵 op 목록은 **0~7까지 8개뿐**이다
(`/isaac-sim/extscache/omni.rtx.settings.core-*/omni/rtx/settings/core/widgets/post_widgets.py:17-26`:
Clamp · Linear · Reinhard · Modified Reinhard · HejlHableAlu · HableUc2 · **Aces** · **Iray**).
S3에서 op 8이 op 7과 같은 값을 낸 것은 **범위를 벗어나 7로 처리된 것**으로 본다.

## 실행 (한 줄씩, 씬마다 따로)

    timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s8_tonemap_adopt.py --scene 17DRP5sb8fy --dataset vln_pe --n_frames 18 --log_dir logs/gs-vlnpe/usd_study

    timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s8_tonemap_adopt.py --scene s8pcmisQ38h --dataset vln_pe --n_frames 18 --log_dir logs/gs-vlnpe/usd_study

Isaac을 **한 번만** 띄우고 씬도 **한 번만** 올린 뒤 설정 조합만 순회한다(04d_tonemap_sweep와 같은 방식).
한 프로세스에서 씬을 3번 이상 올리면 죽으므로 씬은 실행당 하나다.

## 리포트에 들어가는 것

- 설정별 SSIM·밝기 표 (현재 설정을 기준으로 증감 표시)
- **op 6 vs op 7 의 iso 곡선** 꺾은선 차트
- **GT 사진 / 현재 설정 / op 7 최적** 3장을 번갈아 보는 위젯 ← 눈으로 확인하는 지점
"""

import argparse
import importlib.util
import json
import sys
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

TM_OP = '/rtx/post/tonemap/op'
TM_ISO = '/rtx/post/tonemap/filmIso'
TM_CRUSH = '/rtx/post/tonemap/irayReinhard/crushBlacks'
TM_BURN = '/rtx/post/tonemap/irayReinhard/burnHighlights'
RTX_AMBIENT = '/rtx/sceneDb/ambientLightIntensity'

OP_NAMES = {0: 'Clamp', 1: 'Linear', 2: 'Reinhard', 3: 'ModReinhard',
            4: 'HejlHableAlu', 5: 'HableUc2', 6: 'Aces', 7: 'Iray'}


def load_renderer_module():
    """`04_render_obs_isaac.py`를 모듈로 로드 (import 시점에 SimulationApp 부팅). pxr보다 먼저."""
    path = GS_VLNPE / '04_render_obs_isaac.py'
    spec = importlib.util.spec_from_file_location('r4_s8', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['r4_s8'] = mod
    spec.loader.exec_module(mod)
    return mod


def build_configs(args):
    """(그룹, 라벨, 설정) 목록. 첫 항목이 현재 설정(기준)."""
    isos = [float(x) for x in str(args.isos).split(',') if x]
    # **crush를 고정하면 안 된다(실측).** 텍스처를 붙인 뒤 재보니 iso 70에서
    # crush 0.5(0.6834)가 crush 0.0(0.6478)을 이겼다. crush를 0으로 못 박고 iso만
    # 훑으면 최적점을 격자 밖에 두고 재게 된다 — 그래서 두 축을 함께 훑는다.
    crushes = [float(x) for x in str(args.crushes).split(',') if x]
    base = {TM_OP: 6, TM_ISO: args.film_iso, TM_CRUSH: 0.5, TM_BURN: 0.7}
    cfgs = [('current', f'현재 설정 (op6 Aces · iso {args.film_iso:g} · crush 0.5)', dict(base))]
    for iso in isos:
        cfgs.append(('op6', f'op6 Aces · iso {iso:g}', dict(base, **{TM_OP: 6, TM_ISO: iso})))
    for iso in isos:
        for crush in crushes:
            cfgs.append(('op7', f'op7 Iray · iso {iso:g} · crush {crush:g}',
                         dict(base, **{TM_OP: 7, TM_ISO: iso, TM_CRUSH: crush})))
    # 예전에 "함정"이라 부른 설정 — 기준선으로만 남긴다(이제 격자 안에 crush 0.5가 있다)
    cfgs.append(('trap', f'op7 Iray · iso {args.film_iso:g} · crush 0.5 (op만 바꾼 경우)',
                 dict(base, **{TM_OP: 7, TM_CRUSH: 0.5})))
    # 차선책
    cfgs.append(('op1', f'op1 Linear · iso {args.film_iso:g}', dict(base, **{TM_OP: 1})))
    return cfgs


def apply_config(cfg: dict, ambient: float, warmup: int, sim):
    import carb

    s = carb.settings.get_settings()
    s.set_float(RTX_AMBIENT, float(ambient))
    s.set_int(TM_OP, int(cfg[TM_OP]))
    s.set_float(TM_ISO, float(cfg[TM_ISO]))
    s.set_float(TM_CRUSH, float(cfg[TM_CRUSH]))
    s.set_float(TM_BURN, float(cfg[TM_BURN]))
    for _ in range(warmup):
        sim.step()


def main():
    ap = argparse.ArgumentParser(description='S8 — 톤맵 op 7 채택 검토 (GT와 나란히 확인)')
    ap.add_argument('--scene', default=DEFAULT_SCENE)
    ap.add_argument('--dataset', choices=['vln_n1', 'vln_pe'], default='vln_pe')
    ap.add_argument('--usd_root', default=str(DEFAULT_USD_ROOT))
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--n_frames', type=int, default=18, help='기록된 측정과 같은 18을 기본으로')
    ap.add_argument('--isos', default='50,70,90,110', help='스윕할 filmIso 값 (콤마)')
    ap.add_argument('--film_iso', type=float, default=70.0, help='현재 설정의 iso (기준선)')
    ap.add_argument('--light', default='ambient_only')
    ap.add_argument('--rtx_ambient', type=float, default=10.0, help='vln_pe 권장 10.0 / vln_n1 6.0')
    ap.add_argument('--warmup', type=int, default=20, help='설정 변경 후 여유 스텝')
    ap.add_argument('--blink_frames', type=int, default=3, help='GT와 나란히 볼 프레임 수')
    ap.add_argument('--crushes', default='0.0,0.25,0.5,0.75',
                    help='op7의 irayReinhard/crushBlacks 후보. iso와 함께 2차원으로 훑는다')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_pe',
                    help='텍스처 심링크의 원본 폴더 — 04_render_obs_isaac.py의 DEFAULT_MESH_ROOT와 같은 값')
    ap.add_argument('--textures', choices=['symlink', 'raw'], default='symlink',
                    help='symlink=파이프라인과 동일(텍스처 붙음) / raw=심링크 없이 원본 USD 그대로')
    ap.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR))
    args = ap.parse_args()

    base_usd = dict(mp3d_usd_variants(args.usd_root, args.scene)).get('isaacsim')
    if base_usd is None:
        print('[fail] isaacsim_<hash>.usd 없음', file=sys.stderr)
        return 2
    out_dir = Path(args.log_dir) / 's8_tonemap_adopt' / f'{args.scene}_{args.dataset}'
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = PreservationGuard([base_usd]).snapshot_before()

    print(f'[s8] Isaac 부팅 · scene={args.scene} dataset={args.dataset}')
    r4 = load_renderer_module()
    # **이걸 빼먹으면 텍스처 없는 그림을 재게 된다.** 파이프라인은 `load_scene_model()`을
    # 통해 USD를 얻고, 그 안에서 이 심링크를 만들어 준다 — mp3d USD에 텍스처가
    # `/ssd/share/Matterport3D/...` 절대경로로 박혀 있어 심링크 없이는 resolve가 안 된다.
    # 이 스크립트는 USD 파일을 직접 열기 때문에 그 한 단계를 명시적으로 해 줘야 한다.
    if args.textures == 'symlink':
        r4.ensure_texture_symlink(args.mesh_root, args.scene)
    elif args.textures == 'raw':
        pass
    else:
        assert False, f'unreachable textures={args.textures!r}'
    import cv2
    import dataset_utils
    from skimage.metrics import structural_similarity as ssim_metric

    data_root = args.data_root or dataset_utils.default_data_root(args.dataset)
    gt = dataset_utils.load_gt_episode(data_root, args.scene, args.episode, args.dataset)
    W, H = dataset_utils.render_wh(args.dataset)
    n = len(gt['poses_c2w'])
    frame_idx = np.unique(np.linspace(0, n - 1, args.n_frames).astype(int)).tolist()
    poses = [gt['poses_c2w'][f] for f in frame_idx]
    rgb_dir = Path(data_root) / args.scene / 'videos' / 'chunk-000' / 'observation.images.rgb'
    reals = [dataset_utils.load_rgb_frame(rgb_dir, args.episode, f, args.dataset) for f in frame_idx]
    gt_bright = float(np.mean([r.mean() for r in reals]))
    print(f'[s8] GT {len(reals)}장 · GT 평균 밝기 {gt_bright:.2f}')

    cas = r4.build_renderer(W, H, gt['k'], base_usd, args.light, None, args.rtx_ambient, args.film_iso)
    sim = cas[1]

    img_dir = out_dir / 'frames'
    img_dir.mkdir(parents=True, exist_ok=True)
    results, kept = [], {}
    for group, label, cfg in build_configs(args):
        apply_config(cfg, args.rtx_ambient, args.warmup, sim)
        rgb, _ = r4.render_along(cas, poses)
        ssims = [float(ssim_metric(reals[i], rgb[i], channel_axis=2, data_range=255)) for i in range(len(reals))]
        rec = {'group': group, 'label': label, 'op': int(cfg[TM_OP]), 'op_name': OP_NAMES.get(int(cfg[TM_OP]), '?'),
               'iso': float(cfg[TM_ISO]), 'crush': float(cfg[TM_CRUSH]),
               'ssim_median': float(np.median(ssims)), 'ssim_min': float(np.min(ssims)),
               'brightness': float(np.mean([x.mean() for x in rgb]))}
        rec['brightness_gap'] = rec['brightness'] - gt_bright
        results.append(rec)
        kept[label] = rgb
        print(f"[s8] {label:52s} SSIM {rec['ssim_median']:.4f} (min {rec['ssim_min']:.4f})  "
              f"밝기 {rec['brightness']:7.2f} ({rec['brightness_gap']:+.2f})", flush=True)

    guard.snapshot_after()
    cur = results[0]
    best6 = max((r for r in results if r['group'] in ('current', 'op6')), key=lambda r: r['ssim_median'])
    best7 = max((r for r in results if r['group'] == 'op7'), key=lambda r: r['ssim_median'])
    trap = next((r for r in results if r['group'] == 'trap'), None)
    gain = best7['ssim_median'] - cur['ssim_median']

    # ── GT / 현재 / op7최적 나란히 보기
    blink = []
    for i, f in enumerate(frame_idx[:args.blink_frames]):
        p = img_dir / f'gt_f{f:04d}.jpg'
        cv2.imwrite(str(p), cv2.cvtColor(reals[i], cv2.COLOR_RGB2BGR))
        states = [(f'GT 실제 사진 · frame {f}', p)]
        for tag, rec in (('현재', cur), ('op7 최적', best7)):
            q = img_dir / f'{tag}_f{f:04d}.jpg'
            cv2.imwrite(str(q), cv2.cvtColor(kept[rec['label']][i], cv2.COLOR_RGB2BGR))
            states.append((f'{tag} — {rec["label"]}', q))
        blink.append(states)

    # ── iso 곡선
    chart = None
    xs6 = [r['iso'] for r in results if r['group'] == 'op6']
    ys6 = [r['ssim_median'] for r in results if r['group'] == 'op6']
    xs7 = [r['iso'] for r in results if r['group'] == 'op7']
    ys7 = [r['ssim_median'] for r in results if r['group'] == 'op7']
    if xs6 and xs7:
        chart = img_dir / 'iso_curve.jpg'
        viz_utils.line_chart([('op6 Aces', (90, 190, 255), xs6, ys6),
                              ('op7 Iray crush0', (0, 255, 90), xs7, ys7)],
                             chart, x_label='filmIso', y_label='SSIM median')

    (out_dir / 's8_tonemap.json').write_text(json.dumps(
        {'args': vars(args), 'gt_brightness': gt_bright, 'results': results,
         'current': cur, 'best_op6': best6, 'best_op7': best7, 'trap': trap,
         'gain_vs_current': gain, 'preservation': guard.rows()}, indent=2, default=str,
        ensure_ascii=False), encoding='utf-8')

    # ── 리포트
    parts = [TABLE_CSS, glossary_note_html()]
    parts.append('<h2>목적</h2>')
    parts.append(
        '<p>톤맵 설정 번호(<code>/rtx/post/tonemap/op</code>)를 현재 <b>6 (Aces)</b>에서 '
        '<b>7 (Iray)</b>로 바꾸면 렌더가 GT 사진에 더 가까워지는지 본다. '
        'S3에서 씬 하나·데이터셋 하나로 <b>+0.035</b>를 봤는데, 채택하려면 다른 씬에서도 '
        '재현되는지와 <b>같이 바꿔야 하는 값</b>이 무엇인지를 확인해야 한다.</p>'
        '<p><b>함정</b>: op 7에서는 <code>crushBlacks</code>가 <b>실제로 동작한다</b>(op 6에서는 무반응). '
        '기본값 0.5를 그대로 두고 op만 바꾸면 오히려 나빠진다 — 아래 <code>trap</code> 행이 그것이다.</p>')

    parts.append('<h2>커맨드와 결과</h2>')
    parts.append(f'<pre>/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/'
                 f's8_tonemap_adopt.py --scene {esc(args.scene)} --dataset {esc(args.dataset)} '
                 f'--n_frames {args.n_frames} --rtx_ambient {args.rtx_ambient:g} --log_dir {esc(args.log_dir)}</pre>')
    parts.append(stat_row_html([
        ('scene', args.scene), ('dataset', args.dataset), ('프레임', len(reals)),
        ('GT 밝기', f'{gt_bright:.2f}'),
        ('현재 SSIM', f"{cur['ssim_median']:.4f}"),
        ('op7 최적 SSIM', f"{best7['ssim_median']:.4f}"),
        ('증감', f'{gain:+.4f}'),
    ]))

    rows, classes = [], []
    for r in results:
        d = r['ssim_median'] - cur['ssim_median']
        rows.append([r['label'], f"{r['ssim_median']:.4f}", f"{r['ssim_min']:.4f}",
                     '(기준)' if r['group'] == 'current' else f'{d:+.4f}',
                     f"{r['brightness']:.2f}", f"{r['brightness_gap']:+.2f}"])
        classes.append('differ' if r['group'] != 'current' and abs(d) >= 0.005 else '')
    parts.append(table_html(['설정', 'SSIM median', 'SSIM min', '현재 대비', '밝기', 'GT와 밝기차'],
                            rows, row_classes=classes))
    parts.append('<p style="color:var(--text-dim);font-size:.85rem">SSIM은 1에 가까울수록 GT와 닮은 것. '
                 '<code>GT와 밝기차</code>는 0에 가까울수록 좋다 — SSIM만 보면 밝기가 GT에서 '
                 '멀어지는 것을 놓친다.</p>')

    if chart:
        parts.append('<h3>filmIso 곡선 — op6 vs op7</h3>')
        parts.append(f'<img src="{viz_utils.image_to_data_uri(chart)}" style="max-width:100%">')

    parts.append('<h3>GT 사진과 나란히 보기 — 화살표로 번갈아</h3>')
    for i, states in enumerate(blink):
        parts.append(viz_utils.blink_widget_html(f's8_blink_{i}', states, title=f'frame {frame_idx[i]}'))

    parts.append('<h2>Takeaway</h2>')
    tk = []
    ok = gain > 0.01
    tk.append((f'<b>op 7이 이 씬에서 {"재현된다" if ok else "재현되지 않는다"}</b>',
               f"현재 <b>{cur['ssim_median']:.4f}</b> → op7 최적 <b>{best7['ssim_median']:.4f}</b> "
               f"(<b>{gain:+.4f}</b>). op6 안에서의 최적은 {best6['ssim_median']:.4f}이므로, "
               f"iso를 아무리 맞춰도 op6으로는 여기까지다."))
    if trap:
        td = trap['ssim_median'] - cur['ssim_median']
        tk.append(('<b>함정 — crush를 안 만지고 op만 바꾸면 나빠진다</b>',
                   f"op7 · crush 0.5(기본값) = <b>{trap['ssim_median']:.4f}</b> ({td:+.4f}). "
                   f"crush 0.0으로 낮춰야 <b>{best7['ssim_median']:.4f}</b>가 된다. "
                   'op 6에서는 이 값이 무반응이라 아무 영향이 없었고, op 7에서 살아난다.'))
    tk.append(('<b>새 op에서의 노출 최적점</b>',
               f"op7의 최적은 <b>iso {best7['iso']:g}</b>이고 그때 GT와 밝기차 "
               f"<b>{best7['brightness_gap']:+.2f}</b>다(현재는 {cur['brightness_gap']:+.2f}). "
               '밝기차가 커졌다면 SSIM이 올라도 채택을 다시 생각해야 한다.'))
    tk.append(('<b>채택 조건</b>',
               '이 리포트를 <b>씬2·다른 데이터셋에서도</b> 만들어 같은 방향이 나오는지 확인한 뒤에 '
               '기본값을 바꾼다. 한 씬 결과로 파이프라인 기본값을 바꾸지 않는다.'))
    parts.append('<div style="display:flex;flex-direction:column;gap:14px;margin:12px 0 22px">')
    for head, body in tk:
        parts.append('<div style="border:1px solid var(--border);border-radius:10px;background:var(--surface);'
                     f'padding:12px 14px"><div style="margin-bottom:6px">{head}</div>'
                     f'<div style="font-size:.88rem;line-height:1.6;color:var(--text-dim)">{body}</div></div>')
    parts.append('</div>')

    parts.append('<h2>원자료</h2>')
    parts.append(preservation_section(guard))
    parts.append(glossary_html())

    summary = stat_row_html([
        ('현재', f"{cur['ssim_median']:.4f}"), ('op7 최적', f"{best7['ssim_median']:.4f}"),
        ('증감', f'{gain:+.4f}'), ('op7 최적 iso', f"{best7['iso']:g}"),
        ('함정(crush 0.5)', f"{trap['ssim_median']:.4f}" if trap else '—'),
        ('원본 보존', 'ALL UNCHANGED' if guard.all_unchanged else 'MUTATION'),
    ])
    path = viz_utils.save_gallery(out_dir, 'report.html',
                                  f'S8 · 톤맵 op7 채택 검토 — {args.scene} / {args.dataset}',
                                  summary, ''.join(parts), eyebrow='gs_vlnpe usd_study')
    print(f'[s8] report: {path}')
    print(f"[s8] 현재 {cur['ssim_median']:.4f} -> op7 최적 {best7['ssim_median']:.4f} ({gain:+.4f})")
    exit_skipping_isaac_teardown(0 if guard.all_unchanged else 1)


if __name__ == '__main__':
    try:
        main()
    finally:
        exit_skipping_isaac_teardown(0)
