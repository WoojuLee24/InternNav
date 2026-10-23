"""05x (참고용, vln_ce 생산 경로 아님 — vln_ce 는 habitat 으로 렌더한다: 05a) — vln_ce GT pose 를 Isaac(gs_vlnpe 05_export_vlnce)으로 다시 그린 결과를 **원본 데이터셋 이미지**와
**이전 렌더(기준선)**에 프레임 단위로 대조한다. 계단(층 이동) 에피소드를 따로 본다.

    GT 원본 (habitat 렌더, vln_ce)  ──┐
    새 렌더 (Isaac, --render_root)  ──┼─> 프레임별 SSIM · depth 오차 -> 표 · 그림 · report.html
    이전 렌더 (--baseline_root)     ──┘   (같은 에피소드만: 렌더러가 그대로인지 회귀 확인)

## 용어
- **SSIM**: 두 영상의 구조 유사도(0~1, 1 = 같음). 회색조, 프레임 전체.
- **depth 오차**: 두 depth 가 모두 유효(>0)한 픽셀의 |차| 중앙값 [m]. 프레임마다 한 값.
- **계단 에피소드**: nav3d 02 의 `multifloor`(GT 시작·끝 높이 차 > 0.5 m).
- **정합 잔차**: vln_ce GT pose 는 에피소드 시작 기준 상대좌표라, depth 를 mesh 에 강체 정합해 절대좌표로 바꾼다
  (gs_vlnpe `vlnce_align`). 그 잔차(점→mesh 거리 중앙값).

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 | 근거 |
|---|---|---|---|
| C1 구조 | 모든 rig·에피소드의 rgb·depth 프레임 수 == GT | 전부 일치 | 하나라도 다르면 학습 로더가 어긋난다 |
| C2 RGB | rig 마다 SSIM 중앙값 | ≥ 0.75 | 이전 vln_ce 렌더 0.79~0.88, vln_pe 전수 0.83(≥ 0.75 가 56/61 씬) |
| C3 계단 RGB | 계단 에피소드 SSIM 중앙값 − 평지 | ≥ −0.05 | 계단이라고 더 나빠지면 안 된다 |
| C4 depth | 프레임 depth 오차 ≤ 0.05 m 인 비율 | ≥ 95 % | 같은 mesh 를 그리므로 정합·래스터 차이만 남아야 한다 |
| C5 정합 | 에피소드 정합 잔차 | 계단 포함 전부 ≤ 0.02 m | 이전 전수 중앙 0.0004 m. 2 cm 넘으면 pose 가 어긋난 것 |
| C6 회귀 | 이전 렌더와 같은 에피소드의 SSIM 중앙값 | ≥ 0.98 | 같은 코드·설정이면 렌더러 비결정성(평균 0.8/255)만 남는다 |

사용 (isaac 또는 habitat 컨테이너, 저장소 루트 — 이미지만 읽는다):
    python scripts/dataset_converters/nav3d/checks/05x_vlnce_isaac_render_check.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/ (이 파일은 nav3d/checks/ 에 있다)
sys.path.insert(0, str(HERE))
import nav3d_report as R  # noqa: E402

RIGS = ['125cm_0deg', '125cm_30deg', '125cm_45deg', '60cm_15deg', '60cm_30deg']
VIS_RIG = '125cm_30deg'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--render_root', default='data/InternData-N1-v0.5-mini/vln_ce_render_v2/traj_data/r2r')
    ap.add_argument('--baseline_root', default='data/InternData-N1-v0.5-mini/vln_ce_render/traj_data/r2r',
                    help='이전 렌더 (회귀 비교). "none" 이면 끔')
    ap.add_argument('--episodes_json', default='scripts/dataset_converters/nav3d/logs/05_vlnce_gt_render/episodes.json')
    ap.add_argument('--sentinel_dir', default='scripts/dataset_converters/nav3d/logs/05_vlnce_gt_render/05_export_vlnce/sentinels')
    ap.add_argument('--nav02_dir', default='scripts/dataset_converters/nav3d/logs/02_vlnce_gt_compare/scenes')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/05_vlnce_gt_render/check')
    ap.add_argument('--depth_tol_m', type=float, default=0.05)
    return ap.parse_args()


def frame_paths(root: Path, scene: str, kind: str, rig: str, ep: int) -> list:
    d = root / scene / 'videos' / 'chunk-000' / f'observation.images.{kind}.{rig}'
    ext = 'jpg' if kind == 'rgb' else 'png'
    fs = list(d.glob(f'episode_{ep:06d}_*.{ext}'))
    return sorted(fs, key=lambda p: int(p.stem.rsplit('_', 1)[1]))


def load_rgb_gray(p):
    import cv2
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)


def load_depth_m(p):
    import cv2
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=255))


def depth_err(a, b):
    v = (a > 0) & (b > 0)
    return float(np.median(np.abs(a[v] - b[v]))) if v.any() else float('nan'), float(v.mean())


def main():
    args = parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt, rd = Path(args.gt_root), Path(args.render_root)
    bl = None if args.baseline_root == 'none' else Path(args.baseline_root)
    sel = json.loads(Path(args.episodes_json).read_text())
    mf, zgt = {}, {}
    for scene in sel:
        j = Path(args.nav02_dir) / f'{scene}.json'
        if j.exists():
            for e in json.load(open(j))['episodes']:
                mf[(scene, e['ep'])] = (bool(e['multifloor']), float(e['gt_dz_m']))
        z = Path(args.nav02_dir) / f'{scene}_traj.npz'
        if z.exists():
            Z = np.load(z)
            for ep in sel[scene]:
                if f'{ep}_gt' in Z:
                    zgt[(scene, ep)] = Z[f'{ep}_gt'][:, 2]

    rows, frames, reg = [], [], []
    for scene, eps in sel.items():
        for ep in eps:
            is_mf, dz = mf.get((scene, ep), (False, 0.0))
            for rig in RIGS:
                g_rgb, r_rgb = frame_paths(gt, scene, 'rgb', rig, ep), frame_paths(rd, scene, 'rgb', rig, ep)
                g_d, r_d = frame_paths(gt, scene, 'depth', rig, ep), frame_paths(rd, scene, 'depth', rig, ep)
                n_ok = len(g_rgb) == len(r_rgb) == len(g_d) == len(r_d) and len(g_rgb) > 0
                ss, de = [], []
                for i, (a, b, c, d) in enumerate(zip(g_rgb, r_rgb, g_d, r_d)):
                    s_ = ssim(load_rgb_gray(a), load_rgb_gray(b))
                    e_, vf = depth_err(load_depth_m(c), load_depth_m(d))
                    ss.append(s_)
                    de.append(e_)
                    frames.append({'scene': scene, 'ep': ep, 'rig': rig, 'i': i, 'mf': is_mf, 'ssim': s_, 'depth_err_m': e_, 'valid_frac': vf})
                rows.append({'scene': scene, 'ep': ep, 'rig': rig, 'mf': is_mf, 'dz_m': round(dz, 2), 'n_gt': len(g_rgb), 'n_render': len(r_rgb),
                             'struct_ok': n_ok, 'ssim_med': float(np.median(ss)) if ss else float('nan'),
                             'depth_err_med_m': float(np.nanmedian(de)) if de else float('nan')})
                if bl is not None:
                    b_rgb = frame_paths(bl, scene, 'rgb', rig, ep)
                    b_d = frame_paths(bl, scene, 'depth', rig, ep)
                    if b_rgb and len(b_rgb) == len(r_rgb):
                        bs = [ssim(load_rgb_gray(a), load_rgb_gray(b)) for a, b in zip(b_rgb, r_rgb)]
                        dd = [float(np.mean(load_depth_m(a) == load_depth_m(b))) for a, b in zip(b_d, r_d)]
                        reg.append({'scene': scene, 'ep': ep, 'rig': rig, 'ssim_med': float(np.median(bs)), 'ssim_min': float(np.min(bs)),
                                    'depth_identical_frac': float(np.mean(dd))})
            print(f'  {scene} ep{ep} {"STAIRS" if is_mf else "flat"} done', flush=True)

    # 정합 잔차 (05_export_vlnce sentinel)
    align = []
    for f in sorted(Path(args.sentinel_dir).glob('*.json')):
        d = json.load(open(f))
        for pe in d.get('per_episode', []):
            align.append({'scene': d['scene'], 'rig': d['rig'], 'ep': pe['episode'], 'mf': mf.get((d['scene'], pe['episode']), (False, 0))[0],
                          'align_residual_m': pe['align_residual_m']})

    F = frames
    fs = np.array([f['ssim'] for f in F])
    fmf = np.array([f['mf'] for f in F])
    fde = np.array([f['depth_err_m'] for f in F])
    by_rig = []
    for rig in RIGS:
        k = np.array([f['rig'] == rig for f in F])
        by_rig.append({'rig': rig, '프레임': int(k.sum()), 'SSIM 중앙 (전체)': f'{np.median(fs[k]):.3f}',
                       'SSIM 중앙 (평지)': f'{np.median(fs[k & ~fmf]):.3f}' if (k & ~fmf).any() else '-',
                       'SSIM 중앙 (계단)': f'{np.median(fs[k & fmf]):.3f}' if (k & fmf).any() else '-',
                       'SSIM p10': f'{np.percentile(fs[k], 10):.3f}',
                       'depth 오차 ≤ tol': f'{100 * np.mean(fde[k] <= args.depth_tol_m):.1f} %',
                       'depth 오차 중앙 [m]': f'{np.nanmedian(fde[k]):.4f}'})

    # 판정
    c1 = all(r['struct_ok'] for r in rows)
    c2_vals = {rig: float(np.median(fs[np.array([f['rig'] == rig for f in F])])) for rig in RIGS}
    c2 = all(v >= 0.75 for v in c2_vals.values())
    d_mf = float(np.median(fs[fmf]) - np.median(fs[~fmf])) if fmf.any() and (~fmf).any() else float('nan')
    c3 = bool(np.isfinite(d_mf) and d_mf >= -0.05)
    c4_frac = float(np.mean(fde <= args.depth_tol_m))
    c4 = c4_frac >= 0.95
    ar = np.array([a['align_residual_m'] for a in align]) if align else np.array([np.nan])
    c5 = bool(np.all(ar <= 0.02))
    reg_med = float(np.median([r['ssim_med'] for r in reg])) if reg else float('nan')
    c6 = bool(reg and reg_med >= 0.98)
    gates = [
        {'검증': 'C1 구조 (프레임 수 == GT)', '결과': R.pill(c1), '값': f"{sum(r['struct_ok'] for r in rows)}/{len(rows)} (rig×에피소드)", '기준': '전부 일치'},
        {'검증': 'C2 RGB SSIM (rig 별 중앙)', '결과': R.pill(c2), '값': ' · '.join(f'{k} {v:.3f}' for k, v in c2_vals.items()), '기준': '≥ 0.75'},
        {'검증': 'C3 계단 − 평지 SSIM', '결과': R.pill(c3), '값': f'{d_mf:+.3f} (계단 {np.median(fs[fmf]):.3f} / 평지 {np.median(fs[~fmf]):.3f})', '기준': '≥ −0.05'},
        {'검증': 'C4 depth 오차 ≤ 0.05 m 프레임', '결과': R.pill(c4), '값': f'{100 * c4_frac:.1f} % (중앙 {np.nanmedian(fde):.4f} m)', '기준': '≥ 95 %'},
        {'검증': 'C5 정합 잔차', '결과': R.pill(c5), '값': f'중앙 {np.nanmedian(ar):.4f} · 최대 {np.nanmax(ar):.4f} m (계단 최대 '
         f'{max([a["align_residual_m"] for a in align if a["mf"]] or [float("nan")]):.4f})', '기준': '전부 ≤ 0.02 m'},
        {'검증': 'C6 회귀 (이전 렌더 대비 SSIM)', '결과': R.pill(c6) if reg else R.pill('기준선 없음'),
         'value_': '', '값': f'중앙 {reg_med:.4f} · 최소 {min([r["ssim_min"] for r in reg] or [float("nan")]):.4f} · depth 동일 픽셀 '
         f'{100 * np.mean([r["depth_identical_frac"] for r in reg]):.1f} % ({len(reg)} rig×에피소드)' if reg else '-', '기준': '≥ 0.98'},
    ]

    figs = make_figures(out, args, F, rows, sel, mf, zgt, gt, rd, bl)
    summary = {'gates': {g['검증']: g['값'] for g in gates}, 'pass': [c1, c2, c3, c4, c5, c6], 'by_rig': by_rig, 'episodes': rows,
               'regression': reg, 'align': align}
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False, default=float)

    ep_rows = []
    for scene, eps in sel.items():
        for ep in eps:
            rr = [r for r in rows if r['scene'] == scene and r['ep'] == ep]
            al = [a['align_residual_m'] for a in align if a['scene'] == scene and a['ep'] == ep]
            ep_rows.append({'scene': scene, 'ep': ep, '종류': f"계단 ({rr[0]['dz_m']:+.1f} m)" if rr[0]['mf'] else '평지', '프레임': rr[0]['n_gt'],
                            'SSIM 중앙 (5 rig 평균)': f"{np.mean([r['ssim_med'] for r in rr]):.3f}",
                            'depth 오차 중앙 [m]': f"{np.nanmedian([r['depth_err_med_m'] for r in rr]):.4f}",
                            '정합 잔차 [m]': f'{np.median(al):.4f}' if al else '-'})
    n_ep = len(ep_rows)
    n_mf = sum(1 for r in ep_rows if r['종류'] != '평지')
    body = R.header('nav3d 05a · vln_ce GT 재렌더 검증', 'vln_ce GT 경로 재렌더 — 원본·이전 렌더 대조',
                    f'vln_ce GT 카메라 자세를 그대로 재생해 Isaac 으로 다시 그린 이미지(gs_vlnpe <span class="mono">05_export_vlnce</span>)를 '
                    f'원본 데이터셋 이미지와 프레임마다 비교한다. 씬 {len(sel)} · 에피소드 {n_ep} (계단 {n_mf}) · rig 5 · 프레임 {len(F):,}. '
                    f'이 단계가 통과해야 같은 렌더러로 생성 경로를 그릴 수 있다.')
    body += R.section('판정', '<p class="muted">기준은 결과를 보기 전에 정했다(스크립트 docstring 표). SSIM = 구조 유사도(1 = 같음), '
                      'depth 오차 = 두 depth 모두 유효한 픽셀의 |차| 중앙값.</p>' + R.table(gates, ['검증', '결과', '값', '기준'], raw=('결과',)))
    body += R.section('rig 별', R.table(by_rig, list(by_rig[0].keys())))
    body += R.section('에피소드 별', R.table(ep_rows, list(ep_rows[0].keys())) + R.figure(figs['dist'], 'rig 별 SSIM 분포 — 평지(회색) vs 계단(주황). 점선 = 기준 0.75.'))
    stairs_html = ''.join(R.figure(f, c) for f, c in figs['stairs'])
    body += R.section('계단 에피소드 — 눈으로 확인', '<p class="muted">rig 125cm_30deg. 위: GT 발 높이(검정)와 프레임별 SSIM(주황). '
                      '아래: 계단 구간에서 고른 프레임 — GT rgb / 새 렌더 rgb / GT depth / 새 렌더 depth.</p>' + stairs_html)
    body += R.section('SSIM 이 가장 낮은 프레임', R.figure(figs['worst'], 'SSIM 하위 8 프레임 (모든 rig). 왼쪽 GT, 가운데 새 렌더, 오른쪽 |차|×3.'))
    if figs.get('reg'):
        body += R.section('이전 렌더와 회귀 비교', R.table([{k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()} for r in reg[:30]],
                                                       ['scene', 'ep', 'rig', 'ssim_med', 'ssim_min', 'depth_identical_frac'])
                          + R.figure(figs['reg'], '같은 프레임: 이전 렌더 / 새 렌더 / |차|×10.'))
    body += R.section('입력', f'<p class="mono">GT {esc(args.gt_root)}<br>새 렌더 {esc(args.render_root)}<br>이전 렌더 {esc(args.baseline_root)}<br>'
                      f'에피소드 {esc(args.episodes_json)}</p>' + R.raw_json('summary', {k: v for k, v in summary.items() if k != 'episodes'}))
    R.write(out, 'vln_ce GT 재렌더 검증', body)
    print('gates:', [g['검증'].split()[0] + ' ' + ('PASS' if p else 'FAIL') for g, p in zip(gates, summary['pass'])])
    print(f'-> {out}/report.html')


esc = R.esc


def make_figures(out, args, F, rows, sel, mf, zgt, gt, rd, bl):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs = {}
    # 분포
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for j, rig in enumerate(RIGS):
        for k, (flag, col) in enumerate([(False, '#888888'), (True, '#e67e22')]):
            v = [f['ssim'] for f in F if f['rig'] == rig and f['mf'] == flag]
            if v:
                bp = ax.boxplot(v, positions=[j * 3 + k], widths=0.8, patch_artist=True, showfliers=False)
                for b in bp['boxes']:
                    b.set_facecolor(col)
    ax.axhline(0.75, color='k', ls=':', lw=1)
    ax.set_xticks([j * 3 + 0.5 for j in range(len(RIGS))])
    ax.set_xticklabels(RIGS)
    ax.set_ylabel('SSIM vs GT')
    ax.set_title('SSIM per rig: flat (gray) vs stairs (orange)')
    fig.tight_layout()
    fig.savefig(out / 'ssim_dist.png', dpi=90)
    plt.close(fig)
    figs['dist'] = 'ssim_dist.png'

    def rgb(p):
        return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)

    def dep(p):
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    # 계단 에피소드
    figs['stairs'] = []
    for scene, eps in sel.items():
        for ep in eps:
            if not mf.get((scene, ep), (False,))[0]:
                continue
            ff = [f for f in F if f['scene'] == scene and f['ep'] == ep and f['rig'] == VIS_RIG]
            if not ff:
                continue
            z = zgt.get((scene, ep))
            n = len(ff)
            if z is not None and len(z) == n:
                dzf = np.abs(np.gradient(z))
                on = np.where(dzf > 0.02)[0]
            else:
                on = np.array([], int)
            pick = (np.linspace(on.min(), on.max(), 6).round().astype(int) if len(on) >= 6
                    else np.linspace(0, n - 1, 6).round().astype(int))
            g_rgb, r_rgb = frame_paths(gt, scene, 'rgb', VIS_RIG, ep), frame_paths(rd, scene, 'rgb', VIS_RIG, ep)
            g_d, r_d = frame_paths(gt, scene, 'depth', VIS_RIG, ep), frame_paths(rd, scene, 'depth', VIS_RIG, ep)
            fig = plt.figure(figsize=(18, 13))
            gsp = fig.add_gridspec(5, 6, height_ratios=[1.0, 1, 1, 1, 1])
            ax = fig.add_subplot(gsp[0, :])
            if z is not None and len(z) == n:
                ax.plot(range(n), z, 'k-', label='GT foot z [m]')
            ax2 = ax.twinx()
            ax2.plot(range(n), [f['ssim'] for f in ff], color='#e67e22', label='SSIM')
            ax2.set_ylim(0.4, 1.0)
            for p in pick:
                ax.axvline(p, color='#2b5fa8', lw=0.8, ls='--')
            ax.set_xlabel('frame')
            ax.set_ylabel('GT foot z [m]')
            ax2.set_ylabel('SSIM')
            ax.set_title(f'{scene} ep{ep} ({mf[(scene, ep)][1]:+.1f} m) — rig {VIS_RIG}; dashed = frames shown below')
            for c, p in enumerate(pick):
                for r_, (img, cm) in enumerate([(rgb(g_rgb[p]), None), (rgb(r_rgb[p]), None), (dep(g_d[p]), 'turbo'), (dep(r_d[p]), 'turbo')]):
                    a = fig.add_subplot(gsp[1 + r_, c])
                    a.imshow(img, cmap=cm, vmin=0 if cm else None, vmax=8 if cm else None)
                    a.set_xticks([])
                    a.set_yticks([])
                    if c == 0:
                        a.set_ylabel(['GT rgb', 'ours rgb', 'GT depth', 'ours depth'][r_])
                    if r_ == 0:
                        a.set_title(f'f{p} SSIM {ff[p]["ssim"]:.2f}', fontsize=9)
            fig.tight_layout()
            fn = f'stairs_{scene}_{ep}.png'
            fig.savefig(out / fn, dpi=70)
            plt.close(fig)
            figs['stairs'].append((fn, f'{scene} ep{ep}: 높이 {mf[(scene, ep)][1]:+.1f} m. 계단 구간 SSIM 중앙 '
                                       f'{np.median([ff[p]["ssim"] for p in pick]):.3f}.'))
    # 최악 프레임
    worst = sorted(F, key=lambda f: f['ssim'])[:8]
    fig, axs = plt.subplots(len(worst), 3, figsize=(13, 3.4 * len(worst)))
    for r_, f in enumerate(worst):
        a = rgb(frame_paths(gt, f['scene'], 'rgb', f['rig'], f['ep'])[f['i']])
        b = rgb(frame_paths(rd, f['scene'], 'rgb', f['rig'], f['ep'])[f['i']])
        for c, (img, t) in enumerate([(a, 'GT'), (b, 'ours'), (np.clip(np.abs(a.astype(int) - b.astype(int)) * 3, 0, 255).astype(np.uint8), '|diff|x3')]):
            axs[r_, c].imshow(img)
            axs[r_, c].axis('off')
            axs[r_, c].set_title(f"{t} — {f['scene']} ep{f['ep']} {f['rig']} f{f['i']} SSIM {f['ssim']:.2f}{' STAIRS' if f['mf'] else ''}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'worst.png', dpi=60)
    plt.close(fig)
    figs['worst'] = 'worst.png'
    # 회귀 예시
    if bl is not None:
        ex = [(s, e) for s, eps in sel.items() for e in eps if frame_paths(bl, s, 'rgb', VIS_RIG, e)][:3]
        if ex:
            fig, axs = plt.subplots(len(ex), 3, figsize=(13, 3.4 * len(ex)), squeeze=False)
            for r_, (s, e) in enumerate(ex):
                pb, pr = frame_paths(bl, s, 'rgb', VIS_RIG, e), frame_paths(rd, s, 'rgb', VIS_RIG, e)
                i = len(pb) // 2
                a, b = rgb(pb[i]), rgb(pr[i])
                for c, (img, t) in enumerate([(a, 'previous render'), (b, 'new render'),
                                              (np.clip(np.abs(a.astype(int) - b.astype(int)) * 10, 0, 255).astype(np.uint8), '|diff|x10')]):
                    axs[r_, c].imshow(img)
                    axs[r_, c].axis('off')
                    axs[r_, c].set_title(f'{t} — {s} ep{e} f{i}', fontsize=9)
            fig.tight_layout()
            fig.savefig(out / 'regression.png', dpi=60)
            plt.close(fig)
            figs['reg'] = 'regression.png'
    return figs


if __name__ == '__main__':
    main()
