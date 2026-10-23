"""05a — vln_ce GT 를 **habitat** 으로 다시 렌더해 원본(릴리스) 이미지와 대조한다. 계단(층 이동) 에피소드를 따로 본다.

vln_ce 원본은 habitat 으로 렌더됐다. 생성 경로도 같은 렌더러(instruction_labeling `habitat_rig_sim.RigSim`,
640×480 · hfov 79° · 높이 1.25 m · pitch 0/30°)로 그릴 것이므로, 먼저 GT 를 재현하는지 확인한다.

    instruction_labeling/render/p1r_replay_verify.py   릴리스 action 을 그대로 재생 -> pose · RGB · depth 대조 (전 에피소드)
    이 스크립트                                          p1r 결과 + nav3d 02 의 계단 표시 -> 평지/계단 판정
                                                        + 계단 에피소드를 다시 재생해 눈으로 보는 그림

## 용어
- **재생**: raw R2R 시작 자세에 놓고 릴리스 parquet 의 action(전진 0.25 m / 회전 15°)을 그대로 `sim.step`.
- **pose 오차**: 우리 센서 c2w vs 릴리스 `pose.<rig>` (시작 기준) — 위치 [mm] · 회전 [°], 에피소드 최대.
- **SSIM**: 구조 유사도(1 = 같음). 우리 렌더를 릴리스와 같은 JPEG 테이블로 인코딩한 뒤 비교.
- **depth 일치**: uint16 mm 가 정확히 같은 픽셀 비율.
- **계단 에피소드**: nav3d 02 의 `multifloor`(GT 시작·끝 높이 차 > 0.5 m).

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 | 근거 |
|---|---|---|---|
| H1 매칭 | 릴리스 에피소드가 raw 시작 자세와 짝지어지나 | 100 % | 시작 자세가 없으면 재생 불가 |
| H2 pose | 에피소드 최대 위치 오차 ≤ 5 mm · 회전 ≤ 0.1° 인 비율 | ≥ 99 % | 같은 navmesh·같은 action 이면 수치 오차만 남는다 |
| H3 RGB | 에피소드 평균 SSIM ≥ 0.95 인 비율 | ≥ 99 % | 같은 렌더러 — JPEG 잡음 수준만 남아야 한다 |
| H4 depth | 에피소드 depth 일치 ≥ 0.98 인 비율 | ≥ 99 % | mm 정수화 경계에서만 1 mm 차이 |
| H5 계단 | 계단 − 평지 SSIM 중앙 ≥ −0.01, 계단 pose 오차도 H2 안 | 둘 다 | 계단에서 sliding·높이 처리가 달라지지 않는지 |
| H4b depth (채움) | 우리 depth 를 `depth_fill`(3×3 최소 이웃 × 19, 릴리스 구멍 채우기 재현)로 채운 뒤, 에피소드 평균 "1 cm 안" 픽셀 비율 ≥ 0.98 인 비율 | ≥ 99 % | 채운 값은 mm 까지 같지 않다(중앙 4~7 mm) — 1 cm 로 본다 |
| H4c depth (채움) | 채운 뒤 1 m 넘게 틀린 픽셀의 프레임 최대 비율 | 모든 에피소드 ≤ 2 % | 구멍(0) ↔ 값 의 큰 차이가 남았는지 |

H4(채우기 전 mm 일치)는 2026-10-15 부터 참고값 — 릴리스가 구멍을 채웠다는 것이 원인으로 확인됨(본문 depth 불일치 절).
H4b·H4c 도 참고값 — 채우기(A)는 평균만 조금 낫고 일부 에피소드는 나빠져 채우지 않기로 함(2026-10-15 사용자 결정). 생성 데이터 depth = habitat 원본.
H4b·H4c 는 렌더를 다시 돌려 전 에피소드를 `--depth_every` 프레임 간격으로 잰다(p1r 는 depth 를 저장하지 않음).

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/05a_vlnce_gt_render_check.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/ (이 파일은 nav3d/checks/ 에 있다)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'instruction_labeling' / 'render'))
import nav3d_report as R  # noqa: E402

LOG = 'scripts/dataset_converters/nav3d/logs/05_vlnce_gt_render_habitat'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='scripts/dataset_converters/instruction_labeling/configs/render.yaml')
    ap.add_argument('--p1r_dir', default=f'{LOG}/p1r', help='p1r_replay_verify --out')
    ap.add_argument('--nav02_dir', default='scripts/dataset_converters/nav3d/logs/02_vlnce_gt_compare/scenes')
    ap.add_argument('--episodes_json', default='scripts/dataset_converters/nav3d/logs/05_vlnce_gt_render/episodes.json',
                    help='눈으로 볼 계단 에피소드는 여기서 multifloor 인 것')
    ap.add_argument('--out_dir', default=f'{LOG}/check')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--depth_every', type=int, default=5, help='H4b·H4c depth 평가 프레임 간격 (p1r pixel_every 와 같게)')
    return ap.parse_args()


def main():
    import pandas as pd
    args = parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.p1r_dir) / 'per_episode.csv')
    p1r = json.loads((Path(args.p1r_dir) / 'summary.json').read_text())
    rigs = [c.split('.')[0] for c in df.columns if c.endswith('.ssim_mean')]
    mf, nfr = {}, {}
    for scene in df['scene'].unique():
        j = Path(args.nav02_dir) / f'{scene}.json'
        for e in json.load(open(j))['episodes']:
            mf[(scene, e['ep'])] = (bool(e['multifloor']), float(e['gt_dz_m']))
            nfr[(scene, e['ep'])] = e['n_frames']
    ok = df[df['status'] == 'ok'].copy()
    ok['mf'] = [mf.get((s, int(e)), (False, 0.0))[0] for s, e in zip(ok['scene'], ok['episode_index'])]
    ok['_k'] = list(zip(ok['scene'], ok['episode_index']))
    ok['dz'] = [mf.get((s, int(e)), (False, 0.0))[1] for s, e in zip(ok['scene'], ok['episode_index'])]
    # nav02 의 ep 번호가 릴리스 episode_index 와 같은지 (프레임 수로 확인)
    same = np.mean([nfr.get((s, int(e))) == int(f) for s, e, f in zip(ok['scene'], ok['episode_index'], ok['frames'])])

    ssim_ep = ok[[f'{r}.ssim_mean' for r in rigs]].min(1)          # 두 rig 중 나쁜 쪽
    pos_ep = ok[[f'{r}.pos_mm_max' for r in rigs]].max(1)
    rot_ep = ok[[f'{r}.rot_deg_max' for r in rigs]].max(1)
    dep_ep = ok[[f'{r}.depth_exact' for r in rigs]].min(1)
    h1 = len(ok) / len(df)
    h2 = float(np.mean((pos_ep <= 5.0) & (rot_ep <= 0.1)))
    h3 = float(np.mean(ssim_ep >= 0.95))
    h4 = float(np.mean(dep_ep >= 0.98))
    m = ok['mf'].values
    d_mf = float(np.median(ssim_ep[m]) - np.median(ssim_ep[~m])) if m.any() and (~m).any() else float('nan')
    st_pose = bool(np.all((pos_ep[m] <= 5.0) & (rot_ep[m] <= 0.1))) if m.any() else False
    dfe = depth_fill_eval(args, ok, Path(args.out_dir))
    near_f = np.array([dfe[f'{s}|{int(e)}']['filled_le10mm_mean'] for s, e in zip(ok['scene'], ok['episode_index'])])
    far_f = np.array([dfe[f'{s}|{int(e)}']['filled_gt1m_max'] for s, e in zip(ok['scene'], ok['episode_index'])])
    near_r = np.array([dfe[f'{s}|{int(e)}']['raw_le10mm_mean'] for s, e in zip(ok['scene'], ok['episode_index'])])
    far_r = np.array([dfe[f'{s}|{int(e)}']['raw_gt1m_max'] for s, e in zip(ok['scene'], ok['episode_index'])])
    h4b = float(np.mean(near_f >= 0.98))
    h4c = bool(np.all(far_f <= 0.02))
    gates = [
        {'검증': 'H1 매칭', '결과': R.pill(h1 == 1.0), '값': f'{len(ok)}/{len(df)} 에피소드', '기준': '100 %'},
        {'검증': 'H2 pose', '결과': R.pill(h2 >= 0.99), '값': f'{100 * h2:.2f} % (위치 최대 {pos_ep.max():.2f} mm · 회전 최대 {rot_ep.max():.3f}°)',
         '기준': '≤ 5 mm · ≤ 0.1° 인 에피소드 ≥ 99 %'},
        {'검증': 'H3 RGB SSIM', '결과': R.pill(h3 >= 0.99), '값': f'{100 * h3:.2f} % (에피소드 SSIM 중앙 {np.median(ssim_ep):.4f} · 최소 {ssim_ep.min():.4f})',
         '기준': '에피소드 평균 SSIM ≥ 0.95 인 것 ≥ 99 %'},
        {'검증': 'H4 depth (채우기 전, 참고)', '결과': R.pill('참고'), '값': f'{100 * h4:.2f} % (일치 중앙 {np.median(dep_ep):.4f} · 최소 {dep_ep.min():.4f})',
         '기준': '(참고) mm 일치 ≥ 0.98 인 것 ≥ 99 %'},
        {'검증': 'H4b depth 채움 · 1 cm (참고)', '결과': R.pill('참고'),
         '값': f'{100 * h4b:.2f} % (에피소드 평균 1 cm 안: 채움 중앙 {np.median(near_f):.4f} · 최소 {near_f.min():.4f} / 채우기 전 최소 {near_r.min():.4f})',
         '기준': '1 cm 안 ≥ 0.98 인 에피소드 ≥ 99 %'},
        {'검증': 'H4c depth 채움 · 1 m 초과 (참고)', '결과': R.pill('참고'),
         '값': f'프레임 최대 {100 * far_f.max():.2f} % (채우기 전 {100 * far_r.max():.2f} %)', '기준': '모든 에피소드 ≤ 2 %'},
        {'검증': 'H5 계단', '결과': R.pill(bool(np.isfinite(d_mf) and d_mf >= -0.01 and st_pose)),
         '값': f'계단 {int(m.sum())} ep · SSIM 계단−평지 {d_mf:+.4f} · 계단 위치 최대 {pos_ep[m].max() if m.any() else float("nan"):.2f} mm',
         '기준': '≥ −0.01 · 계단도 H2 안'},
    ]
    by_rig = []
    for r in rigs:
        s = p1r[r]
        by_rig.append({'rig': r, '위치 오차 p50 / p99 / 최대 [mm]': f"{s['pos_mm']['p50']:.3f} / {s['pos_mm']['p99']:.3f} / {s['pos_mm']['max']:.3f}",
                       '회전 최대 [°]': f"{s['rot_deg_max']:.4f}", 'RGB MAE (JPEG 바닥)': f"{s['rgb_mae_mean']:.2f} ({s['jpeg_floor_mae_mean']:.2f})",
                       'PSNR 최소': f"{s['psnr_min']:.1f}", 'SSIM 평균 / 최소': f"{s['ssim_mean']:.4f} / {s['ssim_min']:.4f}",
                       'depth 일치 평균': f"{s['depth_exact_mean']:.4f}", '밝기 차 평균': f"{s['mean_brightness_shift']:+.2f}"})
    by_scene = []
    for scene, g in ok.groupby('scene'):
        k = g.index
        for flag, name in [(False, '평지'), (True, '계단')]:
            kk = k[g['mf'].values == flag]
            if len(kk):
                by_scene.append({'scene': scene, '종류': name, '에피소드': len(kk), '프레임': int(ok.loc[kk, 'frames'].sum()),
                                 'SSIM 중앙': f'{np.median(ssim_ep[kk]):.4f}', 'SSIM 최소': f'{ssim_ep[kk].min():.4f}',
                                 '위치 최대 [mm]': f'{pos_ep[kk].max():.2f}', 'depth 일치 최소': f'{dep_ep[kk].min():.4f}'})
    figs = make_figures(out, args, ok, ssim_ep, pos_ep, rigs)
    dm = depth_mismatch_probe(out, args, ok, rigs)
    worst = ok.assign(ssim=ssim_ep, pos=pos_ep, dep=dep_ep).sort_values('ssim').head(10)
    worst_rows = [{'scene': r.scene, 'ep': int(r.episode_index), '종류': '계단' if r.mf else '평지', '프레임': int(r.frames),
                   'SSIM': f'{r.ssim:.4f}', '위치 최대 [mm]': f'{r.pos:.2f}', 'depth 일치': f'{r.dep:.4f}'} for r in worst.itertuples()]
    passed = [g['결과'].count('PASS') == 1 for g in gates if '참고' not in g['검증']]
    dtab = []
    for flag, name in [(False, '평지'), (True, '계단')]:
        k = ok['mf'].values == flag
        if k.any():
            dtab.append({'구분': name, '에피소드': int(k.sum()),
                         '1 cm 안 (채우기 전 → 채움)': f'{np.mean(near_r[k]) * 100:.2f} → {np.mean(near_f[k]) * 100:.2f} %',
                         '1 m 초과 프레임 최대 (전 → 채움)': f'{far_r[k].max() * 100:.2f} → {far_f[k].max() * 100:.2f} %',
                         '1 m 초과 평균 (전 → 채움)': f"{np.mean([dfe[f'{s}|{int(e)}']['raw_gt1m_mean'] for s, e in zip(ok['scene'][k], ok['episode_index'][k])]) * 100:.3f} → "
                                                f"{np.mean([dfe[f'{s}|{int(e)}']['filled_gt1m_mean'] for s, e in zip(ok['scene'][k], ok['episode_index'][k])]) * 100:.3f} %",
                         '0 위치 불일치 평균 (전 → 채움)': f"{np.mean([dfe[f'{s}|{int(e)}']['raw_zero_mis_mean'] for s, e in zip(ok['scene'][k], ok['episode_index'][k])]) * 100:.3f} → "
                                                   f"{np.mean([dfe[f'{s}|{int(e)}']['filled_zero_mis_mean'] for s, e in zip(ok['scene'][k], ok['episode_index'][k])]) * 100:.3f} %"})
    summary = {'gates': {g['검증']: g['값'] for g in gates}, 'pass': passed, 'depth_mismatch': dm['rows'], 'nav02_ep_matches_release_index': float(same),
               'p1r': p1r, 'by_scene': by_scene}
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False, default=float)

    lead = (f'vln_ce 릴리스 action 을 habitat(instruction_labeling <span class="mono">RigSim</span>)으로 그대로 재생해 pose · RGB · depth 를 원본과 대조했다. '
            f'씬 {ok["scene"].nunique()} · 에피소드 {len(ok)} (계단 {int(m.sum())}) · 프레임 {int(ok["frames"].sum()):,} · rig {", ".join(rigs)}. '
            f'<b>판정 {sum(passed)}/{len(passed)} 통과.</b> 이 렌더러로 생성 경로를 그린다(다음 단계 05b).')
    body = R.header('nav3d 05a · vln_ce GT 재렌더 (habitat)', 'vln_ce GT 재렌더 — habitat, 원본 대조', lead)
    body += R.section('판정', '<p class="muted">기준은 결과를 보기 전에 정했다. pose 오차 = 우리 센서 자세와 릴리스 pose 의 차이(에피소드 최대). '
                      'SSIM = 구조 유사도(1 = 같음), 우리 렌더를 릴리스와 같은 JPEG 테이블로 인코딩한 뒤 비교. depth 일치 = mm 정수가 같은 픽셀 비율.</p>'
                      + R.table(gates, ['검증', '결과', '값', '기준'], raw=('결과',)))
    body += R.section('rig 별 (p1r 요약)', R.table(by_rig, list(by_rig[0].keys()))
                      + '<p class="muted">RGB MAE 괄호 = 같은 영상을 JPEG 로 한 번 더 인코딩할 때 생기는 오차(잡음 바닥). MAE 가 이 근처면 사실상 같은 영상이다.</p>')
    body += R.section('씬 · 평지/계단 별', R.table(by_scene, list(by_scene[0].keys())) + R.figure(figs['dist'], '에피소드 SSIM(두 rig 중 낮은 쪽)과 최대 위치 오차 — 평지(회색) vs 계단(주황).'))
    body += R.section('계단 에피소드 — 눈으로 확인', '<p class="muted">rig 125cm_30deg. 위: 에이전트 높이(검정)와 프레임별 SSIM(주황). '
                      '아래: 계단 구간 6 프레임 — 릴리스 rgb / habitat rgb / |차|×8 / 릴리스 depth / habitat depth.</p>'
                      + ''.join(R.figure(f, c) for f, c in figs['stairs']))
    body += R.section('depth 구멍 채우기 (H4b · H4c)', '<p class="muted">원인: 릴리스 depth 는 habitat 이 표면을 못 맞춘 구멍(0)을 가장자리부터 약 20 px 채웠다 '
                      '(구멍의 릴리스 RGB 도 검정 — 뒷면이 아니라 사후 채우기). 재현: <span class="mono">depth_fill.fill_depth_holes</span> = 3×3 최소 이웃 × 19. '
                      '채운 영역 IoU 0.95~0.99, 값은 mm 까지 같지 않다(중앙 4~7 mm). 전 에피소드를 '
                      f'{args.depth_every} 프레임 간격으로 다시 렌더해 쟀다.</p>' + R.table(dtab, list(dtab[0].keys())))
    body += R.section('depth 불일치 — 어떤 픽셀이 다른가', '<p class="muted">씬마다 depth 일치가 가장 낮은 에피소드를 다시 재생해, 다른 픽셀만 모아 본다. '
                      '"우리 = 0" 은 habitat 이 그 방향에서 표면을 못 맞춘 픽셀(mesh 구멍·뒷면).</p>'
                      + R.table(dm['rows'], list(dm['rows'][0].keys())) + ''.join(R.figure(f, c) for f, c in dm['figs']))
    body += R.section('SSIM 낮은 에피소드 10', R.table(worst_rows, list(worst_rows[0].keys())))
    body += R.section('입력', f'<p class="mono">p1r {R.esc(args.p1r_dir)} · config {R.esc(args.config)} · nav02 ep 번호 = 릴리스 episode_index 일치 {100 * same:.1f} %</p>'
                      + R.raw_json('summary', summary))
    R.write(out, 'vln_ce GT 재렌더 habitat', body)
    print('gates:', ['%s %s' % (g['검증'].split()[0], 'PASS' if p else 'FAIL') for g, p in zip([g for g in gates if '참고' not in g['검증']], passed)])
    print(f'-> {out}/report.html')


def depth_fill_eval(args, ok, out: Path) -> dict:
    """전 에피소드를 다시 재생해 채우기 전/후 depth 를 릴리스와 비교 (depth_every 프레임 간격, 모든 rig). 결과는 캐시."""
    import cv2
    from habitat_rig_sim import RigSim
    from release_io import load_cfg, match_raw, release_episode_data, release_episodes, release_image_path, scene_glb
    from p1r_replay_verify import depth_to_mm
    from depth_fill import fill_depth_holes
    cache = out / 'depth_fill_eval.json'
    res = json.loads(cache.read_text()) if cache.exists() else {}
    cfg = load_cfg(args.config)
    dmax = float(cfg['camera']['depth_max_m'])
    rigs = list(cfg['camera']['rigs'])
    for scene, g in ok.groupby('scene'):
        todo = [int(e) for e in g['episode_index'] if f'{scene}|{int(e)}' not in res]
        if not todo:
            continue
        sim = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
        rel = {x['episode_index']: x for x in release_episodes(cfg, scene)}
        for ep in todo:
            raw = match_raw(cfg, scene, rel[ep]['task'])
            d = release_episode_data(cfg, scene, ep, rigs)
            sim.set_start(raw['start_position'], raw['start_rotation'])
            T = len(d['action'])
            acc = {k: [] for k in ('raw_le10mm', 'filled_le10mm', 'raw_gt1m', 'filled_gt1m', 'raw_zero_mis', 'filled_zero_mis', 'raw_exact', 'filled_exact')}
            for f in range(T):
                if f > 0:
                    sim.step(d['action'][f])
                if not (f % args.depth_every == 0 or f == T - 1):
                    continue
                obs = sim.observe()
                for r in rigs:
                    om = depth_to_mm(obs[r][1], dmax).astype(np.int64)
                    rm = cv2.imread(str(release_image_path(cfg, scene, 'depth', r, ep, f)), -1).astype(np.int64)
                    for tag, x in (('raw', om), ('filled', fill_depth_holes(om))):
                        dd = np.abs(x - rm)
                        acc[f'{tag}_le10mm'].append(float(np.mean(dd <= 10)))
                        acc[f'{tag}_gt1m'].append(float(np.mean(dd > 1000)))
                        acc[f'{tag}_zero_mis'].append(float(np.mean((x == 0) != (rm == 0))))
                        acc[f'{tag}_exact'].append(float(np.mean(dd == 0)))
            res[f'{scene}|{ep}'] = {**{f'{k}_mean': float(np.mean(v)) for k, v in acc.items()},
                                    **{f'{k}_max': float(np.max(v)) for k, v in acc.items()}}
        sim.close()
        cache.write_text(json.dumps(res))
        print(f'  depth_fill_eval {scene}: {len(todo)} ep', flush=True)
    return res


def depth_mismatch_probe(out, args, ok, rigs) -> dict:
    """씬마다 depth 일치가 가장 낮은 에피소드·rig 하나를 다시 재생해 불일치 픽셀의 성질을 잰다."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from habitat_rig_sim import RigSim
    from release_io import load_cfg, match_raw, release_episode_data, release_episodes, release_image_path, scene_glb
    from p1r_replay_verify import depth_to_mm
    cfg = load_cfg(args.config)
    dmax = float(cfg['camera']['depth_max_m'])
    rows, figs = [], []
    for scene, g in ok.groupby('scene'):
        col = {r: g[f'{r}.depth_exact'] for r in rigs}
        rig = min(rigs, key=lambda r: col[r].min())
        ep = int(g.loc[col[rig].idxmin(), 'episode_index'])
        sim = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
        e = {x['episode_index']: x for x in release_episodes(cfg, scene)}[ep]
        raw = match_raw(cfg, scene, e['task'])
        d = release_episode_data(cfg, scene, ep, [rig])
        sim.set_start(raw['start_position'], raw['start_rotation'])
        Rv, Ov, n_px, worst = [], [], 0, None
        for f in range(len(d['action'])):
            if f > 0:
                sim.step(d['action'][f])
            o = depth_to_mm(sim.observe()[rig][1], dmax).astype(np.int64)
            r = cv2.imread(str(release_image_path(cfg, scene, 'depth', rig, ep, f)), -1).astype(np.int64)
            m = o != r
            n_px += m.size
            Rv.append(r[m])
            Ov.append(o[m])
            if worst is None or m.mean() > worst[0]:
                worst = (m.mean(), f, r, o)
        sim.close()
        Rv, Ov = np.concatenate(Rv), np.concatenate(Ov)
        rows.append({'scene': scene, 'ep · rig': f'{ep} · {rig}', '불일치 픽셀 비율': f'{len(Rv) / n_px * 100:.2f} %',
                     '그중 우리 = 0': f'{np.mean(Ov == 0) * 100:.1f} %' if len(Ov) else '-',
                     '그중 릴리스 = 0': f'{np.mean(Rv == 0) * 100:.1f} %' if len(Rv) else '-',
                     '릴리스 값 p5 / p50 / p95 [m]': ' / '.join(f'{v / 1000:.2f}' for v in np.percentile(Rv, [5, 50, 95])) if len(Rv) else '-',
                     '|차| 중앙 [m]': f'{np.median(np.abs(Ov - Rv)) / 1000:.2f}' if len(Rv) else '-'})
        if worst is not None and worst[0] > 0:
            _, f, r, o = worst
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].imshow(r, cmap='turbo', vmin=0, vmax=dmax * 1000)
            ax[0].set_title(f'release depth f{f}')
            ax[1].imshow(o, cmap='turbo', vmin=0, vmax=dmax * 1000)
            ax[1].set_title('habitat depth (ours)')
            ax[2].imshow((o != r), cmap='gray')
            ax[2].set_title(f'mismatch ({worst[0] * 100:.1f} % of frame)')
            for a in ax:
                a.axis('off')
            fig.tight_layout()
            fn = f'depth_mismatch_{scene}.png'
            fig.savefig(out / fn, dpi=70)
            plt.close(fig)
            figs.append((fn, f'{scene} ep{ep} {rig} f{f}: 불일치가 가장 큰 프레임. 진한 남색 = depth 0(표면 없음).'))
    return {'rows': rows, 'figs': figs}


def make_figures(out, args, ok, ssim_ep, pos_ep, rigs):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from habitat_rig_sim import RigSim
    from release_io import load_cfg, match_raw, release_episode_data, release_episodes, release_image_path, scene_glb
    from p1r_replay_verify import compare_rgb, depth_to_mm
    figs = {}
    m = ok['mf'].values
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for flag, col, nm in [(False, '#888888', 'flat'), (True, '#e67e22', 'stairs')]:
        axs[0].hist(ssim_ep[m == flag], bins=40, color=col, alpha=0.7, label=nm)
        axs[1].hist(pos_ep[m == flag], bins=40, color=col, alpha=0.7, label=nm)
    axs[0].axvline(0.95, color='k', ls=':')
    axs[1].axvline(5.0, color='k', ls=':')
    axs[0].set_xlabel('episode SSIM (min over rigs)')
    axs[1].set_xlabel('episode max position error [mm]')
    for a in axs:
        a.legend()
        a.set_yscale('log')
    fig.tight_layout()
    fig.savefig(out / 'dist.png', dpi=90)
    plt.close(fig)
    figs['dist'] = 'dist.png'

    cfg = load_cfg(args.config)
    sub = int(cfg['image']['jpeg_subsampling'])
    dmax = float(cfg['camera']['depth_max_m'])
    rig = '125cm_30deg' if '125cm_30deg' in rigs else rigs[0]
    sel = json.loads(Path(args.episodes_json).read_text())
    figs['stairs'] = []
    for scene, eps in sel.items():
        want = [e for e in eps if ((ok['scene'] == scene) & (ok['episode_index'] == e) & ok['mf']).any()]
        if not want:
            continue
        sim = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
        rel_eps = {e['episode_index']: e for e in release_episodes(cfg, scene)}
        for ep in want:
            raw = match_raw(cfg, scene, rel_eps[ep]['task'])
            d = release_episode_data(cfg, scene, ep, [rig])
            sim.set_start(raw['start_position'], raw['start_rotation'])
            T = len(d['action'])
            ys, ss, frames = [], [], []
            for f in range(T):
                if f > 0:
                    sim.step(d['action'][f])
                ys.append(sim.agent_position()[1])
                o = sim.observe()[rig]
                c, rel, oq = compare_rgb(o[0], release_image_path(cfg, scene, 'rgb', rig, ep, f), sub, ssim=True)
                ss.append(c['ssim'])
                frames.append((rel, oq, cv2.imread(str(release_image_path(cfg, scene, 'depth', rig, ep, f)), -1) / 1000.0,
                               depth_to_mm(o[1], dmax) / 1000.0))
            ys = np.array(ys)
            on = np.where(np.abs(np.gradient(ys)) > 0.02)[0]
            pick = (np.linspace(on.min(), on.max(), 6).round().astype(int) if len(on) >= 6
                    else np.linspace(0, T - 1, 6).round().astype(int))
            fig = plt.figure(figsize=(18, 15))
            gsp = fig.add_gridspec(6, 6, height_ratios=[1.0, 1, 1, 1, 1, 1])
            ax = fig.add_subplot(gsp[0, :])
            ax.plot(ys, 'k-', label='agent height (habitat y) [m]')
            ax2 = ax.twinx()
            ax2.plot(ss, color='#e67e22')
            ax2.set_ylim(0.8, 1.0)
            ax2.set_ylabel('SSIM')
            for p in pick:
                ax.axvline(p, color='#2b5fa8', ls='--', lw=0.8)
            ax.set_xlabel('frame')
            ax.set_ylabel('height [m]')
            ax.set_title(f'{scene} ep{ep} — rig {rig}; dashed = frames below')
            for c, p in enumerate(pick):
                rel, oq, dr, do = frames[p]
                diff = np.clip(np.abs(oq.astype(int) - rel.astype(int)) * 8, 0, 255).astype(np.uint8)
                for r_, (img, cm) in enumerate([(rel, None), (oq, None), (diff, None), (dr, 'turbo'), (do, 'turbo')]):
                    a = fig.add_subplot(gsp[1 + r_, c])
                    a.imshow(img, cmap=cm, vmin=0 if cm else None, vmax=8 if cm else None)
                    a.set_xticks([])
                    a.set_yticks([])
                    if c == 0:
                        a.set_ylabel(['release rgb', 'habitat rgb', '|diff| x8', 'release depth', 'habitat depth'][r_])
                    if r_ == 0:
                        a.set_title(f'f{p} SSIM {ss[p]:.3f}', fontsize=9)
            fig.tight_layout()
            fn = f'stairs_{scene}_{ep}.png'
            fig.savefig(out / fn, dpi=65)
            plt.close(fig)
            figs['stairs'].append((fn, f'{scene} ep{ep}: 높이 {ys.max() - ys.min():.2f} m 이동, 계단 구간 SSIM 중앙 {np.median([ss[p] for p in pick]):.4f}.'))
            print(f'  viz {scene} ep{ep}', flush=True)
        sim.close()
    return figs


if __name__ == '__main__':
    main()
