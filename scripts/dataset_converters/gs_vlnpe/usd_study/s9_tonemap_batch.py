"""S9 — 톤맵 설정을 **여러 씬에서 표본 측정**하고 집계한다.

S8은 씬 하나를 본다. 씬 하나 결과로 파이프라인 기본값을 바꿀 수는 없으므로, 여기서 여러 씬을
돌려 **방향이 일관적인지**를 본다. 전수(vln_n1 65씬 / vln_pe 61씬)로 갈지는 이 표본을 보고 정한다.

## 왜 씬마다 프로세스를 새로 띄우나

한 프로세스에서 씬을 3번 이상 올리면 죽는다
(`AnnotatorRegistryError: Annotator rgb is not attached to any render products`, 실측).
그래서 이 러너는 **s8을 씬마다 subprocess로 부르고 json만 모으는 얇은 오케스트레이터**다.
렌더 로직은 s8에만 있고 여기엔 없다.

## 무엇을 집계하나

씬마다 두 가지 개선 후보를 현재 설정과 비교한다.

1. **op 7 (Iray) + crush 0.0** — S3에서 vln_n1 씬1에서 +0.039를 본 것
2. **op 6 유지 + film_iso만 재튜닝** — op을 안 바꾸니 위험이 없는 대안

씬별로 이긴/진/무차를 세고 증감 분포를 낸다. **한 씬이라도 크게 지면 채택하지 않는다**는
판단을 할 수 있도록 최악 씬을 같이 보여준다.

## 실행 (한 줄)

    timeout --signal=KILL 21600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s9_tonemap_batch.py --dataset vln_pe --n_scenes 12 --log_dir logs/gs-vlnpe/usd_study

씬당 약 2분이다(11개 설정 × 18프레임). 12씬이면 25분 안팎.
`--skip_existing`(기본 켜짐)이라 이미 돌린 씬은 json을 재사용한다 — 끊겨도 이어서 돌리면 된다.
전수로 가려면 `--n_scenes 0`(=전부).
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from usd_study_utils import (
    DEFAULT_LOG_DIR,
    DEFAULT_USD_ROOT,
    TABLE_CSS,
    esc,
    glossary_html,
    glossary_note_html,
    pill,
    stat_row_html,
    table_html,
)

import viz_utils

GS_VLNPE = Path(__file__).resolve().parent.parent
ISAAC_PY = '/workspace/isaaclab/_isaac_sim/python.sh'
S8 = Path(__file__).resolve().parent / 's8_tonemap_adopt.py'
# 이보다 작은 차이는 렌더러 노이즈로 본다 — s8 표의 같은 설정 두 행이 0.0000~0.0001이었다.
NOISE = 0.005
DATASET_AMBIENT = {'vln_n1': 6.0, 'vln_pe': 10.0}
DATASET_TRAJ = {'vln_n1': 'data/InternData-N1-v0.5-mini/vln_n1/traj_data/matterport3d_d435i',
                'vln_pe': 'data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r'}


def available_scenes(usd_root, dataset: str) -> list:
    """USD와 GT 궤적이 **둘 다** 있는 씬만. (USD 90개지만 궤적은 vln_n1 65 / vln_pe 61)"""
    usd = {d.parent.parent.name for d in Path(usd_root).glob('*/matterport_mesh/*')
           if d.is_dir() and any(d.glob('isaacsim_*.usd'))}
    traj = {p.name for p in Path(DATASET_TRAJ[dataset]).glob('*') if (p / 'data' / 'chunk-000').is_dir()}
    return sorted(usd & traj)


def pick_scenes(all_scenes, n: int, seed: int, always: list) -> list:
    """재현 가능한 표본. 이미 돌려둔 씬을 앞에 넣어 공짜로 쓴다."""
    if n <= 0 or n >= len(all_scenes):
        return list(all_scenes)
    head = [s for s in always if s in all_scenes]
    rest = [s for s in all_scenes if s not in head]
    rng = random.Random(seed)
    return head + rng.sample(rest, max(0, n - len(head)))


def scene_json(log_dir, scene, dataset) -> Path:
    return Path(log_dir) / 's8_tonemap_adopt' / f'{scene}_{dataset}' / 's8_tonemap.json'


def run_one(scene, args) -> tuple:
    """s8을 subprocess로 돌린다. 반환: (성공, 초, 메모)"""
    out = scene_json(args.log_dir, scene, args.dataset)
    if args.skip_existing and out.is_file():
        return True, 0.0, 'json 재사용'
    cmd = [ISAAC_PY, str(S8), '--scene', scene, '--dataset', args.dataset,
           '--n_frames', str(args.n_frames), '--isos', args.isos,
           '--film_iso', str(args.film_iso),
           '--rtx_ambient', str(args.rtx_ambient), '--log_dir', str(args.log_dir),
           '--mesh_root', args.mesh_root, '--textures', args.textures,
           '--crushes', args.crushes]
    t0 = time.time()
    log_path = Path(args.log_dir) / 's9_tonemap_batch' / f'{args.dataset}' / f'{scene}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=args.per_scene_timeout)
    dt = time.time() - t0
    if not out.is_file():
        return False, dt, f'json 없음 (exit {p.returncode}, 로그: {log_path})'
    return True, dt, f'exit {p.returncode}'


def summarize(scene, d) -> dict:
    """s8 json 하나 -> 씬 한 줄 요약."""
    res = d['results']
    cur = d['current']
    op6 = max((r for r in res if r['group'] in ('current', 'op6')), key=lambda r: r['ssim_median'])
    op7 = max((r for r in res if r['group'] == 'op7'), key=lambda r: r['ssim_median'])
    trap = d.get('trap')
    return {
        'scene': scene,
        'gt_brightness': d['gt_brightness'],
        'cur_ssim': cur['ssim_median'], 'cur_gap': cur['brightness_gap'],
        'op6_ssim': op6['ssim_median'], 'op6_iso': op6['iso'], 'op6_gap': op6['brightness_gap'],
        'op7_ssim': op7['ssim_median'], 'op7_iso': op7['iso'], 'op7_gap': op7['brightness_gap'],
        'gain_op7': op7['ssim_median'] - cur['ssim_median'],
        'gain_iso_only': op6['ssim_median'] - cur['ssim_median'],
        'trap_ssim': (trap or {}).get('ssim_median'),
        'trap_delta': ((trap or {}).get('ssim_median', float('nan')) - cur['ssim_median']) if trap else None,
    }


def verdict_counts(rows, key):
    win = sum(1 for r in rows if r[key] > NOISE)
    lose = sum(1 for r in rows if r[key] < -NOISE)
    return win, lose, len(rows) - win - lose


def main():
    ap = argparse.ArgumentParser(description='S9 — 톤맵 설정 여러 씬 표본 측정 + 집계')
    ap.add_argument('--dataset', choices=['vln_n1', 'vln_pe'], default='vln_pe')
    ap.add_argument('--n_scenes', type=int, default=12, help='0이면 전수')
    ap.add_argument('--seed', type=int, default=20260907, help='표본 추출 시드(재현용)')
    ap.add_argument('--scenes', default=None, help='씬을 직접 지정 (콤마). 주면 --n_scenes 무시')
    ap.add_argument('--usd_root', default=str(DEFAULT_USD_ROOT))
    ap.add_argument('--n_frames', type=int, default=18)
    ap.add_argument('--isos', default='50,70,90,110')
    ap.add_argument('--film_iso', type=float, default=70.0)
    ap.add_argument('--rtx_ambient', type=float, default=None, help='기본: vln_n1 6.0 / vln_pe 10.0')
    ap.add_argument('--crushes', default='0.0,0.25,0.5,0.75',
                    help='op7의 crushBlacks 후보 — iso와 2차원으로 훑는다')
    ap.add_argument('--mesh_root', default='data/scene_data/mp3d_pe')
    ap.add_argument('--textures', choices=['symlink', 'raw'], default='symlink',
                    help='symlink=파이프라인과 동일(텍스처 붙음) / raw=심링크 없이 원본 USD 그대로')
    ap.add_argument('--per_scene_timeout', type=int, default=1800)
    ap.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                    help='이미 있는 json도 다시 렌더')
    ap.add_argument('--log_dir', default=str(DEFAULT_LOG_DIR))
    args = ap.parse_args()
    if args.rtx_ambient is None:
        args.rtx_ambient = DATASET_AMBIENT[args.dataset]

    all_scenes = available_scenes(args.usd_root, args.dataset)
    if args.scenes:
        scenes = [s for s in args.scenes.split(',') if s]
    else:
        scenes = pick_scenes(all_scenes, args.n_scenes, args.seed, ['17DRP5sb8fy', 's8pcmisQ38h'])
    print(f'[s9] {args.dataset}: 가능한 씬 {len(all_scenes)}개 중 {len(scenes)}개 측정 '
          f'(ambient {args.rtx_ambient:g}, seed {args.seed})', flush=True)

    rows, failed = [], []
    for i, sc in enumerate(scenes, 1):
        ok, dt, note = run_one(sc, args)
        if not ok:
            failed.append({'scene': sc, 'note': note})
            print(f'[s9] {i:>3}/{len(scenes)} {sc:<16} FAIL — {note}', flush=True)
            continue
        d = json.loads(scene_json(args.log_dir, sc, args.dataset).read_text(encoding='utf-8'))
        r = summarize(sc, d)
        rows.append(r)
        print(f"[s9] {i:>3}/{len(scenes)} {sc:<16} 현재 {r['cur_ssim']:.4f} · "
              f"op7 {r['gain_op7']:+.4f} · iso만 {r['gain_iso_only']:+.4f}  ({dt:.0f}s, {note})", flush=True)

    if not rows:
        print('[s9] 집계할 결과가 없다', file=sys.stderr)
        return 2

    out_dir = Path(args.log_dir) / 's9_tonemap_batch' / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    g7 = np.array([r['gain_op7'] for r in rows])
    gi = np.array([r['gain_iso_only'] for r in rows])
    w7, l7, t7 = verdict_counts(rows, 'gain_op7')
    wi, li, ti = verdict_counts(rows, 'gain_iso_only')
    worst7 = min(rows, key=lambda r: r['gain_op7'])
    best7 = max(rows, key=lambda r: r['gain_op7'])
    isos_best = {}
    for r in rows:
        isos_best[r['op6_iso']] = isos_best.get(r['op6_iso'], 0) + 1

    agg = {'args': vars(args), 'scenes_measured': len(rows), 'scenes_available': len(all_scenes),
           'failed': failed, 'noise_threshold': NOISE,
           'op7': {'win': w7, 'lose': l7, 'tie': t7, 'median': float(np.median(g7)),
                   'min': float(g7.min()), 'max': float(g7.max())},
           'iso_only': {'win': wi, 'lose': li, 'tie': ti, 'median': float(np.median(gi)),
                        'min': float(gi.min()), 'max': float(gi.max())},
           'best_iso_histogram': isos_best, 'rows': rows}
    (out_dir / 's9_summary.json').write_text(json.dumps(agg, indent=2, default=str, ensure_ascii=False),
                                             encoding='utf-8')

    # ── 씬별 증감 곡선
    chart = None
    if len(rows) >= 3:
        chart = out_dir / 'gains.jpg'
        xs = list(range(1, len(rows) + 1))
        viz_utils.line_chart([('op7 gain', (0, 255, 90), xs, [r['gain_op7'] + 0.05 for r in rows]),
                              ('iso-only gain', (90, 190, 255), xs, [r['gain_iso_only'] + 0.05 for r in rows]),
                              ('zero', (150, 150, 150), xs, [0.05] * len(rows))],
                             chart, x_label='scene index', y_label='SSIM gain +0.05 offset')

    parts = [TABLE_CSS, glossary_note_html()]
    parts.append('<h2>목적</h2>')
    parts.append(
        f'<p>톤맵 설정을 바꿀지 정하려면 <b>여러 씬에서 방향이 일관적인지</b> 봐야 한다. '
        f'씬 하나 결과로는 못 정한다 — 실제로 S8에서 vln_n1 씬1은 +0.039였는데 vln_pe 씬1은 '
        f'0.000, 씬2는 −0.011이었다.</p>'
        f'<p><b>{esc(args.dataset)}</b>에서 가능한 씬 <b>{len(all_scenes)}개</b> 중 '
        f'<b>{len(rows)}개</b>를 측정했다(seed <code>{args.seed}</code>). '
        f'비교 후보는 둘이다 — <b>op7(Iray)+crush 0</b>과 <b>op6 유지 + iso만 재튜닝</b>. '
        f'뒤쪽은 op을 안 바꾸니 위험이 없는 대안이다.</p>')

    parts.append('<h2>커맨드와 결과</h2>')
    parts.append(f'<pre>timeout --signal=KILL 21600 /workspace/isaaclab/_isaac_sim/python.sh '
                 f'scripts/dataset_converters/gs_vlnpe/usd_study/s9_tonemap_batch.py '
                 f'--dataset {esc(args.dataset)} --n_scenes {args.n_scenes} --log_dir {esc(args.log_dir)}</pre>')
    parts.append(stat_row_html([
        ('측정 씬', f'{len(rows)} / {len(all_scenes)}'),
        ('op7 이김', w7), ('op7 짐', l7), ('무차', t7),
        ('op7 증감 중앙값', f'{np.median(g7):+.4f}'),
        ('op7 최악 씬', f"{worst7['gain_op7']:+.4f}"),
        ('iso만 증감 중앙값', f'{np.median(gi):+.4f}'),
    ]))

    parts.append('<h3>후보 두 개 비교</h3>')
    parts.append(table_html(
        ['후보', '이김', '짐', '무차', '증감 중앙값', '최악', '최고'],
        [['op7 (Iray) + crush 0.0', w7, l7, t7, f'{np.median(g7):+.4f}', f'{g7.min():+.4f}', f'{g7.max():+.4f}'],
         ['op6 유지 + iso 재튜닝', wi, li, ti, f'{np.median(gi):+.4f}', f'{gi.min():+.4f}', f'{gi.max():+.4f}']]))
    parts.append(f'<p style="color:var(--text-dim);font-size:.85rem">±{NOISE} 이내는 무차로 센다 '
                 '(s8 표에서 같은 설정 두 행의 차이가 0.0000~0.0001이었다).</p>')

    if chart:
        parts.append('<h3>씬별 증감</h3>')
        parts.append(f'<img src="{viz_utils.image_to_data_uri(chart)}" style="max-width:100%">')
        parts.append('<p style="color:var(--text-dim);font-size:.85rem">회색 선이 0(현재 설정)이다. '
                     '보기 편하게 전체를 +0.05 올려 그렸다.</p>')

    parts.append('<h3>씬별 상세</h3>')
    rows_sorted = sorted(rows, key=lambda r: r['gain_op7'])
    tbl, classes = [], []
    for r in rows_sorted:
        tbl.append([r['scene'], f"{r['gt_brightness']:.1f}", f"{r['cur_ssim']:.4f}",
                    f"{r['gain_op7']:+.4f}", f"{r['op7_iso']:g}",
                    f"{r['gain_iso_only']:+.4f}", f"{r['op6_iso']:g}",
                    f"{r['cur_gap']:+.1f}", f"{r['op6_gap']:+.1f}"])
        classes.append('differ' if abs(r['gain_op7']) > NOISE else '')
    parts.append(table_html(['씬', 'GT 밝기', '현재 SSIM', 'op7 증감', 'op7 최적 iso',
                             'iso만 증감', '최적 iso', '현재 밝기차', 'iso재튜닝 밝기차'],
                            tbl, row_classes=classes))
    parts.append('<p style="color:var(--text-dim);font-size:.85rem">op7 증감이 나쁜 순으로 정렬했다. '
                 '<b>GT 밝기가 씬마다 다르다</b>는 점에 주의 — 하나의 iso를 모든 씬에 쓰는 것이 '
                 '적절한지가 이 표의 관전 포인트다.</p>')

    parts.append('<h3>씬별 최적 iso 분포</h3>')
    parts.append(table_html(['최적 iso', '씬 수'],
                            [[f'{k:g}', v] for k, v in sorted(isos_best.items())]))

    if failed:
        parts.append('<h3>실패한 씬</h3>')
        parts.append(table_html(['씬', '사유'], [[f['scene'], f['note']] for f in failed], mono_cols={1}))

    parts.append('<h2>Takeaway</h2>')
    tk = []
    op7_ok = l7 == 0 and np.median(g7) > NOISE
    tk.append((f'<b>op7 채택 {"가능" if op7_ok else "보류"}</b>',
               f'{len(rows)}개 씬에서 이김 <b>{w7}</b> · 짐 <b>{l7}</b> · 무차 {t7}, '
               f'증감 중앙값 <b>{np.median(g7):+.4f}</b>, 최악 <b>{worst7["gain_op7"]:+.4f}</b>'
               f'({esc(worst7["scene"])}). '
               + ('한 씬도 지지 않고 중앙값이 노이즈보다 크므로 채택 근거가 된다.' if op7_ok else
                  '<b>지는 씬이 있으면 채택하지 않는다</b> — 데이터셋 전체에 적용되는 설정이라 '
                  '평균이 좋아도 특정 씬이 나빠지면 그 씬 데이터가 손해를 본다.')))
    iso_ok = li == 0 and np.median(gi) > NOISE
    tk.append((f'<b>iso 재튜닝 {"은 안전한 개선" if iso_ok else "도 일관적이지 않다"}</b>',
               f'이김 <b>{wi}</b> · 짐 <b>{li}</b> · 무차 {ti}, 증감 중앙값 <b>{np.median(gi):+.4f}</b>. '
               'op을 안 바꾸므로 `crushBlacks` 같은 부작용이 없다. '
               f'씬별 최적 iso 분포는 {esc(str({f"{k:g}": v for k, v in sorted(isos_best.items())}))}.'))
    tk.append(('<b>전수로 갈지</b>',
               f'표본 {len(rows)}개에서 방향이 뚜렷하면 전수({len(all_scenes)}개, 약 '
               f'{len(all_scenes) * 2}분)를 돌릴 값어치가 있다. '
               '엇갈리면 전수를 돌려도 하나의 설정으로 수렴하지 않을 것이므로, '
               '<b>씬/데이터셋별로 다른 값을 쓸지</b>를 먼저 정해야 한다.'))
    parts.append('<div style="display:flex;flex-direction:column;gap:14px;margin:12px 0 22px">')
    for head, body in tk:
        parts.append('<div style="border:1px solid var(--border);border-radius:10px;background:var(--surface);'
                     f'padding:12px 14px"><div style="margin-bottom:6px">{head}</div>'
                     f'<div style="font-size:.88rem;line-height:1.6;color:var(--text-dim)">{body}</div></div>')
    parts.append('</div>')
    parts.append(glossary_html())

    summary = stat_row_html([
        ('dataset', args.dataset), ('측정 씬', f'{len(rows)}/{len(all_scenes)}'),
        ('op7 이김/짐', f'{w7}/{l7}'), ('op7 중앙값', f'{np.median(g7):+.4f}'),
        ('iso만 이김/짐', f'{wi}/{li}'), ('iso만 중앙값', f'{np.median(gi):+.4f}'),
    ])
    path = viz_utils.save_gallery(out_dir, 'report.html',
                                  f'S9 · 톤맵 표본 측정 — {args.dataset} {len(rows)}씬',
                                  summary, ''.join(parts), eyebrow='gs_vlnpe usd_study')
    print(f'[s9] report: {path}')
    print(f'[s9] op7 이김 {w7} / 짐 {l7} / 무차 {t7} · 중앙값 {np.median(g7):+.4f}')
    print(f'[s9] iso만 이김 {wi} / 짐 {li} / 무차 {ti} · 중앙값 {np.median(gi):+.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
