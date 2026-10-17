"""07b — 07 labeling 결과 비교: GT 데이터 label vs 생성 데이터(nav3d wp·sg) label vs 사람 문장, 그리고 노은역 smoke.

## 용어
- **사람 문장**: R2R GT 지시문. 한 trajectory 에 3 개(형제).
- **label**: 07 이 VLM(b0 방법)으로 쓴 문장. source = 입력 이미지·경로가 어디서 왔나 (gt / nav3d_wp / nav3d_sg / noeun).
- **같은 자**: 모든 행을 **같은 참조**(그 episode 의 사람 문장을 뺀 나머지 형제 2 개)로 잰다. 사람 행 = 사람 문장 1 개 vs 나머지 2 개(사람 상한).
- **seed 잡음**: 같은 gt 입력을 seed 만 바꿔 다시 쓴 label 과의 지표 차이 — VLM 샘플링만으로 생기는 흔들림.
- 지표: BLEU-4 · METEOR · ROUGE-L · CIDEr (pycocoevalcap, 높을수록 사람 문장과 비슷).

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 |
|---|---|---|
| L1 실행 | 모든 run 이 오류 없이 끝났나 | 예외·빈 응답 0 · 품질검사 통과 ≥ 90 % |
| L2 생성 = GT | nav3d_wp 이미지·경로로 쓴 label 이 GT 이미지·경로로 쓴 label 만큼 사람 문장에 가까운가 | METEOR(wp) ≥ METEOR(gt) − max(0.02, 2×seed 잡음), CIDEr 도 같은 식(최소 폭 0.05) |
| N1 노은 실행 | 노은 label 이 오류 없이 나오나 | 예외·빈 응답 0 · 품질검사 통과 ≥ 80 % |
| (참고) sg | 시작·끝만 준 경로라 사람 경로와 다를 수 있다 | 판정 안 함 |

사용 (habitat 컨테이너, 저장소 루트):
    PYTHONPATH=/ws/src/InternNav/.venv_label/site /usr/bin/python scripts/dataset_converters/nav3d/07b_compare_labels.py
"""

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'instruction_labeling' / 'eval'))
sys.path.insert(0, str(HERE.parent / 'instruction_labeling' / 'render'))
sys.path.insert(0, str(HERE.parent / 'instruction_labeling' / 'label'))
import nav3d_report as R  # noqa: E402

D = 'data/InternData-N1-v0.5-mini'
TURN = re.compile(r'\b(turn around|turn left|turn right|bear slightly left|bear slightly right|go up the stairs|go down the stairs)\b', re.I)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label_root', default=f'{D}/instruction_labels_nav3d')
    ap.add_argument('--name', default='b0_model-qwen3vl8b-local')
    ap.add_argument('--noise_name', default='b0_model-qwen3vl8b-local_seed1', help='gt 를 seed 만 바꿔 다시 쓴 label')
    ap.add_argument('--noeun_name', default='b0_model-qwen3vl8b-local_rig-60cm_15deg')
    ap.add_argument('--nav3d_root', default=f'{D}/vln_ce_nav3d/traj_data/r2r')
    ap.add_argument('--noeun_root', default=f'{D}/vln_ce_noeun_nav3d/traj_data/r2r')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/07_label_paths')
    ap.add_argument('--n_examples', type=int, default=4, help='key frame 그림을 그릴 gt·wp episode 수')
    return ap.parse_args()


def load_meta(d: Path) -> dict:
    return {int(p.stem.split('_')[1]): json.loads(p.read_text()) for p in sorted((d / 'meta').glob('ep_*.json'))}


def run_stats(m: dict) -> dict:
    g = [x['generated'] for x in m.values()]
    return {'n': len(g), 'errors': sum(x['error'] is not None for x in g), 'quality_ok': sum(x['quality_ok'] for x in g),
            'attempts_mean': float(np.mean([x['attempts'] for x in g])) if g else 0.0,
            'words_mean': float(np.mean([len(x['text'].split()) for x in g])) if g else 0.0}


def route_seq(text: str) -> list:
    return [t.lower() for t in TURN.findall(text)]


def img_path(meta: dict, args, frame, rig: str) -> Path:
    from release_io import REPO, load_cfg
    s = meta['src']['source']
    if s == 'gt':
        root = load_cfg(HERE.parent / 'instruction_labeling' / 'configs' / 'render.yaml')['paths']['release_root']
    elif s in ('nav3d_wp', 'nav3d_sg'):
        root = REPO / args.nav3d_root / s
    elif s == 'noeun':
        root = REPO / args.noeun_root
    else:
        assert False, f'unreachable source={s!r}'
    return root / meta['scene'] / 'videos' / 'chunk-000' / f'observation.images.rgb.{rig}' / f'episode_{meta["episode_index"]:06d}_{frame}.jpg'


def strip(metas: list, titles: list, args, rig: str, out: Path, fn: str) -> str:
    """source 별 한 줄: VLM 에 들어간 key frame (최대 16 장, 8 장씩 줄바꿈)."""
    import cv2
    rows = []
    for meta, title in zip(metas, titles):
        ims = []
        for f, lab in zip(meta['generated']['inputs']['frames'], meta['generated']['inputs']['frame_labels']):
            im = cv2.imread(str(img_path(meta, args, f, rig)))
            im = cv2.resize(im, (200, 150))
            cv2.putText(im, f'{lab} f{f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(im, f'{lab} f{f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            ims.append(im)
        while len(ims) % 8:
            ims.append(np.full((150, 200, 3), 255, np.uint8))
        band = np.full((22, 1600, 3), 255, np.uint8)
        cv2.putText(band, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows.append(band)
        rows += [np.hstack(ims[i:i + 8]) for i in range(0, len(ims), 8)]
    cv2.imwrite(str(out / fn), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return fn


def main():
    args = parse()
    from release_io import REPO
    from text_metrics import coco_scores
    from sources import load_raw
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    LR = REPO / args.label_root
    runs = {s: load_meta(LR / args.name / s) for s in ('gt', 'nav3d_wp', 'nav3d_sg')}
    runs['gt_seed1'] = load_meta(LR / args.noise_name / 'gt')
    noeun = load_meta(LR / args.noeun_name / 'noeun')
    raw = {e['episode_id']: e for e in load_raw('train')}
    by_t = defaultdict(list)
    for e in raw.values():
        by_t[e['trajectory_id']].append(e['episode_id'])

    # 같은 자: 공통 episode, 참조 = 그 episode 를 뺀 형제 사람 문장
    common = sorted(set.intersection(*[set(m) for m in runs.values()]))
    refs = {e: [raw[x]['instruction']['instruction_text'] for x in by_t[raw[e]['trajectory_id']] if x != e] for e in common}
    cand = {'사람 (형제 상한)': {e: raw[e]['instruction']['instruction_text'] for e in common}}
    cand.update({f'label {s}': {e: runs[s][e]['generated']['text'] for e in common} for s in runs})
    scores = {k: coco_scores({str(e): v[e] for e in common}, {str(e): refs[e] for e in common}) for k, v in cand.items()}
    # 같은 episode 끼리 label 대 label (단일 참조 BLEU-4 · METEOR): gt label 을 참조로
    l2l = {s: coco_scores({str(e): runs[s][e]['generated']['text'] for e in common}, {str(e): [runs['gt'][e]['generated']['text']] for e in common})
           for s in ('gt_seed1', 'nav3d_wp', 'nav3d_sg')}
    # 경로 단어(route words) 의 회전·계단 순서가 gt 와 같은가
    rw = {s: float(np.mean([route_seq(runs[s][e]['generated']['inputs']['route_words']) == route_seq(runs['gt'][e]['generated']['inputs']['route_words'])
                            for e in common])) for s in ('nav3d_wp', 'nav3d_sg')}
    stats = {s: run_stats(m) for s, m in runs.items()}
    stats['noeun'] = run_stats(noeun)

    noise_m = abs(scores['label gt_seed1']['METEOR'] - scores['label gt']['METEOR'])
    noise_c = abs(scores['label gt_seed1']['CIDEr'] - scores['label gt']['CIDEr'])
    tol_m, tol_c = max(0.02, 2 * noise_m), max(0.05, 2 * noise_c)
    l1 = all(v['errors'] == 0 and v['quality_ok'] >= 0.9 * v['n'] for k, v in stats.items() if k != 'noeun')
    l2 = (scores['label nav3d_wp']['METEOR'] >= scores['label gt']['METEOR'] - tol_m) and (scores['label nav3d_wp']['CIDEr'] >= scores['label gt']['CIDEr'] - tol_c)
    n1 = stats['noeun']['n'] > 0 and stats['noeun']['errors'] == 0 and stats['noeun']['quality_ok'] >= 0.8 * stats['noeun']['n']
    gates = [
        {'검증': 'L1 실행', '결과': R.pill(l1), '값': ' · '.join(f"{k} 오류 {v['errors']} 품질 {v['quality_ok']}/{v['n']}" for k, v in stats.items() if k != 'noeun'),
         '기준': '오류 0 · 품질 통과 ≥ 90 %'},
        {'검증': 'L2 생성 = GT', '결과': R.pill(l2),
         '값': f"METEOR wp {scores['label nav3d_wp']['METEOR']:.3f} vs gt {scores['label gt']['METEOR']:.3f} (허용 −{tol_m:.3f}) · "
               f"CIDEr wp {scores['label nav3d_wp']['CIDEr']:.3f} vs gt {scores['label gt']['CIDEr']:.3f} (허용 −{tol_c:.3f})",
         '기준': 'wp ≥ gt − max(최소 폭, 2×seed 잡음)'},
        {'검증': 'N1 노은 실행', '결과': R.pill(n1), '값': f"{stats['noeun']['n']} ep · 오류 {stats['noeun']['errors']} · 품질 {stats['noeun']['quality_ok']}/{stats['noeun']['n']}",
         '기준': '오류 0 · 품질 통과 ≥ 80 %'},
    ]
    cols = ['BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
    mt = [{'문장': k, 'episode': len(common), **{c: f'{v[c]:.3f}' for c in cols}} for k, v in scores.items()]
    lt = [{'비교': f'label {s} ↔ label gt', 'BLEU-4': f"{v['BLEU-4']:.3f}", 'METEOR': f"{v['METEOR']:.3f}",
           '경로 단어 회전·계단 순서 일치': '-' if s == 'gt_seed1' else f'{100 * rw[s]:.0f} %'} for s, v in l2l.items()]
    st = [{'run': k, 'episode': v['n'], '오류': v['errors'], '품질 통과': f"{v['quality_ok']}/{v['n']}",
           '평균 시도': f"{v['attempts_mean']:.2f}", '평균 단어': f"{v['words_mean']:.1f}"} for k, v in stats.items()]
    ex = [{'scene': runs['gt'][e]['scene'], 'ep': runs['gt'][e]['src']['release_episode_index'],
           '사람': raw[e]['instruction']['instruction_text'], 'label gt': runs['gt'][e]['generated']['text'],
           'label wp': runs['nav3d_wp'][e]['generated']['text'], 'label sg': runs['nav3d_sg'][e]['generated']['text']} for e in common]

    figs = []
    pick = common[::max(1, len(common) // args.n_examples)][:args.n_examples]
    for e in pick:
        ms = [runs['gt'][e], runs['nav3d_wp'][e], runs['nav3d_sg'][e]]
        fn = strip(ms, ['gt (release images, R2R path)', 'nav3d_wp', 'nav3d_sg'], args, '125cm_0deg', out, f'keyframes_{ms[0]["scene"]}_{ms[0]["src"]["release_episode_index"]}.jpg')
        figs.append((fn, f"{ms[0]['scene']} ep{ms[0]['src']['release_episode_index']} — VLM 에 들어간 key frame (rig 125cm_0deg). "
                         f"사람: {raw[e]['instruction']['instruction_text']} | gt: {ms[0]['generated']['text']} | wp: {ms[1]['generated']['text']} | sg: {ms[2]['generated']['text']}"))
    nfigs, nt = [], []
    for e, m in sorted(noeun.items()):
        fn = strip([m], [f"noeun ep{e} {m['src']['kind']} storey {m['src']['storeys']}"], args, '60cm_15deg', out, f'noeun_keyframes_{e}.jpg')
        nfigs.append((fn, f"노은 ep{e} ({m['src']['kind']}, 층 {m['src']['storeys'][0]}→{m['src']['storeys'][1]}) — label: {m['generated']['text']}"))
        nt.append({'ep': e, '종류': m['src']['kind'], '층': f"{m['src']['storeys'][0]}→{m['src']['storeys'][1]}", '프레임': m['frames_total'],
                   'key frame': len(m['generated']['inputs']['frames']), '경로 단어': m['generated']['inputs']['route_words'].replace('\n', ' '),
                   'label': m['generated']['text'], '품질': 'OK' if m['generated']['quality_ok'] else ','.join(m['generated']['quality_issues'])})

    json.dump({'gates': {g['검증']: g['값'] for g in gates}, 'pass': [l1, l2, n1], 'scores': scores, 'label_to_label': l2l, 'route_seq_match': rw,
               'stats': stats, 'noise': {'METEOR': noise_m, 'CIDEr': noise_c}}, open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False, default=float)
    lead = (f'instruction_labeling b0(e2e: key frame 이미지 + 경로 단어 → VLM 한 번)을 GT 데이터·nav3d 생성 데이터·노은역 데이터에 <b>같은 설정</b>으로 붙였다. '
            f'VLM = <b>Qwen3-VL-8B-Instruct</b> (이 PC 에 labeling 기본 모델 Gemma-4-31B 가 없어 대체 — 절대 수치는 labeling 팀 결과와 비교하지 않는다). '
            f'MP3D {len(common)} episode × (gt · wp · sg · gt seed1) + 노은 {stats["noeun"]["n"]} episode. <b>판정 {sum([l1, l2, n1])}/3 통과.</b>')
    body = R.header('nav3d 07 · labeling 연결 smoke', 'labeling 연결 — GT · 생성 · 노은역', lead)
    body += R.section('판정', '<p class="muted">기준은 결과를 보기 전에 정했다. 같은 자 = 모든 행을 그 episode 의 사람 문장을 뺀 형제 2 개로 잰다. '
                      'seed 잡음 = gt 입력 그대로 seed 만 바꾼 label 과의 차이.</p>' + R.table(gates, ['검증', '결과', '값', '기준'], raw=('결과',)))
    body += R.section('사람 문장과 비교 (같은 자)', '<p class="muted">사람 행 = 사람 문장 1 개 vs 형제 2 개 (상한). label gt = 릴리스 이미지 + R2R 경로, '
                      'wp = nav3d 가 GT 경유점을 이어 만든 경로의 렌더, sg = 시작·끝만 준 경로의 렌더.</p>' + R.table(mt, ['문장', 'episode'] + cols))
    body += R.section('label 끼리 비교', '<p class="muted">참조 = 같은 episode 의 label gt 한 문장. gt_seed1 행이 VLM 샘플링만의 흔들림(바닥선). '
                      '경로 단어 일치 = route words 에서 회전·계단 표현 순서가 gt 와 같은 episode 비율.</p>' + R.table(lt, ['비교', 'BLEU-4', 'METEOR', '경로 단어 회전·계단 순서 일치']))
    body += R.section('run 별 실행', R.table(st, list(st[0].keys())))
    body += R.section('VLM 입력 key frame (MP3D)', ''.join(R.figure(f, c) for f, c in figs))
    body += R.section('노은역 smoke', '<p class="muted">rig 60cm_15deg (D455 장착값) 이미지. 경로 = 04 경로를 RDP 로 꺾인 점만 남김. 방 정보·few-shot 없음(b0 기본).</p>'
                      + R.table(nt, list(nt[0].keys()) if nt else ['ep']) + ''.join(R.figure(f, c) for f, c in nfigs))
    body += R.section('문장 전부 (MP3D)', R.table(ex, ['scene', 'ep', '사람', 'label gt', 'label wp', 'label sg']))
    R.write(out, 'labeling 연결 smoke', body)
    print('gates:', [f"{g['검증']} {'PASS' if p else 'FAIL'}" for g, p in zip(gates, [l1, l2, n1])])
    print(json.dumps({k: {c: v[c] for c in cols} for k, v in scores.items()}, ensure_ascii=False))
    print(f'-> {out}/report.html')


if __name__ == '__main__':
    main()
