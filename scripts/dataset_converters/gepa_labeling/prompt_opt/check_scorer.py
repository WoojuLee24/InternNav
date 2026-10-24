"""채점기 검증 (val_unseen dev): 사람(GT) 문장·b1·대조군에 채점기를 돌리고, 일부러 틀리게 고친 문장(left↔right)을 잡는지 본다.

    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/prompt_opt/check_scorer.py --labels b1 b1_facts-memo-v1 b1_facts-memo-v1_knobs-v1
    ... --fact-checker rule   (사실 검사기 고르기; 기본 configs/gepa.yaml score.fact_checker)
출력: 표 (사실 오류율 규칙/gemma, 손잡이 지킴, 형제 첫 3단어 같음·내용어 겹침) + logs/gepa_labeling/check_scorer/<label>.jsonl (문장별 이유)
"""
import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scorer import CFG, REPO, knob_of, score_paths  # noqa: E402

sys.path.insert(0, str(REPO / 'scripts/dataset_converters/instruction_labeling/label'))
import annotate as A  # noqa: E402
import config as il_config  # noqa: E402
from sources import gt_instructions, items  # noqa: E402

LAB = REPO / 'data/InternData-N1-v0.5-mini/instruction_labels'


def load_label(name):
    d = json.load(gzip.open(LAB / name / 'val_unseen_dev' / f'val_unseen_{name}.json.gz'))
    texts = {e['episode_id']: e['instruction']['instruction_text'] for e in d['episodes']}
    knobs = {}
    for f in (LAB / name / 'val_unseen_dev' / 'meta').glob('*.json'):
        m = json.load(open(f))
        knobs[m['episode_id']] = m['generated']['inputs'].get('knobs')
    return texts, knobs


def swap_lr(t):
    return re.sub(r'\b(left|right)\b', lambda m: {'left': 'right', 'right': 'left'}[m.group(1)], t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', default=['b1'])
    ap.add_argument('--gemma', action='store_true', help='fact_checker=rule 일 때 옛 gemma 사실 검사도 (기본: configs/gepa.yaml score.use_gemma_facts)')
    ap.add_argument('--fact-checker', default=None, help='사실 검사기 (configs/gepa.yaml fact_checkers 의 이름; 기본 score.fact_checker). 예: rule, gemma, sonnet')
    a = ap.parse_args()
    rc = il_config.load('b1', [])['route']
    its = A.select_subset(items('val_unseen'), 'dev')
    by = collections.defaultdict(list)
    for it in its:
        by[it['ep']['trajectory_id']].append(it['ep'])
    gt = gt_instructions('val_unseen')
    sets = {'GT (사람)': ({e['episode_id']: gt[e['episode_id']] for it in its for e in [it['ep']]}, {})}
    for L in a.labels:
        sets[L] = load_label(L)
    b1t = sets.get('b1', load_label('b1'))[0]
    sets['b1 left<->right 뒤집음'] = ({k: swap_lr(v) for k, v in b1t.items()}, {})
    out = REPO / 'logs/gepa_labeling/check_scorer'
    out.mkdir(parents=True, exist_ok=True)
    fc = a.fact_checker or CFG['score'].get('fact_checker', 'rule')
    print(f'사실 검사기: {fc}' + ('' if fc == 'rule' else ' (아래 "사실 오류(규칙)" 칸 = 모델 지적 + 규칙의 최소 정보 검사)'))
    print(f'{"세트":28s} | 사실 오류(규칙) | 사실 오류(gemma만) | 사실 오류(합) | 손잡이 지킴 | 형제 첫3 같음 | 형제 내용어 겹침 | 점수 | 사람과 분포 차이 끝맺음/설명/길이')
    AX = {'ending': ['stop', 'wait', 'none'], 'granularity': ['turn', 'mixed', 'landmark'], 'length': ['short', 'medium', 'long']}
    ref = None
    for name, (texts, knobs) in sets.items():
        paths = [{'eps': eps, 'texts': [texts[e['episode_id']] for e in eps], 'knobs': [knobs.get(e['episode_id']) for e in eps]} for eps in by.values()]
        res = score_paths(paths, rc, use_gemma=True if a.gemma else None, checker=a.fact_checker)
        per = [x for r in res for x in r['per']]
        rule = sum(any(not i.startswith('(checker)') for i in x['issues']) for x in per) / len(per)
        gem_only = sum(all(i.startswith('(checker)') for i in x['issues']) and bool(x['issues']) for x in per) / len(per)
        anyerr = sum(bool(x['issues']) for x in per) / len(per)
        km = [x['knob'] for x in per if x['knob'] is not None]
        print(f'{name:28s} | {100 * rule:6.1f} % | {100 * gem_only:6.1f} % | {100 * anyerr:6.1f} % | '
              f'{(100 * sum(km) / len(km)) if km else float("nan"):5.1f} % | {100 * sum(r["first3"] for r in res) / len(res):5.1f} % | '
              f'{sum(r["jaccard"] for r in res) / len(res):.2f} | {sum(r["score"] for r in res) / len(res):.3f} | ', end='')
        got = [knob_of(x['text'], ep, rc) for r, eps in zip(res, by.values()) for x, ep in zip(r['per'], eps) if x['text']]
        dist = {a: [sum(g[a] == v for g in got) / len(got) for v in vs] for a, vs in AX.items()}
        ref = ref or dist  # 첫 세트 = 사람(GT)
        print(' / '.join(f'{50 * sum(abs(x - y) for x, y in zip(dist[a], ref[a])):.0f}' for a in AX),
              '  (' + '; '.join(f'{a[:4]} ' + '/'.join(f'{100 * x:.0f}' for x in dist[a]) for a in AX) + ')')
        with open(out / f'{name.split()[0].replace("/", "_")}{"_swap" if "뒤집음" in name else ""}{"" if fc == "rule" else "_" + fc}.jsonl', 'w') as f:
            for x in per:
                f.write(json.dumps({k: x[k] for k in ('text', 'issues', 'knob_req', 'knob_got')}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
