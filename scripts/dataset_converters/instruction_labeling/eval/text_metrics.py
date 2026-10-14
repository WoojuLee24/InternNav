"""P3: 생성 지시문을 GT(같은 trajectory 의 사람 지시문)와 비교하는 문장 지표.

공정 비교 ("같은 자"): episode e 에 대해
  GT 행   : GT_e        vs refs = 같은 trajectory 의 나머지 GT 2개   (사람 상한, leave-one-out)
  label 행: label_e     vs refs = 같은 2개 (GT_e 제외)               -> GT 행과 참조가 완전히 같다
  (참고) label_all3     : label_e vs GT 3개
지표
  BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE (pycocoevalcap, java 필요; SPICE 는 실패하면 생략)
  규칙 지표 (vln-s2-pipeline tools/compare_gt.py 의 analyse/summarise 재사용):
    turn_seq_wrong, first_turn_vs_gt, last_turn_wrong, false_turn_around, stairs_*, goal_mismatch, noun_precision 등
  다양성: 고유 문장 비율, 같은 trajectory 형제 문장끼리 BLEU-4 (높을수록 획일적), 내용어 수, 첫 단어 상위, 평균 단어 수

실행 환경: pycocoevalcap·nltk 가 있는 navcomposer venv.
    /home/irteam/venvs/navcomposer/bin/python scripts/dataset_converters/instruction_labeling/eval/text_metrics.py --split val_unseen --subset dev --label b0=data/InternData-N1-v0.5-mini/instruction_labels/b0/val_unseen_dev/val_unseen_b0.json.gz --out logs/instruction_labeling/p3/b0_dev
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
S2 = Path('/home/irteam/git/vln-s2-pipeline')
sys.path.insert(0, str(S2))
sys.path.insert(0, str(S2 / 'tools'))
sys.path.insert(0, str(HERE.parent / 'label'))
from route_words import route_geometry  # noqa: E402

STOP = set('a an the and to of in on at by into onto from with your you then until when there this that it is are be '
           'go walk turn left right stop wait straight forward through past continue keep take head enter exit '
           'up down towards toward next near before after once just slightly bear around'.split())


def coco_scores(cands: dict, refs: dict) -> dict:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tok = PTBTokenizer()
    res = tok.tokenize({k: [{'caption': v}] for k, v in cands.items()})
    gts = tok.tokenize({k: [{'caption': r} for r in v] for k, v in refs.items()})
    out = {}
    b, _ = Bleu(4).compute_score(gts, res, verbose=0)
    out.update({f'BLEU-{i + 1}': round(x, 4) for i, x in enumerate(b)})
    out['METEOR'] = round(Meteor().compute_score(gts, res)[0], 4)
    out['ROUGE-L'] = round(Rouge().compute_score(gts, res)[0], 4)
    out['CIDEr'] = round(Cider().compute_score(gts, res)[0], 4)
    try:
        from pycocoevalcap.spice.spice import Spice
        out['SPICE'] = round(Spice().compute_score(gts, res)[0], 4)
    except Exception as e:  # noqa: BLE001
        out['SPICE'] = f'skip ({type(e).__name__})'
    return out


def diversity(texts: dict, by_t: dict) -> dict:
    vals = list(texts.values())
    words = [re.findall(r"[a-z']+", t.lower()) for t in vals]
    content = Counter(w for ws in words for w in ws if w not in STOP and len(w) > 2)
    first = Counter(ws[0] for ws in words if ws)
    sib = {}
    for tid, eids in by_t.items():
        e = [x for x in eids if x in texts]
        for i, x in enumerate(e):
            sib[x] = [texts[y] for y in e if y != x]
    sib_bleu = None
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        tok = PTBTokenizer()
        ks = [k for k in texts if sib.get(k)]
        res = tok.tokenize({k: [{'caption': texts[k]}] for k in ks})
        gts = tok.tokenize({k: [{'caption': r} for r in sib[k]] for k in ks})
        sib_bleu = round(Bleu(4).compute_score(gts, res, verbose=0)[0][3], 4)
    except Exception:  # noqa: BLE001
        pass
    low = [v.lower() for v in vals]
    phr = {f'"{w}" %': round(100 * sum(w in t for t in low) / max(1, len(low)), 1)
           for w in ('exit', 'leave', 'enter', 'bedroom', 'hallway', 'kitchen', 'bathroom', 'room')}
    return {**phr, 'n': len(vals), 'unique_ratio': round(len(set(v.strip() for v in vals)) / max(1, len(vals)), 4),
            'mean_words': round(sum(len(w) for w in words) / max(1, len(words)), 1),
            'content_vocab': len(content), 'sibling_BLEU-4': sib_bleu,
            'first_word_top5': ', '.join(f'{w} {c / len(vals):.0%}' for w, c in first.most_common(5))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='val_unseen')
    ap.add_argument('--subset', default='', help="'' 또는 'dev' — GT 행도 같은 episode 집합으로 제한")
    ap.add_argument('--label', action='append', required=True, help='name=path/to/<split>_<label>.json.gz (반복 가능)')
    ap.add_argument('--turn_deg', type=float, default=30.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    import compare_gt as cg  # vln-s2-pipeline 규칙 지표 (analyse/summarise)
    gt = json.load(gzip.open(REPO / f'data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r/{a.split}/{a.split}.json.gz'))['episodes']
    if a.subset == 'dev':
        ids = set(json.loads((HERE / 'dev_subset.json').read_text())['trajectory_ids'])
        gt = [e for e in gt if e['trajectory_id'] in ids]
    by_t = defaultdict(list)
    for e in gt:
        by_t[e['trajectory_id']].append(e['episode_id'])
    gt_text = {e['episode_id']: e['instruction']['instruction_text'] for e in gt}
    tid_of = {e['episode_id']: e['trajectory_id'] for e in gt}
    geo = {e['episode_id']: route_geometry(e) for e in gt}

    def refs(eid, loo=True):
        return [gt_text[x] for x in by_t[tid_of[eid]] if not (loo and x == eid)]

    sets = [('GT', gt_text)]
    for x in a.label:
        name, path = x.split('=', 1)
        d = {e['episode_id']: e['instruction']['instruction_text'] for e in json.load(gzip.open(REPO / path))['episodes']}
        missing = set(gt_text) - set(d)
        assert not missing, f'{name}: GT episode {len(missing)}개가 label 에 없다'
        sets.append((name, {k: d[k] for k in gt_text}))

    results = {}
    for name, texts in sets:
        eids = [k for k in texts if len(refs(k)) >= 1]
        r = {'episodes': len(eids)}
        r.update(coco_scores({k: texts[k] for k in eids}, {k: refs(k) for k in eids}))
        if name != 'GT':
            r3 = coco_scores({k: texts[k] for k in eids}, {k: refs(k, loo=False) for k in eids})
            r.update({f'{m} (vs 3 GT)': v for m, v in r3.items() if m in ('BLEU-4', 'METEOR', 'CIDEr', 'SPICE')})
        rows = []
        for k in eids:
            fl, df = cg.analyse(texts[k], refs(k), geo[k], a.turn_deg)
            rows.append({'episode_id': k, 'trajectory_id': tid_of[k], 'text': texts[k], 'flags': fl, 'diff': df})
        r.update(cg.summarise(rows))
        r.update(diversity(texts, by_t))
        results[name] = r
        with open(out / f'{name}_episodes.jsonl', 'w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(name, 'done', flush=True)

    json.dump(results, open(out / 'summary.json', 'w'), ensure_ascii=False, indent=1)
    names = list(results)
    keys = list(dict.fromkeys(k for s in results.values() for k in s))
    lines = [f'# 문장 지표 — {a.split} {a.subset or "all"} (GT 행 = 사람 상한, leave-one-out)\n',
             '| metric | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    lines += [f'| {k} | ' + ' | '.join(str(results[n].get(k, '')) for n in names) + ' |' for k in keys]
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
