"""0단계: 말투 특징(rules + judge) 으로 사람 문장을 유형으로 묶는다. 기준은 train 으로만 만들고 val 에는 유형만 붙인다.

    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/style_types/cluster.py
출력:
  <out_dir>/<split>_types.jsonl   episode_id, type, prob
  <out_dir>/model.pkl             유형 분류기 (scaler + GMM + 범주 목록) — GEPA 채점기의 "유형 일치" 에 다시 쓴다
  <report_dir>/types_report.md    유형별 특징 요약, 대표 문장, 형제 검사, 경로 영향 검사
"""
import collections
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract import CFG, REPO  # noqa: E402

C = CFG['cluster']


def load(split: str) -> list:
    out = REPO / CFG['out_dir']
    R = {json.loads(l)['episode_id']: json.loads(l) for l in open(out / f'{split}_rules.jsonl')}
    for l in open(out / f'{split}_judge.jsonl'):
        j = json.loads(l)
        for ax in CFG['judge']['axes']:
            a, b = (j.get('a') or {}).get(ax), (j.get('b') or {}).get(ax)
            R[j['episode_id']][ax] = a if a == b and a else 'unsure'  # 두 프롬프트가 다르면 애매함
    return [R[k] for k in sorted(R)]


class Featurizer:
    def fit(self, rows):
        X = np.array([[float(r[k]) for k in C['numeric']] for r in rows])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.cats = {k: sorted({str(r[k]) for r in rows}) for k in C['categorical']}
        return self

    def __call__(self, rows):
        X = (np.array([[float(r[k]) for k in C['numeric']] for r in rows]) - self.mu) / self.sd
        oh = [np.array([[float(str(r[k]) == v) for v in self.cats[k]] for r in rows]) for k in C['categorical']]
        return np.hstack([X] + oh)


def describe(rows, labels, probs, k):
    lines = []
    for t in range(k):
        rs = [r for r, l in zip(rows, labels) if l == t]
        if not rs:
            continue
        n = len(rs)

        def dist(key):
            c = collections.Counter(str(r[key]) for r in rs)
            return ', '.join(f'{v} {100 * m / n:.0f}%' for v, m in c.most_common(4))
        lines.append(f'### 유형 {t} — {n} 문장 ({100 * n / len(rows):.1f} %)')
        lines.append(f'- 단어 {np.mean([r["n_words"] for r in rs]):.1f} ± {np.std([r["n_words"] for r in rs]):.1f}, 문장 {np.mean([r["n_sents"] for r in rs]):.1f}, '
                     f'회전 말한 비율(경로 대비) {np.mean([r["turn_ratio"] for r in rs]):.2f}, 설명 방식 비율(1=회전 명령만) {np.mean([r["step_share"] for r in rs]):.2f}, 물체 {np.mean([r["n_objects"] for r in rs]):.1f}, 방 {np.mean([r["n_rooms"] for r in rs]):.1f}')
        for key in ['goal_only', 'opener', 'ending', 'stop_ref', 'length', 'start_turn_said', 'numbers', 'then']:
            lines.append(f'- {key}: {dist(key)}')
        lines.append(f'- (경로 영향 검사) 경로 회전 수 평균 {np.mean([r["path_turns"] for r in rs]):.2f}')
        idx = [i for i, l in enumerate(labels) if l == t]
        top = sorted(idx, key=lambda i: -probs[i])[:5]
        lines += ['- 대표 문장:'] + [f'  - {rows[i]["text"][:220]}' for i in top]
        lines.append('')
    return lines


def sibling_test(rows, labels):
    by = collections.defaultdict(list)
    for r, l in zip(rows, labels):
        by[r['trajectory_id']].append(l)
    pairs = [a == b for g in by.values() for a, b in itertools.combinations(g, 2)]
    p = np.bincount(labels) / len(labels)
    return float(np.mean(pairs)), float((p ** 2).sum())


def main():
    out, rep = REPO / CFG['out_dir'], REPO / CFG['report_dir']
    rep.mkdir(parents=True, exist_ok=True)
    tr = load('train')
    fz = Featurizer().fit(tr)
    X = fz(tr)
    lo, hi = C['k_range']
    fits = {k: GaussianMixture(k, covariance_type='diag', reg_covar=1e-3, n_init=4, random_state=C['seed']).fit(X) for k in range(lo, hi + 1)}
    bic = {k: m.bic(X) for k, m in fits.items()}
    rs = np.random.RandomState(C['seed']).choice(len(X), min(C['silhouette_sample'], len(X)), replace=False)
    sil = {k: silhouette_score(X[rs], m.predict(X[rs])) for k, m in fits.items()}
    k = max(sil, key=sil.get)  # BIC 는 0/1 특징 때문에 k 가 클수록 계속 내려가 고를 수 없다
    gmm = fits[k]
    pickle.dump({'featurizer': fz, 'gmm': gmm, 'k': k}, open(out / 'model.pkl', 'wb'))
    lines = ['# 사람(GT) 문장 말투 유형 (2차, B 방식) — train 기준', '',
             '실루엣: ' + ', '.join(f'k={kk} {v:.3f}' for kk, v in sil.items()) + f' → **k={k}**', '',
             'BIC (참고): ' + ', '.join(f'k={kk} {v:.0f}' for kk, v in bic.items()), '']
    for split in ['train', 'val_seen', 'val_unseen']:
        rows = tr if split == 'train' else load(split)
        P = gmm.predict_proba(fz(rows))
        lab, pr = P.argmax(1), P.max(1)
        with open(out / f'{split}_types.jsonl', 'w') as f:
            for r, l, q in zip(rows, lab, pr):
                f.write(json.dumps({'episode_id': r['episode_id'], 'type': int(l), 'prob': round(float(q), 3)}) + '\n')
        same, exp = sibling_test(rows, lab)
        dist = np.bincount(lab, minlength=k) / len(lab)
        lines.append(f'- {split}: 유형 비율 ' + ' / '.join(f'{100 * d:.0f}' for d in dist)
                     + f' % · 형제 같은 유형 {100 * same:.1f} % (무작위 기대 {100 * exp:.1f} %) · 확신(prob≥0.9) {100 * np.mean(pr >= 0.9):.0f} %')
        if split == 'train':
            tr_lab, tr_pr = lab, pr
    lines += ['', '## 유형별 특징 (train)', ''] + describe(tr, tr_lab, tr_pr, k)
    (rep / 'types_report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:12]))
    print(f'-> {rep / "types_report.md"}')


if __name__ == '__main__':
    main()
