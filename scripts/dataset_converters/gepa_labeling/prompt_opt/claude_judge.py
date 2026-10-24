"""Claude 채점기 시험: 경로 사실(회전·계단·방을 순서대로) 과 문장을 주고 사실과 어긋나는 말만 찾게 한다. 토큰 사용량도 잰다.

    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/prompt_opt/claude_judge.py --run v1
시험 문장 (val_unseen dev 303 경로문장):
  정답 시험: 사람 303, b1 303 원문 (틀렸다고 하면 오경보; 사람 문장에는 진짜 실수도 섞임)
  오답 시험: check_mutations.py 의 오답 종류마다 --n-mut 개를 무작위로 (사람·b1 반반, 표현도 무작위)
Claude 는 `claude -p` (Claude Code, --safe-mode: CLAUDE.md·도구 없음, 자체 system prompt 만) 로 부른다.
출력: logs/gepa_labeling/check_scorer/claude_judge/<run>/{items.jsonl, calls.jsonl, summary.txt}
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_mutations as M  # noqa: E402
from scorer import path_facts  # noqa: E402

from fact_judge import UNKNOWN, judge, load_regions, route_events  # noqa: E402

C = M.C


def build_items(a, rc):
    its = C.A.select_subset(C.items('val_unseen'), 'dev')
    eps = {it['ep']['episode_id']: it['ep'] for it in its}
    gt = C.gt_instructions('val_unseen')
    b1, _ = C.load_label('b1')
    sets = {'사람': {k: gt[k] for k in eps}, 'b1': b1}
    names = rc['rooms']['names']
    facts = {k: path_facts(ep, rc) for k, ep in eps.items()}
    ctx = {k: {'seed': int(k), 'house_rooms': [names[r['label']] for r in load_regions(str(M.house_path(ep))) if names.get(r['label'])]}
           for k, ep in eps.items()}
    items = []
    for base, texts in sets.items():
        for k in eps:
            items.append({'id': f'{base}:orig:{k}', 'ep': k, 'kind': '원문', 'base': base, 'form': None, 'text': texts[k],
                          'rule_cats': sorted(M.cats(texts[k], facts[k]))})
    rng = random.Random(a.seed)
    for mname, (fn, forms, cat) in M.MUTS.items():
        pool = []
        for base, texts in sets.items():
            for k, f in facts.items():
                if cat in M.cats(texts[k], f):
                    continue
                for form in forms:
                    m = fn(texts[k], f, form, ctx[k])
                    if m is not None and m != texts[k]:
                        pool.append((base, k, form, m))
        by_base = {b: [x for x in pool if x[0] == b] for b in sets}
        pick = []
        for b, xs in by_base.items():
            pick += rng.sample(xs, min(len(xs), a.n_mut // 2))
        for base, k, form, m in pick:
            items.append({'id': f'{base}:{mname}:{k}:{forms.index(form)}', 'ep': k, 'kind': mname, 'base': base, 'form': form,
                          'text': m, 'rule_hit': cat in M.cats(m, facts[k])})
    return items, eps


def summarize(items, calls, out):
    L = []
    U = collections.Counter()
    for c in calls:
        if c['ok']:
            u = c['usage']
            U['input'] += u['input_tokens']
            U['cache_write'] += u['cache_creation_input_tokens']
            U['cache_read'] += u['cache_read_input_tokens']
            U['output'] += u['output_tokens']
            U['thinking'] += (u.get('output_tokens_details') or {}).get('thinking_tokens', 0)
            U['cost'] += c['cost'] or 0
            U['sec'] += c['sec']
    n_ok = sum('errors' in x for x in items)  # 토큰은 채점된 문장 수로 나눈다 (두 번째 확인 호출 몫도 포함)
    L.append(f'호출 {len(calls)}번 (실패 {sum(not c["ok"] for c in calls)}), 채점된 문장 {n_ok}')
    tot_in = U['input'] + U['cache_write'] + U['cache_read']
    L.append(f'토큰 합계: 입력 {tot_in:,} (새 입력 {U["input"]:,} + 캐시 쓰기 {U["cache_write"]:,} + 캐시 읽기 {U["cache_read"]:,}), '
             f'출력 {U["output"]:,} (그중 생각 {U["thinking"]:,})')
    L.append(f'문장당: 입력 {tot_in / max(1, n_ok):.0f}, 출력 {U["output"] / max(1, n_ok):.0f} 토큰; 호출당 {U["sec"] / max(1, len(calls)):.0f} 초; '
             f'정가 환산 ${U["cost"]:.2f} (문장당 ${U["cost"] / max(1, n_ok):.4f})')
    for n in (4700,):
        L.append(f'GEPA 1회 ({n} 문장) 추정: 입력 {tot_in / max(1, n_ok) * n / 1e6:.2f}M, 출력 {U["output"] / max(1, n_ok) * n / 1e6:.2f}M 토큰, '
                 f'정가 환산 ${U["cost"] / max(1, n_ok) * n:.0f}')
    L.append('')
    L.append('[정답 시험] 원문을 틀렸다고 한 비율 (낮을수록 좋음)')
    L.append('세트 | 모델 1단계 | 모델 최종 | 규칙 채점기')
    for base in ('사람', 'b1'):
        xs = [x for x in items if x['kind'] == '원문' and x['base'] == base and 'errors' in x]
        L.append(f'{base} | {100 * sum(bool(x["errors_raw"]) for x in xs) / max(1, len(xs)):5.1f} % | '
                 f'{100 * sum(bool(x["errors"]) for x in xs) / max(1, len(xs)):5.1f} % | '
                 f'{100 * sum(bool(x["rule_cats"]) for x in xs) / max(1, len(xs)):5.1f} %  (n={len(xs)})')
    L.append('')
    L.append('[오답 시험] 틀리게 고친 문장을 잡은 비율 (높을수록 좋음; 같은 문장으로 규칙 채점기와 비교)')
    L.append('오답 | 모델 1단계 | 모델 최종 | 모델 최종 (같은 종류 오류) | 규칙 채점기 | n')
    T = {'방향': {'turn', 'order'}, '순서': {'turn', 'order'}, '계단': {'stairs'}, '방-집': {'room'}, '방-지남': {'room'}}
    for mname in M.MUTS:
        xs = [x for x in items if x['kind'] == mname and 'errors' in x]
        L.append(f'{mname} | {100 * sum(bool(x["errors_raw"]) for x in xs) / max(1, len(xs)):5.1f} % | '
                 f'{100 * sum(bool(x["errors"]) for x in xs) / max(1, len(xs)):5.1f} % | '
                 f'{100 * sum(any(e.get("type") in T[mname] for e in x["errors"]) for x in xs) / max(1, len(xs)):5.1f} % | '
                 f'{100 * sum(x["rule_hit"] for x in xs) / max(1, len(xs)):5.1f} % | {len(xs)}')
    txt = '\n'.join(L)
    (out / 'summary.txt').write_text(txt + '\n')
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--backend', default='claude', choices=['claude', 'openai'], help='claude = `claude -p`, openai = OpenAI 호환 서버 (gemma)')
    ap.add_argument('--base-url', default='http://localhost:8100/v1', help='--backend openai 일 때 서버 주소')
    ap.add_argument('--model', default='claude-opus-5-5')
    ap.add_argument('--effort', default='medium')
    ap.add_argument('--batch', type=int, default=10, help='한 번 호출에 넣는 문장 수')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n-mut', type=int, default=60, help='오답 종류마다 시험할 문장 수 (사람·b1 반반)')
    ap.add_argument('--limit', type=int, default=None, help='앞에서 이만큼만 (시험용)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--facts', default='v1', choices=list(UNKNOWN), help='경로 사실 형식. v2 = 방 정보가 없는 곳은 판단하지 말라고 씀')
    ap.add_argument('--verify', action='store_true', help='두 번째 확인: 지적한 오류마다 진짜 모순인지 같은 모델에 다시 묻는다')
    a = ap.parse_args()
    rc = C.il_config.load('b1', [])['route']
    out = M.REPO / 'logs/gepa_labeling/check_scorer/claude_judge' / a.run
    out.mkdir(parents=True, exist_ok=True)
    a.cwd = str(out)  # CLAUDE.md 가 없는 곳에서 부른다 (--safe-mode 와 이중으로)
    items, eps = build_items(a, rc)
    random.Random(a.seed).shuffle(items)  # 한 호출에 원문·오답이 섞이게 (같은 경로가 몰려 답이 서로 영향 주지 않게)
    items = items[:a.limit] if a.limit else items
    ev = {k: route_events(eps[k], rc, a.facts) for k in {x['ep'] for x in items}}
    (out / 'facts.json').write_text(json.dumps(ev, ensure_ascii=False, indent=1))
    print(f'문장 {len(items)}개 (한 번에 {a.batch}), {a.backend} 모델 {a.model} effort {a.effort}, facts {a.facts}, verify {a.verify}', flush=True)
    jc = {k: getattr(a, k) for k in ('backend', 'base_url', 'model', 'effort', 'facts', 'verify', 'batch', 'workers', 'cwd')}
    calls = judge(items, ev, jc)
    with open(out / 'items.jsonl', 'w') as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    with open(out / 'calls.jsonl', 'w') as f:
        for c in calls:
            f.write(json.dumps({k: v for k, v in c.items() if k != 'res'}, ensure_ascii=False) + '\n')
    summarize(items, calls, out)


if __name__ == '__main__':
    main()
