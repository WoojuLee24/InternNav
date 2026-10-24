"""2단계 채점기: 경로 하나(형제 3문장) 의 점수와 "이유 글".

  사실 (문장별 0/1)  : 틀린 말만 감점, 생략은 감점하지 않는다. score.fact_checker 로 고른다:
                       rule  = 규칙(회전 방향·순서, 계단, 마지막 문장의 방, 최소 정보) + use_gemma_facts 면 gemma
                       모델  = fact_judge.py (경로 사실과 문장을 모델이 비교, verify 면 지적을 다시 따짐) + 규칙의 최소 정보
  손잡이 (0~1)       : 생성 문장을 0단계 규칙(style_types/extract.py rule_features) 으로 다시 재서 끝맺음·설명 방식·길이 중 맞은 비율
  다양성 (0~1)       : 1 − 평균(형제 첫 3단어 같은 쌍 비율, 형제 내용어 Jaccard)
  점수 = 사실 평균 × (w_knob·손잡이 평균 + w_div·다양성)
경로 사실은 GT waypoint(reference_path) 와 MP3D 방 정보로만 만든다 (GT 문장은 쓰지 않는다).
"""

import itertools
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / 'scripts/dataset_converters/instruction_labeling/label'))
sys.path.insert(0, str(HERE.parent / 'style_types'))
from route_words import route_words  # noqa: E402

CFG = yaml.safe_load(open(HERE.parent / 'configs' / 'gepa.yaml'))
STOPW = set('the a an and to of in on at by you your into from with then until is are it this that there'.split())


def path_facts(ep: dict, rc: dict, seed: int = 0) -> dict:
    """회전(30도 이상 전부, 살짝 꺾음 포함: 사람은 말할 수 있다), 출발 회전, 계단, 방 순서."""
    _, g = route_words(ep, rc, seed)
    turns = []
    if abs(g['initial']) >= rc['turn_deg']:
        turns.append(('start', g['initial']))
    for i in range(len(g['segments'])):
        if i in g['turns'] and abs(g['turns'][i]) >= rc['turn_deg']:
            turns.append((i, g['turns'][i]))
    dys = [s['dy'] for s in g['segments'] if abs(s['dy']) > rc['stairs_dy_m']]
    return {'turns': turns, 'up': any(d > 0 for d in dys), 'down': any(d < 0 for d in dys),
            'rooms': g.get('rooms') or [], 'n_said_turns': sum(1 for _, a in turns if abs(a) >= rc['slight_deg']), 'g': g}


def facts_text(f: dict) -> str:
    def word(a):
        side = 'right' if a > 0 else 'left'
        return f'turn around (to the {side})' if abs(a) >= 135 else f'turn {side} ({abs(a):.0f} degrees)'
    lines = []
    st = [a for k, a in f['turns'] if k == 'start']
    if st:
        lines.append(f'- At the start the path does not go the way you face: first {word(st[0])}.')
    rest = [word(a) for k, a in f['turns'] if k != 'start']
    lines.append('- Turns along the path, in order: ' + (', '.join(rest) if rest else 'none'))
    lines.append('- Stairs: ' + ('up' if f['up'] else '') + (' and ' if f['up'] and f['down'] else '') + ('down' if f['down'] else '') if f['up'] or f['down'] else '- Stairs: none')
    if f['rooms']:
        lines.append('- Rooms in order: ' + ' -> '.join(f['rooms']) + f'. The path ends in the {f["rooms"][-1]}.')
    return '\n'.join(lines)


def _matches(word: str, a: float) -> bool:
    S = CFG['score']
    if word == 'around':
        return abs(a) >= S['around_min_deg']
    side = 'right' if a > 0 else 'left'
    return word == side and abs(a) < S['side_max_deg']


def rule_fact_issues(text: str, f: dict) -> list:
    R, S = CFG['rules'], CFG['score']
    tl = text.lower()
    issues = []
    said = []
    for m in re.finditer(R['turn_cmd'], tl):
        w = m.group(1) or 'around'  # u-turn / turn back / the way you came
        said.append((w, m.group(0)))
    j = 0  # 말한 회전이 경로 회전의 부분열(순서 유지) 인가
    for w, phrase in said:
        while j < len(f['turns']) and not _matches(w, f['turns'][j][1]):
            j += 1
        if j == len(f['turns']):
            issues.append(f'"{phrase}": the path has no matching {w} turn at this point (true turns: '
                          + ', '.join(f"{'right' if a > 0 else 'left'} {abs(a):.0f}deg" for _, a in f['turns']) + ')')
            break
        j += 1
    if re.search(R['stairs_up'], tl) and not f['up']:
        issues.append('says to go up stairs, but the path does not go up')
    if re.search(R['stairs_down'], tl) and not f['down']:
        issues.append('says to go down stairs, but the path does not go down')
    sents = [x for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x]
    last = sents[-1].lower() if sents else ''
    if f['rooms']:
        syn = R['rooms']
        goal = f['rooms'][-1]
        on_path = {w for r in f['rooms'] for w in syn.get(r, [r])}
        for m in re.finditer(R['room_stop'], last):
            place = (m.group(1) or m.group(2) or '').strip()
            named = {room for room, ws in syn.items() for w in ws if re.search(rf'\b{re.escape(w)}\b', place)}
            if named and not any(w in place for w in on_path):
                issues.append(f'"{m.group(0).strip()}": the path ends in the {goal} (rooms on the path: {", ".join(f["rooms"])})')
    mi = S['min_info']
    if (f['n_said_turns'] >= mi['turns'] or len(f['rooms']) >= mi['rooms']) and not re.search(R['cue'], tl):
        issues.append('only names the destination; this path needs at least one cue about the way (a turn, room or thing passed)')
    return issues


def gemma_fact_issues(items: list) -> list:
    """items = [(text, facts_text)] -> [[이유, ...]] (어긋나는 구절만)."""
    from openai import OpenAI
    G = CFG['gemma_facts']
    client = OpenAI(base_url=G['base_url'], api_key=G['api_key'])

    def one(it):
        text, ft = it
        for attempt in range(3):
            try:
                r = client.chat.completions.create(model=G['model'], max_tokens=G['max_tokens'], temperature=G['temperature'], seed=attempt,
                                                   messages=[{'role': 'user', 'content': G['prompt'].format(facts=ft, text=text.replace('"', "'"))}])
                m = re.search(r'\{.*\}', r.choices[0].message.content or '', re.S)
                d = json.loads(m.group(0))
                return [f'"{c.get("phrase", "")}": {c.get("why", "")}' for c in d.get('contradictions', [])]
            except Exception:  # noqa: BLE001 — 파싱 실패는 다시 묻는다
                continue
        return []  # 끝내 답을 못 받으면 감점하지 않는다 (규칙 검사는 그대로)

    with ThreadPoolExecutor(G['workers']) as ex:
        return list(ex.map(one, items))


def knob_of(text: str, ep: dict, rc: dict) -> dict:
    from extract import rule_features
    r = rule_features(dict(ep, instruction={'instruction_text': text}), rc)
    e = {'stop': 'stop', 'wait': 'wait', 'stay': 'wait', 'none': 'none'}[r['ending']]
    s = r['step_share']
    return {'ending': e, 'granularity': 'turn' if s >= 0.6 else 'landmark' if s <= 0.4 else 'mixed', 'length': r['length'], 'n_words': r['n_words']}


def diversity(texts: list) -> tuple:
    def cw(t):
        return {w for w in re.findall(r"[a-z']+", t.lower()) if w not in STOPW and len(w) > 2}
    pairs = list(itertools.combinations(texts, 2))
    if not pairs:
        return 1.0, 0.0, 0.0
    f3 = sum(a.lower().split()[:3] == b.lower().split()[:3] for a, b in pairs) / len(pairs)
    jac = sum(len(cw(a) & cw(b)) / max(1, len(cw(a) | cw(b))) for a, b in pairs) / len(pairs)
    return 1 - (f3 + jac) / 2, f3, jac


def score_paths(paths: list, rc: dict, use_gemma: bool = None, checker: str = None) -> list:
    """paths = [{'eps': [ep, ...], 'texts': [...], 'knobs': [{ending, granularity, length} | None, ...]}]
    -> [{'score', 'fact', 'knob', 'div', 'per': [{text, issues, knob_req, knob_got}], 'feedback'}]
    checker = configs/gepa.yaml fact_checkers 의 이름 (기본 score.fact_checker; 그 항목이 없으면 'rule' = 규칙 + use_gemma_facts)."""
    S = CFG['score']
    name = checker or S.get('fact_checker', 'rule')
    for p in paths:
        f = path_facts(p['eps'][0], rc)
        p['_facts'], p['_ft'] = f, facts_text(f)
    if CFG.get('fact_checkers', {}).get(name, {'type': 'rule'})['type'] == 'rule':
        issues = _rule_issues(paths, use_gemma)
    elif CFG['fact_checkers'][name]['type'] == 'model':
        issues = _model_issues(paths, rc, CFG['fact_checkers'][name])
    else:
        assert False, f"unreachable fact_checkers.{name}.type={CFG['fact_checkers'][name]['type']!r}"
    return _assemble(paths, issues, rc)


def _rule_issues(paths: list, use_gemma: bool = None) -> list:
    """규칙 (+ use_gemma_facts 면 gemma 한 번 검사) -> 경로마다 [문장마다 [이유, ...]]."""
    S = CFG['score']
    use_gemma = S['use_gemma_facts'] if use_gemma is None else use_gemma
    flat = [(t, p['_ft']) for p in paths for t in p['texts']]
    gem = gemma_fact_issues(flat) if use_gemma else [[] for _ in flat]
    out, k = [], 0
    for p in paths:
        row = []
        for t in p['texts']:
            row.append((rule_fact_issues(t, p['_facts']) if t else ['empty']) + [f'(checker) {x}' for x in gem[k]])
            k += 1
        out.append(row)
    return out


def _model_issues(paths: list, rc: dict, jc: dict) -> list:
    """모델 검사 (fact_judge.py: 1단계 오류 찾기 + verify 면 2단계 다시 따지기) + 규칙의 최소 정보 검사 (목적지만 말함).
    모델 답을 끝내 못 받은 문장은 모델 쪽 감점 없음 (최소 정보 검사는 그대로)."""
    from fact_judge import judge, route_events
    jc = dict(jc, cwd=str(REPO / jc.get('cwd', 'logs/gepa_labeling')))
    items, ev = [], {}
    for pi, p in enumerate(paths):
        ev[pi] = route_events(p['eps'][0], rc, jc['facts'])
        for si, t in enumerate(p['texts']):
            if t:
                items.append({'id': f'{pi}_{si}', 'ep': pi, 'text': t})
    calls = judge(items, ev, jc)
    miss = sum('errors' not in x for x in items)
    if miss or not all(c['ok'] for c in calls):
        print(f'[fact_checker] 답을 못 받은 문장 {miss}/{len(items)}, 실패한 호출 {sum(not c["ok"] for c in calls)}/{len(calls)}', flush=True)
    got = {x['id']: x.get('errors', []) for x in items}
    out = []
    for pi, p in enumerate(paths):
        row = []
        for si, t in enumerate(p['texts']):
            if not t:
                row.append(['empty'])
                continue
            mi = [i for i in rule_fact_issues(t, p['_facts']) if i.startswith('only names')]  # 최소 정보: 모델은 생략을 감점하지 않으므로 규칙으로
            row.append([f'"{e.get("phrase", "")}": {e.get("why", "")}' for e in got[f'{pi}_{si}']] + mi)
        out.append(row)
    return out


def _assemble(paths: list, issues: list, rc: dict) -> list:
    S = CFG['score']
    out = []
    for p, row in zip(paths, issues):
        per = []
        for ep, t, kn, iss in zip(p['eps'], p['texts'], p['knobs'], row):
            got = knob_of(t, ep, rc) if t else None
            km = None if kn is None or got is None else sum(got[a] == kn[a] for a in ('ending', 'granularity', 'length')) / 3
            per.append({'text': t, 'issues': iss, 'knob_req': kn, 'knob_got': got, 'knob': km})
        fact = sum(not x['issues'] for x in per) / len(per)
        knob = sum(x['knob'] for x in per if x['knob'] is not None) / max(1, sum(x['knob'] is not None for x in per))
        div, f3, jac = diversity([x['text'] for x in per if x['text']])
        score = fact * (S['w_knob'] * knob + S['w_div'] * div)
        fb = [f'path facts:\n{p["_ft"]}', f'score {score:.2f} = facts {fact:.2f} x (knobs {knob:.2f}, diversity {div:.2f}; '
              f'siblings with the same first 3 words {f3:.2f}, content-word overlap {jac:.2f})']
        for i, x in enumerate(per, 1):
            req = x['knob_req'] or {}
            miss = [f'{a}: asked {req[a]}, got {x["knob_got"][a]}' for a in ('ending', 'granularity', 'length') if req and x['knob_got'] and req[a] != x['knob_got'][a]]
            fb.append(f'sentence {i} (asked: {req}): "{x["text"]}"\n  wrong facts: {x["issues"] or "none"}\n  style misses: {miss or "none"}')
        out.append({'score': score, 'fact': fact, 'knob': knob, 'div': div, 'first3': f3, 'jaccard': jac, 'per': per, 'feedback': '\n'.join(fb)})
    return out
