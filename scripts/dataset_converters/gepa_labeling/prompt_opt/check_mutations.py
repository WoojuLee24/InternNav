"""채점기 오답 시험 (항목별): 문장을 일부러 한 가지씩 틀리게 고쳐서 채점기가 그 항목에서 잡는지 본다.

    /home/irteam/venvs/gepa/bin/python scripts/dataset_converters/gepa_labeling/prompt_opt/check_mutations.py
틀리게 고치는 표현은 채점기 정규식과 따로 만든 목록(FORMS)에서 고른다 — 채점기가 모르는 표현으로 틀려도 잡는지 보려고.
오답 종류 (각 항목은 원문이 그 항목에서 이미 걸리지 않은 문장만 분모; 표현마다 같은 분모로 따로 잰다):
  방향   : 이동 회전 구절(MOTION, 위치 표현 "on your left" 제외) 의 방향을 뒤집고 표현을 FORMS['방향'] 의 하나로
  순서   : 이동 회전 구절들의 방향 순서를 거꾸로 (거꾸로 해도 같으면 제외)
  계단   : 경로에 없는 쪽(오름/내림) 계단 문장을 마지막 문장 앞에 끼움
  방-집  : 마지막 문장을 "멈추는 곳 = 이 집에 있지만 경로에 없는 방" 으로 (MP3D .house 방 정보)
  방-지남: 마지막 문장을 "멈추는 곳 = 경로에서 지나온 방 (끝 방 아님)" 으로
정답 시험: 원문(사람·b1) 의 항목별 경보율.  --turn-min-deg 로 회전 목록 하한을 바꿔 본다 (기본 = route 설정 turn_deg).
출력: 표 + logs/gepa_labeling/check_scorer/mutations_<세트>.jsonl (놓친 문장)
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_scorer as C  # noqa: E402
from scorer import CFG, REPO, path_facts, rule_fact_issues  # noqa: E402

sys.path.insert(0, str(REPO / 'scripts/dataset_converters/instruction_labeling/label'))
from rooms import load_regions  # noqa: E402
from sources import house_path  # noqa: E402

R = CFG['rules']
FLIP = {'left': 'right', 'right': 'left'}
# 이동 회전 구절 (채점기 turn_cmd 와 따로): 동사 + 꾸밈말 0~3개 + left/right. 뒤에 of/side/hand 등이 오면 위치 표현
_MOD = r'(?:a|an|another|the|your|first|next|second|third|immediate|immediately|slight|slightly|sharp|sharply|hard|quick|to|towards|toward)'
MOTION = re.compile(r'\b(turn|make|take|hang|go|veer|bear|head|swing|curve|walk|move|step|pivot|hook)\s+(?:' + _MOD + r'\s+){0,3}(left|right)\b'
                    r'(?!\s+(?:of|side|hand|wall|corner|door|doorway|room)\b)(?:\s+turn)?')
INTENS = re.compile(r'\s+(?:past|through|in front|there|here|by|next|before|after|up|down|at|outside|inside|out|away|behind)\b')  # "go right past" = 곧장
FORMS = {
    '방향': ['turn {d}', 'make a {d}', 'take a {d}', 'hang a {d}', 'go {d}', 'veer {d}', 'bear {d}',
             'turn to your {d}', 'turn to the {d}', 'take the next {d}', 'make a {d} turn', 'swing {d}', 'curve {d}', 'head {d}'],
    '계단+': ['go up the stairs', 'climb the stairs', 'head upstairs', 'take the stairs up', 'walk up the steps',
              'go up to the next floor', 'head to the upper floor', 'ascend the staircase', 'climb up one flight', 'go up a level'],
    '계단-': ['go down the stairs', 'descend the stairs', 'head downstairs', 'take the stairs down', 'walk down the steps',
              'go down to the lower floor', 'head to the basement', 'descend the staircase', 'go down one flight', 'go down a level'],
    '방': ['Stop in the {r}.', 'Wait inside the {r}.', 'Your destination is the {r}.', 'End your walk in the {r}.',
           'Stop once you reach the {r}.', 'Go into the {r} and wait.', 'Stop when you are in the {r}.', 'Finish in the {r}.',
           'The goal is in the {r}.', 'Wait for further instructions in the {r}.'],
}


def cat_of(issue):
    if 'true turns' in issue:
        return '방향/순서'
    elif 'the path ends in' in issue:  # 방 이유 글에 "rooms on the path: ..., stairs" 가 들어갈 수 있어 계단보다 먼저
        return '방'
    elif issue.startswith('says to go'):
        return '계단'
    elif 'only names' in issue:
        return '단서'
    else:
        assert False, f'unreachable issue={issue!r}'


def cats(text, f):
    return {cat_of(i) for i in rule_fact_issues(text, f)}


def motions(t):
    out = []
    for m in MOTION.finditer(t.lower()):
        if m.group(1) in ('go', 'walk', 'move', 'step', 'head') and m.group(2) == 'right' and INTENS.match(t.lower(), m.end()):
            continue
        out.append(m)
    return out


def put(t, spans):
    """spans = [(start, end, new)] 를 뒤에서부터 바꿔 넣는다 (대소문자는 원래 구절을 따름)."""
    for s, e, new in sorted(spans, reverse=True):
        old = t[s:e]
        t = t[:s] + (new.upper() if old.isupper() else new[:1].upper() + new[1:] if old[:1].isupper() else new) + t[e:]
    return t


def sents(t):
    return [x for x in re.split(r'(?<=[.!?])\s+', t.strip()) if x]


def mut_dir(t, f, form, ctx):
    ms = motions(t)
    return put(t, [(m.start(), m.end(), form.format(d=FLIP[m.group(2)])) for m in ms]) if ms else None


def mut_order(t, f, form, ctx):
    ms = motions(t)
    seq = [m.group(2) for m in ms]
    if seq == seq[::-1]:
        return None
    return put(t, [(m.start(2), m.end(2), d) for m, d in zip(ms, seq[::-1])])


def mut_stairs(t, f, form, ctx):
    if form in FORMS['계단+'] and f['up'] or form in FORMS['계단-'] and f['down']:
        return None  # 경로에 이미 있는 쪽이면 틀린 말이 아님
    s = sents(t)
    ins = 'Then ' + form + '.'
    return ' '.join(s[:-1] + [ins] + s[-1:]) if len(s) > 1 else ins + ' ' + t.strip()


def _syn(r):
    return set(R['rooms'].get(r, [r]))


def mut_room(t, f, form, ctx, passed=False):
    if not f['rooms']:
        return None
    on_path = set().union(*(_syn(r) for r in f['rooms']))
    goal = _syn(f['rooms'][-1])
    if passed:  # 경로에서 지나온 방 (끝 방과 같은 말로 부를 수 있는 방은 제외)
        cands = [r for r in f['rooms'][:-1] if r != 'stairs' and R['rooms'].get(r) and not (_syn(r) & goal)]
    else:  # 이 집에 있지만 경로에 없는 방
        cands = [r for r in ctx['house_rooms'] if r != 'stairs' and R['rooms'].get(r) and not (_syn(r) & on_path)]
    if not cands:
        return None
    r = sorted(set(cands))[ctx['seed'] % len(set(cands))]
    return ' '.join(sents(t)[:-1] + [form.format(r=r)])


MUTS = {  # 이름: (함수, 표현 목록, 채점 항목)
    '방향': (mut_dir, FORMS['방향'], '방향/순서'),
    '순서': (mut_order, ['(원래 표현)'], '방향/순서'),
    '계단': (mut_stairs, FORMS['계단+'] + FORMS['계단-'], '계단'),
    '방-집': (mut_room, FORMS['방'], '방'),
    '방-지남': (lambda *a: mut_room(*a, passed=True), FORMS['방'], '방'),
}
CATS = ['방향/순서', '계단', '방', '단서']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--turn-min-deg', type=float, nargs='+', default=None,
                    help='회전 목록에 넣는 최소 각도 (여러 개 주면 방향·순서만 각각 비교). 기본 = route turn_deg')
    ap.add_argument('--forms', action='store_true', help='표현별 검출률도 출력')
    a = ap.parse_args()
    rc = C.il_config.load('b1', [])['route']
    names = rc['rooms']['names']
    its = C.A.select_subset(C.items('val_unseen'), 'dev')
    gt = C.gt_instructions('val_unseen')
    b1, _ = C.load_label('b1')
    eps = {it['ep']['episode_id']: it['ep'] for it in its}
    sets = {'사람': {k: gt[k] for k in eps}, 'b1': b1}
    facts = {k: path_facts(ep, rc) for k, ep in eps.items()}
    ctx = {k: {'seed': int(k), 'house_rooms': [names[r['label']] for r in load_regions(str(house_path(ep))) if names.get(r['label'])]}
           for k, ep in eps.items()}
    out = REPO / 'logs/gepa_labeling/check_scorer'
    out.mkdir(parents=True, exist_ok=True)
    print(f'문장 {len(eps)}개 (val_unseen dev), 회전 목록 하한 기본 {rc["turn_deg"]}°\n')

    print('[정답 시험] 원문 경보율 (낮을수록 좋음; 사람 문장 경보에는 사람의 진짜 실수도 섞임)')
    print(f'{"세트":5s} | ' + ' | '.join(f'{c:7s}' for c in CATS) + ' | 아무 항목')
    for name, texts in sets.items():
        cs = [cats(texts[k], facts[k]) for k in eps]
        print(f'{name:5s} | ' + ' | '.join(f'{100 * sum(c in x for x in cs) / len(cs):5.1f} %' for c in CATS)
              + f' | {100 * sum(bool(x) for x in cs) / len(cs):5.1f} %')

    print('\n[오답 시험] 일부러 틀리게 고친 문장을 그 항목에서 잡은 비율 (높을수록 좋음). 분모 = 문장 x 표현')
    print(f'{"세트":5s} | {"오답":6s} | 잡음/분모 | 검출률 | 가장 못 잡는 표현 (검출률)')
    for name, texts in sets.items():
        log = open(out / f'mutations_{name}.jsonl', 'w')
        for mname, (fn, forms, cat) in MUTS.items():
            by_form = collections.defaultdict(lambda: [0, 0])
            for k, f in facts.items():
                t = texts[k]
                if cat in cats(t, f):
                    continue
                for form in forms:
                    m = fn(t, f, form, ctx[k])
                    if m is None or m == t:
                        continue
                    hit = cat in cats(m, f)
                    by_form[form][0] += hit
                    by_form[form][1] += 1
                    if not hit:
                        log.write(json.dumps({'mut': mname, 'form': form, 'orig': t, 'mutated': m,
                                              'turns': [(i, round(x)) for i, x in f['turns']], 'up': f['up'], 'down': f['down'],
                                              'rooms': f['rooms']}, ensure_ascii=False) + '\n')
            hit, n = sum(v[0] for v in by_form.values()), sum(v[1] for v in by_form.values())
            worst = sorted(by_form.items(), key=lambda kv: kv[1][0] / kv[1][1])[:3]
            print(f'{name:5s} | {mname:6s} | {hit:4d}/{n:4d} | {100 * hit / max(1, n):5.1f} % | '
                  + ', '.join(f'"{fm}" {100 * v[0] / v[1]:.0f}%' for fm, v in worst))
            if a.forms and len(by_form) > 1:
                for fm, v in by_form.items():
                    print(f'{"":5s} | {"":6s} | {v[0]:4d}/{v[1]:4d} | {100 * v[0] / v[1]:5.1f} % | {fm}')
        log.close()

    if a.turn_min_deg:
        print('\n[회전 목록 하한 비교] 방향·순서 항목만. 방향 오답은 채점기가 아는 표현("turn {d}")과 전체 표현을 따로')
        print(f'{"하한":>5s} | {"세트":5s} | 정답 경보 | 방향("turn") | 방향(전체) | 순서')
        for deg in a.turn_min_deg:
            for name, texts in sets.items():
                fx = {k: dict(f, turns=[(i, x) for i, x in f['turns'] if abs(x) >= deg]) for k, f in facts.items()}
                fa = sum('방향/순서' in cats(texts[k], fx[k]) for k in eps) / len(eps)
                row = []
                for mname, forms in (('방향', ['turn {d}']), ('방향', FORMS['방향']), ('순서', ['(원래 표현)'])):
                    fn = MUTS[mname][0]
                    hit = n = 0
                    for k, f in fx.items():
                        if '방향/순서' in cats(texts[k], f):
                            continue
                        for form in forms:
                            m = fn(texts[k], f, form, ctx[k])
                            if m is None or m == texts[k]:
                                continue
                            n += 1
                            hit += '방향/순서' in cats(m, f)
                    row.append(f'{100 * hit / max(1, n):5.1f} % ({n})')
                print(f'{deg:4.0f}° | {name:5s} | {100 * fa:6.1f} % | ' + ' | '.join(row))


if __name__ == '__main__':
    main()
