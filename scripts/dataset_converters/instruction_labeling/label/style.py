"""style 그룹 -> episode 별 규칙 줄·추가 블록·품질검사 설정.

규칙 순서 (b0 와 같음): length, detail, turn_language, stop, numbers, concise, (structure)
episode 별 무작위 선택(stop gtdist, length gtdist, fewshot)은 (seed, episode_id, 이름) 으로 고정한다.
fewshot 예시는 GT **train** 문장에서만 뽑고, 같은 trajectory 는 뺀다 (평가 split 문장은 쓰지 않는다).
"""

import random
import re

from sources import load_raw

_TRAIN = None


def _train_pool():
    global _TRAIN
    if _TRAIN is None:
        eps = load_raw('train')
        pool = [(e['trajectory_id'], e['instruction']['instruction_text'].strip()) for e in eps]
        _TRAIN = ([len(t.split()) for _, t in pool], pool)
    return _TRAIN


def sibling_profile(cfg: dict, ep: dict, sib_idx: int, names: list) -> str:
    """형제 순번 -> 서술 방식 이름. trajectory 마다 섞은 순서 (style.sibling / style.fewshot_match 공통)."""
    order = random.Random(f'{cfg["seed"]}:{ep["trajectory_id"]}:sibling').sample(names, len(names))
    return order[sib_idx % len(order)]


_MATCH = {}


def gt_style(t: str, fm: dict) -> str:
    """GT 문장 문체 묶음: brief | step | landmark | other (style.yaml fewshot_match)."""
    if len(t.split()) <= fm['brief_max_words']:
        return 'brief'
    k = len(re.findall(fm['turn_re'], t, flags=re.I))
    if k >= fm['step_min_turns']:
        return 'step'
    if re.search(fm['landmark_re'], t, flags=re.I):
        return 'landmark'
    return 'other'


def _match_pool(fm: dict, name: str) -> list:
    key = (id(fm), name)
    if key not in _MATCH:
        _MATCH[key] = [(tid, t) for tid, t in _train_pool()[1] if gt_style(t, fm) == name]
    return _MATCH[key]


def episode_style(cfg: dict, ep: dict, sib_idx: int = 0) -> dict:
    s = cfg['style']
    rng = random.Random(f'{cfg["seed"]}:{ep["episode_id"]}:style')
    if s['length_mode'] == 'gtdist':
        length = s['length'].format(n=rng.choice(_train_pool()[0]))
    else:
        length = s['length']
    stop = s['stop_mode']
    if stop == 'gtdist':
        d = s['stop_gtdist']
        stop = rng.choices(list(d), weights=list(d.values()))[0]
    elif s['stop_mix']:  # 형제마다 다른 끝맺음: trajectory 별로 섞은 순서에서 형제 번호로 고른다
        order = random.Random(f'{cfg["seed"]}:{ep["trajectory_id"]}:stop_mix').sample(s['stop_mix'], len(s['stop_mix']))
        stop = order[sib_idx % len(order)]
    rules = [length, s['detail'], s['turn_language'], s['stop_lines'][stop], s['numbers'], s['concise']]
    if s['structure']:
        rules.append(s['structure'])
    profile = None
    if s.get('sibling'):  # 형제마다 다른 서술 방식 줄 (길이가 있으면 길이 규칙 줄을 대체)
        prof = {p['name']: p for p in s['sibling']}
        profile = sibling_profile(cfg, ep, sib_idx, list(prof))
        if prof[profile].get('length'):
            rules[0] = prof[profile]['length']
        rules.append(prof[profile]['line'])
    extra = ''
    if s['fewshot']:
        pool = _train_pool()[1]
        if s.get('fewshot_match'):  # 형제 서술 방식과 같은 GT 문체 묶음에서만 예시를 뽑는다
            fm = s['fewshot_match']
            profile = profile or sibling_profile(cfg, ep, sib_idx, fm['profiles'])
            pool = _match_pool(fm, profile)
        ex = [t for tid, t in rng.sample(pool, 4 * s['fewshot']) if tid != ep['trajectory_id']][:s['fewshot']]
        extra = s['fewshot_header'] + ''.join(f'- {t}\n' for t in ex).rstrip('\n')
    q = dict(cfg['quality'])
    if stop in s['stop_no_require']:
        q['require_stop_condition'] = False
    out = {'rules': '\n'.join(f'- {r}' for r in rules), 'extra': extra, 'quality': q, 'stop': stop}
    if profile:
        out['sibling_profile'] = profile
    return out
