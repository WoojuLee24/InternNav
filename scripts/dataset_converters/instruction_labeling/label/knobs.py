"""style.knobs=v1: 형제마다 다른 "손잡이" 조합 (끝맺음·설명 방식·길이). style.episode_style 이 guard clause 로 위임한다.

손잡이 = 사람이 고르는 말투 축 (gepa_labeling 0단계: 형제끼리 같은 값 비율이 무작위와 같음 → 경로가 아니라 사람이 정함).
첫머리·회전 말하기·출발 회전은 경로가 정하는 축이라 손잡이에 넣지 않는다.
조합 비율 = train GT 문장 비율 (style.yaml knobs.v1.rates). 형제 3개에 서로 다른 조합을 (seed, trajectory_id) 로 고정 추출.
GT 예시 3개·나머지 규칙 줄은 b1 과 같다 (episode_style 을 손잡이 없이 부른 결과에서 가져온다).
"""

import random


def assign(cfg: dict, ep: dict, sib_idx: int) -> dict:
    """형제 순번 -> {'ending', 'granularity', 'length'}. 같은 trajectory 의 형제는 서로 다른 조합 (비율 가중, 비복원)."""
    rates = [list(r) for r in cfg['style']['knobs']['rates']]
    rng = random.Random(f'{cfg["seed"]}:{ep["trajectory_id"]}:knobs')
    order = []
    pool = rates[:]
    while pool and len(order) <= sib_idx:
        k = rng.choices(range(len(pool)), weights=[r[3] for r in pool])[0]
        order.append(pool.pop(k))
    e, g, l, _ = order[sib_idx % len(order)]
    return {'ending': e, 'granularity': g, 'length': l}


def knob_line(kn: dict, K: dict) -> str:
    return K['instruction'].format(ending=K['ending'][kn['ending']], granularity=K['granularity'][kn['granularity']], length=K['length'][kn['length']])


def knob_style(cfg: dict, ep: dict, sib_idx: int) -> dict:
    from style import episode_style
    s = cfg['style']
    K = s['knobs']
    base = episode_style(dict(cfg, style=dict(s, knobs=None)), ep, sib_idx)  # b1 과 같은 GT 예시 (같은 rng 순서)
    kn = assign(cfg, ep, sib_idx)
    # b1 순서에서 length·stop 줄 대신 손잡이 한 줄. concise("Concise, direct") 줄은 길이 손잡이(long)와 부딪혀 뺀다
    detail = [] if kn['granularity'] in K.get('drop_detail_for', []) else [s['detail']]  # v2: turn 위주면 랜드마크 필수 줄을 뺀다
    rules = [knob_line(kn, K)] + detail + [s['turn_language'], s['numbers']]
    if s['structure']:
        rules.append(s['structure'])
    q = dict(base['quality'])
    q['min_words'], q['max_words'] = K['quality']['words'][kn['length']]
    if 'require_landmark' in K['quality']:  # v3: 물건 언급 필수 끄기 (손잡이 설정에서만)
        q['require_landmark'] = K['quality']['require_landmark']
    if kn['ending'] == 'none':
        q['require_stop_condition'] = False
        q['forbid'] = list(q.get('forbid', [])) + [K['quality']['none_forbid']]
    return {'rules': '\n'.join(f'- {r}' for r in rules), 'extra': base['extra'], 'quality': q, 'stop': kn['ending'], 'knobs': kn}
