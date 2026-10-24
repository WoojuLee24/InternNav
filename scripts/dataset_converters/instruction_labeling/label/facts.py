"""route.facts=memo-v1: 번호 명령문 목록("1. Turn around. 2. Walk forward ...") 대신 같은 경로 사실을 서술 메모로 준다.

사실 내용은 route_words 와 같다 (같은 geometry·같은 said()·같은 계단/거리 규칙) — 표현만 명령문이 아닌 서술.
b1 의 명령문 목록은 gemma 가 순서·표현을 그대로 옮겨 써서 형제 문장이 비슷해졌다 (형제 첫 3단어 같음 76 %).
e2e.build_request 가 guard clause 로 이 파일의 build_request_facts 에 위임한다 (b1 기본 경로는 그대로).
"""

import base64

from e2e import frame_caption, select_frames
from route_words import route_words, said
from style import episode_style


def _kind(a: float, rc: dict) -> str:
    side = 'right' if a > 0 else 'left'
    if abs(a) >= rc['around_deg']:
        return 'around'
    if abs(a) < rc['slight_deg']:
        return f'slight_{side}'
    return side


def facts_items(g: dict, rc: dict) -> tuple:
    """(출발 회전 종류 또는 None, 순서대로의 항목 키 목록). route_words 의 반복과 같은 규칙."""
    start = _kind(g['initial'], rc) if g['start_turn_said'] else None
    steps, walk = [], 0.0

    def flush():
        nonlocal walk
        if walk > 0:
            steps.append('few' if walk < rc['short_m'] else 'long' if walk >= rc['long_m'] else 'short')
        walk = 0.0

    for i, sg in enumerate(g['segments']):
        if i in g['turns'] and said(g['turns'][i], rc):
            flush()
            steps.append(_kind(g['turns'][i], rc))
        if abs(sg['dy']) > rc['stairs_dy_m']:
            flush()
            st = 'up' if sg['dy'] > 0 else 'down'
            if not (steps and steps[-1] == st):
                steps.append(st)
        else:
            walk += sg['dist']
    flush()
    return start, steps


def facts_memo(g: dict, rc: dict) -> str:
    F = rc['facts']
    start, steps = facts_items(g, rc)
    lines = [F['header']]
    if start:
        lines.append('- ' + F['start'].format(side=F['side'][start.split('_')[-1] if start.startswith('slight') else start]))
    if steps:
        lines.append('- ' + F['sequence' if start else 'sequence_first'].format(items=', '.join(F['items'][s] for s in steps)))
    lines.append('- ' + F['end'])
    rooms = g.get('rooms') or []
    if len(rooms) == 1:
        lines.append('- ' + F['rooms_single'].format(goal=rooms[0]))
    elif len(rooms) >= 2:
        mid = F['rooms_middle'].format(rooms=', '.join(rooms[1:-1])) if len(rooms) > 2 else ''
        lines.append('- ' + F['rooms'].format(first=rooms[0], middle=mid, goal=rooms[-1]))
    return '\n'.join(lines)


def build_request_facts(cfg: dict, ep: dict, track: dict) -> tuple:
    """e2e.build_request 와 같은 반환 (content, inputs, es). 사진·사진 설명·규칙·GT 예시는 b1 과 같고 경로 블록만 메모."""
    assert 'instruction' not in ep, 'GT 누출: 생성 입력에 instruction 이 있다'
    rc = cfg['route']
    sel = select_frames(cfg, ep, track)
    es = episode_style(cfg, ep, track['item'].get('sib_idx', 0))
    _, g = route_words(ep, rc, cfg['seed'])  # g: geometry + start_turn_said + rooms (route_words 와 같은 판단)
    memo = facts_memo(g, rc)
    content = []
    for k, f in enumerate(sel, 1):
        content.append({'type': 'text', 'text': f'Image {k}: {frame_caption(f, g, rc)}'})
        for r in cfg['frames']['rigs']:
            path = f['paths'][r] if 'paths' in f else track['img'](f['frame'], r)
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()}})
    sd = cfg['delivery']
    tmpl = cfg['style']['template']
    assert 'PATH MOTION SEQUENCE:\n{route}' in tmpl
    tmpl = tmpl.replace('PATH MOTION SEQUENCE:\n{route}', '{route}')  # 머리말은 메모 첫 줄 (facts.header)
    prompt = tmpl.format(route=memo, rules=es['rules'], extra=es['extra'],
                         scene_intro=sd['intro'].format(n_images=len(sel)), scene_context=sd['context'])
    content.append({'type': 'text', 'text': prompt})
    inputs = {'frames': [f['frame'] for f in sel], 'frame_labels': [f['label'] for f in sel], 'rigs': cfg['frames']['rigs'],
              'route_words': memo, 'rooms': g.get('rooms'), 'initial_turn_deg': round(g['initial'], 1),
              'start_turn_said': g['start_turn_said'], 'stop_style': es['stop'], 'stop_cue': None}
    if 'knobs' in es:
        inputs['knobs'] = es['knobs']
    return content, inputs, es
