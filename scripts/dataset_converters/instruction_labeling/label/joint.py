"""delivery=joint3-v1: 같은 경로의 형제 문장을 한 번의 요청으로 함께 쓴다 ("서로 다른 사람 n 명").

annotate.py 는 episode 마다 generate 를 부른다. 같은 trajectory 의 첫 호출이 형제 전부를 한 번에 만들고,
나머지 형제 호출은 그 결과를 기다려 자기 순번의 문장을 가져간다 (trajectory 별 lock).
사진·사실 메모·GT 예시는 형제 0 의 요청과 같고, 사람마다 손잡이 한 줄(style.knobs)이 다르다.
"""

import base64
import json
import re
import threading
from collections import Counter

from e2e import call_seed, frame_caption, quality_check, select_frames
from facts import facts_memo
from knobs import assign, knob_line, knob_style
from route_words import route_words
from sources import items

_LOCK = threading.Lock()
_TRAJ = {}      # (split, trajectory_id) -> {'lock', 'result'}
_NSIB = {}      # split -> Counter(trajectory_id)


def n_siblings(split: str, tid: int) -> int:
    with _LOCK:
        if split not in _NSIB:
            _NSIB[split] = Counter(it['ep']['trajectory_id'] for it in items(split))
    return _NSIB[split][tid]


def build_joint_request(cfg: dict, ep: dict, track: dict, n: int) -> tuple:
    assert 'instruction' not in ep, 'GT 누출: 생성 입력에 instruction 이 있다'
    rc, s, K, D = cfg['route'], cfg['style'], cfg['style']['knobs'], cfg['delivery']
    sel = select_frames(cfg, ep, track)
    _, g = route_words(ep, rc, cfg['seed'])
    content = []
    for k, f in enumerate(sel, 1):
        content.append({'type': 'text', 'text': f'Image {k}: {frame_caption(f, g, rc)}'})
        for r in cfg['frames']['rigs']:
            path = f['paths'][r] if 'paths' in f else track['img'](f['frame'], r)
            content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()}})
    knobs = [assign(cfg, ep, i) for i in range(n)]
    persons = []
    for i, kn in enumerate(knobs, 1):
        line = knob_line(kn, K)
        if kn['granularity'] not in K.get('drop_detail_for', []):
            line += ' ' + s['detail'] + '.'
        persons.append(f'- Person {i}: {line}')
    rules = '\n'.join(f'- {r}' for r in [s['turn_language'], s['numbers']])
    extra = knob_style(cfg, ep, 0)['extra']  # 형제 0 의 GT 예시 (b1 과 같은 뽑기)
    prompt = D['template'].format(scene_intro=D['intro'].format(n_images=len(sel)), n=n, route=facts_memo(g, rc),
                                  rules=rules, persons='\n'.join(persons), extra=extra)
    content.append({'type': 'text', 'text': prompt})
    return content, knobs, g, sel


def parse_list(s: str, n: int):
    m = re.search(r'\[.*\]', s or '', re.S)
    try:
        xs = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        xs = None
    if not isinstance(xs, list) or len(xs) != n or not all(isinstance(x, str) for x in xs):
        return None
    return [' '.join(x.strip().strip('"').split()) for x in xs]


def _generate_traj(client, cfg: dict, ep: dict, track: dict, n: int) -> dict:
    content, knobs, g, sel = build_joint_request(cfg, ep, track, n)
    gc = cfg['gen']
    qs = [knob_style(cfg, ep, i)['quality'] for i in range(n)]
    best, error = None, None
    for attempt in range(gc['max_retries']):
        try:
            r = client.chat.completions.create(
                model=cfg['vllm']['model'], messages=[{'role': 'user', 'content': content}],
                max_tokens=cfg['delivery']['max_tokens'], temperature=gc['temperature'] + gc['retry_temp_step'] * attempt,
                seed=call_seed(cfg['seed'], ep['trajectory_id'], 'joint', attempt))
            texts = parse_list(r.choices[0].message.content, n)
            if texts is None:
                error = 'parse_fail'
                continue
            issues = [quality_check(t, q) for t, q in zip(texts, qs)]
            cand = {'texts': texts, 'issues': issues, 'attempts': attempt + 1}
            if best is None or sum(map(len, issues)) < sum(map(len, best['issues'])):
                best = cand
            error = None
            if not any(issues):
                break
        except Exception as ex:  # noqa: BLE001 — 기록하고 재시도
            error = repr(ex)[:300]
    if best is None:
        best = {'texts': [''] * n, 'issues': [['empty']] * n, 'attempts': gc['max_retries']}
    best.update(knobs=knobs, error=error if not any(best['texts']) else None, g=g, sel=sel)
    return best


def generate(client, cfg: dict, ep: dict, track: dict) -> dict:
    split = track['item']['split']
    key = (split, ep['trajectory_id'])
    with _LOCK:
        slot = _TRAJ.setdefault(key, {'lock': threading.Lock(), 'result': None})
    with slot['lock']:  # 첫 형제가 만들고, 나머지는 기다렸다가 가져간다
        if slot['result'] is None:
            slot['result'] = _generate_traj(client, cfg, ep, track, n_siblings(split, ep['trajectory_id']))
    r = slot['result']
    i = track['item'].get('sib_idx', 0)
    g, sel = r['g'], r['sel']
    inputs = {'frames': [f['frame'] for f in sel], 'frame_labels': [f['label'] for f in sel], 'rigs': cfg['frames']['rigs'],
              'route_words': facts_memo(g, cfg['route']), 'rooms': g.get('rooms'), 'initial_turn_deg': round(g['initial'], 1),
              'start_turn_said': g['start_turn_said'], 'stop_style': r['knobs'][i]['ending'], 'stop_cue': None,
              'knobs': r['knobs'][i], 'joint_index': i, 'joint_texts': r['texts']}
    return {'text': r['texts'][i], 'quality_ok': not r['issues'][i], 'quality_issues': r['issues'][i], 'error': r['error'],
            'attempts': r['attempts'], 'inputs': inputs}
