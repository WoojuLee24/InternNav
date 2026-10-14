"""e2e 생성기 (b0 이식): 이미지 여러 장 + route words 를 VLM 한 번 호출에 넣어 지시문 한 개를 받는다.

vln-s2-pipeline `annotate.py` 의 e2e_episode / _frame_caption / call_seed / rac._quality_check 를 옮겼다.
바뀐 점: 프레임은 `frames.select_keyframes` (실제 궤적), 이미지는 InternNav 조건 렌더.
"""

import base64
import hashlib
import re

from frames import select_keyframes, subsample
from sources import view_path
from route_words import route_words, said, turn_words
from style import episode_style

LANDMARK_NOUNS = {'table', 'chair', 'couch', 'sofa', 'desk', 'bed', 'stairs', 'staircase', 'doorway', 'door',
                  'window', 'kitchen', 'bedroom', 'hallway', 'living', 'dining', 'bathroom', 'cabinet', 'shelf',
                  'rug', 'counter', 'fireplace'}


def call_seed(*parts) -> int:
    return int(hashlib.sha1(':'.join(map(str, parts)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def quality_check(text: str, q: dict) -> list:
    words = text.split()
    issues = []
    if len(words) < q.get('min_words', 8):
        issues.append(f'too_short({len(words)}w)')
    if len(words) > q.get('max_words', 80):
        issues.append(f'too_long({len(words)}w)')
    if q.get('require_stop_condition', True) and not any(
            w in text.lower() for w in ['stop', 'wait', 'stand', 'remain', 'halt']):
        issues.append('no_stop')
    if q.get('require_landmark', False) and not any(n in text.lower() for n in LANDMARK_NOUNS):
        issues.append('no_landmark')
    issues += [f'forbid:{r}' for r in q.get('forbid', []) if re.search(r, text, re.I)]
    return issues


def frame_caption(f: dict, g: dict, rc: dict) -> str:
    lab = f['label']
    if 'caption' in f:  # 덧붙인 시점(도착 좌·우)은 정의 파일의 캡션을 그대로 쓴다
        return f['caption']
    if lab == 'start':
        return 'start position, the direction you face before moving'
    if lab == 'goal':
        return 'destination, where you stop (facing the way you arrived)'
    t = g['turns'].get(f['waypoint_idx'])
    nxt = '' if t is None or not said(t, rc) else f'; here you {turn_words(t, rc)}'
    return f'along the route, facing the way forward{nxt}'


def select_frames(cfg: dict, ep: dict, track: dict) -> list:
    """입력 프레임 선택 (e2e·two_stage 공통). views 정의가 있으면 출발 시점은 start 뒤, 도착 시점은 맨 끝에 붙인다."""
    assert 'instruction' not in ep, 'GT 누출: 생성 입력에 instruction 이 있다'
    views = cfg['frames']['select_def'].get('views') or []
    sel = select_keyframes(ep, track['pose'], track['action'], float(cfg['frames']['turn_deg']))
    budget = int(cfg['frames']['max_images']) // len(cfg['frames']['rigs'])  # 이미지 상한 = 프레임 수 × rig 수
    sel = subsample(sel, budget - len(views))
    if not views:
        return sel
    extra = {'start': [], 'goal': []}
    for v in views:
        f = {'frame': v['label'], 'label': v['label'], 'waypoint_idx': None, 'caption': v['caption'],
             'paths': {r: view_path(track['item'], v['at'], v['turn_deg'], r) for r in cfg['frames']['rigs']}}
        for p in f['paths'].values():
            assert p.is_file(), f'추가 시점이 없다: {p} — render/extra_views.py --select {cfg["frames"]["select"]} 를 먼저 실행한다'
        extra[v['at']].append(f)
    return sel[:1] + extra['start'] + sel[1:] + extra['goal']


def build_request(cfg: dict, ep: dict, track: dict) -> tuple:
    """VLM 요청 content, 기록용 입력 요약, episode 별 style."""
    rc = cfg['route']
    sel = select_frames(cfg, ep, track)
    route, g = route_words(ep, rc, cfg['seed'])
    es = episode_style(cfg, ep, track['item'].get('sib_idx', 0))
    content = []
    rigs = cfg['frames']['rigs']
    for k, f in enumerate(sel, 1):
        content.append({'type': 'text', 'text': f'Image {k}: {frame_caption(f, g, rc)}'})
        for r in rigs:
            path = f['paths'][r] if 'paths' in f else track['img'](f['frame'], r)
            b64 = base64.b64encode(path.read_bytes()).decode()
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
    sd = cfg['delivery']
    prompt = cfg['style']['template'].format(route=route, rules=es['rules'], extra=es['extra'],
                                             scene_intro=sd['intro'].format(n_images=len(sel)),
                                             scene_context=sd['context'])
    content.append({'type': 'text', 'text': prompt})
    inputs = {'frames': [f['frame'] for f in sel], 'frame_labels': [f['label'] for f in sel], 'rigs': rigs,
              'route_words': route, 'rooms': g.get('rooms'), 'initial_turn_deg': round(g['initial'], 1), 'start_turn_said': g['start_turn_said'],
              'stop_style': es['stop']}
    return content, inputs, es


def generate(client, cfg: dict, ep: dict, track: dict) -> dict:
    content, inputs, es = build_request(cfg, ep, track)
    gc, q = cfg['gen'], es['quality']
    text, issues, error, attempt = '', [], None, 0
    for attempt in range(gc['max_retries']):
        try:
            resp = client.chat.completions.create(
                model=cfg['vllm']['model'], messages=[{'role': 'user', 'content': content}],
                max_tokens=gc['max_tokens'], temperature=gc['temperature'] + gc['retry_temp_step'] * attempt,
                seed=call_seed(cfg['seed'], ep['episode_id'], 'e2e', attempt))
            text = ' '.join((resp.choices[0].message.content or '').strip().strip('"').split())
            error = None if text else 'empty_response'
            issues = quality_check(text, q) if text else ['empty']
            if not issues:
                break
        except Exception as e:  # noqa: BLE001  — 기록하고 재시도, 끝까지 실패하면 error 로 집계
            error, issues = repr(e)[:300], ['exception']
    return {'text': text, 'quality_ok': not issues, 'quality_issues': issues, 'error': error,
            'attempts': attempt + 1, 'inputs': inputs}
