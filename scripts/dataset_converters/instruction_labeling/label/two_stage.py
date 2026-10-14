"""two_stage 생성기: v278fix 의 장면 정보 전달 방식 (b0 와 다른 점은 이것 하나).

  stage 1 (scene) : 프레임마다 VLM 이 장면을 글로 요약 `ROOM: … | LANDMARKS: … | DIRECTION: …` (v278fix gate3 프롬프트),
                    마지막 프레임은 도착 지점 랜드마크 한 문장 (gate3 goal 프롬프트)
  stage 2 (writer): 이미지 없이 **텍스트만** — 프레임 캡션 + stage 1 요약 + 도착 랜드마크 + route words + 규칙
b0(e2e) 와 같게 둔 것: 프레임 선택, 프레임 캡션, route words, 문장 규칙, 생성 파라미터, 품질검사, 재시도, 모델.
프롬프트와 파라미터는 scene_delivery 정의(configs/scene_delivery/2stage-v*.yaml, cfg['delivery'])에서 온다.
"""

import base64

from e2e import call_seed, frame_caption, quality_check, select_frames
from route_words import route_words
from style import episode_style


def parse_scene(raw: str) -> dict:
    """v278fix `_parse_scene_description` 과 같음. 형식이 어긋나면 parse_ok=False 로 남긴다 (조용한 실패 방지)."""
    r = {'room': 'unknown', 'landmarks': [], 'direction': 'unclear', 'raw': raw}
    for part in raw.split('|'):
        part = part.strip()
        if part.startswith('ROOM:'):
            r['room'] = part[5:].strip().lower()
        elif part.startswith('LANDMARKS:'):
            r['landmarks'] = [x.strip() for x in part[10:].split(',') if x.strip()]
        elif part.startswith('DIRECTION:'):
            r['direction'] = part[10:].strip().lower()
    r['parse_ok'] = r['room'] != 'unknown' or bool(r['landmarks'])
    return r


def _image_call(client, cfg, img_path, prompt, max_tokens, seed):
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    resp = client.chat.completions.create(
        model=cfg['vllm']['model'],
        messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            {'type': 'text', 'text': prompt}]}],
        max_tokens=max_tokens, temperature=cfg['delivery']['temperature'], seed=seed)
    return (resp.choices[0].message.content or '').strip()


def scene_stage(client, cfg, ep, track, sel) -> dict:
    sc = cfg['delivery']
    rig = cfg['frames']['rig']
    per_frame, errors = [], []
    for f in sel:
        try:
            raw = _image_call(client, cfg, f['paths'][rig] if 'paths' in f else track['img'](f['frame'], rig),
                              sc['frame_prompt'], int(sc['max_tokens']),
                              call_seed(cfg['seed'], ep['episode_id'], 'scene', f['frame']))
            per_frame.append(parse_scene(raw))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e)[:200])
            per_frame.append({'room': 'unknown', 'landmarks': [], 'direction': 'unclear', 'raw': '', 'parse_ok': False})
    goal = ''
    try:
        last = [f for f in sel if 'paths' not in f][-1]  # 도착 프레임 (덧붙인 좌·우 시점 제외)
        goal = _image_call(client, cfg, track['img'](last['frame'], rig), sc['goal_prompt'],
                           int(sc['goal_max_tokens']), call_seed(cfg['seed'], ep['episode_id'], 'goal'))
    except Exception as e:  # noqa: BLE001
        errors.append(repr(e)[:200])
    return {'per_frame': per_frame, 'goal_landmark': goal, 'errors': errors}


def writer_prompt(cfg: dict, ep: dict, sel: list, s1: dict, sib_idx: int = 0) -> tuple:
    """stage 1 결과 -> writer 프롬프트 (VLM 호출 없음; 재현 검사에도 쓴다)."""
    rc = cfg['route']
    route, g = route_words(ep, rc, cfg['seed'])
    es = episode_style(cfg, ep, sib_idx)
    sd = cfg['delivery']
    lines = [sd['frame_line'].format(k=k, caption=frame_caption(f, g, rc), room=d['room'].title(),
                                     landmarks=', '.join(d['landmarks']), direction=d['direction'])
             for k, (f, d) in enumerate(zip(sel, s1['per_frame']), 1)]
    context = sd['context'].format(scene_context='\n'.join(lines) or sd['empty_context'],
                                   goal_landmark=s1['goal_landmark'] or sd['empty_goal'])
    prompt = cfg['style']['template'].format(route=route, rules=es['rules'], extra=es['extra'],
                                             scene_intro=sd['intro'], scene_context=context)
    return prompt, route, g, es


def generate(client, cfg: dict, ep: dict, track: dict) -> dict:
    gc = cfg['gen']
    sel = select_frames(cfg, ep, track)
    s1 = scene_stage(client, cfg, ep, track, sel)
    prompt, route, g, es = writer_prompt(cfg, ep, sel, s1, track['item'].get('sib_idx', 0))
    q = es['quality']
    text, issues, error, attempt = '', [], None, 0
    for attempt in range(gc['max_retries']):
        try:
            resp = client.chat.completions.create(
                model=cfg['vllm']['model'], messages=[{'role': 'user', 'content': prompt}],
                max_tokens=gc['max_tokens'], temperature=gc['temperature'] + gc['retry_temp_step'] * attempt,
                seed=call_seed(cfg['seed'], ep['episode_id'], 'writer', attempt))
            text = ' '.join((resp.choices[0].message.content or '').strip().strip('"').split())
            error = None if text else 'empty_response'
            issues = quality_check(text, q) if text else ['empty']
            if not issues:
                break
        except Exception as e:  # noqa: BLE001
            error, issues = repr(e)[:300], ['exception']
    if s1['errors'] and error is None:
        error = f'scene_stage: {s1["errors"][0]}'
    inputs = {'frames': [f['frame'] for f in sel], 'frame_labels': [f['label'] for f in sel], 'rigs': [cfg['frames']['rig']],
              'stop_style': es['stop'], 'route_words': route, 'rooms': g.get('rooms'), 'initial_turn_deg': round(g['initial'], 1), 'start_turn_said': g['start_turn_said'],
              'scene': s1, 'scene_parse_fail': sum(not d['parse_ok'] for d in s1['per_frame']), 'writer_prompt': prompt}
    return {'text': text, 'quality_ok': not issues, 'quality_issues': issues, 'error': error,
            'attempts': attempt + 1, 'inputs': inputs}
