"""방법(method) + 그룹별 인자 -> 생성 코드가 쓰는 resolved cfg.

  configs/methods/<method>.yaml   그룹별 인자 값의 조합 (b0, 나중에 b1 …). `base:` 로 상속 가능
  --set <group>.<arg>=<value>     인자 하나 바꾸기 (값이 그룹 전체면 --set <group>=<value>)
  결과 이름 = <method> + 바꾼 인자 suffix, 예: b0_start_turn-gtrate-v1, b0_delivery-2stage-v1
  인자가 가리키는 텍스트·표(route.yaml, style.yaml, delivery/, model/)를 **내용째** cfg 에 넣는다
  -> gen_hash 가 내용에 묶여서, 자원을 고치면 이전 결과와 섞이지 않는다 (고칠 때는 새 버전 키를 추가).
"""

import copy
import hashlib
import json
from pathlib import Path

import yaml

CONF = Path(__file__).resolve().parents[1] / 'configs'
GROUPS = ('frames', 'route', 'delivery', 'style', 'model', 'gen')
SCALAR_GROUPS = ('delivery', 'model')  # 그룹 값 자체가 이름 하나


def _yaml(p):
    return yaml.safe_load(open(p))


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_method(method: str) -> dict:
    raw = _yaml(CONF / 'methods' / f'{method}.yaml')
    assert raw.get('method') == method, f'{method}.yaml 의 method={raw.get("method")!r}'
    base = raw.pop('base', None)
    return deep_merge(load_method(base), raw) if base else raw


def _parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except Exception:  # noqa: BLE001
        return v


def apply_sets(m: dict, sets: list) -> tuple:
    """sets = ['route.start_turn=gtrate-v1', 'delivery=2stage-v1'] -> (method dict, suffix 목록)."""
    m = copy.deepcopy(m)
    names = {}
    for g, v in m['args'].items():
        if isinstance(v, dict):
            for a in v:
                names.setdefault(a, []).append(g)
    suffix = []
    for s in sets:
        key, val = s.split('=', 1)
        if '.' in key:
            g, a = key.split('.', 1)
            assert g in m['args'] and isinstance(m['args'][g], dict) and a in m['args'][g], f'알 수 없는 인자 {key}'
            m['args'][g][a] = _parse_value(val)
            tag = a if len(names[a]) == 1 else f'{g}.{a}'
        else:
            assert key in SCALAR_GROUPS, f'{key} 는 그룹 전체 값이 아니다 — <group>.<arg>=<value> 로 쓴다'
            m['args'][key] = val
            tag = key
        suffix.append(f'{tag}-' + val.strip('[]').replace(' ', '').replace(',', '+'))  # 파일 이름에 안전하게
    return m, suffix


def resolve(m: dict) -> dict:
    """method dict -> 생성 코드용 cfg (텍스트 자원까지 내용으로)."""
    a = m['args']
    route_res, style_res = _yaml(CONF / 'route.yaml'), _yaml(CONF / 'style.yaml')
    fr, rt, st, gn = a['frames'], a['route'], a['style'], a['gen']
    frames_res = _yaml(CONF / 'frames.yaml')
    assert fr['select'] in frames_res['select'], f'unknown frames.select={fr["select"]!r}'
    route = {k: rt[k] for k in ('turn_deg', 'slight_deg', 'around_deg', 'stairs_dy_m', 'short_m', 'long_m')}
    route['words'] = route_res['words'][rt['words']]
    route['slight'] = rt['slight']
    rooms_key = rt.get('rooms', 'none')
    route['rooms'] = None if rooms_key == 'none' else route_res['rooms'][rooms_key]
    cue_key = rt.get('stop_cue', 'none')
    route['stop_cue'] = None if cue_key == 'none' else route_res['stop_cue'][cue_key]
    assert rt['slight'] in ('say', 'omit'), f'unreachable route.slight={rt["slight"]!r}'
    if rt['start_turn'] == 'always':
        route['start_turn_rate'] = None
    else:
        route['start_turn_rate'] = route_res['start_turn_rate'][rt['start_turn']]
    assert st['stop'] in list(style_res['stop']) + ['gtdist'] + list(style_res.get('stop_mix', {})), f'unreachable style.stop={st["stop"]!r}'
    # 선택된 정지 표현 줄만 넣는다 (style.yaml 에 새 키를 추가해도 기존 방법의 gen_hash 가 바뀌지 않게)
    if st['stop'] == 'gtdist':
        stop_keys = list(style_res['stop_gtdist'])
    elif st['stop'] in style_res.get('stop_mix', {}):
        stop_keys = list(style_res['stop_mix'][st['stop']])
    else:
        stop_keys = [st['stop']]
    stop_lines = {k: style_res['stop'][k] for k in stop_keys}
    style = {'template': style_res['templates'][st['template']],
             'length_mode': st['length'], 'length': style_res['length'][str(st['length']) if st['length'] != 'gtdist' else 'gtdist'],
             'detail': style_res['detail'][st['detail']], 'turn_language': style_res['turn_language'][st['turn_language']],
             'stop_mode': st['stop'], 'stop_lines': stop_lines,
             'stop_gtdist': style_res['stop_gtdist'] if st['stop'] == 'gtdist' else None,
             'stop_mix': style_res['stop_mix'][st['stop']] if st['stop'] in style_res.get('stop_mix', {}) else None,
             'stop_no_require': [k for k in stop_lines if k in style_res['stop_no_require']],
             'numbers': style_res['numbers'][st['numbers']], 'concise': style_res['concise'][st['concise']],
             'structure': style_res['structure'][st['structure']], 'fewshot': int(st['fewshot']),
             'fewshot_header': style_res['fewshot_header'][st['fewshot_header']]}
    delivery = _yaml(CONF / 'delivery' / f'{a["delivery"]}.yaml')
    assert delivery['name'] == a['delivery']
    model = _yaml(CONF / 'model' / f'{a["model"]}.yaml')
    assert model['name'] == a['model']
    return {
        'method': m['method'], 'seed': m['seed'], 'workers': m['workers'], 'fail_abort_rate': m['fail_abort_rate'],
        'args': a,
        'frames': {'select': fr['select'], 'select_def': frames_res['select'][fr['select']],
                   'approach': None if fr.get('approach', 'none') == 'none' else frames_res['approach'][fr['approach']],
                   'turn_deg': fr['keyframe_deg'], 'rig': fr['rig'],
                   'rigs': [fr['rig']] + list(fr['extra_rigs']), 'max_images': fr['max_images']},
        'route': route, 'delivery': delivery, 'style': style,
        'vllm': {'model': model['model'], 'base_url': model['base_url'], 'api_key': model['api_key'],
                 'managed': model.get('managed')},
        'gen': {k: gn[k] for k in ('temperature', 'retry_temp_step', 'max_tokens', 'max_retries')},
        'quality': {k: gn[k] for k in ('min_words', 'max_words', 'require_stop_condition', 'require_landmark')},
    }


def load(method: str, sets: list) -> dict:
    m, suffix = apply_sets(load_method(method), sets)
    cfg = resolve(m)
    cfg['name'] = '_'.join([method] + suffix)
    cfg['sets'] = list(sets)
    return cfg


def _drop_none(x):
    """값이 None 인 키는 hash 에서 뺀다 -> 새 기능을 'None = 꺼짐' 으로 추가하면 기존 방법의 hash 가 안 바뀐다."""
    if isinstance(x, dict):
        return {k: _drop_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_drop_none(v) for v in x]
    return x


def gen_hash(cfg: dict, code_version: int) -> str:
    """텍스트에 영향을 주는 것만 (실행 설정·서버 주소·이름 제외).
    규칙: 새 옵션은 resolve() 에서 꺼졌을 때 None 이 되게 추가한다 (그래야 기존 결과의 hash 가 유지된다)."""
    base = {k: v for k, v in cfg.items() if k not in ('name', 'sets', 'method', 'workers', 'fail_abort_rate', 'args')}
    base['vllm'] = {'model': cfg['vllm']['model']}
    base['code_version'] = code_version
    return hashlib.sha1(json.dumps(_drop_none(base), sort_keys=True).encode()).hexdigest()[:12]
