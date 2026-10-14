"""reference_path -> 경로를 숫자 없는 단어로 (b0 의 route_words 이식).

vln-s2-pipeline `gate2_path/route_geometry.py` + `annotate.py` (route_words, _turn_words) 이식. 문구는 configs/route.yaml.
GT 지시문은 읽지 않는다 — reference_path 와 start_rotation 만 쓴다.

나침반 heading: 0 = -Z, +90 = +X. 부호 있는 회전각 b - a in [-180, 180): > 0 = 시계방향 = 오른쪽.
"""

import math
import random

CODE_VERSION = 2  # 경로 단어 '코드' 버전 (문구·확률표는 configs/route.yaml 에서 내용째 gen_hash 에 들어간다)


def yaw_from_quat(q):
    """R2R start_rotation [x, y, z, w] (순수 yaw) -> 나침반 heading."""
    return -math.degrees(2 * math.atan2(q[1], q[3]))


def heading(a, b):
    return math.degrees(math.atan2(b[0] - a[0], -(b[2] - a[2])))


def diff(a, b):
    return (b - a + 180) % 360 - 180


def route_geometry(ep):
    """{'segments': [{heading, dist, dy}], 'initial': 시작 방향 -> 첫 구간 각, 'turns': {wp: angle}}"""
    p = ep['reference_path']
    seg = [{'heading': heading(p[i], p[i + 1]), 'dist': math.dist((p[i][0], p[i][2]), (p[i + 1][0], p[i + 1][2])),
            'dy': p[i + 1][1] - p[i][1]} for i in range(len(p) - 1)]
    init = diff(yaw_from_quat(ep['start_rotation']), seg[0]['heading']) if seg else 0.0
    turns = {i: diff(seg[i - 1]['heading'], seg[i]['heading']) for i in range(1, len(seg))}
    return {'segments': seg, 'initial': init, 'turns': turns}


def turn_words(a, rc):
    w = rc['words']
    side = 'right' if a > 0 else 'left'
    if abs(a) >= rc['around_deg']:
        return w['around']
    if abs(a) < rc['slight_deg']:
        return w['slight'].format(side=side)
    return w['turn'].format(side=side)


def said(a, rc) -> bool:
    """이 회전을 말하는가: turn_deg 이상, route.slight=omit 이면 slight_deg 이상."""
    if rc['slight'] == 'say':
        return abs(a) >= rc['turn_deg']
    elif rc['slight'] == 'omit':
        return abs(a) >= max(rc['turn_deg'], rc['slight_deg'])
    else:
        assert False, f"unreachable route.slight={rc['slight']!r}"


def say_start_turn(ep, rc, angle, seed) -> bool:
    """route.start_turn: always -> 말할 만한 각이면 항상 / 확률표 -> 각도 구간의 GT 비율로 (episode 마다 고정)."""
    if not said(angle, rc):
        return False
    rate = rc['start_turn_rate']
    if rate is None:
        return True
    p = next((r for lo, hi, r in rate if lo <= abs(angle) < hi), 1.0)
    return random.Random(f"{seed}:{ep['episode_id']}:start_turn").random() < p


def route_words(ep, rc, seed=0):
    """'1. Turn around.\n2. Walk forward a short way.\n...' 와 geometry."""
    g = route_geometry(ep)
    steps, walk = [], 0.0
    w = rc['words']

    def flush():
        nonlocal walk
        if walk > 0:
            steps.append('Walk forward ' + ('a few steps' if walk < rc['short_m'] else
                                            'a long way' if walk >= rc['long_m'] else 'a short way') + '.')
        walk = 0.0

    g['start_turn_said'] = say_start_turn(ep, rc, g['initial'], seed)
    if g['start_turn_said']:
        steps.append(w['start_turn'].format(turn=turn_words(g['initial'], rc).capitalize()))
    for i, sg in enumerate(g['segments']):
        if i in g['turns'] and said(g['turns'][i], rc):
            flush()
            steps.append(turn_words(g['turns'][i], rc).capitalize() + '.')
        if abs(sg['dy']) > rc['stairs_dy_m']:
            flush()
            st_ = 'Go up the stairs.' if sg['dy'] > 0 else 'Go down the stairs.'
            if not (steps and steps[-1] == st_):
                steps.append(st_)
        else:
            walk += sg['dist']
    flush()
    steps.append('Stop (destination).')
    text = '\n'.join(f'{k}. {t}' for k, t in enumerate(steps, 1))
    if rc.get('rooms'):  # route.rooms: 지나는 방 순서 (MP3D scene annotation)
        from rooms import rooms_line
        from sources import house_path
        line, seq = rooms_line(ep, str(house_path(ep)), rc['rooms'])
        g['rooms'] = seq['rooms']
        g['rooms_coverage'] = seq['coverage']
        if line:
            text += '\n' + line
    return text, g
