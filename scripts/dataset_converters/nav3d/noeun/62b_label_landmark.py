"""noeun 62b · 랜드마크 경로 라벨 — 62_label.py 의 b1 · '후'(post) 생성을 노은역 랜드마크 학습 데이터(vlnce_train_t12)에 붙인다.

62 는 R2R raw json(`raw_data/r2r/<split>/<split>.json.gz`, habitat 좌표) 을 입력으로 받는데, 랜드마크 데이터에는 raw json 이 없다.
62b 는 62 를 고치지 않고 (기존 파일 수정 금지) 62 의 `items()` 만 바꿔 끼운다 — 생성(run · e2e.generate · joint_results),
설정(METHODS · NOEUN_SETS = rig 60cm_15deg · rooms noeun-v1), 출력 형식, gen_hash 는 62 와 같다.

입력 item 만드는 법 (ep 번호 i = 51b 렌더 순서 = sel_train 의 성공 순서 = render_meta.json 의 episode):
  reference_path : **걸은 자세** (sel_train frames pos, 바닥 = trunk z − 0.55 — 51b 와 같음) 를 habitat (x, z, −y) 로 바꾸고 RDP(--simplify_m).
                   계획 경로가 아니라 걸은 경로 = 사진이 찍힌 경로. 시작 프레임 + action 1(앞으로) 로 도달한 프레임만 쓴다
                   (제자리 회전 중 몸 흔들림 ≤ 0.4 m 가 꺾임으로 남지 않게). frame k 의 action = k 로 오게 한 동작 (0 번은 -1).
  start_rotation : 첫 프레임 몸 yaw -> label_paths.quat_from_nav3d_yaw.
  goals          : 걸은 마지막 점 (radius 3.0, 62 원본 raw 와 같음).  info.geodesic_distance = paths json path_length_m.
  src            : paths json 의 key · kind · stop_cue(랜드마크) — 생성에는 안 쓰고 meta 에 남긴다 (나중 점검용).

사용 (저장소 루트, habitat 환경 python, vLLM 서버가 떠 있어야 함):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/62b_label_landmark.py --data_root <vlnce_train_t12> --methods b1,post --base_url http://127.0.0.1:8100/v1 --limit 4
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/62b_label_landmark.py --data_root <vlnce_train_t12> --dry_run --limit 20   (서버 없이 입력·경로 단어·구역만 확인)
"""

import argparse
import importlib.util
import json
import math
import threading
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('noeun62', HERE / '62_label.py')
L62 = importlib.util.module_from_spec(sp)
sys.modules['noeun62'] = L62
sp.loader.exec_module(L62)          # 62 의 import(label_paths · e2e · noeun_rooms …) 와 경로 설정을 그대로 쓴다
LP = L62.LP

DS = 'landmark_t12'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', required=True, help='vlnce_train_t12 (traj_data/r2r/noeunstation + noeun_meta)')
    ap.add_argument('--out_root', default='', help='빈값 = <data_root>/../instruction_labels')
    ap.add_argument('--methods', default='b1,post', help='b1 | post (쉼표)')
    ap.add_argument('--model', default='gemma31b-bf16')
    ap.add_argument('--n_sib', type=int, default=3)
    ap.add_argument('--simplify_m', type=float, default=0.3)
    ap.add_argument('--base_height', type=float, default=0.55, help='trunk 높이 − 이 값 = 바닥 (51b 와 같음)')
    ap.add_argument('--limit', type=int, default=0, help='앞에서 ep N 개 (0 = 전부)')
    ap.add_argument('--ep_list', default='', help='쉼표로 ep 번호 (시험용, --limit 보다 먼저)')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--managed', action='store_true')
    ap.add_argument('--base_url', default='')
    ap.add_argument('--gpus', default='')
    ap.add_argument('--dry_run', action='store_true', help='서버 없이 입력 · 경로 단어 · 구역 순서만 출력')
    ap.add_argument('--landmark_nouns', default='mp3d',
                    help='품질검사 no_landmark 가 "물건을 말했다" 고 인정하는 단어 목록. mp3d = e2e.LANDMARK_NOUNS 그대로 (집 물건 22 개, 기본) | '
                         '<이름> = landmark_nouns/<이름>.json (예 noeun-v1) | json 파일 경로. mp3d 가 아니면 출력이 <out_root>/nouns-<이름>/ 아래로 간다')
    ap.add_argument('--rooms', default='noeun-v1', help='route.rooms 키: noeun-v1 (62 와 같음, 기본) | noeun-v2 (+ 바깥 street · 층 가장자리, 61b)')
    ap.add_argument('--stop_cue', default='none', help='route.stop_cue 키: none (기본, b1 · 후 그대로) | noeun-paths-v1 (경로의 랜드마크를 한 줄로; 후 는 facts 메모 끝에 한 줄)')
    ap.add_argument('--split', default='train', choices=['train', 'eval', 'eval_seen'],
                    help='train (기본, 학습 1,200) | eval (평가 A · 안 본 랜드마크) | eval_seen (평가 B · 본 랜드마크) — noeun_meta/sel_<split>.json 을 읽고 출력 폴더 landmark_t12_<split>')
    a = ap.parse_args()
    a.datasets, a.splits = DS, a.split
    a.out_root = a.out_root or str(Path(a.data_root).parent / 'instruction_labels')
    return a


def apply_landmark_nouns(a):
    """--landmark_nouns: e2e.LANDMARK_NOUNS 를 데이터셋 목록으로 바꿔 끼운다 (e2e 는 고치지 않음).
    gen_hash 에는 안 들어가므로 (코드 상수) 섞이지 않게 출력 폴더를 나누고 목록을 함께 저장한다."""
    if a.landmark_nouns == 'mp3d':
        return
    p = Path(a.landmark_nouns)
    p = p if p.suffix == '.json' else HERE / 'landmark_nouns' / f'{a.landmark_nouns}.json'
    d = json.loads(p.read_text())
    L62.e2e.LANDMARK_NOUNS = set(d['nouns'])
    a.out_root = str(Path(a.out_root) / f'nouns-{d["name"]}')
    Path(a.out_root).mkdir(parents=True, exist_ok=True)
    (Path(a.out_root) / 'landmark_nouns.json').write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f'[62b] 품질검사 단어 {d["name"]} ({len(d["nouns"])} 개) -> 출력 {a.out_root}', flush=True)


def body_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def items(a, ds: str, split: str) -> list:
    assert ds == DS and split in ('train', 'eval', 'eval_seen'), f'unreachable ds/split={ds}/{split}'
    meta = Path(a.data_root) / 'noeun_meta'
    W = [r for r in json.loads((meta / f'sel_{split}.json').read_text())['results'] if r['success']]
    RM = json.loads((meta / 'render_meta.json').read_text())['episodes']
    P = {e['episode_id']: e for e in json.loads((meta / 'paths_meshfloor_straight_v5.json').read_text())['episodes']}
    assert len(W) == len(RM) and all(r['key'] == m['key'] and len(r['frames']) == m['frames'] for r, m in zip(W, RM)), f'sel_{split} 과 render_meta 순서가 다르다'
    root = Path(a.data_root) / 'traj_data' / 'r2r'
    eps = [int(x) for x in a.ep_list.split(',')] if a.ep_list else list(range(a.limit or len(W)))
    assert a.n_sib <= 10
    out = []
    for i in eps:
        r = W[i]
        pid = int(r['key'].split(':')[1])
        pe = P[pid]
        # 시작 + 앞으로(action 1) 로 도달한 프레임만: 제자리 회전 중 몸이 0.4 m 까지 흔들려 RDP 가 그걸 꺾임으로 남기면 첫 구간 방향이 틀어진다
        xyz = np.array([f['pos'] for k, f in enumerate(r['frames']) if k == 0 or f['action'] == 1], float)
        xyz[:, 2] -= a.base_height
        Wh = LP.nu.ours_to_hab(xyz)
        ref = LP.rdp(Wh, a.simplify_m)
        e0 = {'scene_id': 'noeun/noeun_station/noeun_station_collision.usdz', 'start_position': ref[0].tolist(),
              'start_rotation': LP.quat_from_nav3d_yaw(body_yaw(r['frames'][0]['quat_wxyz'])),
              'goals': [{'position': ref[-1].tolist(), 'radius': 3.0}], 'reference_path': ref.tolist(),
              'info': {'geodesic_distance': float(pe['path_length_m'])}}
        for k in range(a.n_sib):
            out.append({'ep': dict(e0, episode_id=i * 10 + k, trajectory_id=i), 'scene': 'noeunstation', 'root': root, 'episode_index': i,
                        'split': f'noeun_{ds}_{split}', 'sib_idx': k,
                        'src': {'source': 'noeun_landmark', 'dataset': ds, 'split': split, 'key': r['key'], 'kind': pe['kind'],
                                'from_storey': pe.get('from_storey'), 'to_storey': pe.get('to_storey'), 'stop_cue': pe.get('stop_cue'),
                                'final_xy': [float(v) for v in r['frames'][-1]['pos'][:2]],
                                'final_yaw': body_yaw(r['frames'][-1]['quat_wxyz'])}})
    return out


_LM_NAMES = {}


def anchor_name(anc: dict) -> str:
    """기준점 이름: anchor ids 중 이름이 있는 첫 랜드마크 (landmark_nouns/landmark_names_v1.json, goals.json 에서 뽑음). 없으면 종류(key)."""
    if not _LM_NAMES:
        _LM_NAMES.update(json.loads((HERE / 'landmark_nouns' / 'landmark_names_v1.json').read_text())['names'])
    return next((_LM_NAMES[str(i)] for i in anc['ids'] if str(i) in _LM_NAMES), anc['key'])


def paths_stop_cue(sc: dict, src: dict) -> dict:
    """route.stop_cue kind=paths: 경로를 만들 때 고른 랜드마크 -> 한 줄. 방향은 마지막 자세 기준 (nav3d z-up, yaw 반시계 +)."""
    c = src['stop_cue']
    name = c['name']
    if c['part'] != 'whole':
        where = sc['parts'][c['part']].format(name=name)
    elif c.get('anchor'):
        where = sc['anchor'].format(name=name, anchor=anchor_name(c['anchor']))
    elif c['ordinal'] >= 2:
        where = sc['ordinal'].format(name=name, nth=sc['nth'][c['ordinal']])
    else:
        where = sc['whole'].format(name=name)
    tx, ty = (c['target_xyz'] if c['part'] != 'whole' else c['xyz'])[:2]
    fx, fy = src['final_xy']
    bear = math.degrees(math.atan2(ty - fy, tx - fx) - src['final_yaw'])
    bear = (bear + 180) % 360 - 180                                   # + = 왼쪽 (반시계)
    if abs(bear) <= sc['ahead_deg']:
        side = 'ahead'
    elif abs(bear) > sc['behind_deg']:
        side = None
    elif bear > 0:
        side = 'left'
    else:
        side = 'right'
    line = sc['line'].format(where=where, side=sc['side_words'][side] if side else '')
    return {'line': line, 'cands': [{'name': name, 'part': c['part'], 'ordinal': c['ordinal'], 'anchor': (c.get('anchor') or {}).get('key'),
                                     'side': side, 'bearing_deg': round(bear, 1), 'dist_m': c['dist_m']}]}


_CTX = threading.local()                 # 후(joint): 지금 쓰는 경로의 멈출 곳 줄


def _facts_memo(g: dict, rc: dict) -> str:
    """joint.facts_memo 대신: 원래 메모 + 멈출 곳 줄 한 줄 (b1 이 경로 단어 뒤에 붙이는 것과 같은 문장)."""
    memo = _orig_facts_memo(g, rc)
    line = getattr(_CTX, 'line', '')
    return memo + '\n- ' + line if line else memo


def _joint_results(client, cfg: dict, its: list) -> dict:
    """LP.joint_results 대신: stop_cue 가 켜졌으면 경로마다 멈출 곳 줄을 정해 facts 메모에 넣고, inputs.stop_cue 에 기록."""
    sc = cfg['route'].get('stop_cue')
    if not sc:
        return _orig_joint_results(client, cfg, its)
    out = {}
    by_t = {}
    for it in its:
        by_t.setdefault(it['ep']['trajectory_id'], []).append(it)
    for sib in by_t.values():
        cue = paths_stop_cue(sc, sib[0]['src'])
        _CTX.line = cue['line']
        try:
            r = _orig_joint_results(client, cfg, sib)
        finally:
            _CTX.line = ''
        for v in r.values():
            v['inputs']['stop_cue'] = cue['cands']
        out.update(r)
    return out


def _load(method: str, sets: list) -> dict:
    """config.load 대신: route.facts(후) + route.stop_cue 는 config 가 막는다 (facts 경로에 stop_cue 자리가 없어서).
    62b 는 위 _facts_memo 로 자리를 만들었으니, stop_cue 를 빼고 읽은 뒤 붙인다 — 이름 · gen_hash 에 stop_cue 가 들어간다."""
    cue = [s for s in sets if s.startswith('route.stop_cue=')]
    if not cue or not any(s.startswith('route.facts=') for s in sets):
        return _orig_load(method, sets)
    key = cue[0].split('=', 1)[1]
    cfg = _orig_load(method, [s for s in sets if s not in cue])
    cfg['route']['stop_cue'] = L62.LC._yaml(L62.LC.CONF / 'route.yaml')['stop_cue'][key]
    assert cfg['route']['stop_cue']['kind'] == 'paths', '후 에는 kind=paths stop_cue 만 (62b 가 만든 자리)'
    cfg['name'] += f'_stop_cue-{key}'
    cfg['sets'] = list(sets)
    return cfg


import joint  # noqa: E402  (62 가 잡아 둔 label/ 경로)
_orig_facts_memo, _orig_joint_results, _orig_load = joint.facts_memo, LP.joint_results, L62.LC.load
_orig_stop_cue_line = L62.e2e.stop_cue_line


def stop_cue_line(rc: dict, ep: dict, track: dict) -> dict:
    """e2e.stop_cue_line 대신: kind=paths 만 여기서, 나머지는 원래 함수 그대로."""
    sc = rc.get('stop_cue')
    if sc and sc['kind'] == 'paths':
        return paths_stop_cue(sc, track['item']['src'])
    return _orig_stop_cue_line(rc, ep, track)


def dry_run(a):
    """서버 없이: 경로 단어(route_words) · 구역 순서(noeun_rooms) · 구역 coverage · 시작 방향 확인."""
    import config as LC
    import noeun_rooms
    import route_words as RW
    cfg = LC.load('b1', L62.METHODS['b1'] + [f'model={a.model}'] + L62.NOEUN_SETS)
    rc_rooms = cfg['route']['rooms']
    its = [it for it in items(a, DS, 'train') if it['sib_idx'] == 0]
    cov = []
    for it in its:
        e = it['ep']
        g = RW.route_geometry(e)
        rs = noeun_rooms.room_sequence(e, '', rc_rooms)
        cov.append(rs['coverage'])
        sc = it['src']['stop_cue'] or {}
        print(f"ep {it['episode_index']:4d} {it['src']['key']:<22} ref {len(e['reference_path']):3d} pts · 처음 회전 {g['initial']:+6.1f}° · "
              f"구역 {rs['coverage']:.2f} {' > '.join(rs['rooms'])} · 랜드마크 {sc.get('name')} (#{sc.get('ordinal')})")
    print(f'[62b dry] ep {len(its)} · 구역 coverage 최소 {min(cov):.2f} 중앙 {float(np.median(cov)):.2f}')


def main():
    a = parse()
    L62.items = items                    # 62 의 run() 이 이 함수로 입력을 만든다
    L62.NOEUN_SETS = ['frames.rig=60cm_15deg', f'route.rooms={a.rooms}'] + ([f'route.stop_cue={a.stop_cue}'] if a.stop_cue != 'none' else [])
    L62.e2e.stop_cue_line = stop_cue_line
    joint.facts_memo = _facts_memo          # build_joint_request · joint_results 가 이 이름으로 부른다
    LP.joint_results = _joint_results       # 62 run() 이 LP.joint_results 로 부른다
    L62.LC.load = _load
    if a.dry_run:
        L62.rooms.room_sequence = L62.noeun_rooms.room_sequence
        return dry_run(a)
    apply_landmark_nouns(a)
    L62.parse = lambda: a
    L62.main()


if __name__ == '__main__':
    main()
