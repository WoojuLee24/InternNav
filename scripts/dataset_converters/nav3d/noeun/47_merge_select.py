"""noeun 47 — 나눠 걸은 결과(walk_*.json 여러 개)를 합치고, split 마다 성공 경로를 골라 sel_<split>.json 을 만들고 라벨을 점검한다.
파이프라인(noeun_landmark_pipeline.sh) 의 '고르기' 와 같은 규칙: split 마다 episode_id 순서로, 성공한 것을 앞에서부터.
    python3 47_merge_select.py --walk_dir <dir> --tags f0 f1 tail --paths_json paths_meshfloor_straight_v5.json --eps_json eps_noeun_v5_narrow.json
결과: <walk_dir>/walk_all.json · sel_{train,eval,eval_seen}.json · stats.json"""
import argparse, collections, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--walk_dir', required=True)
ap.add_argument('--tags', nargs='+', required=True)
ap.add_argument('--paths_json', required=True)
ap.add_argument('--eps_json', required=True, help='narrow_wp 표시용')
ap.add_argument('--need', default='eval=80,eval_seen=80,train=1200')
ap.add_argument('--allow_missing', action='store_true', help='걷지 않은 ep 가 있어도 진행 (시험용)')
a = ap.parse_args()
W = Path(a.walk_dir)
R, bg = {}, None
for t in a.tags:
    d = json.load(open(W / f'walk_{t}.json')); bg = d.get('background') or bg
    for r in d['results']:
        i = int(r['key'].split(':')[1]); assert i not in R, f'{r["key"]} 가 두 번 걸림'
        R[i] = r
E = {e['episode_id']: e for e in json.load(open(a.paths_json))['episodes']}
NW = {e['key']: bool(e.get('narrow_wp')) for e in json.load(open(a.eps_json))['episodes']}
miss = sorted(set(E) - set(R))
print(f'걸은 ep {len(R)} / {len(E)} · 안 걸린 {len(miss)} {miss[:10]}')
assert a.allow_missing or not miss
json.dump({'background': bg, 'results': [R[i] for i in sorted(R)]}, open(W / 'walk_all.json', 'w'))

need = {k: int(v) for k, v in (x.split('=') for x in a.need.split(','))}
sel, stat = collections.defaultdict(list), collections.defaultdict(collections.Counter)
for i in sorted(E):
    s = E[i]['split']; r = R.get(i)
    stat[s]['walked'] += r is not None; stat[s]['ok'] += bool(r and r['success'])
    if r and r['success'] and len(sel[s]) < need[s]: sel[s].append(r)


def yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def labels(rs):
    c = collections.Counter()
    for r in rs:
        fr = r['frames']; a_ = [f['action'] for f in fr[1:]]
        t = [(i, x) for i, x in enumerate(a_) if x in (2, 3)]
        z = sum(1 for (i, x), (j, y) in zip(t, t[1:]) if x != y and j - i <= 2)
        c['act'] += len(a_); c['turn'] += len(t); c['zig'] += z; c['ep_zig3'] += z >= 3
        for i in range(1, len(fr)):
            if fr[i]['action'] == 1:
                c['fwd'] += 1
                d = abs((yaw(fr[i]['quat_wxyz']) - yaw(fr[i - 1]['quat_wxyz']) + math.pi) % (2 * math.pi) - math.pi)
                c['fwd_yaw_gt7.5'] += math.degrees(d) > 7.5
        c['frames'] += len(fr); c['narrow_ep'] += NW.get(r['key'], False)
    c['zig_pct'] = round(100 * c['zig'] / max(c['turn'], 1), 1); c['turn_pct'] = round(100 * c['turn'] / max(c['act'], 1), 1)
    return dict(c)


cls = collections.defaultdict(collections.Counter)
for r in R.values():
    k = 'narrow' if NW.get(r['key']) else r['key'].split(':')[2]
    cls[k]['n'] += 1; cls[k]['ok'] += r['success']
cls['all'] = collections.Counter(n=len(R), ok=sum(r['success'] for r in R.values()))
out = {'class': {k: dict(v) for k, v in cls.items()}, 'split': {s: dict(stat[s]) for s in stat},
       'selected': {s: len(sel[s]) for s in need}, 'labels': {s: labels(sel[s]) for s in need}}
for s, rs in sel.items():
    json.dump({'background': bg, 'results': rs}, open(W / f'sel_{s}.json', 'w'))
json.dump(out, open(W / 'stats.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
short = [s for s in need if len(sel[s]) < need[s]]
assert a.allow_missing or not short, f'성공 경로가 모자란다: {short}'
