"""06k 병렬 실행 — 에피소드를 N 조각으로 나눠 walk_rbq_worker.py 를 N 개 동시에 돌리고, 결과를 합쳐 경로 종류별로 요약한다.

한 worker 는 physics 가 시간의 ~85 % (PhysX step ≈ 1.75 ms, GPU/CPU dynamics·Kit 루프 생략과 무관 — 실측) 라서
빠르게 하려면 병렬뿐이다. worker 당 `--threads 4` → 24 코어에 5 개.

실험용 묶음 `--subset exp30`: 경로 종류(평지 S0 · 평지 S1 · 층 이동 S0→S1 · S0→S2 · S1→S2) 마다 6 개를 고르게 뽑는다 (30 ep, 5 worker 로 ~10 분).

사용 (isaac 컨테이너, 저장소 루트; 이 스크립트는 일반 python3 로 돈다):
    python3 scripts/dataset_converters/nav3d/noeun/42_walk_rbq.py --episodes_json <json> --tag <이름> --subset exp30 -- --steer_gain 1.5 --turn_threshold_deg 30
    ('--' 뒤 인자는 walk_rbq_worker.py 로 그대로 넘긴다)
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
ISAAC_PY = '/workspace/isaaclab/_isaac_sim/python.sh'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episodes_json', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--subset', default='all', choices=['all', 'exp30'])
    ap.add_argument('--paths_json', default='scripts/dataset_converters/nav3d/logs/noeun/paths/noeun_station_paths_meshfloor.json',
                    help='경로 종류(층) 를 읽을 04 json (episode_id 로 매칭)')
    ap.add_argument('--log_dir', default='scripts/dataset_converters/nav3d/logs/noeun/walk')
    ap.add_argument('--timeout_s', type=int, default=21600)
    ap.add_argument('--stagger_s', type=float, default=30.0, help='worker 시작 간격 — 동시에 띄우면 RTX 초기화가 CUDA 오류로 멈춘다 (실측)')
    ap.add_argument('--retries', type=int, default=1, help='결과가 모자란 조각을 다시 돌리는 횟수')
    ap.add_argument('rest', nargs=argparse.REMAINDER)
    return ap.parse_args()


def group_of(key, P):
    e = P.get(key.split(':')[1])
    if e is None:
        return '?'
    return f"{'평지' if e['kind'] == 'flat' else '층 이동'} S{e['from_storey']}→S{e['to_storey']}"


def main():
    args = parse()
    rest = [a for a in args.rest if a != '--']
    spec = json.loads(Path(args.episodes_json).read_text())
    P = {str(e['episode_id']): e for e in json.loads(Path(args.paths_json).read_text())['episodes']}
    eps = spec['episodes']
    if args.subset == 'exp30':
        g = defaultdict(list)
        for e in eps:
            g[group_of(e['key'], P)].append(e)
        eps = []
        for k in sorted(g):
            L = g[k]
            eps += [L[round(i * (len(L) - 1) / 5)] for i in range(6)] if len(L) >= 6 else L
    log = Path(args.log_dir)
    sub = log / f'eps_{args.tag}.json'
    sub.write_text(json.dumps(dict(spec, episodes=eps)))
    t0 = time.time()
    print(f'[06k par] {len(eps)} ep · worker {args.workers} × threads {args.threads} · 인자 {rest}', flush=True)

    def launch(i):
        cmd = [ISAAC_PY, str(HERE / 'noeun' / 'walk_rbq_worker.py'), '--episodes_json', str(sub), '--tag', f'{args.tag}_s{i}',
               '--shard', f'{i}/{args.workers}', '--threads', str(args.threads), '--log_dir', str(log)] + rest
        return subprocess.Popen(cmd, stdout=open(log / f'{args.tag}_s{i}.log', 'w'), stderr=subprocess.STDOUT)

    def done(i):
        f = log / f'walk_{args.tag}_s{i}.json'
        n = len(eps[i::args.workers])
        return f.exists() and len(json.loads(f.read_text())['results']) >= n

    todo = list(range(args.workers))
    for attempt in range(1 + args.retries):
        procs = {}
        for j, i in enumerate(todo):
            (log / f'walk_{args.tag}_s{i}.json').unlink(missing_ok=True)
            procs[i] = launch(i)
            if j + 1 < len(todo):
                time.sleep(args.stagger_s)
        for i, p in procs.items():
            try:
                p.wait(timeout=max(1, args.timeout_s - (time.time() - t0)))
            except subprocess.TimeoutExpired:
                p.kill()
        todo = [i for i in todo if not done(i)]
        if not todo:
            break
        print(f'[06k par] 결과가 모자란 조각 {todo} — 다시 ({attempt + 1}/{args.retries})', flush=True)
    R = []
    for i in range(args.workers):
        f = log / f'walk_{args.tag}_s{i}.json'
        if f.exists():
            R += json.loads(f.read_text())['results']
    (log / f'walk_{args.tag}.json').write_text(json.dumps({'args': vars(args), 'background': spec.get('background_usd'), 'results': R}))
    g = defaultdict(lambda: [0, 0])
    for r in R:
        k = group_of(r['key'], P)
        g[k][0] += r['success']
        g[k][1] += 1
    why = defaultdict(int)
    for r in R:
        if not r['success']:
            why[r['why'].split(' (')[0]] += 1
    print(f'[06k par] {args.tag}: {len(R)}/{len(eps)} ep · 성공 {sum(r["success"] for r in R)} · {time.time() - t0:.0f}s', flush=True)
    print('  ' + ' · '.join(f'{k} {v[0]}/{v[1]}' for k, v in sorted(g.items())), flush=True)
    print(f'  실패 사유 {dict(why)}', flush=True)


if __name__ == '__main__':
    main()
