"""07r — 경로를 꺾인 점으로 줄이는 방법(route_vertices.py)별로, labeling 경로 단어가 GT 와 얼마나 같은지 잰다 (VLM 없음).

같은 GT 에피소드에 대해
  GT          : R2R reference_path 그대로 -> route_words  (기준)
  GT 촘촘     : reference_path 를 5 cm 로 촘촘히 한 뒤 방법 적용 -> route_words  (방법 자체의 오차; 경로 차이 없음)
  nav3d wp/sg : nav3d 03 경로에 방법 적용 -> route_words
일치 = 경로 단어에서 회전·계단 표현(turn left/right, bear slightly left/right, turn around, go up/down the stairs) 순서가 GT 와 같은 비율.
보조: 회전 단어 수 차 중앙, 첫 회전 일치율.

## 판정 기준 (결과를 보기 전에 정함)
| | 기준 |
|---|---|
| T1 방법 오차 | 'GT 촘촘' 일치율 ≥ 80 % (경로가 같으면 거의 같은 단어가 나와야 한다) |
| T2 개선 | wp 일치율이 rdp(기존)보다 높다 |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/label_route_words_check.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/
sys.path.insert(0, str(HERE / 'lib'))
LABEL = HERE.parent / 'instruction_labeling'
for p in (str(HERE), str(LABEL / 'label'), str(LABEL / 'render')):
    if p not in sys.path:
        sys.path.insert(0, p)
import config as LC  # noqa: E402
import nav3d_utils as nu  # noqa: E402
from route_vertices import simplify  # noqa: E402
from route_words import route_words  # noqa: E402

TURN = re.compile(r'(turn around|turn left|turn right|bear slightly left|bear slightly right|go up the stairs|go down the stairs)', re.I)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--episodes_json', default='scripts/dataset_converters/nav3d/logs/common/episodes.json')
    ap.add_argument('--nav03_dir', default='scripts/dataset_converters/nav3d/logs/common/refine_paths/scenes')
    ap.add_argument('--variants', default='rdp:0.3,rdp:0.6,turn:2.0:25,turn:2.0:30,turn:3.0:25,turn:1.5:25',
                    help="rdp:<eps> | rdp_merge:<eps>:<merge_m> | turn:<window_m>:<turn_deg>")
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/checks/label_route_words_check/summary.json')
    return ap.parse_args()


def seq(text):
    return [t.lower() for t in TURN.findall(text)]


def main():
    args = parse()
    from release_io import load_cfg, match_raw, release_episodes
    rc = load_cfg(LABEL / 'configs' / 'render.yaml')
    cfg = LC.load('b0', [])
    rcfg = cfg['route']
    sel = json.loads(Path(args.episodes_json).read_text())
    eps = []
    for scene, ids in sel.items():
        rel = {e['episode_index']: e for e in release_episodes(rc, scene)}
        Z = np.load(Path(args.nav03_dir) / f'{scene}_traj.npz')
        for ep in ids:
            raw = match_raw(rc, scene, rel[ep]['task'])
            gt_path = nu.hab_to_ours(np.asarray(raw['reference_path']))
            eps.append({'scene': scene, 'ep': ep, 'raw': raw, 'gt': gt_path,
                        'wp': Z[f'{ep}_wp_ref'].astype(float) if f'{ep}_wp_ref' in Z.files else None,
                        'sg': Z[f'{ep}_sg_ref'].astype(float) if f'{ep}_sg_ref' in Z.files else None})

    def words(raw, P_ours):
        e = dict(raw, reference_path=nu.ours_to_hab(P_ours).tolist())
        return route_words(e, rcfg, 0)[0]

    res = {}
    for v in args.variants.split(','):
        k = v.split(':')
        if k[0] == 'rdp':
            kw = {'method': 'rdp', 'rdp_eps': float(k[1])}
        elif k[0] == 'rdp_merge':
            kw = {'method': 'rdp_merge', 'rdp_eps': float(k[1]), 'merge_m': float(k[2])}
        elif k[0] == 'turn':
            kw = {'method': 'turn', 'window_m': float(k[1]), 'turn_deg': float(k[2])}
        else:
            assert False, f'unreachable variant={v!r}'
        row = {}
        for src in ('gt_dense', 'wp', 'sg'):
            m, dn, first, n = [], [], [], 0
            for e in eps:
                base = seq(route_words(e['raw'], rcfg, 0)[0])
                if src == 'gt_dense':
                    P = nu.densify(e['gt'], 0.05)
                else:
                    P = e[src]
                    if P is None:
                        continue
                s = seq(words(e['raw'], simplify(P, **kw)))
                m.append(s == base)
                dn.append(len(s) - len(base))
                first.append((s[:1] == base[:1]))
                n += 1
            row[src] = {'n': n, 'match': float(np.mean(m)), 'len_diff_median': float(np.median(dn)), 'len_diff_mean': float(np.mean(dn)),
                        'first_match': float(np.mean(first))}
        res[v] = row
        print(f"{v:14s} " + ' · '.join(f"{s} 일치 {r['match'] * 100:4.0f} % (단어 수 차 평균 {r['len_diff_mean']:+.2f}, 첫 회전 {r['first_match'] * 100:3.0f} %)" for s, r in row.items()), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'variants': res, 'n_episodes': len(eps)}, open(out, 'w'), indent=1)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
