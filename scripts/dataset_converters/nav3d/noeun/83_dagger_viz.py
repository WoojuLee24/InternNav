"""noeun 83 · DAgger 롤아웃 점검 그림 — 82 결과 에피소드마다 (1) 위에서 본 경로 (2) 프레임에 전문가 pixel goal 을 그린다.

(1) topdown_<ep>.png: 정답 경로(회색) · 롤아웃(모델 실행 파랑 / 전문가 실행 주황) · 시작(초록) · 끝(빨강), 색 진하기 = 높이 무관.
(2) frames_<ep>.jpg: 고른 프레임에 전문가 정답 — pixel(초록 원 + goal_len) / 턴(글자) / STOP — 과 이탈 거리.
마지막에 전체 라벨 종류 · 끝난 이유 · 이탈 거리 요약을 출력한다.

    /usr/bin/python3 scripts/dataset_converters/nav3d/noeun/83_dagger_viz.py --root <82 out_root> --raw_json <82 에 준 raw_json> --out <dir>
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ACT = {0: 'STOP', 1: 'F', 2: 'L', 3: 'R', -1: '-'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--raw_json', required=True)
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--n_frames', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = {e['episode_id']: e for e in json.load(gzip.open(a.raw_json))['episodes']}
    kinds, ends, devs = Counter(), Counter(), []
    for sc in sorted((Path(a.root) / 'traj_data' / 'r2r').glob('dagger_*')):
        for m in (json.loads(l) for l in open(sc / 'meta' / 'episodes.jsonl')):
            i = m['episode_index']
            t = pq.read_table(sc / 'data' / 'chunk-000' / f'episode_{i:06d}.parquet').to_pydict()
            W = np.array(t['dagger.world_pose'])
            by = np.array(t['dagger.exec_by'])
            ref = json.loads(raw[i]['reference_path']) if isinstance(raw[i]['reference_path'], str) else raw[i]['reference_path']
            R = np.array([[p[0], -p[2], p[1]] for p in ref])
            ends[m['end_reason']] += 1
            devs += t['dagger.dev_m']
            for L, st, ea in zip(t[f'relative_goal_frame_id.{a.rig}'], t['dagger.expert_stop'], t['dagger.expert_actions']):
                kinds['stop' if st else ('pixel' if L >= 3 else ('pixel<3' if L >= 0 else ('turn' if ea[0] in (2, 3) else 'none')))] += 1

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(R[:, 0], R[:, 1], '-', color='0.6', lw=4, label='reference')
            ax.scatter(W[by == 0, 0], W[by == 0, 1], s=6, c='tab:blue', label='model exec')
            ax.scatter(W[by == 1, 0], W[by == 1, 1], s=6, c='tab:orange', label='expert exec')
            ax.plot(*R[0, :2], 'go', ms=10)
            ax.plot(*R[-1, :2], 'r*', ms=14)
            ax.set_aspect('equal')
            ax.legend(loc='best', fontsize=8)
            ax.set_title(f"{sc.name} ep {i}: {m['length']} f · {m['end_reason']} · max dev {m['max_dev_m']:.2f} m")
            fig.savefig(out / f'topdown_{sc.name}_{i}.png', dpi=90)
            plt.close(fig)

            n = len(t['frame_index'])
            pick = np.unique(np.linspace(0, n - 1, a.n_frames).astype(int))
            tiles = []
            for f in pick:
                im = Image.open(sc / 'videos' / 'chunk-000' / f'observation.images.rgb.{a.rig}' / f'episode_{i:06d}_{f}.jpg').convert('RGB')
                d = ImageDraw.Draw(im)
                L, (u, v) = t[f'relative_goal_frame_id.{a.rig}'][f], t[f'goal.{a.rig}'][f]
                if t['dagger.expert_stop'][f]:
                    lab = 'expert STOP'
                elif L >= 0:
                    d.ellipse([u - 12, v - 12, u + 12, v + 12], outline=(0, 255, 0), width=4)
                    lab = f'expert pixel len {L}'
                else:
                    lab = 'expert ' + ''.join(ACT[x] for x in t['dagger.expert_actions'][f][:6])
                d.rectangle([0, 0, 640, 26], fill=(0, 0, 0))
                d.text((6, 6), f"f{f} {lab} · exec {ACT[t['dagger.exec_action'][f]]}({'E' if t['dagger.exec_by'][f] else 'M'}) · dev {t['dagger.dev_m'][f]:.2f}m", fill=(255, 255, 0))
                tiles.append(im.resize((320, 240)))
            cols = 4
            grid = Image.new('RGB', (320 * cols, 240 * ((len(tiles) + cols - 1) // cols)))
            for k, im in enumerate(tiles):
                grid.paste(im, (320 * (k % cols), 240 * (k // cols)))
            grid.save(out / f'frames_{sc.name}_{i}.jpg')
    print('라벨 종류', dict(kinds))
    print('끝난 이유', dict(ends))
    if devs:
        print('이탈 거리 (m) 백분위 50/90/99', np.percentile(devs, [50, 90, 99]).round(2).tolist())


if __name__ == '__main__':
    main()
