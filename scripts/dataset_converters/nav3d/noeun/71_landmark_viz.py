"""noeun 71 · 랜드마크 만들기 단계별 **검증 그림**.

각 단계가 무엇을 했는지 눈으로 확인하려고 만든다. 범례는 한글로 크게 쓴다 (글자 12 pt 이상).

만드는 그림:
  1 scan_coverage.png   씬 스캔이 역을 얼마나 덮었나 — 층별 평면도에 설 자리 · 촬영 방향
  2 vocab_scatter.png   어휘 검증 — 이름별 (검출률, score) 분포와 통과선
  3 landmark_map.png    층별 평면도에 랜드마크 위치 · 종류별 색
  4 landmark_samples.jpg 랜드마크 예시 사진 (검출 상자 그려서)

    python3 scripts/dataset_converters/nav3d/noeun/71_landmark_viz.py --snaps logs/scene_scan --landmarks logs/scene_landmarks --out logs/scene_landmarks/viz
"""
import argparse, collections, json, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 — 컨테이너에 한글 폰트가 없어 범례가 □□□ 로 깨진다. 저장소에 넣어둔 나눔고딕을 등록해 쓴다.
KO = None
_F = Path(__file__).resolve().parents[4] / '.venv_label' / 'fonts' / 'NanumGothic-Regular.ttf'
if _F.exists():
    font_manager.fontManager.addfont(str(_F))
    KO = font_manager.FontProperties(fname=str(_F)).get_name()
    plt.rcParams['font.family'] = KO
else:
    for cand in ('NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR'):
        try:
            font_manager.findfont(cand, fallback_to_default=False)
            KO = cand; plt.rcParams['font.family'] = KO; break
        except Exception:
            pass
plt.rcParams['axes.unicode_minus'] = False
BIG, MID, SML = 17, 14, 12          # 제목 · 범례 · 눈금 — 읽을 수 있는 크기


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snaps', required=True)
    ap.add_argument('--landmarks', required=True)
    ap.add_argument('--verify', default='data/data/noeun_eval/navmesh/noeun_station/verify.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--top_names', type=int, default=12, help='지도에 색으로 구분할 상위 종류 수')
    return ap.parse_args()


def storey_of(z, storeys):
    return int(np.argmin([abs(z - s['z']) for s in storeys])) if storeys else 0


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = Path(a.landmarks)
    snaps = [json.loads(l) for l in (Path(a.snaps) / 'snaps.jsonl').open()]
    V = json.load(open(L / 'vocab.json'))
    storeys = json.load(open(a.verify))['storeys']
    names = ['지하 2층 (승강장)', '지하 1층 (대합실)', '지상 (출구)'][:len(storeys)] or [f'층 {i}' for i in range(len(storeys))]
    if not KO:
        names = ['B2 platform', 'B1 concourse', 'Ground exit'][:len(storeys)]
    print(f'[71] 한글 폰트: {KO or "없음 — 영문으로 그린다"}')

    # ── 1 스캔 덮임 ────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, len(storeys), figsize=(6.2 * len(storeys), 6.4))
    axs = np.atleast_1d(axs)
    for i, (ax, st) in enumerate(zip(axs, storeys)):
        P = np.array([s['robot_pos'] for s in snaps])
        m = np.array([storey_of(p[2], storeys) == i for p in P])
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], s=46, c='tab:blue', alpha=.85,
                       label='촬영한 자리' if KO else 'camera spot')
        ax.set_title(f'{names[i]}  ·  {int(m.sum())}' + ('장' if KO else ' shots'), fontsize=BIG)
        ax.set_aspect('equal'); ax.grid(alpha=.25)
        ax.tick_params(labelsize=SML)
        ax.set_xlabel('x [m]', fontsize=MID); ax.set_ylabel('y [m]', fontsize=MID)
        ax.legend(fontsize=MID, loc='best', framealpha=.95)
    fig.suptitle('① 씬 스캔 — 역 안 어디서 사진을 찍었나 (격자 5 m · 자리마다 4 방향)'
                 if KO else '1) Scene scan coverage (5 m grid, 4 directions each)', fontsize=BIG + 2)
    fig.tight_layout(); fig.savefig(out / 'scan_coverage.png', dpi=115); plt.close(fig)
    print(f'  저장 {out}/scan_coverage.png')

    # ── 2 어휘 검증 ────────────────────────────────────────────────────────
    # 검출률은 거의 전부 100 % 라 변별이 안 된다 -> **등장 횟수(가로) × 신뢰도(세로)** 로 바꾸고,
    # 검출률은 색(통과/탈락)으로만 표시한다. 이름표는 상위 몇 개만, 겹치지 않게 번갈아 띄운다.
    rows = V['rows']
    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    ok = [r for r in rows if r['pass']]; ng = [r for r in rows if not r['pass']]
    ax.scatter([r['said'] for r in ng], [r['score'] for r in ng], s=46, c='tab:red', alpha=.6,
               label=f'탈락 {len(ng)}개 — 검출률 80 % 미만 또는 score 0.35 미만' if KO else f'rejected {len(ng)}')
    ax.scatter([r['said'] for r in ok], [r['score'] for r in ok], s=46, c='tab:green', alpha=.65,
               label=f'통과 {len(ok)}개' if KO else f'passed {len(ok)}')
    ax.axhline(0.35, ls='--', c='0.35', lw=1.4)
    ax.text(ax.get_xlim()[1] * .99, 0.357, 'score 0.35 (통과선)' if KO else 'score 0.35',
            fontsize=MID, color='0.3', ha='right')
    lab = sorted(ok, key=lambda r: -r['said'])[:10]
    for i, r in enumerate(lab):
        ax.annotate(r['name'], (r['said'], r['score']), fontsize=MID - 1,
                    xytext=(8, 10 if i % 2 == 0 else -14), textcoords='offset points',
                    arrowprops=dict(arrowstyle='-', lw=.7, color='0.55'))
    ax.set_xscale('log')
    ax.set_xlabel('그 이름이 나온 사진 수 (로그 눈금)' if KO else '#images mentioning the name (log)', fontsize=MID)
    ax.set_ylabel('검출 신뢰도 평균 (score)' if KO else 'mean detection score', fontsize=MID)
    ax.set_title('② 어휘 검증 — Gemma 가 말한 이름을 검출기가 찾아내는가'
                 if KO else '2) Vocabulary check', fontsize=BIG)
    ax.tick_params(labelsize=SML); ax.grid(alpha=.25, which='both')
    ax.legend(fontsize=MID, loc='lower right', framealpha=.95)
    fig.tight_layout(); fig.savefig(out / 'vocab_scatter.png', dpi=115); plt.close(fig)
    print(f'  저장 {out}/vocab_scatter.png')

    # ── 3 랜드마크 지도 ────────────────────────────────────────────────────
    f_lm = L / 'landmarks.json'
    if not f_lm.exists():
        print('  landmarks.json 아직 없음 — 지도는 건너뜀'); return
    lms = json.load(open(f_lm))
    cnt = collections.Counter(r['name'] for r in lms)
    top = [n for n, _ in cnt.most_common(a.top_names)]
    cmap = plt.get_cmap('tab20')
    col = {n: cmap(i % 20) for i, n in enumerate(top)}
    # 층마다 세로로 쌓고(가로로 늘어놓으면 평면도가 납작해진다), 범례는 **그림 오른쪽에 한 번만** 그린다.
    fig, axs = plt.subplots(len(storeys), 1, figsize=(13.5, 4.9 * len(storeys)))
    axs = np.atleast_1d(axs)
    from matplotlib.lines import Line2D
    for i, (ax, st) in enumerate(zip(axs, storeys)):
        sub = [r for r in lms if storey_of(r['xyz'][2], storeys) == i]
        oth = [r for r in sub if r['name'] not in col]
        if oth:
            ax.scatter([r['xyz'][0] for r in oth], [r['xyz'][1] for r in oth], s=26, c='0.68')
        for n in top:
            g = [r for r in sub if r['name'] == n]
            if not g: continue
            ax.scatter([r['xyz'][0] for r in g], [r['xyz'][1] for r in g], s=62, color=col[n],
                       edgecolors='white', linewidths=.7)
        ax.set_title(f'{names[i]}  ·  ' + (f'랜드마크 {len(sub)}개' if KO else f'{len(sub)} landmarks'), fontsize=BIG)
        ax.set_aspect('equal'); ax.grid(alpha=.25); ax.tick_params(labelsize=SML)
        ax.set_xlabel('x [m]  (역 평면 좌표)' if KO else 'x [m]', fontsize=MID)
        ax.set_ylabel('y [m]', fontsize=MID)
    handles = [Line2D([], [], marker='o', ls='', ms=10, color=col[n],
                      markeredgecolor='white', label=f'{n} ({cnt[n]})') for n in top]
    n_oth = sum(1 for r in lms if r['name'] not in col)
    handles.append(Line2D([], [], marker='o', ls='', ms=9, color='0.68',
                          label=(f'기타 {len(cnt) - len(top)}종 {n_oth}개' if KO else f'other {n_oth}')))
    fig.legend(handles=handles, fontsize=MID, loc='center left', bbox_to_anchor=(0.845, 0.5),
               framealpha=.96, title='랜드마크 종류 (전체 개수)' if KO else 'type (total)',
               title_fontsize=MID + 1)
    fig.suptitle('③ 랜드마크 지도 — 역 안 어디에 무엇이 있나'
                 if KO else '3) Landmark map', fontsize=BIG + 3)
    fig.tight_layout(rect=[0, 0, 0.84, 0.97]); fig.savefig(out / 'landmark_map.png', dpi=115); plt.close(fig)
    print(f'  저장 {out}/landmark_map.png')
    print(f'[71] 랜드마크 {len(lms)} · 종류 {len(cnt)}')


if __name__ == '__main__':
    main()
