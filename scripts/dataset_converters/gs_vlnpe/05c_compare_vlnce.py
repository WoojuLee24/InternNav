"""vln_ce GT와 우리 렌더를 **같은 자리에서 토글**해 비교한다 (rgb + depth).

`05b_verify_vlnce.py`는 게이트 판정이 목적이라 SSIM 숫자만 낸다. 이 파일은 **눈으로 보는 것**이
목적이다 — 같은 프레임을 GT ↔ 우리 렌더로 번갈아 보여주고, 색 차이를 수치로도 같이 낸다.

토글은 **CSS만으로** 만든다
---------------------------
`viz_utils.blink_widget_html`은 자바스크립트가 이미지를 채우는 방식이라, 스크립트를 안 돌리는
뷰어에서 **빈칸으로 보인다**(실측 — `04e_sample_compare` 리포트에서 겪었다).
여기서는 라디오 버튼 + 형제 선택자로 JS 없이 전환한다.

depth는 그냥 보면 안 보인다
---------------------------
uint16 밀리미터라 육안으로 구분이 안 된다. `geometry_utils.colorize_depth`로 컬러맵을 입히되,
**GT와 우리가 같은 범위로 정규화**해야 비교가 된다 — 각자 min/max로 정규화하면 실제 차이가
사라진다. 그래서 두 이미지의 유효 픽셀을 합쳐 공통 범위를 먼저 구한다.

커맨드
------
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05c_compare_vlnce.py --render_root data/InternData-N1-v0.5-mini/vln_ce_render/traj_data/r2r --gt_root data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r --scenes all --rigs 125cm_30deg,60cm_15deg --n_frames 4 --log_dir logs/gs-vlnpe

Isaac Sim을 띄우지 않는다 — 디스크만 읽는다.
"""

import argparse
import base64
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCRIPT_NAME = '05c_compare_vlnce'
DEPTH_SCALE_MM = 1000.0

# JS 없이 도는 토글. 라디오를 숨기고, 체크된 라디오의 **형제** 이미지만 보여준다.
# `nth-of-type`으로 짝을 맞추므로 위젯마다 CSS를 새로 만들 필요가 없다.
TOGGLE_CSS = """
<style>
.tg { border:1px solid rgba(255,255,255,.15); border-radius:8px; overflow:hidden;
      background:rgba(0,0,0,.25); display:inline-block; margin:6px 8px 12px 0; }
.tg > input { display:none; }
.tg .fr { position:relative; line-height:0; background:#000; }
.tg .fr > img { display:none; width:100%; height:auto; image-rendering:pixelated; }
.tg > input:nth-of-type(1):checked ~ .fr > img:nth-of-type(1),
.tg > input:nth-of-type(2):checked ~ .fr > img:nth-of-type(2),
.tg > input:nth-of-type(3):checked ~ .fr > img:nth-of-type(3) { display:block; }
.tg .bar { display:flex; gap:0; font-size:12px; }
.tg .bar > label { flex:1; text-align:center; padding:5px 8px; cursor:pointer;
                   background:rgba(255,255,255,.06); border-top:1px solid rgba(255,255,255,.12);
                   user-select:none; }
.tg .bar > label:hover { background:rgba(255,255,255,.14); }
.tg > input:nth-of-type(1):checked ~ .bar > label:nth-of-type(1),
.tg > input:nth-of-type(2):checked ~ .bar > label:nth-of-type(2),
.tg > input:nth-of-type(3):checked ~ .bar > label:nth-of-type(3) {
    background:rgba(120,200,255,.28); font-weight:600; }
.tg .cap { font-size:12px; opacity:.75; padding:5px 8px 0; text-align:center; }
.row { display:flex; flex-wrap:wrap; align-items:flex-start; gap:4px; overflow-x:auto; }
</style>
"""


def data_uri(path: Path) -> str:
    mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
    return f'data:{mime};base64,{base64.b64encode(path.read_bytes()).decode("ascii")}'


def toggle(wid: str, states: list, caption: str, width_px: int) -> str:
    """`states = [(라벨, 이미지경로), ...]` -> CSS 전용 토글 위젯."""
    radios = ''.join(f'<input type="radio" name="{wid}" id="{wid}_{i}"'
                     f'{" checked" if i == 0 else ""}>' for i in range(len(states)))
    imgs = ''.join(f'<img src="{data_uri(p)}" alt="{lab}">' for lab, p in states)
    labels = ''.join(f'<label for="{wid}_{i}">{lab}</label>'
                     for i, (lab, _) in enumerate(states))
    return (f'<div class="tg" style="width:{width_px}px">{radios}'
            f'<div class="fr">{imgs}</div><div class="bar">{labels}</div>'
            f'<div class="cap">{caption}</div></div>')


def scene_paths(root: Path, scene: str) -> dict:
    base = Path(root) / scene
    return {'base': base, 'parquet_dir': base / 'data' / 'chunk-000',
            'videos_dir': base / 'videos' / 'chunk-000'}


def ep_ids(root: Path, scene: str) -> list:
    d = scene_paths(root, scene)['parquet_dir']
    return sorted(int(p.stem.split('_')[1]) for p in d.glob('episode_*.parquet')) if d.is_dir() else []


def rigs_present(videos_dir: Path) -> list:
    out = {d.name[len('observation.images.rgb.'):]
           for d in videos_dir.glob('observation.images.rgb.*')}
    return sorted(r for r in out if r != 'topdown')


def colorize_pair(gt_mm: np.ndarray, ours_mm: np.ndarray):
    """두 depth를 **공통 범위**로 정규화해 컬러맵. 각자 정규화하면 차이가 사라진다."""
    from geometry_utils import colorize_depth

    g = gt_mm.astype(np.float32) / DEPTH_SCALE_MM
    o = ours_mm.astype(np.float32) / DEPTH_SCALE_MM
    vg, vo = g > 0, o > 0
    both = np.concatenate([g[vg], o[vo]]) if (vg.any() or vo.any()) else np.array([0.0, 1.0])
    lo, hi = float(np.percentile(both, 1)), float(np.percentile(both, 99))
    span = max(hi - lo, 1e-6)

    def cmap(d, v):
        n = np.clip((d - lo) / span, 0, 1)
        return colorize_depth(n * span + lo, v)

    return cmap(g, vg), cmap(o, vo)


def frame_stats(gt_rgb: np.ndarray, our_rgb: np.ndarray,
                gt_mm: np.ndarray, our_mm: np.ndarray) -> dict:
    """색·밝기·깊이 차이를 수치로. 눈으로 본 차이를 숫자로 확인하려는 것."""
    from skimage.metrics import structural_similarity as ssim_metric

    g = gt_rgb.astype(np.float64)
    o = our_rgb.astype(np.float64)
    both = (gt_mm > 0) & (our_mm > 0)
    dm = (np.abs(gt_mm.astype(np.float64) - our_mm.astype(np.float64))[both] / DEPTH_SCALE_MM
          if both.any() else np.array([np.nan]))
    return {
        'ssim': float(ssim_metric(gt_rgb, our_rgb, channel_axis=2, data_range=255)),
        'mean_gt': g.reshape(-1, 3).mean(axis=0).round(1).tolist(),
        'mean_ours': o.reshape(-1, 3).mean(axis=0).round(1).tolist(),
        'brightness_gt': float(g.mean()), 'brightness_ours': float(o.mean()),
        'depth_abs_median_m': float(np.nanmedian(dm)),
        'depth_valid_overlap': float(both.mean()),
    }


def build(args) -> Path:
    import cv2
    import viz_utils

    gt_root, rd = Path(args.gt_root), Path(args.render_root)
    scenes = (sorted(p.name for p in rd.glob('*') if (p / 'data' / 'chunk-000').is_dir())
              if args.scenes == 'all' else [s for s in args.scenes.split(',') if s])
    out_dir = Path(args.log_dir) / SCRIPT_NAME
    img_dir = out_dir / 'frames'
    img_dir.mkdir(parents=True, exist_ok=True)

    body, all_stats = [], []
    for scene in scenes:
        G, R = scene_paths(gt_root, scene), scene_paths(rd, scene)
        rigs = rigs_present(R['videos_dir'])
        if args.rigs != 'all':
            want = [x for x in args.rigs.split(',') if x]
            rigs = [r for r in rigs if r in want]
        eps = ep_ids(rd, scene)
        if not rigs or not eps:
            continue
        body.append(f'<h2>{scene}</h2>')

        for rig in rigs:
            ep = eps[0]
            import pyarrow.parquet as pq
            n = pq.read_metadata(R['parquet_dir'] / f'episode_{ep:06d}.parquet').num_rows
            idx = np.unique(np.linspace(0, n - 1, min(args.n_frames, n)).astype(int))
            cells, rig_stats = [], []
            for i in idx:
                gr = G['videos_dir'] / f'observation.images.rgb.{rig}' / f'episode_{ep:06d}_{i}.jpg'
                orr = R['videos_dir'] / f'observation.images.rgb.{rig}' / f'episode_{ep:06d}_{i}.jpg'
                gd = G['videos_dir'] / f'observation.images.depth.{rig}' / f'episode_{ep:06d}_{i}.png'
                od = R['videos_dir'] / f'observation.images.depth.{rig}' / f'episode_{ep:06d}_{i}.png'
                if not all(p.is_file() for p in (gr, orr, gd, od)):
                    continue
                a = cv2.cvtColor(cv2.imread(str(gr)), cv2.COLOR_BGR2RGB)
                b = cv2.cvtColor(cv2.imread(str(orr)), cv2.COLOR_BGR2RGB)
                gmm = cv2.imread(str(gd), cv2.IMREAD_UNCHANGED)
                omm = cv2.imread(str(od), cv2.IMREAD_UNCHANGED)
                if a.shape != b.shape or gmm.shape != omm.shape:
                    continue
                st = frame_stats(a, b, gmm, omm)
                rig_stats.append(st)
                all_stats.append(st)

                # depth를 공통 범위로 컬러맵
                cg, co = colorize_pair(gmm, omm)
                pg = img_dir / f'{scene}_{rig}_e{ep}_f{i}_depth_gt.png'
                po = img_dir / f'{scene}_{rig}_e{ep}_f{i}_depth_ours.png'
                cv2.imwrite(str(pg), cv2.cvtColor(cg, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(po), cv2.cvtColor(co, cv2.COLOR_RGB2BGR))

                wid = f'{scene}_{rig}_{ep}_{i}'.replace('.', '_')
                cells.append(toggle(
                    f'rgb_{wid}', [('GT', gr), ('우리 렌더', orr)],
                    f'rgb f{i} · SSIM {st["ssim"]:.3f} · 밝기 {st["brightness_gt"]:.0f} → '
                    f'{st["brightness_ours"]:.0f}', args.img_px))
                cells.append(toggle(
                    f'dep_{wid}', [('GT', pg), ('우리 렌더', po)],
                    f'depth f{i} · |차이| 중앙 {st["depth_abs_median_m"]:.3f} m', args.img_px))

            if not rig_stats:
                continue
            ss = np.array([s['ssim'] for s in rig_stats])
            bg = np.array([s['brightness_gt'] for s in rig_stats])
            bo = np.array([s['brightness_ours'] for s in rig_stats])
            mg = np.array([s['mean_gt'] for s in rig_stats]).mean(axis=0)
            mo = np.array([s['mean_ours'] for s in rig_stats]).mean(axis=0)
            dd = np.array([s['depth_abs_median_m'] for s in rig_stats])
            body.append(
                f'<h3>{rig} <span style="font-weight:400;opacity:.75;font-size:13px">'
                f'— episode {ep} · SSIM {np.median(ss):.4f} · 밝기 {bg.mean():.1f} → {bo.mean():.1f} '
                f'({bo.mean() - bg.mean():+.1f}) · RGB 평균 '
                f'({mg[0]:.0f},{mg[1]:.0f},{mg[2]:.0f}) → ({mo[0]:.0f},{mo[1]:.0f},{mo[2]:.0f}) · '
                f'depth |차이| {np.median(dd):.3f} m</span></h3>')
            body.append('<div class="row">' + ''.join(cells) + '</div>')

    ss = np.array([s['ssim'] for s in all_stats])
    bg = np.array([s['brightness_gt'] for s in all_stats])
    bo = np.array([s['brightness_ours'] for s in all_stats])
    mg = np.array([s['mean_gt'] for s in all_stats]).mean(axis=0)
    mo = np.array([s['mean_ours'] for s in all_stats]).mean(axis=0)
    dd = np.array([s['depth_abs_median_m'] for s in all_stats])
    ov = np.array([s['depth_valid_overlap'] for s in all_stats])

    summary = (
        f'<p><b>보는 법</b>: 각 이미지 아래 <b>GT</b> / <b>우리 렌더</b> 버튼을 누르면 '
        f'같은 자리에서 바뀐다. 위가 rgb, 아래가 depth다. '
        f'depth는 두 이미지를 <b>같은 범위</b>로 정규화해 컬러맵을 입혔다 — '
        f'각자 정규화하면 실제 차이가 사라진다.</p>'
        f'<p>프레임 {len(all_stats)}개 · SSIM 중앙 <b>{np.median(ss):.4f}</b></p>'
        f'<p><b>밝기</b> GT {bg.mean():.1f} → 우리 {bo.mean():.1f} '
        f'(<b>{bo.mean() - bg.mean():+.1f}</b>) · '
        f'<b>채널 평균</b> GT ({mg[0]:.0f}, {mg[1]:.0f}, {mg[2]:.0f}) → '
        f'우리 ({mo[0]:.0f}, {mo[1]:.0f}, {mo[2]:.0f})</p>'
        f'<p><b>depth</b> |차이| 중앙 {np.median(dd):.3f} m · '
        f'양쪽 유효 픽셀 겹침 {ov.mean() * 100:.1f} %</p>')
    return viz_utils.save_gallery(out_dir, 'report.html',
                                  f'{SCRIPT_NAME} — GT ↔ 우리 렌더 토글',
                                  summary, TOGGLE_CSS + ''.join(body))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render_root', default='data/InternData-N1-v0.5-mini/vln_ce_render/traj_data/r2r')
    ap.add_argument('--gt_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--scenes', default='all')
    ap.add_argument('--rigs', default='125cm_30deg', help='콤마 구분 또는 all')
    ap.add_argument('--n_frames', type=int, default=4, help='rig당 볼 프레임 수')
    ap.add_argument('--img_px', type=int, default=320, help='표시 너비[px]')
    ap.add_argument('--log_dir', default='logs/gs-vlnpe')
    args = ap.parse_args()

    if not Path(args.render_root).is_dir():
        print(f'[{SCRIPT_NAME}] 산출물 루트가 없다: {args.render_root}', file=sys.stderr)
        return 2
    report = build(args)
    print(f'[{SCRIPT_NAME}] report -> {report}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
