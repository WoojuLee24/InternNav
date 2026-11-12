"""noeun 86 · 사진 비교 — VLN-PE 평가 카메라 사진(scripts/eval/noeun_vlnpe_snap.py) 과 같은 c2w 를 학습 데이터 렌더러(51 · 82 · 84 와 같은 build_renderer)로 찍어 나란히 놓는다 (Isaac).

같은 자세 · 같은 장면 파일 · 같은 조명 설정(ambient_only, rtx_ambient 10, film_iso 70) 이면 두 사진은 같아야 한다.
다르면 그 차이가 '학습 사진 ↔ 실제 주행 사진' 차이다.
수치 (사진마다): 밝기 평균(0~255) · 색 평균(R G B) · 선명도(라플라시안 분산, 클수록 선명) · 두 사진 차 평균(|평가-학습|) · PSNR(dB, 클수록 비슷).

    /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/86_render_compare.py --snap logs/render_compare/vlnpe --out logs/render_compare
출력: <out>/pair_ep<ep>.jpg (위 = 평가, 아래 = 학습 렌더러, 열 = 사진) · <out>/metrics.json · 마지막에 요약 출력
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
CONV = REPO / 'scripts' / 'dataset_converters'
sys.path.insert(0, str(REPO))
for p in (CONV / 'nav3d' / 'utils', CONV / 'gs_vlnpe', CONV / 'gs_vlnpe' / 'apply_real'):
    sys.path.insert(0, str(p))


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snap', required=True, help='noeun_vlnpe_snap.py 의 --out')
    ap.add_argument('--usd', default='data/GS_USDZ/noeun_station_collision.usdz')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    snaps = [json.loads(l) for l in open(Path(a.snap) / 'snaps.jsonl')]

    ex = _load('ex05', CONV / 'gs_vlnpe' / 'apply_real' / '05_export_vlnce.py')
    r4 = ex.load_renderer_module()
    prof = ex.load_camera_profiles().get(a.rig)
    cas = r4.build_renderer(prof.width, prof.height, prof.k, REPO / a.usd, 'ambient_only', None, 10.0, 70.0, prof.render_near_m, prof.render_far_m)
    rgb_tr, _ = r4.render_along(cas, np.array([s['c2w'] for s in snaps]))

    import cv2
    from PIL import Image

    def stats(im):
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        return {'bright': float(g.mean()), 'rgb': [float(im[..., c].mean()) for c in range(3)], 'sharp': float(cv2.Laplacian(g, cv2.CV_64F).var())}

    rows = []
    by_ep = {}
    for s, tr in zip(snaps, rgb_tr):
        ev = np.asarray(Image.open(Path(a.snap) / s['rgb']).convert('RGB'))
        tr = np.asarray(tr, dtype=np.uint8)
        assert ev.shape == tr.shape, (ev.shape, tr.shape)
        d = np.abs(ev.astype(np.float64) - tr.astype(np.float64))
        mse = float((d ** 2).mean())
        rows.append({'ep': s['ep'], 'step': s['step'], 'eval': stats(ev), 'train': stats(tr), 'absdiff': float(d.mean()),
                     'psnr': float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))})
        by_ep.setdefault(s['ep'], []).append((ev, tr, rows[-1]))
    for ep, items in by_ep.items():
        W, H = 320, 240
        grid = Image.new('RGB', (W * len(items), H * 2))
        for k, (ev, tr, r) in enumerate(items):
            grid.paste(Image.fromarray(ev).resize((W, H)), (W * k, 0))
            grid.paste(Image.fromarray(tr).resize((W, H)), (W * k, H))
            from PIL import ImageDraw

            dr = ImageDraw.Draw(grid)
            dr.text((W * k + 4, 4), f"eval s{r['step']} psnr {r['psnr']:.1f}", fill=(255, 255, 0))
            dr.text((W * k + 4, H + 4), 'train renderer', fill=(255, 255, 0))
        grid.save(out / f'pair_ep{ep:02d}.jpg', quality=90)
    json.dump(rows, open(out / 'metrics.json', 'w'), indent=1)

    def med(f):
        return round(float(np.median([f(r) for r in rows])), 2)

    print(f'사진 {len(rows)} 장')
    print('밝기 평가/학습', med(lambda r: r['eval']['bright']), med(lambda r: r['train']['bright']))
    print('RGB 평가', [med(lambda r, c=c: r['eval']['rgb'][c]) for c in range(3)], '학습', [med(lambda r, c=c: r['train']['rgb'][c]) for c in range(3)])
    print('선명도 평가/학습', med(lambda r: r['eval']['sharp']), med(lambda r: r['train']['sharp']))
    print('차 평균 · PSNR 중앙', med(lambda r: r['absdiff']), med(lambda r: r['psnr']))


if __name__ == '__main__':
    main()
