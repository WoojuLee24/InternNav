"""depth 세 소스가 얼마나 다른가 — **범위별**(5/10/15/20 m) 3자 비교. (이전 R3 타일링 검증 대체)

## 세 소스
| # | 소스 | 에셋 | 렌더러 |
|---|---|---|---|
| ① | **저장 depth** (학습이 실제로 먹는 것) | `mp3d_ce/<scan>.glb` | habitat (데이터셋 생성 시) |
| ② | **우리 렌더** | `mp3d_n1/<scan>/*.obj` → 지오메트리 전용 `.ply` | Open3D |
| ③ | **habitat 직접 렌더** | `mp3d_ce/<scan>.glb` | habitat (지금) |

**에셋도 렌더러도 다르다.** ①②는 G1이 이미 봤지만, 그것만으로는 ②③이 서로 같은지 알 수 없다
(둘 다 ①에 가까우면서 서로 다를 수 있다). 그리고 ③↔① 비교는 **우리 habitat 설정이 맞는지의 검증**이다
— 데이터셋이 habitat에서 나왔으므로 거의 같아야 하고, 아니면 좌표 변환·센서 설정이 틀린 것이다.

## 왜 범위별로 재나
파이프라인은 depth를 **5 m로 clip**한다(학습 preprocess + BEV range). 그런데 나중에 범위를 늘리면
먼 지오메트리가 들어오고, 에셋 차이(obj vs glb)·렌더러 차이가 그때 커질 수 있다. 5/10/15/20 m에서
따로 재서 "어디까지 믿을 수 있나"를 답한다.
⚠️ Open3D far plane이 기존 코드에 10 m로 하드코딩(`04_render_obs.RENDER_FAR_M`)돼 있어, 15/20 m를 재려면
`build_renderer` 뒤에 **projection을 다시 설정**해야 한다(기존 파일은 수정하지 않는다).

## 타일링은 제거됐다
이전 판은 "타일로 잘라도 같은 그림인가"를 물었다. 실측 결과 **씬 전체를 지오메트리 전용으로 저장하면
타일 하나보다 작다**(7.4 MB vs 6.0 MB, 로드 0.09 s, RSS 0.08 GB) → 타일링 자체가 불필요해졌다
(`01_build_scene_geo.py` docstring에 근거). 그래서 margin 스윕·타일 경계 검증이 전부 사라졌다.

## 그림 (프레임당 2장)
`<nm>.jpg`   5패널 `① 저장 | ② 우리 | ③ habitat | ④ |①−②| | ⑤ |①−③|`
`<nm>_rgb.jpg` full 씬 RGB 렌더 (G1의 `g1b_rgb_*`와 같은 형식)
①~③ 색범위 `0~vis_depth` 고정, ④⑤ `0~10 cm` 고정 — 스케일이 다르면 눈으로 비교가 안 된다.
기본값은 **G1(00)과 동일 조건**이라 ①②④가 G1의 3패널과 픽셀 단위로 대조된다.

실행: /usr/bin/python scripts/dataset_converters/3dloader_vlnce/02_verify_depth_sources.py
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / 'gs_vlnpe')); sys.path.insert(0, str(_HERE))
from geometry_utils import save_jpg  # noqa: E402
from viz_utils import save_gallery  # noqa: E402
from vlnce_align import K_VLNCE, build_rawdata_index  # noqa: E402
from habitat_render import HabitatDepthRenderer  # noqa: E402
_geo_mod = importlib.import_module('01_build_scene_geo')
_vpm = importlib.import_module('00_verify_pose_mesh')
_vng = importlib.import_module('03_verify_navmesh_gt')
_render = importlib.import_module('04_render_obs')

NATIVE_W, NATIVE_H = 640, 480
ERR_VIS_MAX_M = 0.10        # 오차 패널 고정 색범위. G1과 같은 값이라 두 리포트를 눈으로 비교 가능
MISSING_RGB = (255, 0, 255)  # 자홍 = 한쪽엔 지오메트리가 있고 다른 쪽엔 없는 픽셀


def make_renderer(far_m):
    """`build_renderer` 재사용 + **far plane만 덮어쓴다**. 기존 파일(`04_render_obs`)은 수정하지 않는다."""
    r = _render.build_renderer(NATIVE_W, NATIVE_H, K_VLNCE)
    k = np.asarray(K_VLNCE, dtype=np.float64).copy()
    k[0, 2] += _render.RENDER_PRINCIPAL_POINT_OFFSET_PX
    k[1, 2] += _render.RENDER_PRINCIPAL_POINT_OFFSET_PX
    r.scene.camera.set_projection(k, _render.RENDER_NEAR_M, float(far_m),
                                  float(NATIVE_W), float(NATIVE_H))
    return r


def _cf(x, valid, hi):
    """고정 색범위. `geometry_utils.colorize_depth`는 이미지마다 자기 min/max로 정규화해 비교가 안 된다."""
    return _vpm.colorize_fixed(x, valid, 0.0, hi)


def _err_panel(a, b, valid, missing, hi=ERR_VIS_MAX_M):
    """|a−b| + **사라진 픽셀 자홍** + 실제 값 각인. 고정 스케일에서 sub-mm는 둘 다 검게 나온다."""
    import cv2
    d = np.abs(a - b)
    img = np.ascontiguousarray(_cf(np.where(valid, d, 0.0), valid, hi))
    img[missing] = MISSING_RGB
    txt = (f'med {np.median(d[valid]) * 1000:.2f}mm  max {d[valid].max() * 1000:.1f}mm'
           if valid.any() else 'no overlap')
    if missing.any():
        txt += f'  mask {100 * missing.mean():.2f}%'
    cv2.putText(img, txt, (6, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _join(panels):
    """흰 구분선 + 번호. 없으면 5패널이 한 장으로 뭉개져 무엇이 무엇인지 알 수 없다."""
    import cv2
    out = []
    for i, a in enumerate(panels):
        a = np.ascontiguousarray(a)
        cv2.putText(a, str(i + 1), (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        out.append(a)
        if i != len(panels) - 1:
            out.append(np.full((a.shape[0], 3, 3), 255, dtype=np.uint8))
    return np.hstack(out)


def valid_of(d, hi, stored=False):
    """유효 픽셀 마스크. **세 소스를 같은 기준으로** 잘라야 그림·수치가 비교된다.

    저장 depth는 0이 '측정 없음'이고, 렌더는 배경이 inf(Open3D) 또는 0(habitat)이다.
    """
    d = np.asarray(d)
    return np.isfinite(d) & (d > 0.05) & (d < float(hi))


def compare(a, b, hi):
    """a 기준 b와의 차이. -> dict(median, p90, max, mask_frac, n)

    `mask_frac` = 한쪽만 유효한 픽셀 비율. 이전 G4는 이 픽셀을 분모에서 **제외**해 실패를 못 잡았다 —
    여기서는 따로 센다.
    """
    va, vb = valid_of(a, hi), valid_of(b, hi)
    both = va & vb
    n = int(va.sum())
    if n == 0 or not both.any():
        return dict(median=float('nan'), p90=float('nan'), max=float('nan'),
                    mask_frac=float('nan'), n=n)
    d = np.abs(a - b)[both]
    return dict(median=float(np.median(d)), p90=float(np.percentile(d, 90)), max=float(d.max()),
                mask_frac=float((va ^ vb).sum()) / n, n=n)


def run(args):
    """프레임별로 세 소스를 렌더하고 범위별 지표를 낸다. -> (rows, imgs)"""
    import open3d as o3d
    frames = _vpm.load_episode(args.data_root, args.scene, args.rig, args.episode, args.n_frames)
    raw = build_rawdata_index(Path(args.raw_root))
    tasks = [json.loads(l) for l in open(Path(args.data_root) / args.scene / 'meta' / 'episodes.jsonl')]
    instr = tasks[args.episode]['tasks'][0].strip()
    assert (args.scene, instr) in raw, f'{args.scene} ep{args.episode}: raw 매칭 실패'
    T = _vng.sf2mesh(raw[(args.scene, instr)])

    geo = _geo_mod.geo_path(args.geo_dir, args.scene)
    assert geo.exists(), f'{geo} 없음 — 01_build_scene_geo.py 를 먼저 돌려라'
    # far plane은 **가장 큰 판정 범위**로 한 번만 잡는다(그보다 가까운 범위는 마스크로 자르면 된다).
    ranges = sorted(float(x) for x in args.ranges.split(','))
    r3d = make_renderer(max(ranges) * 1.2)
    model = o3d.io.read_triangle_model(str(geo))
    r3d.scene.clear_geometry(); r3d.scene.add_model('m', model)

    rows, imgs = [], []
    with HabitatDepthRenderer(f'{args.glb_root}/{args.scene}/{args.scene}.glb') as hab:
        for f, pose, sd, _sv in frames:
            c2w = T @ pose
            _render.set_camera_pose(r3d, c2w)
            od = np.asarray(r3d.render_to_depth_image(z_in_view_space=True))
            orgb = np.asarray(r3d.render_to_image())
            hd = hab.render(c2w)
            per = {}
            for hi in ranges:
                per[hi] = dict(ours=compare(sd, od, hi),      # ① vs ②
                               hab=compare(sd, hd, hi),       # ① vs ③  (habitat 설정 검증)
                               cross=compare(od, hd, hi))     # ② vs ③  (렌더러·에셋 gap)
            rows.append(dict(frame=int(f), per=per))
            nm = f'src_{args.scene}_{args.rig}_ep{args.episode:03d}_f{f:04d}'
            vs, vo, vh = (valid_of(sd, args.vis_depth), valid_of(od, args.vis_depth),
                          valid_of(hd, args.vis_depth))
            save_jpg(_join([_cf(sd, vs, args.vis_depth), _cf(od, vo, args.vis_depth),
                            _cf(hd, vh, args.vis_depth),
                            _err_panel(sd, od, vs & vo, vs ^ vo),
                            _err_panel(sd, hd, vs & vh, vs ^ vh)]), Path(args.out_dir) / f'{nm}.jpg')
            save_jpg(orgb, Path(args.out_dir) / f'{nm}_rgb.jpg')
            imgs.append((nm, rows[-1]))
    return rows, imgs, ranges


def main():
    ap = argparse.ArgumentParser()
    # 기본값은 **G1(00)과 동일 조건** — ①②④ 패널이 G1의 3패널과 픽셀 단위로 대조된다.
    ap.add_argument('--scene', default='s8pcmisQ38h')
    ap.add_argument('--rig', default='125cm_0deg')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--n_frames', type=int, default=6, help='G1과 같은 값이면 같은 프레임이 뽑힌다')
    ap.add_argument('--ranges', default='5,10,15,20', help='판정할 depth 범위[m]. 5 = 파이프라인 실사용')
    ap.add_argument('--vis_depth', type=float, default=10.0,
                    help='패널 색범위[m] (표시 전용). **G1이 10 m**라 기본 10이면 색이 같다')
    ap.add_argument('--data_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--raw_root', default='data/InternData-N1-v0.5-mini/vln_ce/raw_data/r2r')
    ap.add_argument('--glb_root', default='data/scene_data/mp3d_ce/mp3d')
    ap.add_argument('--geo_dir', default=_geo_mod.DEFAULT_OUT)
    ap.add_argument('--out_dir', default='logs/embodiment_augment/depth_sources')
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    rows, imgs, ranges = run(args)

    def agg(key, hi, stat):
        v = [r['per'][hi][key][stat] for r in rows]
        return float(np.nanmedian(v)) if stat != 'max' else float(np.nanmax(v))

    PAIRS = [('ours', '① 저장 ↔ ② 우리(Open3D·obj)', '우리 렌더가 학습 depth를 재현하나 (#00과 같은 질문)'),
             ('hab', '① 저장 ↔ ③ habitat(지금)', '**우리 habitat 설정 검증** — 데이터셋이 habitat 산출물이라 거의 같아야'),
             ('cross', '② 우리 ↔ ③ habitat', '**렌더러·에셋 gap** (obj vs glb, Open3D vs habitat)')]
    tbl = ''
    for key, label, why in PAIRS:
        cells = ''.join(f'<td>{agg(key, hi, "median")*1000:.2f}</td>'
                        f'<td>{agg(key, hi, "p90")*1000:.2f}</td>'
                        f'<td>{agg(key, hi, "max")*1000:.0f}</td>'
                        f'<td>{100*agg(key, hi, "mask_frac"):.2f}%</td>' for hi in ranges)
        tbl += f'<tr><td><b>{label}</b><br><small>{why}</small></td>{cells}</tr>'
        for hi in ranges:
            print(f'[SRC] {label:34s} ≤{hi:4.0f} m  median {agg(key,hi,"median")*1000:7.2f} mm  '
                  f'p90 {agg(key,hi,"p90")*1000:7.2f} mm  max {agg(key,hi,"max")*1000:8.0f} mm  '
                  f'마스크불일치 {100*agg(key,hi,"mask_frac"):5.2f}%')

    hdr = ''.join(f'<th colspan=4>≤ {hi:.0f} m'
                  + (' <small>(파이프라인 실사용)</small>' if hi == min(ranges) else '') + '</th>'
                  for hi in ranges)
    sub = ''.join('<th>median</th><th>p90</th><th>max</th><th>마스크<br>불일치</th>' for _ in ranges)
    summary = (
        f'<p><b>{args.scene}</b> · ep{args.episode} · rig {args.rig} · '
        f'프레임 {[r["frame"] for r in rows]} · 색범위 0~{args.vis_depth:.0f} m — '
        f'<b>#00과 동일 조건</b>이라 패널 ①②④를 #00의 3패널과 직접 대조할 수 있다.</p>'
        f'<h3>실험결과 — 범위별 3자 비교 (단위 mm)</h3>'
        f'<table border=1 cellpadding=6><tr><th rowspan=2>비교</th>{hdr}</tr><tr>{sub}</tr>{tbl}</table>'
        f'<p><b>게이트는 ≤{min(ranges):.0f} m 열이다</b> — 학습이 depth를 그 값으로 clip하고 BEV range도 '
        f'같으므로 그 밖은 쓰지 않는다. 더 넓은 열은 "나중에 범위를 늘리면 무엇이 나빠지나"에 대한 답이다.</p>'
        f'<p><b>마스크 불일치</b> = 한쪽엔 지오메트리가 있고 다른 쪽엔 없는 픽셀 비율. 이전 G4는 이 픽셀을 '
        f'분모에서 <b>제외</b>해 실패를 구조적으로 못 잡았다 — 여기서는 따로 센다.</p>'
        f'<h3>타일링은 제거했다</h3>'
        f'<p>이전 판(R3)은 "타일로 잘라도 같은 그림인가"를 물었다. 실측하니 <b>씬 전체를 지오메트리 전용으로 '
        f'저장하면 타일 하나보다 작다</b>(7.4 MB vs 6.0 MB · 로드 0.09 s · RSS 0.08 GB) — 무거운 것은 '
        f'지오메트리가 아니라 <b>텍스처</b>였다(367 MB jpg → RSS 8.5 GB). v1은 depth만 렌더하니 텍스처가 '
        f'전부 낭비다. 그래서 타일링·margin 스윕·타일 경계 검증이 <b>전부 불필요해졌다</b>. '
        f'depth가 같은지는 확인했다 — 5 m 안에서 1,628,548 픽셀 중 오차 &gt;1 mm 인 것 <b>0개</b>.</p>'
        f'<h3>한계</h3><ul>'
        f'<li>씬 1채 · 에피소드 1개 · {len(rows)}프레임. 씬마다 에셋 품질이 다를 수 있다.</li>'
        f'<li><b>RGB는 비교하지 않았다</b> — v1은 depth/BEV만 바꾼다. RGB로 가면 텍스처가 다시 필요하고 '
        f'타일링 결정도 재검토해야 한다.</li>'
        f'<li>max가 수 m로 튀는 픽셀이 남는다. #00이 꼬리로 보고한 것과 같은 <b>실루엣·거울/창</b> 픽셀로 '
        f'보이나 <b>원인을 특정하지 않았다</b>.</li></ul>')

    body = []
    for nm, r in imgs:
        g = r['per'][min(ranges)]
        body.append(
            f'<h3>frame {r["frame"]} — ≤{min(ranges):.0f} m 기준: 우리 median '
            f'{g["ours"]["median"]*1000:.2f} mm · habitat median {g["hab"]["median"]*1000:.2f} mm · '
            f'둘 사이 median {g["cross"]["median"]*1000:.2f} mm</h3>'
            f'<p>① 저장 depth | ② 우리(Open3D·지오메트리 ply) | ③ habitat 직접 렌더 | '
            f'④ <b>|①−②|</b> | ⑤ <b>|①−③|</b> '
            f'(①~③ 0~{args.vis_depth:.0f} m 고정, ④⑤ 0~{ERR_VIS_MAX_M*100:.0f} cm 고정 · '
            f'자홍 = 한쪽만 지오메트리가 있는 픽셀)</p>'
            f'<img src="{nm}.jpg" style="width:100%">'
            f'<img src="{nm}_rgb.jpg" style="width:33%">')

    out = Path(args.out_dir)
    (out / 'summary.html').write_text(summary, encoding='utf-8')
    # body.html을 쓰면 발행기가 **본문이 참조한 그림만** 인라인한다(갤러리 모드는 옛 jpg까지 긁어온다).
    (out / 'body.html').write_text(''.join(body), encoding='utf-8')
    save_gallery(out, 'report.html', 'depth 세 소스 비교 — 저장 / 우리(Open3D) / habitat', summary,
                 ''.join(body))
    print(f'[SRC] report -> {out}/report.html')
    return 0


if __name__ == '__main__':
    sys.exit(main())
