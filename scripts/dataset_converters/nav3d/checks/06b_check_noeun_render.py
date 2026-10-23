"""06b — 06 이 만든 노은역 vln_ce 데이터셋을 검사하고 눈으로 보는 페이지를 만든다.

## 용어
- **몸 / 카메라**: 06 docstring. 카메라 = 몸 + 전방 0.365 m, 높이 0.584 m, 아래로 14.76°.
- **카메라 바닥 높이**: 카메라에서 수직 아래로 쏜 광선이 충돌 mesh 에 닿기까지 거리.
- **제자리 회전 이동량**: 회전 action 프레임에서 카메라가 움직인 거리. 카메라가 몸 앞에 있으므로 0 이 아니다
  — 15° 회전이면 2 · 0.365 · sin(7.5°) = 0.095 m 가 맞는 값이다(릴리스는 카메라가 몸 중심이라 0).

## 판정 기준 (결과를 보기 전에 정함)
| | 묻는 것 | 기준 | 근거 |
|---|---|---|---|
| R1 형식 | 에피소드마다 rgb·depth 프레임 수 == parquet 행 · 21 컬럼 · rgb 640×480 · depth uint16 | 전부 | 학습 로더가 그대로 읽어야 한다 |
| R2 pose | `pose.60cm_15deg` 0행 = (0, 0, 0.584) · 전진 수평 0.25 m · 회전 15° · 회전 이동량 0.095 m | 전부 ±1 mm · ±0.01° | 06 이 의도한 리그와 같은지 (전진은 수평 거리 — 계단에서는 높이도 바뀐다) |
| R3 경로 | 몸 위치 ↔ 04 경로 xy 거리 | 에피소드 최대 ≤ 0.30 m · 종점 ≤ 0.5 m | 이산화 폭 (05b P2 와 같음) |
| R4 높이 | 평지 에피소드 카메라 바닥 높이 | 프레임의 ≥ 95 % 가 0.584 ± 0.06 m | 바닥을 따라가는지(00: mesh 바닥 ↔ GS 5 cm) |
| R5 depth | 프레임 depth 유효(>0) 비율 | ≥ 0.5 인 프레임 ≥ 99 % | 카메라가 벽 속·허공을 보지 않는지 |
| R6 rgb | 프레임 평균 밝기 | > 15/255 인 프레임 ≥ 99 % | GS 가 그려졌는지(검은 화면 = NuRec 로드 실패) |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/06b_check_noeun_render.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/ (이 파일은 nav3d/checks/ 에 있다)
sys.path.insert(0, str(HERE))
import nav3d_report as R  # noqa: E402
import nav3d_utils as nu  # noqa: E402

CHUNK = 200                     # strip 한 장의 프레임 수 (320 px × 200 = 64,000 px < JPEG 한계)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log_dir', default='scripts/dataset_converters/nav3d/logs/06_render_noeun')
    ap.add_argument('--paths_json', default='scripts/dataset_converters/nav3d/logs/04_export_paths/noeun_station/noeun_station_paths.json')
    ap.add_argument('--mesh', default='data/scene_data/noeun_station/noeun_station_mid.ply')
    ap.add_argument('--viz_w', type=int, default=320)
    return ap.parse_args()


def main():
    import cv2
    import open3d as o3d
    import pyarrow.parquet as pq
    import trimesh
    args = parse()
    log = Path(args.log_dir)
    fm = json.loads((log / 'frames_meta.json').read_text())
    a6 = fm['args']
    rig, H = a6['rig'], a6['mount_height_m']
    root = Path(a6['out_root']) / 'traj_data' / 'r2r' / a6['out_scene']
    paths = json.loads(Path(args.paths_json).read_text())['episodes']
    mesh = trimesh.load(args.mesh, process=False)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    out = log / 'check'
    (out / 'strips').mkdir(parents=True, exist_ok=True)
    rows, man = [], []
    turn_move_expect = 2 * a6['forward_m'] * np.sin(np.deg2rad(7.5))
    for e in fm['episodes']:
        ei, n = e['episode'], e['frames']
        p = root / 'data' / 'chunk-000' / f'episode_{ei:06d}.parquet'
        t = pq.read_table(p)
        rgbs = sorted((root / 'videos' / 'chunk-000' / f'observation.images.rgb.{rig}').glob(f'episode_{ei:06d}_*.jpg'),
                      key=lambda q: int(q.stem.rsplit('_', 1)[1]))
        deps = sorted((root / 'videos' / 'chunk-000' / f'observation.images.depth.{rig}').glob(f'episode_{ei:06d}_*.png'),
                      key=lambda q: int(q.stem.rsplit('_', 1)[1]))
        im0 = cv2.imread(str(rgbs[0])) if rgbs else None
        d0 = cv2.imread(str(deps[0]), -1) if deps else None
        r1 = (t.num_rows == n == len(rgbs) == len(deps) and t.num_columns == 21 and im0 is not None and im0.shape == (480, 640, 3)
              and d0 is not None and d0.dtype == np.uint16)
        pose = np.stack([np.asarray(x, np.float64).reshape(4, 4) for x in t.column(f'pose.{rig}').to_pylist()])
        act = np.asarray(t.column('action').to_pylist())
        dpos = np.linalg.norm(np.diff(pose[:, :3, 3], axis=0), axis=1)
        dxy_step = np.linalg.norm(np.diff(pose[:, :2, 3], axis=0), axis=1)   # 전진은 수평 0.25 m (계단에서는 높이도 바뀐다)
        fwd = pose[:, :3, 2]
        dang = np.degrees(np.arccos(np.clip((fwd[1:, :2] * fwd[:-1, :2]).sum(1) / np.maximum(
            np.linalg.norm(fwd[1:, :2], axis=1) * np.linalg.norm(fwd[:-1, :2], axis=1), 1e-12), -1, 1)))
        a = act[1:]
        r2_t0 = np.abs(pose[0, :3, 3] - [0, 0, H]).max()
        fw_err = np.abs(dxy_step[a == 1] - 0.25).max() if (a == 1).any() else 0.0
        tn_err = np.abs(dang[np.isin(a, [2, 3])] - 15).max() if np.isin(a, [2, 3]).any() else 0.0
        tm_err = np.abs(dpos[np.isin(a, [2, 3])] - turn_move_expect).max() if np.isin(a, [2, 3]).any() else 0.0
        body = np.c_[np.array(e['body_xy']), np.array(e['body_z'])]
        traj = np.asarray(paths[e['src_episode_id']]['trajectory'])
        dxy = np.min(np.linalg.norm(body[:, None, :2] - traj[None, ::2, :2], axis=2), axis=1)
        cam = np.array(e['cam_c2w_cv'])[:, :3, 3]
        h = sc.cast_rays(o3d.core.Tensor(np.c_[cam, np.tile([0, 0, -1], (len(cam), 1))].astype(np.float32)))['t_hit'].numpy()
        vf, br, strip_r, strip_d = [], [], [], []
        for q_r, q_d in zip(rgbs, deps):
            im = cv2.imread(str(q_r))
            d = cv2.imread(str(q_d), -1)
            vf.append(float(np.mean(d > 0)))
            br.append(float(im.mean()))
            strip_r.append(cv2.resize(im, (args.viz_w, args.viz_w * 3 // 4), interpolation=cv2.INTER_AREA))
            import matplotlib
            col = (matplotlib.colormaps['turbo'](np.clip(d / 10000.0, 0, 1))[..., :3] * 255).astype(np.uint8)[..., ::-1].copy()
            col[d == 0] = 0
            strip_d.append(cv2.resize(col, (args.viz_w, args.viz_w * 3 // 4), interpolation=cv2.INTER_NEAREST))
        tag = f'ep{ei:03d}'
        for c0 in range(0, len(strip_r), CHUNK):             # JPEG 폭 한계 65,535 px -> 200 프레임씩
            for k, st in (('rgb', strip_r), ('dep', strip_d)):
                ok_w = cv2.imwrite(str(out / 'strips' / f'{tag}_{k}_{c0 // CHUNK}.jpg'), np.concatenate(st[c0:c0 + CHUNK], 1),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                assert ok_w, f'strip 저장 실패 {tag} {k} {c0}'
        rows.append({'ep': ei, 'kind': e['kind'], 'frames': n, 'r1': bool(r1), 'r2_t0_err_m': float(r2_t0), 'r2_fwd_err_m': float(fw_err),
                     'r2_turn_err_deg': float(tn_err), 'r2_turnmove_err_m': float(tm_err),
                     'r3_off_max_m': float(dxy.max()), 'r3_end_m': float(e['end_gap_m']),
                     'cam_h': h.tolist(), 'valid_frac': vf, 'bright': br})
        man.append({'ep': ei, 'kind': e['kind'], 'frames': n, 'stairs': e['kind'] == 'multifloor',
                    'z': [round(z, 2) for z in e['body_z']], 'cam_h': [round(float(x), 3) for x in h],
                    'valid': [round(v, 3) for v in vf], 'path_m': round(paths[e['src_episode_id']]['path_length_m'], 1)})
        print(f'  ep {ei} {e["kind"]} r1 {r1} off {dxy.max():.2f}', flush=True)
    report(out, a6, rows, man, turn_move_expect)


def report(out, a6, rows, man, tme):
    H = a6['mount_height_m']
    r1 = all(r['r1'] for r in rows)
    r2 = all(r['r2_t0_err_m'] <= 1e-3 and r['r2_fwd_err_m'] <= 1e-3 and r['r2_turn_err_deg'] <= 0.01 and r['r2_turnmove_err_m'] <= 1e-3 for r in rows)
    r3 = all(r['r3_off_max_m'] <= 0.30 and r['r3_end_m'] <= 0.5 for r in rows)
    has_flat = any(r['kind'] == 'flat' for r in rows)
    fl = np.concatenate([np.array(r['cam_h']) for r in rows if r['kind'] == 'flat']) if has_flat else np.array([np.nan])
    r4v = float(np.mean(np.abs(fl - H) <= 0.06)) if has_flat else float('nan')
    vf = np.concatenate([r['valid_frac'] for r in rows])
    br = np.concatenate([r['bright'] for r in rows])
    r5v, r6v = float(np.mean(vf >= 0.5)), float(np.mean(br > 15))
    gates = [
        {'검증': 'R1 형식', '결과': R.pill(r1), '값': f"{sum(r['r1'] for r in rows)}/{len(rows)} 에피소드", '기준': '프레임 == 행 · 21 컬럼 · 640×480 · uint16'},
        {'검증': 'R2 pose', '결과': R.pill(r2), '값': f"0행 {max(r['r2_t0_err_m'] for r in rows) * 1000:.2f} mm · 전진 {max(r['r2_fwd_err_m'] for r in rows) * 1000:.2f} mm · "
         f"회전 {max(r['r2_turn_err_deg'] for r in rows):.4f}° · 회전 이동량 {max(r['r2_turnmove_err_m'] for r in rows) * 1000:.2f} mm (기대 {tme:.3f} m)",
         '기준': '±1 mm · ±0.01°'},
        {'검증': 'R3 경로', '결과': R.pill(r3), '값': f"몸↔경로 최대 {max(r['r3_off_max_m'] for r in rows):.2f} m · 종점 최대 {max(r['r3_end_m'] for r in rows):.2f} m",
         '기준': '≤ 0.30 · ≤ 0.5 m'},
        {'검증': 'R4 높이 (평지)', '결과': R.pill(r4v >= 0.95) if has_flat else R.pill('평지 없음'),
         '값': f'{100 * r4v:.1f} % (중앙 {np.nanmedian(fl):.3f} m)' if has_flat else '-', '기준': f'{H} ± 0.06 m 인 프레임 ≥ 95 %'},
        {'검증': 'R5 depth 유효', '결과': R.pill(r5v >= 0.99), '값': f'{100 * r5v:.2f} % (유효 비율 중앙 {np.median(vf):.3f} · 최소 {vf.min():.3f})', '기준': '유효 ≥ 0.5 인 프레임 ≥ 99 %'},
        {'검증': 'R6 rgb 밝기', '결과': R.pill(r6v >= 0.99), '값': f'{100 * r6v:.2f} % (중앙 {np.median(br):.0f}/255 · 최소 {br.min():.0f})', '기준': '> 15/255 인 프레임 ≥ 99 %'},
    ]
    passed = [g['결과'].count('PASS') == 1 for g in gates if '평지 없음' not in g['결과']]
    json.dump({'gates': {g['검증']: g['값'] for g in gates}, 'pass': passed,
               'episodes': [{k: v for k, v in r.items() if k not in ('cam_h', 'valid_frac', 'bright')} for r in rows]},
              open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False)
    nm = sum(r['kind'] == 'multifloor' for r in rows)
    body = R.header('nav3d 06b · 노은역 렌더 검사', '노은역 vln_ce 렌더 — 검사',
                    f'04 경로 {len(rows)} (층 이동 {nm} · 평지 {len(rows) - nm}) 을 Isaac 으로 렌더(rgb = GS, depth = 충돌 mesh). '
                    f'rig 칸 <span class="mono">{a6["rig"]}</span>, 높이 {H} m · pitch {a6["pitch_down_deg"]}° · 전방 {a6["forward_m"]} m. '
                    f'<b>판정 {sum(passed)}/{len(passed)} 통과.</b>')
    body += R.section('판정', '<p class="muted">기준은 결과를 보기 전에 정했다. 카메라 바닥 높이 = 카메라에서 수직 아래 충돌 mesh 까지 거리. '
                      f'제자리 회전 이동량 = 회전 프레임에서 카메라가 움직인 거리(카메라가 몸 앞에 있어 {tme:.3f} m 가 맞다).</p>'
                      + R.table(gates, ['검증', '결과', '값', '기준'], raw=('결과',)))
    tab = [{'ep': r['ep'], '종류': '층 이동' if r['kind'] == 'multifloor' else '평지', '프레임': r['frames'], '형식': 'OK' if r['r1'] else 'X',
            '몸↔경로 [m]': f"{r['r3_off_max_m']:.2f}", '종점 [m]': f"{r['r3_end_m']:.2f}",
            '카메라 높이 중앙 [m]': f"{np.median(r['cam_h']):.3f}", 'depth 유효 최소': f"{min(r['valid_frac']):.2f}",
            '밝기 최소': f"{min(r['bright']):.0f}"} for r in rows]
    body += R.section('에피소드 별', R.table(tab, list(tab[0].keys())))
    viewer = f"""<section><h2>프레임 보기</h2>
<div class="bar" style="display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center">
<label>에피소드 <select id="ep"></select></label>
<label>보기 <select id="show"><option value="both">RGB + depth</option><option value="rgb">RGB</option><option value="dep">depth</option></select></label>
<label>프레임 간격 <select id="step"><option>1</option><option selected>2</option><option>4</option></select></label>
<span class="muted" style="font-size:.8rem">↑ ↓ 에피소드</span></div>
<p class="muted" id="info"></p>
<div id="grid" style="display:grid;grid-template-columns:minmax(0,720px);justify-content:center;gap:8px"></div></section>
<script>
const M = {json.dumps(man)};
const CH = {CHUNK};
const $ = id => document.getElementById(id);
let ei = 0;
M.forEach((e, i) => {{ const o = document.createElement('option'); o.value = i; o.textContent = `ep${{e.ep}} · ${{e.stairs ? '층 이동' : '평지'}} · ${{e.path_m}} m · ${{e.frames}} f`; $('ep').appendChild(o); }});
function build() {{
  const e = M[ei], step = +$('step').value, show = $('show').value; $('ep').value = ei;
  $('info').textContent = `ep${{e.ep}} — 머리: 프레임 · 몸 바닥 z · 카메라 바닥 높이 · depth 유효 비율`;
  const g = $('grid'); g.innerHTML = '';
  for (let f = 0; f < e.frames; f += step) {{
    const c = document.createElement('div'); c.style.cssText = 'border:1px solid var(--line);border-radius:6px;background:var(--panel);padding:4px';
    c.innerHTML = `<div class="mono" style="display:flex;justify-content:space-between;color:var(--muted)"><b style="color:var(--fg)">f${{f}}</b><span>z ${{e.z[f].toFixed(2)}} · 카메라 ${{e.cam_h[f].toFixed(2)}} m · depth ${{(e.valid[f] * 100).toFixed(0)}}%</span></div>`;
    const box = document.createElement('div'); const ks = show === 'both' ? ['rgb', 'dep'] : [show];
    box.style.cssText = `display:grid;grid-template-columns:repeat(${{ks.length}},minmax(0,1fr));gap:4px;margin-top:3px`;
    ks.forEach(k => {{ const d = document.createElement('div'); d.className = 'im'; d.dataset.k = k; d.dataset.f = f;
      d.style.cssText = 'width:100%;aspect-ratio:4/3;background-repeat:no-repeat;background-color:#000;border-radius:3px'; box.appendChild(d); }});
    c.appendChild(box); g.appendChild(c);
  }}
  paint();
}}
function paint() {{
  const e = M[ei], tag = 'ep' + String(e.ep).padStart(3, '0');
  document.querySelectorAll('.im').forEach(d => {{ const w = d.clientWidth || 320, h = w * 3 / 4, f = +d.dataset.f;
    const c = Math.floor(f / CH), nf = Math.min(CH, e.frames - c * CH);
    d.style.backgroundImage = `url(strips/${{tag}}_${{d.dataset.k}}_${{c}}.jpg)`; d.style.backgroundSize = `${{w * nf}}px ${{h}}px`; d.style.backgroundPosition = `${{-(f - c * CH) * w}}px 0`; }});
}}
$('ep').onchange = ev => {{ ei = +ev.target.value; build(); }};
['show', 'step'].forEach(k => $(k).onchange = build);
document.addEventListener('keydown', ev => {{ if (ev.target.tagName === 'SELECT') return;
  if (ev.key === 'ArrowDown') {{ ei = Math.min(M.length - 1, ei + 1); build(); ev.preventDefault(); }}
  else if (ev.key === 'ArrowUp') {{ ei = Math.max(0, ei - 1); build(); ev.preventDefault(); }} }});
window.addEventListener('resize', paint);
build();
</script>"""
    body += viewer
    R.write(out, '노은역 렌더 검사', body)
    print('gates:', ['%s %s' % (g['검증'].split()[0], 'PASS' if p else 'FAIL') for g, p in zip([g for g in gates if '평지 없음' not in g['결과']], passed)])
    print(f'-> {out}/report.html')


if __name__ == '__main__':
    main()
