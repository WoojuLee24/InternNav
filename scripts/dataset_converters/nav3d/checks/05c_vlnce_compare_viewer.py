"""05c — vln_ce GT 재렌더를 **프레임 하나씩 일대일로** 넘겨 보는 뷰어 두 개(RGB · depth)를 만든다.

05a 가 수치로 본 것을 눈으로 확인하는 단계. 릴리스 action 을 habitat(RigSim)으로 다시 재생해 프레임마다
릴리스 이미지와 우리 이미지를 나란히(+ 차이, + 같은 자리에서 번갈아 보기) 놓는다. ← → 로 프레임, ↑ ↓ 로 에피소드.

출력 `<out>/rgb/` · `<out>/depth/` 각각: `index.html`(뷰어) + `manifest.json`(프레임별 지표) + `strips/*.jpg`
(에피소드·rig 마다 프레임을 가로로 이어 붙인 한 장 — 파일 수를 줄여 Artifact 로 올리기 위해).

| 뷰어 | 패널 | 프레임 지표 |
|---|---|---|
| RGB | 릴리스 · 우리(릴리스와 같은 JPEG 테이블로 인코딩) · \\|차\\|×8 | SSIM, MAE |
| depth | 릴리스 · 우리 (turbo 0~10 m, 검정 = 0 = 표면 없음) · 불일치 지도 | mm 불일치 비율, 우리=0 & 릴리스>0 비율 |

불일치 지도 색: 어두운 회색 = 같음 · 회색 = 1 mm(정수화 경계) · 노랑 = > 1 mm · **자홍 = 우리 0, 릴리스 값 있음**.

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/05c_vlnce_compare_viewer.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # nav3d/ (이 파일은 nav3d/checks/ 에 있다)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'instruction_labeling' / 'render'))

W, H = 320, 240          # 뷰어 프레임 크기 (원본 640×480 의 절반)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='scripts/dataset_converters/instruction_labeling/configs/render.yaml')
    ap.add_argument('--episodes_json', default='scripts/dataset_converters/nav3d/logs/05_vlnce_gt_render/episodes.json')
    ap.add_argument('--nav02_dir', default='scripts/dataset_converters/nav3d/logs/02_vlnce_gt_compare/scenes')
    ap.add_argument('--out_dir', default='scripts/dataset_converters/nav3d/logs/05_vlnce_gt_render_habitat/viewer')
    ap.add_argument('--jpeg_q', type=int, default=80, help='뷰어 strip JPEG 품질 (판정과 무관, 보기용)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--page_only', action='store_true', help='렌더 없이 기존 manifest·strip 으로 통합 페이지(compare.html)만 다시 쓴다')
    return ap.parse_args()


def turbo(depth_m, vmax=10.0):
    import matplotlib
    rgb = (matplotlib.colormaps['turbo'](np.clip(depth_m / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[depth_m <= 0] = 0
    return rgb


def mismatch_map(o_mm, r_mm):
    d = np.abs(o_mm - r_mm)
    img = np.full(o_mm.shape + (3,), 40, np.uint8)
    img[d == 1] = (120, 120, 120)
    img[d > 1] = (255, 210, 0)
    img[(o_mm == 0) & (r_mm > 0)] = (255, 0, 255)
    return img


def mismatch_map_cm(f_mm, r_mm):
    """채운 depth vs 릴리스: 어두운 회색 ≤ 1 cm · 노랑 1 cm ~ 1 m · 빨강 > 1 m · 자홍 = 채워도 0(우리) 인데 릴리스 값 · 하늘 = 우리 값인데 릴리스 0."""
    d = np.abs(f_mm - r_mm)
    img = np.full(f_mm.shape + (3,), 40, np.uint8)
    img[d > 10] = (255, 210, 0)
    img[d > 1000] = (230, 30, 30)
    img[(f_mm == 0) & (r_mm > 0)] = (255, 0, 255)
    img[(f_mm > 0) & (r_mm == 0)] = (0, 200, 255)
    return img


def main():
    import cv2
    from habitat_rig_sim import RigSim
    from release_io import load_cfg, match_raw, release_episode_data, release_episodes, release_image_path, scene_glb
    from p1r_replay_verify import compare_rgb, depth_to_mm
    from depth_fill import fill_depth_holes

    args = parse()
    if args.page_only:
        write_compare_page(Path(args.out_dir))
        return
    cfg = load_cfg(args.config)
    rigs = list(cfg['camera']['rigs'])
    sub = int(cfg['image']['jpeg_subsampling'])
    dmax = float(cfg['camera']['depth_max_m'])
    out = Path(args.out_dir)
    for k in ('rgb', 'depth'):
        (out / k / 'strips').mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(args.episodes_json).read_text())
    man = {'rgb': [], 'depth': []}
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_q]

    def small(img, nearest=False):
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

    def save_strip(path, frames):
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(frames, 1), cv2.COLOR_RGB2BGR), enc)

    for scene, eps in sel.items():
        mf = {e['ep']: e for e in json.load(open(Path(args.nav02_dir) / f'{scene}.json'))['episodes']}
        rel = {e['episode_index']: e for e in release_episodes(cfg, scene)}
        sim = RigSim(cfg, scene_glb(cfg, scene), gpu=args.gpu)
        for ep in eps:
            raw = match_raw(cfg, scene, rel[ep]['task'])
            d = release_episode_data(cfg, scene, ep, rigs)
            T = len(d['action'])
            info = {'scene': scene, 'ep': ep, 'frames': T, 'stairs': bool(mf[ep]['multifloor']), 'dz': round(float(mf[ep]['gt_dz_m']), 2),
                    'instruction': rel[ep]['task'].split('<INSTRUCTION_SEP>')[0].strip()}
            buf = {r: {k: [] for k in ('r_rgb', 'o_rgb', 'd_rgb', 'r_dep', 'o_dep', 'm_dep', 'f_dep', 'n_dep')} for r in rigs}
            met = {r: {'ssim': [], 'mae': [], 'dmis': [], 'dzero': [], 'height': [], 'dfar': [], 'dfar_f': [], 'dnear_f': []} for r in rigs}
            sim.set_start(raw['start_position'], raw['start_rotation'])
            for f in range(T):
                if f > 0:
                    sim.step(d['action'][f])
                obs = sim.observe()
                hgt = float(sim.agent_position()[1])
                for r in rigs:
                    c, rel_rgb, oq = compare_rgb(obs[r][0], release_image_path(cfg, scene, 'rgb', r, ep, f), sub, ssim=True)
                    o_mm = depth_to_mm(obs[r][1], dmax).astype(np.int64)
                    r_mm = cv2.imread(str(release_image_path(cfg, scene, 'depth', r, ep, f)), -1).astype(np.int64)
                    b = buf[r]
                    b['r_rgb'].append(small(rel_rgb))
                    b['o_rgb'].append(small(oq))
                    b['d_rgb'].append(small(np.clip(np.abs(oq.astype(np.int16) - rel_rgb) * 8, 0, 255).astype(np.uint8)))
                    b['r_dep'].append(small(turbo(r_mm / 1000.0, dmax), True))
                    b['o_dep'].append(small(turbo(o_mm / 1000.0, dmax), True))
                    b['m_dep'].append(small(mismatch_map(o_mm, r_mm), True))
                    f_mm = fill_depth_holes(o_mm)
                    b['f_dep'].append(small(turbo(f_mm / 1000.0, dmax), True))
                    b['n_dep'].append(small(mismatch_map_cm(f_mm, r_mm), True))
                    m = met[r]
                    m['ssim'].append(round(c['ssim'], 4))
                    m['mae'].append(round(c['mae'], 2))
                    m['dmis'].append(round(float(np.mean(o_mm != r_mm)), 4))
                    m['dzero'].append(round(float(np.mean((o_mm == 0) & (r_mm > 0))), 4))
                    m['height'].append(round(hgt, 3))
                    m['dfar'].append(round(float(np.mean(np.abs(o_mm - r_mm) > 1000)), 4))
                    m['dfar_f'].append(round(float(np.mean(np.abs(f_mm - r_mm) > 1000)), 4))
                    m['dnear_f'].append(round(float(np.mean(np.abs(f_mm - r_mm) <= 10)), 4))
            for r in rigs:
                tag = f'{scene}_{ep:06d}_{r}'
                for kind, keys in (('rgb', ('r_rgb', 'o_rgb', 'd_rgb')), ('depth', ('r_dep', 'o_dep', 'm_dep', 'f_dep', 'n_dep'))):
                    for k in keys:
                        save_strip(out / kind / 'strips' / f'{tag}_{k}.jpg', buf[r][k])
            for kind in ('rgb', 'depth'):
                man[kind].append(dict(info, rigs={r: {'tag': f'{scene}_{ep:06d}_{r}', **met[r]} for r in rigs}))
            print(f'  {scene} ep{ep} T={T} {"stairs" if info["stairs"] else "flat"}', flush=True)
        sim.close()
    for kind in ('rgb', 'depth'):
        eps = sorted(man[kind], key=lambda e: (not e['stairs'], e['scene'], e['ep']))
        doc = {'kind': kind, 'w': W, 'h': H, 'rigs': rigs, 'episodes': eps,
               'source': {'release': str(cfg['paths']['release_root']), 'renderer': 'habitat RigSim (instruction_labeling)'}}
        (out / kind / 'manifest.json').write_text(json.dumps(doc, ensure_ascii=False))
        (out / kind / 'index.html').write_text(viewer_html(kind, doc))
    write_compare_page(out)
    print(f'-> {out}/compare.html (통합) · {out}/rgb/index.html · {out}/depth/index.html')


def write_compare_page(out: Path) -> None:
    """RGB·depth 를 한 페이지에: 프레임 격자 전체를 ← → 로 GT ↔ 생성(↔ 차이) 한꺼번에 넘긴다."""
    import nav3d_report as R
    rgb = json.loads((out / 'rgb' / 'manifest.json').read_text())
    dep = json.loads((out / 'depth' / 'manifest.json').read_text())
    dm = {(e['scene'], e['ep']): e for e in dep['episodes']}
    eps = []
    for e in rgb['episodes']:
        d = dm[(e['scene'], e['ep'])]
        eps.append({k: e[k] for k in ('scene', 'ep', 'frames', 'stairs', 'dz', 'instruction')} |
                   {'rigs': {r: {'tag': e['rigs'][r]['tag'], 'ssim': e['rigs'][r]['ssim'], 'dmis': d['rigs'][r]['dmis'],
                                 'dzero': d['rigs'][r]['dzero'], 'height': e['rigs'][r]['height'],
                                 'dfar': d['rigs'][r]['dfar'], 'dfar_f': d['rigs'][r]['dfar_f'], 'dnear_f': d['rigs'][r]['dnear_f']} for r in rgb['rigs']}})
    doc = {'w': rgb['w'], 'h': rgb['h'], 'rigs': rgb['rigs'], 'episodes': eps}
    html = f"""<title>vln_ce GT·생성 비교</title>
{R.FONTS}
<style>{R.CSS}
.wrap{{max-width:1240px}}
.bar{{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}}
select,button{{font:inherit;font-size:.86rem;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg);cursor:pointer}}
button[aria-pressed=true]{{background:var(--accent);color:var(--bg);border-color:var(--accent)}}
.state{{display:flex;align-items:center;gap:10px;position:sticky;top:env(safe-area-inset-top,0px);z-index:2;background:var(--bg);padding-block:8px;border-bottom:1px solid var(--line)}}
.state .arrow{{font-size:1.4rem;line-height:1;padding:6px 16px}}
.state .now{{font-size:1.15rem;font-weight:700;min-width:12ch;text-align:center}}
.state .steps{{display:flex;gap:4px}}.state .steps span{{font-family:var(--f-mono);font-size:.78rem;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}}
.state .steps span.on{{background:var(--accent);color:var(--bg);border-color:var(--accent)}}
.grid{{display:grid;grid-template-columns:minmax(0,720px);justify-content:center;gap:8px}}
.imgs{{display:grid;grid-template-columns:repeat(var(--n,2),minmax(0,1fr));gap:8px}}
.cell{{min-width:0;border:1px solid var(--line);border-radius:6px;background:var(--panel);padding:5px}}
.cell .hd{{display:flex;justify-content:space-between;font-family:var(--f-mono);font-size:.78rem;color:var(--muted);gap:6px;margin-bottom:3px}}
.cell .hd b{{color:var(--fg)}}.cell .bad{{color:var(--ng)}}
.im{{width:100%;aspect-ratio:4/3;background-repeat:no-repeat;background-color:#000;border-radius:3px}}
.cap{{font-size:.72rem;color:var(--muted);margin-top:2px}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.8rem;color:var(--muted)}}.legend i{{display:inline-block;width:12px;height:12px;vertical-align:-2px;margin-right:4px;border:1px solid var(--line)}}
.keys{{font-size:.8rem;color:var(--muted)}}kbd{{font-family:var(--f-mono);border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:var(--panel)}}
@media (max-width:640px){{.wrap{{padding-inline:8px}} .cell{{padding:3px}} .cell .hd{{font-size:.7rem}} .state .arrow{{padding:4px 12px}} .state .steps{{display:none}} .imgs{{gap:4px}} h1{{font-size:1.25rem}}}}
</style>
<div class="wrap">
<header><div class="eyebrow">nav3d 05c · vln_ce GT 재렌더 · RGB + depth</div><h1>vln_ce GT · 생성 렌더 비교</h1>
<p class="muted">한 에피소드의 프레임을 격자로 늘어놓고, <b>←/→ 를 누르면 모든 칸이 한꺼번에 GT(릴리스) ↔ 생성(habitat 재렌더)</b>으로 바뀐다.
같은 자리에서 넘겨 보며 차이를 찾는다. 에피소드 {len(eps)} (계단 {sum(e['stairs'] for e in eps)}) · rig {', '.join(doc['rigs'])} · 칸 크기 {doc['w']}×{doc['h']}.</p></header>
<section style="display:grid;gap:10px">
<div class="bar">
<label>에피소드 <select id="ep"></select></label>
<label>rig <select id="rig"></select></label>
<label>보기 <select id="show"><option value="both">RGB + depth</option><option value="rgb">RGB</option><option value="depth">depth</option></select></label>
<label>프레임 간격 <select id="step"><option>1</option><option selected>2</option><option>4</option><option>8</option></select></label>
<span class="keys"><kbd>←</kbd><kbd>→</kbd> GT · 생성 · 생성(채움) · 차이 · <kbd>↑</kbd><kbd>↓</kbd> 에피소드 · <kbd>R</kbd> rig</span>
</div>
<div class="state">
<button class="arrow" id="prev" aria-label="이전 (GT 쪽)">◀</button>
<div class="now" id="now">GT</div>
<button class="arrow" id="next" aria-label="다음 (생성 쪽)">▶</button>
<div class="steps"><span data-s="0">GT (릴리스)</span><span data-s="1">생성 원본</span><span data-s="2">생성 채움</span><span data-s="3">차이 (채움)</span></div>
<span class="muted" id="info"></span>
</div>
<p class="muted" id="instr" style="max-width:none"></p>
<div class="legend" id="leg"><span>depth: turbo 0 → 10 m, <i style="background:#000"></i>0 = 표면 없음</span>
<span>차이(채움) — RGB: |차|×8 · depth: <i style="background:#282828"></i>≤ 1 cm <i style="background:#ffd200"></i>1 cm~1 m <i style="background:#e61e1e"></i>&gt; 1 m <i style="background:#ff00ff"></i>채워도 0 · 릴리스 값 <i style="background:#00c8ff"></i>우리 값 · 릴리스 0</span>
<span>RGB 는 "생성 원본"과 "생성 채움"이 같다(채우기는 depth 만). 머리: SSIM · depth &gt; 1 m 틀린 픽셀 % 원본 → 채움 · 빨강 = SSIM &lt; 0.95 또는 채움 후 &gt; 1 m 가 1 % 넘음</span></div>
<div class="grid" id="grid"></div>
</section>
</div>
<script>
const DOC = {json.dumps(doc, ensure_ascii=False)};
const SRC = {{rgb: ['r_rgb', 'o_rgb', 'o_rgb', 'd_rgb'], depth: ['r_dep', 'o_dep', 'f_dep', 'n_dep']}};
const NAME = ['GT (릴리스)', '생성 원본', '생성 채움', '차이 (채움)'];
const $ = id => document.getElementById(id);
let ei = 0, ri = Math.max(0, DOC.rigs.indexOf('125cm_30deg')), st = 0;
try {{ const s = JSON.parse(localStorage.getItem('v05c_cmp') || '{{}}'); if (s.ei != null && s.ei < DOC.episodes.length) {{ ei = s.ei; ri = s.ri ?? ri; }}
  ['show', 'step'].forEach(k => {{ if (s[k]) $(k).value = s[k]; }}); }} catch (e) {{}}
DOC.episodes.forEach((e, i) => {{ const o = document.createElement('option'); o.value = i;
  o.textContent = `${{e.stairs ? '계단 ' + (e.dz > 0 ? '+' : '') + e.dz + ' m' : '평지'}} · ${{e.scene}} ep${{e.ep}} (${{e.frames}} f)`; $('ep').appendChild(o); }});
DOC.rigs.forEach((r, i) => {{ const o = document.createElement('option'); o.value = i; o.textContent = r; $('rig').appendChild(o); }});
function build() {{
  const e = DOC.episodes[ei], m = e.rigs[DOC.rigs[ri]], step = +$('step').value, show = $('show').value;
  $('ep').value = ei; $('rig').value = ri;
  $('instr').textContent = e.instruction;
  $('info').textContent = `${{e.scene}} ep${{e.ep}} · ${{DOC.rigs[ri]}} · ${{e.frames}} 프레임`;
  const g = $('grid'); g.innerHTML = '';
  for (let f = 0; f < e.frames; f += step) {{
    const c = document.createElement('div'); c.className = 'cell';
    const bad = m.ssim[f] < 0.95 || m.dfar_f[f] > 0.01;
    c.innerHTML = `<div class="hd"><b>f${{f}}</b><span class="${{bad ? 'bad' : ''}}">SSIM ${{m.ssim[f].toFixed(3)}} · &gt;1m ${{(m.dfar[f] * 100).toFixed(1)}}→${{(m.dfar_f[f] * 100).toFixed(1)}}%</span><span>z ${{m.height[f].toFixed(2)}}</span></div>`;
    const box = document.createElement('div'); box.className = 'imgs'; const ks = show === 'both' ? ['rgb', 'depth'] : [show];
    box.style.setProperty('--n', ks.length);
    ks.forEach(k => {{ const w = document.createElement('div'); const d = document.createElement('div'); d.className = 'im'; d.dataset.k = k; d.dataset.f = f;
      const cap = document.createElement('div'); cap.className = 'cap'; cap.textContent = k === 'rgb' ? 'RGB' : 'depth'; w.appendChild(d); w.appendChild(cap); box.appendChild(w); }});
    c.appendChild(box);
    g.appendChild(c);
  }}
  paint();
  try {{ localStorage.setItem('v05c_cmp', JSON.stringify({{ ei, ri, show: $('show').value, step: $('step').value }})); }} catch (x) {{}}
}}
function paint() {{
  const e = DOC.episodes[ei], tag = e.rigs[DOC.rigs[ri]].tag;
  $('now').textContent = NAME[st];
  document.querySelectorAll('.steps span').forEach(s => s.classList.toggle('on', +s.dataset.s === st));
  document.querySelectorAll('.im').forEach(d => {{
    const w = d.clientWidth || DOC.w, h = w * DOC.h / DOC.w, f = +d.dataset.f;
    d.style.backgroundImage = `url(${{d.dataset.k}}/strips/${{tag}}_${{SRC[d.dataset.k][st]}}.jpg)`;
    d.style.backgroundSize = `${{w * e.frames}}px ${{h}}px`;
    d.style.backgroundPosition = `${{-f * w}}px 0`;
  }});
}}
$('prev').onclick = () => {{ st = Math.max(0, st - 1); paint(); }};
$('next').onclick = () => {{ st = Math.min(3, st + 1); paint(); }};
document.querySelectorAll('.steps span').forEach(s => s.onclick = () => {{ st = +s.dataset.s; paint(); }});
['show', 'step'].forEach(k => $(k).onchange = build);
$('ep').onchange = ev => {{ ei = +ev.target.value; build(); }};
$('rig').onchange = ev => {{ ri = +ev.target.value; build(); }};
document.addEventListener('keydown', ev => {{
  if (ev.target.tagName === 'SELECT') return;
  if (ev.key === 'ArrowRight') {{ st = Math.min(3, st + 1); paint(); }}
  else if (ev.key === 'ArrowLeft') {{ st = Math.max(0, st - 1); paint(); }}
  else if (ev.key === 'ArrowDown') {{ ei = Math.min(DOC.episodes.length - 1, ei + 1); build(); }}
  else if (ev.key === 'ArrowUp') {{ ei = Math.max(0, ei - 1); build(); }}
  else if (ev.key === 'r' || ev.key === 'R') {{ ri = (ri + 1) % DOC.rigs.length; build(); }}
  else return;
  ev.preventDefault();
}});
window.addEventListener('resize', paint);
build();
</script>
"""
    (out / 'compare.html').write_text(html)


def viewer_html(kind: str, doc: dict) -> str:
    """Artifact 게시용 페이지 (문서 뼈대 없이). manifest 를 인라인으로 넣어 fetch 없이 뜬다."""
    import nav3d_report as R
    title = 'vln_ce RGB 일대일 비교' if kind == 'rgb' else 'vln_ce depth 일대일 비교'
    labels = (['릴리스 (GT)', '우리 (habitat)', '|차| × 8'] if kind == 'rgb'
              else ['릴리스 (GT)', '우리 (habitat)', '불일치 지도'])
    keys = ['r_rgb', 'o_rgb', 'd_rgb'] if kind == 'rgb' else ['r_dep', 'o_dep', 'm_dep']
    legend = ('' if kind == 'rgb' else
              '<div class="legend"><span><i style="background:#000"></i>depth 0 (표면 없음)</span>'
              '<span>turbo 0 → 10 m</span><span><i style="background:#282828"></i>같음</span>'
              '<span><i style="background:#787878"></i>1 mm</span><span><i style="background:#ffd200"></i>&gt; 1 mm</span>'
              '<span><i style="background:#ff00ff"></i>우리 0 · 릴리스 값 있음</span></div>')
    metric_names = (['SSIM', 'MAE'] if kind == 'rgb' else ['불일치 %', '우리0·릴리스>0 %'])
    return f"""<title>{R.esc(title)}</title>
{R.FONTS}
<style>{R.CSS}
.wrap{{max-width:1240px}}
.bar{{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}}
select,button{{font:inherit;font-size:.86rem;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg);cursor:pointer}}
button[aria-pressed=true]{{background:var(--accent);color:var(--bg);border-color:var(--accent)}}
.panes{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}
.pane{{min-width:0}}.pane .lab{{font-size:.8rem;color:var(--muted);margin-bottom:3px}}
.img{{width:100%;aspect-ratio:4/3;background-repeat:no-repeat;background-color:#000;border-radius:4px;border:1px solid var(--line)}}
.flick .img{{outline:3px solid var(--accent)}}
.meta{{display:flex;flex-wrap:wrap;gap:6px 18px;font-family:var(--f-mono);font-size:.82rem}}
.meta b{{font-weight:600}}
canvas{{width:100%;height:110px;display:block;border:1px solid var(--line);border-radius:4px;background:var(--panel);cursor:pointer}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.8rem;color:var(--muted)}}.legend i{{display:inline-block;width:12px;height:12px;vertical-align:-2px;margin-right:4px;border:1px solid var(--line)}}
.keys{{font-size:.8rem;color:var(--muted)}}kbd{{font-family:var(--f-mono);border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:var(--panel)}}
.instr{{font-size:.86rem;color:var(--muted);max-width:none}}
@media (max-width:640px){{.panes{{grid-template-columns:1fr}}}}
</style>
<div class="wrap">
<header><div class="eyebrow">nav3d 05c · vln_ce GT 재렌더 · {'RGB' if kind == 'rgb' else 'depth'}</div><h1>{R.esc(title)}</h1>
<p class="muted">릴리스 action 을 habitat 으로 그대로 재생한 프레임과 원본 릴리스 프레임을 하나씩 맞대어 본다.
에피소드 {len(doc['episodes'])} (계단 {sum(e['stairs'] for e in doc['episodes'])}) · rig {', '.join(doc['rigs'])} · 뷰어 해상도 {doc['w']}×{doc['h']} (원본 640×480 의 절반).</p></header>
<section>
<div class="bar">
<label>에피소드 <select id="ep"></select></label>
<label>rig <select id="rig"></select></label>
<button id="flick" aria-pressed="false" title="Space">번갈아 보기</button>
<button id="worst">가장 나쁜 프레임</button>
<span class="keys"><kbd>←</kbd><kbd>→</kbd> 프레임 · <kbd>↑</kbd><kbd>↓</kbd> 에피소드 · <kbd>Space</kbd> 릴리스↔우리 · <kbd>R</kbd> rig</span>
</div>
<p class="instr" id="instr"></p>
<div class="meta" id="meta"></div>
{legend}
<div class="panes" id="panes">
<div class="pane" id="p0"><div class="lab">{labels[0]}</div><div class="img" id="i0"></div></div>
<div class="pane" id="p1"><div class="lab">{labels[1]}</div><div class="img" id="i1"></div></div>
<div class="pane" id="p2"><div class="lab">{labels[2]}</div><div class="img" id="i2"></div></div>
</div>
<canvas id="spark" width="1200" height="110" aria-label="프레임별 지표"></canvas>
<p class="keys">그래프: 주황 = {metric_names[0]}, 파랑 = {metric_names[1]}, 회색 = 에이전트 높이(계단). 그래프를 누르면 그 프레임으로 간다.</p>
</section>
</div>
<script>
const DOC = {json.dumps(doc, ensure_ascii=False)};
const KEYS = {json.dumps(keys)};
const KIND = {json.dumps(kind)};
const $ = id => document.getElementById(id);
let ei = 0, ri = Math.max(0, DOC.rigs.indexOf('125cm_30deg')), fi = 0, flick = false, flickOn = false;
try {{ const s = JSON.parse(localStorage.getItem('v05c_' + KIND) || '{{}}'); if (s.ei != null && s.ei < DOC.episodes.length) {{ ei = s.ei; ri = s.ri ?? ri; }} }} catch (e) {{}}
DOC.episodes.forEach((e, i) => {{ const o = document.createElement('option'); o.value = i;
  o.textContent = `${{e.stairs ? '계단 ' + (e.dz > 0 ? '+' : '') + e.dz + ' m' : '평지'}} · ${{e.scene}} ep${{e.ep}} (${{e.frames}} f)`; $('ep').appendChild(o); }});
DOC.rigs.forEach((r, i) => {{ const o = document.createElement('option'); o.value = i; o.textContent = r; $('rig').appendChild(o); }});
function cur() {{ return DOC.episodes[ei].rigs[DOC.rigs[ri]]; }}
function m0(m) {{ return KIND === 'rgb' ? m.ssim : m.dmis.map(v => v * 100); }}
function m1(m) {{ return KIND === 'rgb' ? m.mae : m.dzero.map(v => v * 100); }}
function setImg(el, key) {{
  const tag = cur().tag, w = el.clientWidth || DOC.w, h = w * DOC.h / DOC.w, n = DOC.episodes[ei].frames;
  el.style.backgroundImage = `url(strips/${{tag}}_${{key}}.jpg)`;
  el.style.backgroundSize = `${{w * n}}px ${{h}}px`;
  el.style.backgroundPosition = `${{-fi * w}}px 0`;
}}
function draw() {{
  const e = DOC.episodes[ei], m = cur();
  fi = Math.max(0, Math.min(fi, e.frames - 1));
  $('ep').value = ei; $('rig').value = ri;
  if (flick) {{
    setImg($('i0'), flickOn ? KEYS[1] : KEYS[0]);
    $('p0').querySelector('.lab').textContent = flickOn ? '{labels[1]}' : '{labels[0]}';
  }} else {{ setImg($('i0'), KEYS[0]); $('p0').querySelector('.lab').textContent = '{labels[0]}'; }}
  $('panes').classList.toggle('flick', flick);
  $('p1').hidden = flick; setImg($('i1'), KEYS[1]); setImg($('i2'), KEYS[2]);
  const a = m0(m)[fi], b = m1(m)[fi];
  $('meta').innerHTML = `<span>프레임 <b>${{fi}}</b> / ${{e.frames - 1}}</span>` +
    (KIND === 'rgb' ? `<span>SSIM <b>${{a.toFixed(4)}}</b></span><span>MAE <b>${{b.toFixed(2)}}</b></span>`
                    : `<span>불일치 <b>${{a.toFixed(2)}} %</b></span><span>우리 0·릴리스 값 <b>${{b.toFixed(2)}} %</b></span>`) +
    `<span>높이 <b>${{m.height[fi].toFixed(2)}} m</b></span><span>${{e.scene}} ep${{e.ep}} · ${{DOC.rigs[ri]}}</span>`;
  $('instr').textContent = e.instruction;
  spark();
  try {{ localStorage.setItem('v05c_' + KIND, JSON.stringify({{ ei, ri }})); }} catch (x) {{}}
}}
function css(v) {{ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }}
function spark() {{
  const c = $('spark'), g = c.getContext('2d'), m = cur(), n = DOC.episodes[ei].frames, W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  const line = (arr, col) => {{ const lo = Math.min(...arr), hi = Math.max(...arr), span = (hi - lo) || 1;
    g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
    arr.forEach((v, i) => {{ const x = 8 + i * (W - 16) / Math.max(1, n - 1), y = H - 10 - (v - lo) / span * (H - 20); i ? g.lineTo(x, y) : g.moveTo(x, y); }}); g.stroke(); }};
  line(m.height, '#9aa0a8'); line(m0(m), '#e67e22'); line(m1(m), '#2b5fa8');
  const x = 8 + fi * (W - 16) / Math.max(1, n - 1);
  g.strokeStyle = css('--fg') || '#000'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
}}
$('spark').addEventListener('click', ev => {{ const r = ev.target.getBoundingClientRect(), n = DOC.episodes[ei].frames;
  fi = Math.round((ev.clientX - r.left) / r.width * (n - 1)); draw(); }});
$('ep').onchange = ev => {{ ei = +ev.target.value; fi = 0; draw(); }};
$('rig').onchange = ev => {{ ri = +ev.target.value; draw(); }};
$('flick').onclick = () => {{ flick = !flick; flickOn = false; $('flick').setAttribute('aria-pressed', flick); draw(); }};
$('worst').onclick = () => {{ const m = cur(), v = m0(m); let k = 0;
  v.forEach((x, i) => {{ if (KIND === 'rgb' ? x < v[k] : x > v[k]) k = i; }}); fi = k; draw(); }};
document.addEventListener('keydown', ev => {{
  if (ev.target.tagName === 'SELECT') return;
  const n = DOC.episodes[ei].frames;
  if (ev.key === 'ArrowRight') {{ fi = Math.min(n - 1, fi + 1); }}
  else if (ev.key === 'ArrowLeft') {{ fi = Math.max(0, fi - 1); }}
  else if (ev.key === 'ArrowDown') {{ ei = Math.min(DOC.episodes.length - 1, ei + 1); fi = 0; }}
  else if (ev.key === 'ArrowUp') {{ ei = Math.max(0, ei - 1); fi = 0; }}
  else if (ev.key === ' ') {{ if (!flick) {{ flick = true; $('flick').setAttribute('aria-pressed', true); }} flickOn = !flickOn; }}
  else if (ev.key === 'r' || ev.key === 'R') {{ ri = (ri + 1) % DOC.rigs.length; }}
  else return;
  ev.preventDefault(); draw();
}});
window.addEventListener('resize', draw);
draw();
</script>
"""


if __name__ == '__main__':
    main()
