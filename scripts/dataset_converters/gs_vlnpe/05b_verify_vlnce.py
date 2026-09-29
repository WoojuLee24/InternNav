"""`05_export_vlnce`가 만든 것이 **정말로 vln_ce 구조인지** 게이트로 검증하고 GT와 대조한다.

`05b_verify_vlnpe.py`의 vln_ce 판이다. 게이트 구성은 같되 형식이 달라 검사 항목이 다르다.

| | vln_pe (05b_verify_vlnpe) | vln_ce (이 파일) |
|---|---|---|
| 이미지 | 에피소드당 `.npy` 덩어리 | 프레임당 `.jpg`/`.png` **낱개** |
| depth | float32 ×10 m | **uint16 밀리미터**, 0=무효 |
| 스트림 | rgb·depth 각 1개 | rig 5개 × rgb·depth = **10개** |
| parquet | 14컬럼 | **21컬럼** (`pose`/`goal`/`relative_goal_frame_id` × 5 rig) |

게이트
------
| | 내용 |
|---|---|
| V1 | 트리·스트림 폴더·파일명. **모든 스트림이 프레임 수가 같아야** 한다 |
| V2 | rgb 640×480 uint8 jpg · depth 640×480 uint16 png |
| V3 | parquet 21컬럼·Arrow 타입·`Array2D` 확장 메타가 GT와 동일 (복사이므로 sha256도) |
| V4 | depth 값역·무효(0) 비율이 GT와 같은 수준 |
| V5 | `2dloader_vlnce/episode_io.py`(학습 규약 재현 리더)로 실제 로드가 되는가 |
| V6 | `pose` 불변식 — 0행 translation `(0,0,H)`, 전진 0.25 m / 회전 15° |
| V7 | **GT 대조** — rig별 SSIM (보고용, FAIL 아님) |
| NEG | 복사본을 5가지로 망가뜨려 해당 게이트가 잡는지 |

커맨드
------
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05b_verify_vlnce.py --render_root data/InternData-N1-v0.5-mini/vln_ce_render/traj_data/r2r --gt_root data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r --scenes all --negative --deep --log_dir logs/gs-vlnpe

Isaac Sim을 띄우지 않는다 — 디스크만 읽는다.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(HERE / 'apply_real'), str(REPO / 'scripts' / 'dataset_converters' / '2dloader_vlnce')):
    if p not in sys.path:
        sys.path.insert(0, p)

SCRIPT_NAME = '05b_verify_vlnce'

IMG_W, IMG_H = 640, 480
DEPTH_SCALE_MM = 1000.0
META_FILES = ('info.json', 'episodes.jsonl', 'episodes_stats.jsonl', 'tasks.jsonl')
# 릴리스 실측 — 전진 0.25 m / 제자리 회전 15°
FORWARD_M, TURN_DEG = 0.25, 15.0
ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT = 1, 2, 3


def scene_paths(root: Path, scene: str) -> dict:
    base = Path(root) / scene
    return {'base': base, 'parquet_dir': base / 'data' / 'chunk-000',
            'videos_dir': base / 'videos' / 'chunk-000', 'meta_dir': base / 'meta'}


def sha256(p: Path, chunk_mb: int = 8) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(chunk_mb << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rigs_present(videos_dir: Path) -> list:
    """그 씬에 실제로 만들어진 rig 목록 (rgb 스트림 기준)."""
    out = set()
    for d in videos_dir.glob('observation.images.rgb.*'):
        name = d.name[len('observation.images.rgb.'):]
        if name != 'topdown':
            out.add(name)
    return sorted(out)


def ep_ids(root: Path, scene: str) -> list:
    d = scene_paths(root, scene)['parquet_dir']
    return sorted(int(p.stem.split('_')[1]) for p in d.glob('episode_*.parquet')) if d.is_dir() else []


def ep_frame_count(root: Path, scene: str, ep: int) -> int:
    import pyarrow.parquet as pq

    return pq.read_metadata(scene_paths(root, scene)['parquet_dir'] / f'episode_{ep:06d}.parquet').num_rows


# ---------------------------------------------------------------------------
# V1 — 트리
# ---------------------------------------------------------------------------

def gate_v1(gt_root: Path, rd: Path, scene: str, eps: list, expect_provenance: bool) -> dict:
    R = scene_paths(rd, scene)
    issues = []
    if not R['base'].is_dir():
        return {'ok': False, 'issues': [f'씬 폴더 없음: {R["base"]}']}
    rigs = rigs_present(R['videos_dir'])
    if not rigs:
        issues.append('rgb 스트림 폴더가 하나도 없다')

    for rig in rigs:
        for kind, ext in (('rgb', 'jpg'), ('depth', 'png')):
            d = R['videos_dir'] / f'observation.images.{kind}.{rig}'
            if not d.is_dir():
                issues.append(f'{kind}.{rig} 폴더 없음')
                continue
            for ep in eps:
                n_pq = ep_frame_count(rd, scene, ep)
                got = sorted(int(p.stem.split('_')[-1]) for p in d.glob(f'episode_{ep:06d}_*.{ext}'))
                if len(got) != n_pq:
                    issues.append(f'{kind}.{rig} ep{ep}: 이미지 {len(got)} != parquet {n_pq}행')
                elif got != list(range(n_pq)):
                    issues.append(f'{kind}.{rig} ep{ep}: 프레임 번호가 0..{n_pq - 1} 연속이 아니다')

    for name in META_FILES:
        if not (R['meta_dir'] / name).is_file():
            issues.append(f'meta/{name} 없음')
    if expect_provenance and not (R['meta_dir'] / 'render_provenance.json').is_file():
        issues.append('meta/render_provenance.json 없음 (렌더 산출물 표시가 빠졌다)')
    return {'ok': not issues, 'issues': issues, 'rigs': rigs, 'episodes': len(eps)}


# ---------------------------------------------------------------------------
# V2 / V4 — 이미지 형식·값역
# ---------------------------------------------------------------------------

def gate_v2v4(gt_root: Path, rd: Path, scene: str, eps: list, rigs: list, n_sample: int) -> dict:
    from PIL import Image

    G, R = scene_paths(gt_root, scene), scene_paths(rd, scene)
    v2, v4 = [], []
    zero_ours, zero_gt = [], []
    for rig in rigs:
        for ep in eps[:max(1, min(len(eps), 2))]:
            n = ep_frame_count(rd, scene, ep)
            idx = np.unique(np.linspace(0, n - 1, min(n_sample, n)).astype(int))
            for i in idx:
                rp = R['videos_dir'] / f'observation.images.rgb.{rig}' / f'episode_{ep:06d}_{i}.jpg'
                dp = R['videos_dir'] / f'observation.images.depth.{rig}' / f'episode_{ep:06d}_{i}.png'
                if not rp.is_file() or not dp.is_file():
                    v2.append(f'{rig} ep{ep} f{i}: 파일 없음')
                    continue
                a = np.asarray(Image.open(rp))
                d = np.asarray(Image.open(dp))
                if a.shape != (IMG_H, IMG_W, 3) or a.dtype != np.uint8:
                    v2.append(f'{rig} ep{ep} f{i} rgb: {a.shape}{a.dtype} (기대 ({IMG_H},{IMG_W},3) uint8)')
                if d.shape != (IMG_H, IMG_W) or d.dtype != np.uint16:
                    v2.append(f'{rig} ep{ep} f{i} depth: {d.shape}{d.dtype} (기대 ({IMG_H},{IMG_W}) uint16)')
                    continue
                zero_ours.append(float((d == 0).mean()))
                gdp = G['videos_dir'] / f'observation.images.depth.{rig}' / f'episode_{ep:06d}_{i}.png'
                if gdp.is_file():
                    zero_gt.append(float((np.asarray(Image.open(gdp)) == 0).mean()))
                nz = d[d > 0]
                if nz.size and nz.max() > 60000:
                    v4.append(f'{rig} ep{ep} f{i} depth: 최대 {nz.max()} (mm 단위가 맞나)')
    zo = float(np.median(zero_ours)) if zero_ours else float('nan')
    zg = float(np.median(zero_gt)) if zero_gt else float('nan')
    return {'V2': {'ok': not v2, 'issues': v2},
            'V4': {'ok': not v4, 'issues': v4,
                   'zero_frac_ours': zo, 'zero_frac_gt': zg}}


# ---------------------------------------------------------------------------
# V3 — parquet (GT에서 복사했으므로 바이트까지 같아야)
# ---------------------------------------------------------------------------

def gate_v3(gt_root: Path, rd: Path, scene: str, eps: list, deep: bool) -> dict:
    import pyarrow.parquet as pq

    G, R = scene_paths(gt_root, scene), scene_paths(rd, scene)
    issues = []
    for ep in eps:
        gp = G['parquet_dir'] / f'episode_{ep:06d}.parquet'
        rp = R['parquet_dir'] / f'episode_{ep:06d}.parquet'
        if not rp.is_file():
            issues.append(f'ep{ep}: parquet 없음')
            continue
        gs, rs = pq.read_schema(gp), pq.read_schema(rp)
        if gs.names != rs.names:
            issues.append(f'ep{ep}: 컬럼명 다름 (GT {len(gs.names)} vs 산출 {len(rs.names)})')
        elif not gs.equals(rs):
            diffs = [f'{n}: {gs.field(n).type} vs {rs.field(n).type}'
                     for n in gs.names if gs.field(n).type != rs.field(n).type]
            issues.append(f'ep{ep}: Arrow 타입 다름 {diffs[:3]}')
        # pose.* 의 Array2D 확장 메타 — pyarrow만으로는 못 만드는 부분이라 꼭 본다
        for n in gs.names:
            if not n.startswith('pose.'):
                continue
            gm = (gs.field(n).metadata or {})
            rm = (rs.field(n).metadata or {})
            if gm != rm:
                issues.append(f'ep{ep} {n}: Array2D 확장 메타 다름')
                break
        if deep and sha256(gp) != sha256(rp):
            issues.append(f'ep{ep}: parquet 바이트가 GT와 다르다 (05는 복사만 해야 한다)')
    for name in META_FILES:
        g, r = G['meta_dir'] / name, R['meta_dir'] / name
        if r.is_file() and g.is_file() and sha256(g) != sha256(r):
            issues.append(f'meta/{name}: GT와 바이트가 다르다')
    return {'ok': not issues, 'issues': issues}


# ---------------------------------------------------------------------------
# V5 — 학습 규약 리더로 로드
# ---------------------------------------------------------------------------

def presets_from_rigs(rigs: list) -> list:
    """만들어진 rig에서 학습 preset 문자열을 뽑는다.

    preset은 `<H>cm_<pitch1>_<pitch2>` 형식이고 **같은 높이의 rig 두 개**를 묶는다
    (`episode_io.parse_preset`). pitch_1은 S2가 보는 FPV, pitch_2는 pose/goal/depth 기준이다.
    """
    by_h = {}
    for r in rigs:
        h, p = r.split('cm_')
        by_h.setdefault(int(h), []).append(int(p.replace('deg', '')))
    out = []
    for h, ps in sorted(by_h.items()):
        ps = sorted(ps)
        if len(ps) >= 2:
            out.append(f'{h}cm_{ps[0]}_{ps[-1]}')
    return out


def gate_v5(rd: Path, scene: str, eps: list, rigs: list) -> dict:
    """`2dloader_vlnce/episode_io.py`는 학습 코드를 import하지 않고 **같은 규약만 재현**한 리더다.

    시그니처는 `load_episode(data_root, scene, episode, preset)` — `data_root`는 씬 폴더가
    아니라 **r2r 루트**고, 넷째 인자는 rig가 아니라 **preset**(`125cm_0_30`)이다.
    """
    try:
        import episode_io
    except Exception as exc:  # noqa: BLE001
        return {'ok': None, 'skipped': f'episode_io import 실패 — {exc!r}'}

    presets = presets_from_rigs(rigs)
    if not presets:
        return {'ok': None, 'skipped': f'같은 높이 rig가 2개 이상 필요하다 (있는 것: {rigs})'}

    issues, loaded = [], 0
    for preset in presets:
        for ep in eps[:2]:
            try:
                episode_io.load_episode(str(rd), scene, ep, preset)
                loaded += 1
            except Exception as exc:  # noqa: BLE001
                issues.append(f'{preset} ep{ep}: {exc!r}')
    return {'ok': not issues, 'issues': issues, 'presets': presets, 'loaded': loaded}


# ---------------------------------------------------------------------------
# V6 — pose 불변식
# ---------------------------------------------------------------------------

def gate_v6(rd: Path, scene: str, eps: list, rigs: list) -> dict:
    import pyarrow.parquet as pq

    R = scene_paths(rd, scene)
    issues = []
    for rig in rigs:
        h = int(rig.split('cm')[0]) / 100.0
        for ep in eps[:3]:
            t = pq.read_table(R['parquet_dir'] / f'episode_{ep:06d}.parquet',
                              columns=['action', f'pose.{rig}']).to_pydict()
            P = np.stack([np.asarray(p, dtype=np.float64).reshape(4, 4) for p in t[f'pose.{rig}']])
            A = np.asarray(t['action'], dtype=np.int32)
            t0 = P[0, :3, 3]
            if not (abs(t0[0]) < 1e-5 and abs(t0[1]) < 1e-5 and abs(t0[2] - h) < 1e-4):
                issues.append(f'{rig} ep{ep}: pose[0] translation {t0.round(4).tolist()} != (0,0,{h})')
            xy = P[:, :2, 3]
            fwd = P[:, :3, 2]
            yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
            for i in range(1, len(A)):
                d = float(np.linalg.norm(xy[i] - xy[i - 1]))
                r = abs(float(np.degrees((yaw[i] - yaw[i - 1] + np.pi) % (2 * np.pi) - np.pi)))
                if A[i] == ACTION_FORWARD and abs(d - FORWARD_M) > 0.05:
                    issues.append(f'{rig} ep{ep} f{i}: 전진인데 이동 {d:.4f} m')
                    break
                if A[i] in (ACTION_LEFT, ACTION_RIGHT) and (d > 1e-4 or abs(r - TURN_DEG) > 0.5):
                    issues.append(f'{rig} ep{ep} f{i}: 회전인데 이동 {d:.4f} m / 각 {r:.2f}°')
                    break
    return {'ok': not issues, 'issues': issues}


# ---------------------------------------------------------------------------
# V7 — GT 대조 (SSIM)
# ---------------------------------------------------------------------------

def gate_v7(gt_root: Path, rd: Path, scene: str, eps: list, rigs: list, n_frames: int) -> dict:
    import cv2
    from skimage.metrics import structural_similarity as ssim_metric

    G, R = scene_paths(gt_root, scene), scene_paths(rd, scene)
    per_rig = {}
    for rig in rigs:
        vals = []
        for ep in eps:
            n = ep_frame_count(rd, scene, ep)
            for i in np.unique(np.linspace(0, n - 1, min(n_frames, n)).astype(int)):
                gp = G['videos_dir'] / f'observation.images.rgb.{rig}' / f'episode_{ep:06d}_{i}.jpg'
                rp = R['videos_dir'] / f'observation.images.rgb.{rig}' / f'episode_{ep:06d}_{i}.jpg'
                if not (gp.is_file() and rp.is_file()):
                    continue
                a = cv2.cvtColor(cv2.imread(str(gp)), cv2.COLOR_BGR2RGB)
                b = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
                if a.shape != b.shape:
                    continue
                vals.append(float(ssim_metric(a, b, channel_axis=2, data_range=255)))
        if vals:
            per_rig[rig] = {'n': len(vals), 'median': float(np.median(vals)), 'min': float(np.min(vals))}
    meds = [v['median'] for v in per_rig.values()]
    return {'ok': None, 'per_rig': per_rig,
            'ssim_median': float(np.median(meds)) if meds else None}


# ---------------------------------------------------------------------------
# negative
# ---------------------------------------------------------------------------

def run_negative(gt_root: Path, rd: Path, scene: str, eps: list, rigs: list) -> list:
    """산출물 **복사본**을 5가지로 망가뜨려 해당 게이트가 잡는지. 원본은 안 건드린다."""
    import pyarrow.parquet as pq
    from PIL import Image

    ep, rig = eps[0], rigs[0]
    cases = []

    def prep(tmp: Path) -> Path:
        dst = tmp / 'r2r'
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(scene_paths(rd, scene)['base'], dst / scene)
        return dst

    # ① rgb 1장 삭제 -> V1 (프레임 수 불일치)
    with tempfile.TemporaryDirectory(prefix='ceneg1_') as td:
        d = prep(Path(td))
        (scene_paths(d, scene)['videos_dir'] / f'observation.images.rgb.{rig}'
         / f'episode_{ep:06d}_0.jpg').unlink()
        r = gate_v1(gt_root, d, scene, [ep], expect_provenance=False)
        cases.append({'case': '① rgb 1장 삭제', 'expect': 'V1', 'detected': not r['ok'],
                      'issues': r['issues'][:2]})

    # ② depth를 미터 단위로 저장 -> V2/V4 (dtype 또는 값역)
    with tempfile.TemporaryDirectory(prefix='ceneg2_') as td:
        d = prep(Path(td))
        p = (scene_paths(d, scene)['videos_dir'] / f'observation.images.depth.{rig}'
             / f'episode_{ep:06d}_0.png')
        Image.fromarray((np.asarray(Image.open(p)).astype(np.float32) / 1000.0).astype(np.uint8)).save(p)
        r = gate_v2v4(gt_root, d, scene, [ep], [rig], 4)
        cases.append({'case': '② depth를 m 단위 uint8로', 'expect': 'V2',
                      'detected': not r['V2']['ok'], 'issues': r['V2']['issues'][:2]})

    # ③ parquet 행 제거 -> V3 (바이트/스키마) + V1 (프레임 수)
    with tempfile.TemporaryDirectory(prefix='ceneg3_') as td:
        d = prep(Path(td))
        p = scene_paths(d, scene)['parquet_dir'] / f'episode_{ep:06d}.parquet'
        t = pq.read_table(p)
        pq.write_table(t.slice(0, t.num_rows - 3), p)
        r = gate_v3(gt_root, d, scene, [ep], deep=True)
        cases.append({'case': '③ parquet 행 3개 제거', 'expect': 'V3', 'detected': not r['ok'],
                      'issues': r['issues'][:2]})

    # ④ pose를 흔들어 전진 거리 깨기 -> V6
    with tempfile.TemporaryDirectory(prefix='ceneg4_') as td:
        d = prep(Path(td))
        p = scene_paths(d, scene)['parquet_dir'] / f'episode_{ep:06d}.parquet'
        t = pq.read_table(p).to_pydict()
        P = [np.asarray(x, dtype=np.float64).reshape(4, 4) for x in t[f'pose.{rig}']]
        for m in P[1:]:
            m[0, 3] += 1.0
        t[f'pose.{rig}'] = [m.reshape(4, 4).tolist() for m in P]
        import pyarrow as pa
        pq.write_table(pa.Table.from_pydict(t), p)
        r = gate_v6(d, scene, [ep], [rig])
        cases.append({'case': '④ pose에 +1 m 오프셋', 'expect': 'V6', 'detected': not r['ok'],
                      'issues': r['issues'][:2]})

    # ⑤ meta/tasks.jsonl 변조 -> V3 (바이트 불일치)
    with tempfile.TemporaryDirectory(prefix='ceneg5_') as td:
        d = prep(Path(td))
        f = scene_paths(d, scene)['meta_dir'] / 'tasks.jsonl'
        f.write_text(f.read_text() + '\n')
        r = gate_v3(gt_root, d, scene, [ep], deep=False)
        cases.append({'case': '⑤ meta/tasks.jsonl 변조', 'expect': 'V3', 'detected': not r['ok'],
                      'issues': r['issues'][:2]})
    return cases


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def build_report(results: dict, neg: list, args, out_dir: Path) -> Path:
    import cv2
    import viz_utils

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / 'frames'
    img_dir.mkdir(parents=True, exist_ok=True)

    def pill(ok):
        if ok is None:
            return '<span class="pill warn">보류</span>'
        return '<span class="pill good">PASS</span>' if ok else '<span class="pill bad">FAIL</span>'

    gates = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    rows = []
    for scene, r in results.items():
        s7 = r['V7'].get('ssim_median')
        rows.append([scene, str(len(r['V1'].get('rigs', [])))]
                    + [pill(r[g].get('ok')) for g in gates]
                    + [f'{s7:.4f}' if s7 is not None else '—'])
    header = ['씬', 'rig'] + gates + ['GT 대비 SSIM']
    table = ('<table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in header) + '</tr></thead><tbody>'
             + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows)
             + '</tbody></table>')

    hard = ('V1', 'V2', 'V3', 'V4', 'V6')
    all_ok = all(results[s][g].get('ok') is not False for s in results for g in hard)
    meds = [results[s]['V7']['ssim_median'] for s in results if results[s]['V7'].get('ssim_median')]
    overall = f'<p><b>구조 검증</b> {pill(all_ok)} — V1~V6(하드 게이트). V7은 GT 대조로 보고용이다.</p>'
    if meds:
        overall += (f'<p>GT 대비 SSIM 중앙값 <b>{float(np.median(meds)):.4f}</b> '
                    f'(최악 씬 {min(meds):.4f}, 씬 {len(meds)}개) · '
                    f'참고: vln_pe 전수 0.8295</p>')

    # rig별 SSIM
    rig_rows = {}
    for scene, r in results.items():
        for rig, v in (r['V7'].get('per_rig') or {}).items():
            rig_rows.setdefault(rig, []).append(v['median'])
    if rig_rows:
        overall += ('<p>rig별 SSIM 중앙값 — '
                    + ' · '.join(f'<b>{k}</b> {float(np.median(v)):.4f}'
                                 for k, v in sorted(rig_rows.items())) + '</p>')

    detail = []
    for scene, r in results.items():
        bad = {g: r[g]['issues'] for g in hard if r[g].get('ok') is False}
        if bad:
            detail.append(f'<h3>{scene}</h3><ul>' + ''.join(
                f'<li><b>{g}</b>: ' + '<br>'.join(map(str, v[:5])) + '</li>'
                for g, v in bad.items()) + '</ul>')
        if r['V5'].get('skipped'):
            detail.append(f'<p><b>{scene}</b> V5 보류 — {r["V5"]["skipped"]}</p>')
        z = r['V4']
        if z.get('zero_frac_ours') is not None and np.isfinite(z['zero_frac_ours']):
            detail.append(f'<p><b>{scene}</b> depth 무효(0) 비율 — 우리 '
                          f'{z["zero_frac_ours"] * 100:.2f}% · GT {z["zero_frac_gt"] * 100:.2f}%</p>')

    neg_html = ''
    if neg:
        nrows = ''.join(f'<tr><td>{c["case"]}</td><td>{c["expect"]}</td><td>{pill(c["detected"])}</td>'
                        f'<td>{"<br>".join(map(str, c["issues"]))}</td></tr>' for c in neg)
        n_ok = sum(1 for c in neg if c['detected'])
        neg_html = (f'<h2>negative test</h2><p>산출물 <b>복사본</b>을 일부러 망가뜨려 해당 게이트가 '
                    f'잡는지 본다. 검출 <b>{n_ok}/{len(neg)}</b></p>'
                    f'<table><thead><tr><th>변형</th><th>기대</th><th>검출</th><th>메시지</th></tr>'
                    f'</thead><tbody>{nrows}</tbody></table>')

    # GT vs 렌더 프레임
    frames_html = ''
    if meds:
        scene = min(results, key=lambda s: results[s]['V7'].get('ssim_median') or 1.0)
        rigs = results[scene]['V1'].get('rigs', [])
        eps = ep_ids(Path(args.render_root), scene)
        if rigs and eps:
            G = scene_paths(Path(args.gt_root), scene)['videos_dir']
            R = scene_paths(Path(args.render_root), scene)['videos_dir']
            cells = []
            for rig in rigs[:3]:
                for i in (0, 10):
                    gp = G / f'observation.images.rgb.{rig}' / f'episode_{eps[0]:06d}_{i}.jpg'
                    rp = R / f'observation.images.rgb.{rig}' / f'episode_{eps[0]:06d}_{i}.jpg'
                    if not (gp.is_file() and rp.is_file()):
                        continue
                    for lab, src in (('GT', gp), ('우리 렌더', rp)):
                        dst = img_dir / f'{scene}_{rig}_f{i}_{lab}.jpg'
                        cv2.imwrite(str(dst), cv2.imread(str(src)))
                        cells.append(
                            f'<div style="flex:0 0 auto;text-align:center">'
                            f'<div style="font-size:12px;opacity:.75;margin-bottom:4px">'
                            f'{lab} · {rig} · f{i}</div>'
                            f'<img src="{viz_utils.image_to_data_uri(dst)}" alt="{lab}" '
                            f'style="width:220px;height:auto;border:1px solid rgba(255,255,255,.15);'
                            f'border-radius:4px"></div>')
            if cells:
                frames_html = (f'<h2>GT vs 우리 렌더 — {scene} (SSIM 최악 씬)</h2>'
                               f'<div style="display:flex;gap:10px;flex-wrap:wrap;overflow-x:auto">'
                               f'{"".join(cells)}</div>')

    summary = (f'<p><b>목적</b>: <code>05_export_vlnce</code>가 만든 데이터셋이 릴리스 vln_ce와 '
               f'<b>같은 구조인지</b> 게이트로 검증하고 이미지를 GT와 대조한다.</p>'
               f'{overall}{table}')
    return viz_utils.save_gallery(out_dir, 'report.html',
                                  f'{SCRIPT_NAME} — 씬 {len(results)}개', summary,
                                  ''.join(detail) + neg_html + frames_html)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render_root', default='data/InternData-N1-v0.5-mini/vln_ce_render/traj_data/r2r')
    ap.add_argument('--gt_root', default='data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r')
    ap.add_argument('--scenes', default='all')
    ap.add_argument('--n_frames', type=int, default=6, help='V7에서 에피소드당 SSIM을 잴 프레임 수')
    ap.add_argument('--n_image_sample', type=int, default=4, help='V2/V4에서 볼 프레임 수')
    ap.add_argument('--deep', action='store_true', help='parquet/meta sha256까지 GT와 비교')
    ap.add_argument('--negative', action='store_true')
    ap.add_argument('--self_test', action='store_true',
                    help='GT를 산출물 자리에 넣어 검증기 자체를 검사(provenance 검사 생략)')
    ap.add_argument('--log_dir', default='logs/gs-vlnpe')
    args = ap.parse_args()

    gt_root, rd = Path(args.gt_root), Path(args.render_root)
    if not rd.is_dir():
        print(f'[{SCRIPT_NAME}] 산출물 루트가 없다: {rd}', file=sys.stderr)
        return 2
    scenes = (sorted(p.name for p in rd.glob('*') if (p / 'data' / 'chunk-000').is_dir())
              if args.scenes == 'all' else [s for s in args.scenes.split(',') if s])
    if not scenes:
        print(f'[{SCRIPT_NAME}] 검사할 씬이 없다', file=sys.stderr)
        return 2
    print(f'[{SCRIPT_NAME}] 씬 {len(scenes)}개 · GT {gt_root} · 산출 {rd}', flush=True)

    results = {}
    for i, scene in enumerate(scenes, 1):
        eps = ep_ids(rd, scene)
        r = {'V1': gate_v1(gt_root, rd, scene, eps, expect_provenance=not args.self_test)}
        rigs = r['V1'].get('rigs', [])
        if eps and rigs:
            r.update(gate_v2v4(gt_root, rd, scene, eps, rigs, args.n_image_sample))
            r['V3'] = gate_v3(gt_root, rd, scene, eps, args.deep)
            r['V5'] = gate_v5(rd, scene, eps, rigs)
            r['V6'] = gate_v6(rd, scene, eps, rigs)
            r['V7'] = gate_v7(gt_root, rd, scene, eps, rigs, args.n_frames)
        else:
            for g in ('V2', 'V3', 'V4', 'V5', 'V6'):
                r[g] = {'ok': False, 'issues': ['에피소드 또는 rig 0개']}
            r['V7'] = {'ok': None}
        results[scene] = r
        flags = ' '.join(f'{g}={"OK" if r[g].get("ok") else ("—" if r[g].get("ok") is None else "FAIL")}'
                         for g in ('V1', 'V2', 'V3', 'V4', 'V5', 'V6'))
        s7 = r['V7'].get('ssim_median')
        tail = f'  SSIM {s7:.4f}' if s7 is not None else ''
        print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene}  rig {len(rigs)}  {flags}{tail}', flush=True)

    neg = []
    if args.negative:
        base = next((s for s in scenes if ep_ids(rd, s) and results[s]['V1'].get('rigs')), None)
        if base:
            print(f'[{SCRIPT_NAME}] negative test — {base} 복사본에 5개 변형', flush=True)
            neg = run_negative(gt_root, rd, base, ep_ids(rd, base), results[base]['V1']['rigs'])
            for c in neg:
                print(f'[{SCRIPT_NAME}]   {c["case"]:26s} 기대 {c["expect"]}  '
                      f'{"검출" if c["detected"] else "놓침"}', flush=True)

    out_dir = Path(args.log_dir) / SCRIPT_NAME
    report = build_report(results, neg, args, out_dir)
    (out_dir / 'verify.json').write_text(json.dumps(
        {'scenes': results, 'negative': neg, 'render_root': str(rd), 'gt_root': str(gt_root)},
        indent=2, ensure_ascii=False, default=str))

    hard = ('V1', 'V2', 'V3', 'V4', 'V6')
    failed = [s for s in results if any(results[s][g].get('ok') is False for g in hard)]
    missed = [c['case'] for c in neg if not c['detected']]
    meds = [results[s]['V7']['ssim_median'] for s in results if results[s]['V7'].get('ssim_median')]
    print(f'\n[{SCRIPT_NAME}] 구조 게이트: {len(results) - len(failed)}/{len(results)}씬 PASS', flush=True)
    if failed:
        print(f'[{SCRIPT_NAME}] FAIL 씬: {failed}', flush=True)
    if neg:
        print(f'[{SCRIPT_NAME}] negative: {len(neg) - len(missed)}/{len(neg)} 검출'
              + (f' · 놓침 {missed}' if missed else ''), flush=True)
    if meds:
        print(f'[{SCRIPT_NAME}] GT 대비 SSIM 중앙값 {float(np.median(meds)):.4f} '
              f'(최악 씬 {min(meds):.4f})', flush=True)
    print(f'[{SCRIPT_NAME}] report -> {report}', flush=True)
    return 0 if (not failed and not missed) else 1


if __name__ == '__main__':
    sys.exit(main())
