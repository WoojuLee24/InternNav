"""06이 만든 데이터셋이 **정말로 vln_pe 구조인지** 게이트로 검증하고 GT와 대조한다 (Phase B).

왜 새로 쓰나
------------
`apply_real/format_validation/validate_noeun_vln_formats.py`는 그대로 쓸 수 없다 —
`compare()`가 인자를 무시하고 `exact_storage_format_same`을 하드코딩 `False`로 반환하며
`integrity_passed`가 노은역 프레임 수(8378)를 하드코딩한다. 재사용하는 것은 그 파일
`inspect_vlnpe()`(L133-171)의 **에피소드 단위 구조 체크 항목**뿐이다.

게이트
------
| V1 | 트리: 디렉토리·파일명·개수가 GT와 동일 |
| V2 | npy dtype/shape가 GT와 동일, rgb·depth·parquet 행수 일치 |
| V3 | parquet: 컬럼명·Arrow 타입·`huggingface` 메타 동일 (복사이므로 sha256도 같아야) |
| V4 | depth 값역 `0 < v <= 1`, NaN 0개, rgb uint8 |
| V5 | `LerobotAsLmdb`: 키 수가 GT와 같고 에피소드마다 지시문 3개 |
| V6 | `cma_collate_fn` 첫 배치가 예외 없이 만들어진다 |
| V7 | **GT 대조** — 에피소드별 SSIM (보고용, FAIL 아님) |

`--negative`를 주면 산출물 **복사본**에 5개 변형을 넣어 해당 게이트만 FAIL 하는지 확인한다.
원본 산출물과 GT는 절대 수정하지 않는다.

커맨드
------
한 씬:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05b_verify_vlnpe.py --render_root data/InternData-N1-v0.5-mini/vln_pe_render/traj_data/r2r --gt_root data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --scenes 17DRP5sb8fy --negative --log_dir logs/gs-vlnpe

전수:
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/05b_verify_vlnpe.py --render_root data/InternData-N1-v0.5-mini/vln_pe_render/traj_data/r2r --gt_root data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --scenes all --negative --log_dir logs/gs-vlnpe

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
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

SCRIPT_NAME = '05b_verify_vlnpe'

# 로더가 실제로 읽는 9개 컬럼 (validate_noeun_vln_formats.py:142-146과 같은 목록)
LOADER_COLUMNS = [
    'observation.camera_position', 'observation.camera_orientation', 'observation.camera_yaw',
    'observation.robot_position', 'observation.robot_orientation', 'observation.robot_yaw',
    'observation.progress', 'observation.step', 'observation.action',
]
META_FILES = ['info.json', 'episodes.jsonl', 'episodes_stats.jsonl', 'tasks.jsonl']


def scene_paths(root: Path, scene: str) -> dict:
    base = Path(root) / scene
    return {
        'base': base,
        'parquet_dir': base / 'data' / 'chunk-000',
        'rgb_dir': base / 'videos' / 'chunk-000' / 'observation.images.rgb',
        'depth_dir': base / 'videos' / 'chunk-000' / 'observation.images.depth',
        'meta_dir': base / 'meta',
    }


def sha256(path: Path, chunk_mb: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_mb << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def npy_header(path: Path) -> tuple:
    """(shape, dtype 문자열) — 배열을 로드하지 않는다."""
    with open(path, 'rb') as fh:
        version = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, version)
    return tuple(shape), str(dtype), bool(fortran)


def ep_ids(root: Path, scene: str) -> list:
    d = scene_paths(root, scene)['parquet_dir']
    return sorted(int(p.stem.split('_')[1]) for p in d.glob('episode_*.parquet')) if d.is_dir() else []


# ---------------------------------------------------------------------------
# V1 — 트리
# ---------------------------------------------------------------------------

def gate_v1(gt_root: Path, rd_root: Path, scene: str,
            require_complete: bool = False, expect_provenance: bool = True) -> dict:
    """디렉토리 집합·파일명 집합이 GT와 같은가.

    `require_complete=False`(기본)는 산출물이 GT의 **부분집합**인 것을 허용한다 —
    `--max_episodes`로 일부만 만든 중간 상태를 검사할 수 있어야 하기 때문이다.
    전수 실행을 검사할 때는 `--require_complete`로 "GT에 있는데 빠진 에피소드"도 잡는다.
    """
    G, R = scene_paths(gt_root, scene), scene_paths(rd_root, scene)
    issues = []
    if not R['base'].is_dir():
        return {'ok': False, 'issues': [f'씬 폴더 없음: {R["base"]}']}

    for key, pattern in (('parquet_dir', 'episode_*.parquet'),
                         ('rgb_dir', 'episode_*.npy'),
                         ('depth_dir', 'episode_*.npy')):
        if not R[key].is_dir():
            issues.append(f'{key} 없음')
            continue
        g = {p.name for p in G[key].glob(pattern)}
        r = {p.name for p in R[key].glob(pattern)}
        extra = r - g
        if extra:
            issues.append(f'{key}: GT에 없는 파일 {len(extra)}개 (예: {sorted(extra)[:3]})')
        if not r:
            issues.append(f'{key}: 파일 0개')
        if require_complete:
            missing = g - r
            if missing:
                issues.append(f'{key}: GT에 있는데 빠진 파일 {len(missing)}개 '
                              f'(예: {sorted(missing)[:3]})')

    for name in META_FILES:
        if not (R['meta_dir'] / name).is_file():
            issues.append(f'meta/{name} 없음')
    if expect_provenance and not (R['meta_dir'] / 'render_provenance.json').is_file():
        issues.append('meta/render_provenance.json 없음 (렌더 산출물 표시가 빠졌다)')

    # 세 디렉토리의 에피소드 집합이 서로 같아야 한다
    sets = {k: {p.stem for p in R[k].glob('episode_*')} for k in ('parquet_dir', 'rgb_dir', 'depth_dir')
            if R[k].is_dir()}
    if len(set(map(frozenset, sets.values()))) > 1:
        issues.append(f'parquet/rgb/depth 에피소드 집합 불일치: '
                      + ', '.join(f'{k}={len(v)}' for k, v in sets.items()))
    return {'ok': not issues, 'issues': issues, 'episodes': len(ep_ids(rd_root, scene))}


# ---------------------------------------------------------------------------
# V2/V3/V4 — 에피소드 단위
# ---------------------------------------------------------------------------

def gate_v2v3v4(gt_root: Path, rd_root: Path, scene: str, eps: list, deep: bool) -> dict:
    import pyarrow.parquet as pq

    G, R = scene_paths(gt_root, scene), scene_paths(rd_root, scene)
    v2, v3, v4 = [], [], []
    for ep in eps:
        tag = f'{scene}/ep{ep}'
        g_rgb, r_rgb = G['rgb_dir'] / f'episode_{ep:06d}.npy', R['rgb_dir'] / f'episode_{ep:06d}.npy'
        g_dep, r_dep = G['depth_dir'] / f'episode_{ep:06d}.npy', R['depth_dir'] / f'episode_{ep:06d}.npy'
        g_pq, r_pq = G['parquet_dir'] / f'episode_{ep:06d}.parquet', R['parquet_dir'] / f'episode_{ep:06d}.parquet'

        # --- V2 dtype/shape/개수 ---
        gs, gd, _ = npy_header(g_rgb)
        rs, rd_, rf = npy_header(r_rgb)
        if (gs, gd) != (rs, rd_):
            v2.append(f'{tag} rgb: GT {gs}{gd} vs 산출 {rs}{rd_}')
        if rf:
            v2.append(f'{tag} rgb: fortran_order=True (GT는 False)')
        gs2, gd2, _ = npy_header(g_dep)
        rs2, rd2, rf2 = npy_header(r_dep)
        if (gs2, gd2) != (rs2, rd2):
            v2.append(f'{tag} depth: GT {gs2}{gd2} vs 산출 {rs2}{rd2}')
        if rf2:
            v2.append(f'{tag} depth: fortran_order=True')
        n_rows = pq.read_metadata(r_pq).num_rows
        if not (rs[0] == rs2[0] == n_rows):
            v2.append(f'{tag}: rgb {rs[0]} · depth {rs2[0]} · parquet {n_rows} 행수 불일치')

        # --- V3 parquet 스키마 + 바이트 동일 ---
        gm, rm = pq.read_schema(g_pq), pq.read_schema(r_pq)
        if gm.names != rm.names:
            v3.append(f'{tag}: 컬럼명 다름 (GT {len(gm.names)} vs 산출 {len(rm.names)})')
        elif not gm.equals(rm):
            diffs = [f'{n}: {gm.field(n).type} vs {rm.field(n).type}'
                     for n in gm.names if gm.field(n).type != rm.field(n).type]
            v3.append(f'{tag}: Arrow 타입 다름 {diffs[:3]}')
        missing = [c for c in LOADER_COLUMNS if c not in rm.names]
        if missing:
            v3.append(f'{tag}: 로더 필수 컬럼 없음 {missing}')
        g_hf = (gm.metadata or {}).get(b'huggingface')
        r_hf = (rm.metadata or {}).get(b'huggingface')
        if g_hf != r_hf:
            v3.append(f'{tag}: huggingface 스키마 메타 다름 '
                      f'(GT {len(g_hf or b"")}B vs 산출 {len(r_hf or b"")}B)')
        if deep and sha256(g_pq) != sha256(r_pq):
            v3.append(f'{tag}: parquet 바이트가 GT와 다르다 (06은 복사만 해야 한다)')

        # --- V4 값역 ---
        d = np.load(r_dep, mmap_mode='r')
        dmin, dmax = float(np.min(d)), float(np.max(d))
        n_nan = int(np.count_nonzero(~np.isfinite(np.asarray(d))))
        if n_nan:
            v4.append(f'{tag} depth: 비유한 값 {n_nan}개')
        if not (dmin > 0.0 and dmax <= 1.0):
            v4.append(f'{tag} depth: 값역 [{dmin:.6g}, {dmax:.6g}] (0 < v <= 1 이어야)')
        if rd_ != 'uint8':
            v4.append(f'{tag} rgb: dtype {rd_} (uint8 이어야)')
        if rd2 != 'float32':
            v4.append(f'{tag} depth: dtype {rd2} (float32 이어야)')

    for name in META_FILES:
        g, r = G['meta_dir'] / name, R['meta_dir'] / name
        if r.is_file() and g.is_file() and sha256(g) != sha256(r):
            v3.append(f'{scene}/meta/{name}: GT와 바이트가 다르다 (06은 복사만 해야 한다)')

    return {'V2': {'ok': not v2, 'issues': v2},
            'V3': {'ok': not v3, 'issues': v3},
            'V4': {'ok': not v4, 'issues': v4}}


# ---------------------------------------------------------------------------
# V5 — 로더 왕복
# ---------------------------------------------------------------------------

def gate_v5(rd_root: Path, scene: str, eps: list) -> dict:
    """`episodes_stats.jsonl`의 task_index 범위 -> `tasks.jsonl` 연결이 성립하는가.

    로더(`internnav/utils/loader.py:184-219`)가 쓰는 것과 **같은 규칙**을 여기서 직접 확인한다.
    `LerobotAsLmdb`를 import하면 torch 등 무거운 의존성이 붙어서, 규칙만 그대로 옮긴다.
    """
    M = scene_paths(rd_root, scene)['meta_dir']
    issues = []
    stats = [json.loads(l) for l in (M / 'episodes_stats.jsonl').read_text().splitlines() if l.strip()]
    tasks = [json.loads(l) for l in (M / 'tasks.jsonl').read_text().splitlines() if l.strip()]
    episodes = [json.loads(l) for l in (M / 'episodes.jsonl').read_text().splitlines() if l.strip()]
    by_index = {t['task_index']: t for t in tasks}
    stat_by_ep = {s['episode_index']: s for s in stats}
    ep_by_index = {e['episode_index']: e for e in episodes}

    n3 = 0
    for ep in eps:
        if ep not in stat_by_ep:
            issues.append(f'ep{ep}: episodes_stats.jsonl에 없음')
            continue
        ti = stat_by_ep[ep].get('task_index') or {}
        lo, hi = ti.get('min'), ti.get('max')
        if lo is None or hi is None:
            issues.append(f'ep{ep}: task_index 범위 없음')
            continue
        got = [by_index[i] for i in range(int(lo), int(hi) + 1) if i in by_index]
        if len(got) != int(hi) - int(lo) + 1:
            issues.append(f'ep{ep}: tasks.jsonl에 task_index {lo}~{hi} 중 일부 없음 ({len(got)}개만)')
            continue
        if len(got) == 3:
            n3 += 1
        for t in got:
            toks = t.get('instruction_tokens')
            if toks is None or len(toks) != 200:
                issues.append(f'ep{ep} task{t["task_index"]}: instruction_tokens 길이 '
                              f'{len(toks) if toks is not None else "없음"} (200 이어야)')
                break
            if max(toks) >= 2504:
                issues.append(f'ep{ep} task{t["task_index"]}: 토큰 id {max(toks)} >= 2504 '
                              f'(cma.py vocab_size 초과)')
                break
        if ep in ep_by_index:
            want = [t['task'] for t in got]
            if ep_by_index[ep].get('tasks') != want:
                issues.append(f'ep{ep}: episodes.jsonl의 tasks가 tasks.jsonl과 불일치')
    return {'ok': not issues, 'issues': issues, 'episodes_with_3_instructions': n3}


# ---------------------------------------------------------------------------
# V6 — CMA 데이터로더 첫 배치
# ---------------------------------------------------------------------------

def gate_v6(rd_root: Path, scene: str) -> dict:
    """실제 학습 경로로 배치 하나를 만들어 본다. 무거우므로 실패해도 원인을 그대로 적어 둔다."""
    try:
        import torch  # noqa: F401
        from internnav.dataset.cma_lerobot_dataset import CMALerobotDataset, cma_collate_fn  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {'ok': None, 'skipped': f'import 실패 — {exc!r}'}
    return {'ok': None, 'skipped': '미구현 — 학습 config 없이 데이터셋을 세울 수 없어 보류'}


# ---------------------------------------------------------------------------
# V7 — GT 대조 (SSIM)
# ---------------------------------------------------------------------------

def gate_v7(gt_root: Path, rd_root: Path, scene: str, eps: list, n_frames: int) -> dict:
    from skimage.metrics import structural_similarity as ssim_metric

    G, R = scene_paths(gt_root, scene), scene_paths(rd_root, scene)
    per_ep = []
    for ep in eps:
        g = np.load(G['rgb_dir'] / f'episode_{ep:06d}.npy', mmap_mode='r')
        r = np.load(R['rgb_dir'] / f'episode_{ep:06d}.npy', mmap_mode='r')
        n = min(len(g), len(r))
        idx = np.unique(np.linspace(0, n - 1, min(n_frames, n)).astype(int)).tolist()
        vals = [float(ssim_metric(np.asarray(g[i]), np.asarray(r[i]), channel_axis=2, data_range=255))
                for i in idx]
        per_ep.append({'episode': ep, 'ssim_median': float(np.median(vals)),
                       'ssim_min': float(np.min(vals)), 'frames': len(idx)})
    meds = [e['ssim_median'] for e in per_ep]
    return {'ok': None, 'per_episode': per_ep,
            'ssim_median': float(np.median(meds)) if meds else None,
            'ssim_worst_episode': float(np.min(meds)) if meds else None}


# ---------------------------------------------------------------------------
# negative test — 복사본을 망가뜨려 게이트가 실제로 잡는지 본다
# ---------------------------------------------------------------------------

def run_negative(gt_root: Path, rd_root: Path, scene: str, eps: list,
                 expect_provenance: bool = True) -> list:
    """산출물 **복사본**에 5개 변형을 넣고 각각 해당 게이트만 FAIL 하는지 확인한다.

    원본 산출물과 GT는 건드리지 않는다 — 복사본은 임시 폴더에 만들고 끝나면 지운다.
    """
    import pyarrow.parquet as pq

    ep = eps[0]
    cases = []

    def prep(tmp: Path) -> Path:
        dst = tmp / 'r2r'
        (dst / scene).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(scene_paths(rd_root, scene)['base'], dst / scene)
        return dst

    # ① parquet 행 3개 제거 -> V2 (행수 불일치)
    with tempfile.TemporaryDirectory(prefix='neg1_') as td:
        dst = prep(Path(td))
        p = scene_paths(dst, scene)['parquet_dir'] / f'episode_{ep:06d}.parquet'
        t = pq.read_table(p)
        pq.write_table(t.slice(0, t.num_rows - 3), p)
        r = gate_v2v3v4(gt_root, dst, scene, [ep], deep=False)
        cases.append({'case': '① parquet 행 3개 제거', 'expect': 'V2', 'detected': not r['V2']['ok'],
                      'issues': r['V2']['issues'][:2]})

    # ② depth를 m 단위로 저장 -> V4 (값역 초과)
    with tempfile.TemporaryDirectory(prefix='neg2_') as td:
        dst = prep(Path(td))
        f = scene_paths(dst, scene)['depth_dir'] / f'episode_{ep:06d}.npy'
        np.save(f, (np.load(f) * 10.0).astype(np.float32))
        r = gate_v2v3v4(gt_root, dst, scene, [ep], deep=False)
        cases.append({'case': '② depth를 m 단위로 저장', 'expect': 'V4', 'detected': not r['V4']['ok'],
                      'issues': r['V4']['issues'][:2]})

    # ③ rgb를 float32로 -> V2 (dtype 불일치)
    with tempfile.TemporaryDirectory(prefix='neg3_') as td:
        dst = prep(Path(td))
        f = scene_paths(dst, scene)['rgb_dir'] / f'episode_{ep:06d}.npy'
        np.save(f, np.load(f).astype(np.float32))
        r = gate_v2v3v4(gt_root, dst, scene, [ep], deep=False)
        cases.append({'case': '③ rgb를 float32로', 'expect': 'V2', 'detected': not r['V2']['ok'],
                      'issues': r['V2']['issues'][:2]})

    # ④ episodes_stats의 task_index 범위를 1개로 축소 -> V5
    with tempfile.TemporaryDirectory(prefix='neg4_') as td:
        dst = prep(Path(td))
        f = scene_paths(dst, scene)['meta_dir'] / 'episodes_stats.jsonl'
        lines = []
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d['episode_index'] == ep:
                d['task_index'] = {'min': d['task_index']['min'], 'max': d['task_index']['min'],
                                   'count': d['task_index'].get('count', 0)}
            lines.append(json.dumps(d, ensure_ascii=False))
        f.write_text('\n'.join(lines) + '\n')
        r = gate_v5(dst, scene, [ep])
        # 지시문이 3개가 아니게 되는 것이 검출 신호다
        cases.append({'case': '④ task_index 범위를 1개로', 'expect': 'V5',
                      'detected': r['episodes_with_3_instructions'] == 0,
                      'issues': [f"3개 지시문 에피소드 {r['episodes_with_3_instructions']}개"]})

    # ⑤ rgb npy 1개 삭제 -> V1 (에피소드 집합 불일치)
    with tempfile.TemporaryDirectory(prefix='neg5_') as td:
        dst = prep(Path(td))
        (scene_paths(dst, scene)['rgb_dir'] / f'episode_{ep:06d}.npy').unlink()
        r = gate_v1(gt_root, dst, scene, expect_provenance=expect_provenance)
        cases.append({'case': '⑤ rgb npy 1개 삭제', 'expect': 'V1', 'detected': not r['ok'],
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

    gates = ['V1', 'V2', 'V3', 'V4', 'V5', 'V7']
    rows = []
    for scene, r in results.items():
        cells = [scene] + [pill(r[g].get('ok')) for g in gates]
        s7 = r['V7'].get('ssim_median')
        cells.append(f'{s7:.4f}' if s7 is not None else '—')
        rows.append(cells)
    header = ['씬'] + gates + ['GT 대비 SSIM']
    table = ('<table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in header) + '</tr></thead><tbody>'
             + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows)
             + '</tbody></table>')

    # 전체 판정
    hard = [g for g in ('V1', 'V2', 'V3', 'V4', 'V5')]
    all_ok = all(results[s][g].get('ok') is not False for s in results for g in hard)
    meds = [results[s]['V7']['ssim_median'] for s in results if results[s]['V7'].get('ssim_median')]
    overall = (f'<p><b>구조 검증</b> {pill(all_ok)} — V1~V5 (하드 게이트). '
               f'V7은 GT 대조로 보고용이다.</p>')
    if meds:
        overall += (f'<p>GT 대비 SSIM 중앙값 <b>{float(np.median(meds)):.4f}</b> '
                    f'(최악 씬 {min(meds):.4f}, 씬 {len(meds)}개)</p>')

    # 실패 상세
    detail = []
    for scene, r in results.items():
        bad = {g: r[g]['issues'] for g in hard if r[g].get('ok') is False}
        if bad:
            detail.append(f'<h3>{scene}</h3><ul>' + ''.join(
                f'<li><b>{g}</b>: ' + '<br>'.join(map(str, v[:5])) + '</li>' for g, v in bad.items()) + '</ul>')

    # negative
    neg_html = ''
    if neg:
        nrows = ''.join(f'<tr><td>{c["case"]}</td><td>{c["expect"]}</td><td>{pill(c["detected"])}</td>'
                        f'<td>{"<br>".join(map(str, c["issues"]))}</td></tr>' for c in neg)
        n_ok = sum(1 for c in neg if c['detected'])
        neg_html = (f'<h2>negative test — 게이트가 실제로 잡는가</h2>'
                    f'<p>산출물 <b>복사본</b>을 일부러 망가뜨려 해당 게이트만 FAIL 하는지 본다. '
                    f'검출 <b>{n_ok}/{len(neg)}</b></p>'
                    f'<table><thead><tr><th>변형</th><th>기대 게이트</th><th>검출</th><th>메시지</th>'
                    f'</tr></thead><tbody>{nrows}</tbody></table>')

    # GT vs 렌더 프레임 (최악/최선 씬)
    frames_html = ''
    if meds:
        ranked = sorted(((results[s]['V7']['ssim_median'], s) for s in results
                         if results[s]['V7'].get('ssim_median')))
        picks = [('최악', ranked[0][1]), ('최선', ranked[-1][1])] if len(ranked) > 1 else [('표본', ranked[0][1])]
        blocks = []
        for label, scene in picks:
            eps = results[scene]['V7']['per_episode']
            if not eps:
                continue
            ep = min(eps, key=lambda e: e['ssim_median'])['episode']
            g = np.load(scene_paths(Path(args.gt_root), scene)['rgb_dir'] / f'episode_{ep:06d}.npy',
                        mmap_mode='r')
            r = np.load(scene_paths(Path(args.render_root), scene)['rgb_dir'] / f'episode_{ep:06d}.npy',
                        mmap_mode='r')
            n = min(len(g), len(r))
            idx = np.unique(np.linspace(0, n - 1, min(3, n)).astype(int)).tolist()
            cells = []
            for i in idx:
                gp = img_dir / f'{scene}_ep{ep:06d}_f{i:04d}_gt.jpg'
                rp = img_dir / f'{scene}_ep{ep:06d}_f{i:04d}_render.jpg'
                cv2.imwrite(str(gp), cv2.cvtColor(np.asarray(g[i]), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(rp), cv2.cvtColor(np.asarray(r[i]), cv2.COLOR_RGB2BGR))
                for lab, path in (('GT', gp), ('우리 렌더', rp)):
                    cells.append(
                        f'<div style="flex:0 0 auto;text-align:center">'
                        f'<div style="font-size:12px;opacity:.75;margin-bottom:4px">{lab} · f{i}</div>'
                        f'<img src="{viz_utils.image_to_data_uri(path)}" alt="{lab}" '
                        f'style="width:200px;height:auto;image-rendering:pixelated;'
                        f'border:1px solid rgba(255,255,255,.15);border-radius:4px"></div>')
            blocks.append(f'<h3>{label} 씬 — {scene} · episode {ep}</h3>'
                          f'<div style="display:flex;gap:10px;flex-wrap:wrap;overflow-x:auto">'
                          f'{"".join(cells)}</div>')
        frames_html = '<h2>GT vs 우리 렌더</h2>' + ''.join(blocks)

    summary = (f'<p><b>목적</b>: 06이 만든 데이터셋이 릴리스 vln_pe와 <b>같은 구조인지</b> 게이트로 '
               f'검증하고, 이미지를 GT와 대조한다.</p>{overall}{table}')
    body = ''.join(detail) + neg_html + frames_html
    return viz_utils.save_gallery(out_dir, 'report.html',
                                  f'{SCRIPT_NAME} — 씬 {len(results)}개', summary, body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render_root', default='data/InternData-N1-v0.5-mini/vln_pe_render/traj_data/r2r',
                    help='06이 만든 산출물의 씬 폴더들이 있는 곳')
    ap.add_argument('--gt_root', default='data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r',
                    help='원본 GT. **읽기만 한다**')
    ap.add_argument('--scenes', default='all', help='콤마로 지정하거나 all')
    ap.add_argument('--max_episodes', type=int, default=0, help='씬당 검사할 에피소드 상한. 0이면 전부')
    ap.add_argument('--n_frames', type=int, default=6, help='V7에서 에피소드당 SSIM을 잴 프레임 수')
    ap.add_argument('--deep', action='store_true',
                    help='parquet/meta의 sha256까지 GT와 비교(느리지만 복사가 맞는지 확실히 본다)')
    ap.add_argument('--require_complete', action='store_true',
                    help='GT의 모든 에피소드가 산출물에 있어야 PASS (전수 실행 검사용)')
    ap.add_argument('--self_test', action='store_true',
                    help='검증기 자기검사 — GT를 산출물 자리에 넣고 돌린다. '
                         'render_provenance.json 검사를 건너뛰어 전 게이트 PASS가 기준선이 된다')
    ap.add_argument('--negative', action='store_true', help='negative test도 실행')
    ap.add_argument('--log_dir', default='logs/gs-vlnpe')
    args = ap.parse_args()

    gt_root, rd_root = Path(args.gt_root), Path(args.render_root)
    if not rd_root.is_dir():
        print(f'[{SCRIPT_NAME}] 산출물 루트가 없다: {rd_root}', file=sys.stderr)
        return 2
    scenes = (sorted(p.name for p in rd_root.glob('*') if (p / 'data' / 'chunk-000').is_dir())
              if args.scenes == 'all' else [s for s in args.scenes.split(',') if s])
    if not scenes:
        print(f'[{SCRIPT_NAME}] 검사할 씬이 없다', file=sys.stderr)
        return 2
    print(f'[{SCRIPT_NAME}] 씬 {len(scenes)}개 · GT {gt_root} · 산출 {rd_root}', flush=True)

    results = {}
    for i, scene in enumerate(scenes, 1):
        eps = ep_ids(rd_root, scene)
        if args.max_episodes > 0:
            eps = eps[:args.max_episodes]
        r = {'V1': gate_v1(gt_root, rd_root, scene,
                           require_complete=args.require_complete,
                           expect_provenance=not args.self_test)}
        if eps and r['V1']['ok'] is not False or eps:
            r.update(gate_v2v3v4(gt_root, rd_root, scene, eps, args.deep))
            r['V5'] = gate_v5(rd_root, scene, eps)
            r['V7'] = gate_v7(gt_root, rd_root, scene, eps, args.n_frames)
        else:
            for g in ('V2', 'V3', 'V4', 'V5'):
                r[g] = {'ok': False, 'issues': ['에피소드 0개']}
            r['V7'] = {'ok': None}
        results[scene] = r
        flags = ' '.join(f'{g}={"OK" if r[g].get("ok") else ("—" if r[g].get("ok") is None else "FAIL")}'
                         for g in ('V1', 'V2', 'V3', 'V4', 'V5'))
        s7 = r['V7'].get('ssim_median')
        print(f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene}  {flags}  '
              f'SSIM {s7:.4f}' if s7 is not None else
              f'[{SCRIPT_NAME}] {i}/{len(scenes)} {scene}  {flags}', flush=True)

    neg = []
    if args.negative:
        base = next((s for s in scenes if ep_ids(rd_root, s)), None)
        if base:
            print(f'[{SCRIPT_NAME}] negative test — {base} 복사본에 5개 변형', flush=True)
            neg = run_negative(gt_root, rd_root, base, ep_ids(rd_root, base),
                               expect_provenance=not args.self_test)
            for c in neg:
                print(f'[{SCRIPT_NAME}]   {c["case"]:28s} 기대 {c["expect"]}  '
                      f'{"검출" if c["detected"] else "놓침"}', flush=True)

    out_dir = Path(args.log_dir) / SCRIPT_NAME
    report = build_report(results, neg, args, out_dir)
    (out_dir / 'verify.json').write_text(json.dumps(
        {'scenes': results, 'negative': neg,
         'render_root': str(rd_root), 'gt_root': str(gt_root)},
        indent=2, ensure_ascii=False, default=str))

    hard = ('V1', 'V2', 'V3', 'V4', 'V5')
    failed = [s for s in results if any(results[s][g].get('ok') is False for g in hard)]
    neg_missed = [c['case'] for c in neg if not c['detected']]
    meds = [results[s]['V7']['ssim_median'] for s in results if results[s]['V7'].get('ssim_median')]
    print(f'\n[{SCRIPT_NAME}] 구조 게이트: {len(results) - len(failed)}/{len(results)}씬 PASS', flush=True)
    if failed:
        print(f'[{SCRIPT_NAME}] FAIL 씬: {failed}', flush=True)
    if neg:
        print(f'[{SCRIPT_NAME}] negative: {len(neg) - len(neg_missed)}/{len(neg)} 검출'
              + (f' · 놓침 {neg_missed}' if neg_missed else ''), flush=True)
    if meds:
        print(f'[{SCRIPT_NAME}] GT 대비 SSIM 중앙값 {float(np.median(meds)):.4f} '
              f'(최악 씬 {min(meds):.4f})', flush=True)
    print(f'[{SCRIPT_NAME}] report -> {report}', flush=True)
    return 0 if (not failed and not neg_missed) else 1


if __name__ == '__main__':
    sys.exit(main())
