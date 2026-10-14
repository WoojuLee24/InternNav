"""P1-E 검증: render_split.py 산출물이 릴리스 train 포맷과 같은지, 개수·이미지·pose 가 맞는지 확인한다.

검사 항목 (scene 마다)
  C1 개수     : raw split 의 episode 수 == episodes.jsonl == parquet 수 == _index.jsonl, trajectory 당 episode 수
  C2 스키마   : parquet 스키마 == 릴리스 parquet 스키마 (21컬럼, 타입 포함), info.json 키 == 릴리스 키
  C3 이미지   : 렌더한 rig 의 모든 프레임에 jpg/png 존재, 640x480, jpg 4:2:0, depth uint16
  C4 사본     : 같은 trajectory 의 episode 끼리 jpg 바이트 동일 (릴리스 train 과 같은 저장 방식)
  C5 지시문   : tasks[0] == raw instruction_text (끝 ". " 포함)
  V5/V6       : gs_vlnpe/05b_verify_vlnce.py 의 GT 없는 게이트 (로더 재현 로드, pose 불변식)

    /usr/bin/python scripts/dataset_converters/instruction_labeling/render/verify_split.py --config scripts/dataset_converters/instruction_labeling/configs/render.yaml --split val_unseen
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_io import REPO, load_cfg  # noqa: E402


def load_05b():
    p = REPO / 'scripts/dataset_converters/gs_vlnpe/05b_verify_vlnce.py'
    spec = importlib.util.spec_from_file_location('verify_vlnce_05b', p)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    spec.loader.exec_module(m)
    return m


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def check_scene(cfg, split, scene, raw_eps, index_rows, ref_schema, ref_info_keys, m05b) -> dict:
    root = cfg['paths']['out_root'] / split
    base = root / scene
    rigs = list(cfg['camera']['rigs'])
    issues = []
    eps_meta = [json.loads(line) for line in open(base / 'meta' / 'episodes.jsonl')]
    parquets = sorted((base / 'data' / 'chunk-000').glob('*.parquet'))
    # C1
    by_traj = defaultdict(list)
    for r in index_rows:
        by_traj[r['trajectory_id']].append(r)
    if not (len(raw_eps) == len(eps_meta) == len(parquets) == len(index_rows)):
        issues.append(f'C1 개수 불일치 raw {len(raw_eps)} meta {len(eps_meta)} parquet {len(parquets)} '
                      f'index {len(index_rows)}')
    # C2
    for p in parquets[:3]:
        if pq.read_schema(p) != ref_schema:
            issues.append(f'C2 스키마 다름: {p.name}')
    info = json.loads((base / 'meta' / 'info.json').read_text())
    if set(info) != ref_info_keys:
        issues.append(f'C2 info.json 키 다름: {set(info) ^ ref_info_keys}')
    if info['total_episodes'] != len(eps_meta):
        issues.append('C2 info.total_episodes 불일치')
    # C3
    vid = base / 'videos' / 'chunk-000'
    n_img = 0
    for row, em in zip(sorted(index_rows, key=lambda r: r['episode_index']), eps_meta):
        T = em['length']
        for r in rigs:
            for f in (0, T // 2, T - 1):
                j = vid / f'observation.images.rgb.{r}' / f'episode_{row["episode_index"]:06d}_{f}.jpg'
                d = vid / f'observation.images.depth.{r}' / f'episode_{row["episode_index"]:06d}_{f}.png'
                if not (j.is_file() and d.is_file()):
                    issues.append(f'C3 이미지 없음 ep{row["episode_index"]} f{f} {r}')
                    continue
                im = Image.open(j)
                if im.size != (640, 480) or im.mode != 'RGB':
                    issues.append(f'C3 jpg 형식 {j.name} {im.size} {im.mode}')
                da = np.asarray(Image.open(d))
                if da.shape != (480, 640) or da.dtype not in (np.uint16, np.int32):
                    issues.append(f'C3 depth 형식 {d.name} {da.shape} {da.dtype}')
                n_img += 2
            n_files = len(list((vid / f'observation.images.rgb.{r}').glob(f'episode_{row["episode_index"]:06d}_*.jpg')))
            if n_files != T:
                issues.append(f'C3 프레임 수 ep{row["episode_index"]} {r}: 파일 {n_files} != length {T}')
    # C4
    for tid, rows in by_traj.items():
        if len(rows) < 2:
            continue
        a, b = rows[0]['episode_index'], rows[1]['episode_index']
        for r in rigs:
            ja = vid / f'observation.images.rgb.{r}' / f'episode_{a:06d}_0.jpg'
            jb = vid / f'observation.images.rgb.{r}' / f'episode_{b:06d}_0.jpg'
            if md5(ja) != md5(jb):
                issues.append(f'C4 traj {tid} {r}: 사본이 다름')
    # C5
    raw_by_id = {e['episode_id']: e for e in raw_eps}
    for row in index_rows:
        em = eps_meta[row['episode_index']]
        if em['tasks'][0] != raw_by_id[row['episode_id']]['instruction']['instruction_text']:
            issues.append(f'C5 지시문 불일치 ep{row["episode_index"]}')
    # V5 / V6
    eps_idx = list(range(len(eps_meta)))
    v5 = m05b.gate_v5(root, scene, eps_idx, rigs)
    v6 = m05b.gate_v6(root, scene, eps_idx, rigs)
    for name, g in (('V5', v5), ('V6', v6)):
        if g.get('ok') is False:
            issues.append(f'{name}: {g.get("issues")}')
    return {'ok': not issues, 'issues': issues[:20], 'n_issues': len(issues), 'episodes': len(eps_meta),
            'trajectories': len(by_traj), 'images_checked': n_img,
            'V5': v5.get('ok'), 'V5_skipped': v5.get('skipped'), 'V6': v6.get('ok')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    raw = json.load(gzip.open(cfg['paths']['raw_root'] / args.split / f'{args.split}.json.gz'))['episodes']
    raw_by_scene = defaultdict(list)
    for e in raw:
        raw_by_scene[Path(e['scene_id']).stem].append(e)
    root = cfg['paths']['out_root'] / args.split
    index = [json.loads(line) for fp in sorted((root / '_index').glob('*.jsonl')) for line in open(fp)]
    idx_by_scene = defaultdict(list)
    for r in index:
        idx_by_scene[r['scene']].append(r)
    ref_parquet = REPO / cfg['image']['schema_ref']
    ref_schema = pq.read_schema(ref_parquet)
    ref_info_keys = set(json.loads((ref_parquet.parents[2] / 'meta' / 'info.json').read_text()))
    m05b = load_05b()
    rendered = sorted(p.name for p in root.iterdir() if (p / 'meta').is_dir())
    res = {}
    for scene in rendered:
        res[scene] = check_scene(cfg, args.split, scene, raw_by_scene[scene], idx_by_scene[scene], ref_schema,
                                 ref_info_keys, m05b)
        print(scene, 'OK' if res[scene]['ok'] else 'FAIL', res[scene]['issues'][:3], flush=True)
    missing = sorted(set(raw_by_scene) - set(rendered))
    total = {'scenes_rendered': len(rendered), 'scenes_missing': missing,
             'episodes': sum(r['episodes'] for r in res.values()), 'raw_episodes': len(raw),
             'trajectories': sum(r['trajectories'] for r in res.values()),
             'all_ok': all(r['ok'] for r in res.values()) and not missing}
    (root / '_verify.json').write_text(json.dumps({'total': total, 'scenes': res}, indent=2, ensure_ascii=False))
    print(json.dumps(total, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
