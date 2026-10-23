"""08 — 우리가 만든 데이터를 **InternNav 학습 로더 코드 그대로** 읽어 본다 (vln_ce · vln_pe). 릴리스 한 씬을 대조군으로 같이 돌린다.

| 포맷 | 학습 경로 | 여기서 부르는 것 |
|---|---|---|
| vln_ce | `internvla_n1_lerobot_dataset.NavPixelGoalDataset` (InternVLA-N1) | `get_annotations_from_lerobot_data(data_path, '<h>cm_<p1>deg')` 그대로 + `__getitem__` 이 여는 파일 경로 규칙 그대로 (rgb p1 · rgb p2 · depth p2 · depth p1) |
| vln_pe | `cma_lerobot_dataset.CMALerobotDataset` (CMA 계열) | `internnav.utils.loader.LerobotAsLmdb.get_all_keys / get_data_by_key` 그대로 |

토크나이저·모델 없이 돌 수 있는 데까지 실제 코드를 쓴다. 이미지 텐서 변환(processor)은 부르지 않는다.

## 판정 기준 (결과를 보기 전에 정함)
| | 기준 |
|---|---|
| D1 vln_ce | 모든 episode: actions 길이 == length == poses 길이 == pixel_goals 길이, `__getitem__` 이 열 파일 전부 존재, 첫·끝 프레임 rgb 640×480 · depth uint16 |
| D2 vln_pe | 모든 key 가 예외 없이 읽히고, rgb (T,256,256,3) · depth (T,256,256) · action T 개 · 지시문 토큰이 있다 |
| 대조군 | 릴리스 한 씬이 같은 검사를 통과 (검사 자체가 맞는지) |

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/checks/08_loader_check.py
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
D = 'data/InternData-N1-v0.5-mini'


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vlnce', action='append', default=None,
                    help='<이름>=<data_path(씬 폴더들의 부모)>=<height>,<pitch_1>,<pitch_2> (반복). 기본: 릴리스 대조 · nav3d sg · 노은 meshfloor')
    ap.add_argument('--vlnpe', action='append', default=None, help='<이름>=<traj_data 폴더 (r2r/ 의 부모)> (반복)')
    ap.add_argument('--control_scene', default='17DRP5sb8fy')
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/08_loader_check/summary.json')
    return ap.parse_args()


def one_scene_view(src_scene_dir: Path, sub: str) -> Path:
    """릴리스는 61 씬을 다 훑으므로 한 씬만 심링크한 임시 폴더를 만든다."""
    t = Path(tempfile.mkdtemp(prefix='08_'))
    (t / sub).mkdir(parents=True)
    os.symlink(src_scene_dir, t / sub / src_scene_dir.name)
    return t / sub


def check_vlnce(name, data_path, h, p1, p2) -> dict:
    from PIL import Image
    from internnav.dataset.internvla_n1_lerobot_dataset import get_annotations_from_lerobot_data
    ann = get_annotations_from_lerobot_data(str(data_path), f'{h}cm_{p1}deg')
    issues, n_files, seen = [], 0, set()
    for e in ann['episodes']:
        key = (e['video'], e['id'])
        if key in seen:                          # 지시문마다 같은 episode 가 반복된다
            continue
        seen.add(key)
        T = e['length']
        if not (len(e['actions']) == T == len(e[f'poses_{h}cm_{p1}deg']) == len(e['pixel_goals'])):
            issues.append(f"{e['video']} ep{e['id']}: 길이 불일치")
        for fid in range(T):                     # __getitem__ 의 경로 규칙 그대로
            rgb = os.path.join(e['video'], f'observation.images.rgb.{h}cm_{p1}deg', f"episode_{e['id']:06d}_{fid}.jpg")
            files = [rgb, rgb.replace(f'_{p1}deg', f'_{p2}deg'),
                     rgb.replace(f'_{p1}deg', f'_{p2}deg').replace('rgb', 'depth').replace('.jpg', '.png'),
                     rgb.replace('rgb', 'depth').replace('.jpg', '.png')]
            for f in files:
                n_files += 1
                if not os.path.exists(f):
                    issues.append(f'없음: {f}')
                    break
        for fid in (0, T - 1):
            rgb = os.path.join(e['video'], f'observation.images.rgb.{h}cm_{p1}deg', f"episode_{e['id']:06d}_{fid}.jpg")
            if os.path.exists(rgb):
                im = Image.open(rgb)
                dp = Image.open(rgb.replace('rgb', 'depth').replace('.jpg', '.png'))
                if im.size != (640, 480) or np.asarray(dp).dtype != np.uint16:
                    issues.append(f'{rgb}: rgb {im.size} depth {np.asarray(dp).dtype}')
    return {'name': name, 'data_path': str(data_path), 'setting': f'{h}cm_{p1}_{p2}', 'samples': len(ann['episodes']),
            'episodes': len(seen), 'files_checked': n_files, 'issues': issues[:20], 'n_issues': len(issues), 'ok': not issues and len(seen) > 0}


def check_vlnpe(name, traj_root) -> dict:
    from internnav.utils.loader import LerobotAsLmdb
    L = LerobotAsLmdb(str(traj_root))
    keys = L.get_all_keys()
    issues, shapes = [], {}
    for k in keys:
        try:
            d = L.get_data_by_key(k)
        except Exception as exc:  # noqa: BLE001
            issues.append(f'{k}: {exc!r}'[:200])
            continue
        ed = d['episode_data']
        cam = ed['camera_info']['pano_camera_0']
        rgb, dep, act = np.asarray(cam['rgb']), np.asarray(cam['depth']), ed['action']
        T = len(act)
        if not shapes:
            shapes = {'rgb': list(rgb.shape), 'depth': list(dep.shape), 'action': T, 'camera_position': list(np.asarray(cam['position']).shape),
                      'instructions': len(d['episodes_in_json'])}
        ok = (rgb.shape == (T, 256, 256, 3) and dep.shape == (T, 256, 256) and len(cam['position']) == T and len(ed['robot_info']['position']) == T
              and len(d['episodes_in_json']) == 3 and all(len(t.get('instruction_tokens', [])) == 200 for t in d['episodes_in_json']))
        if not ok:
            issues.append(f'{k}: rgb {rgb.shape} depth {dep.shape} T {T} 지시문 {len(d["episodes_in_json"])}')
    return {'name': name, 'traj_root': str(traj_root), 'keys': len(keys), 'first_sample_fields': shapes, 'issues': issues[:20],
            'n_issues': len(issues), 'ok': not issues and len(keys) > 0}


def main():
    args = parse()
    ce = args.vlnce or [
        f"릴리스 vln_ce (대조, {args.control_scene})=CONTROL_CE=125,0,30",
        f"nav3d sg (MP3D)={D}/vln_ce_nav3d/traj_data/r2r/nav3d_sg=125,0,30",
        f"노은역 meshfloor={D}/vln_ce_noeun_nav3d_meshfloor/traj_data/r2r=60,15,15",
    ]
    pe = args.vlnpe or [f"릴리스 vln_pe (대조, {args.control_scene})=CONTROL_PE", f"노은역 vln_pe={D}/vln_pe_noeun_nav3d/traj_data"]
    res = {'vln_ce': [], 'vln_pe': []}
    for spec in ce:
        name, path, rig = spec.split('=')
        if path == 'CONTROL_CE':
            path = one_scene_view(REPO / D / 'vln_ce' / 'traj_data' / 'r2r' / args.control_scene, 'r2r')
        h, p1, p2 = rig.split(',')
        r = check_vlnce(name, Path(path), h, p1, p2)
        res['vln_ce'].append(r)
        print(f"[08 vln_ce] {name}: {'PASS' if r['ok'] else 'FAIL'} · ep {r['episodes']} · 샘플 {r['samples']} · 파일 {r['files_checked']} · 문제 {r['n_issues']} {r['issues'][:2]}", flush=True)
    for spec in pe:
        name, path = spec.split('=')
        if path == 'CONTROL_PE':
            path = one_scene_view(REPO / D / 'vln_pe' / 'traj_data' / 'r2r' / args.control_scene, 'r2r').parent
        r = check_vlnpe(name, Path(path))
        res['vln_pe'].append(r)
        print(f"[08 vln_pe] {name}: {'PASS' if r['ok'] else 'FAIL'} · key {r['keys']} · 문제 {r['n_issues']} {r['issues'][:2]} · 필드 {r['first_sample_fields']}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False, default=str)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
