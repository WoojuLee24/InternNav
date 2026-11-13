"""노은역 "stop by A" 라벨 품질 측정 — A 를 분할(SAM)하고 목적지까지 거리를 잰다.

왜: 성공 기준이 목적지 3 m 안인데 문장이 가리키는 A 가 멀면 **문장대로 해도 실패**한다.
    노은역 문장은 Gemma 3 1B 가 목적지 프레임만 보고 지었고 (`stop_cue: null`),
    MP3D 쪽 `instruction_labeling/label/stop_cues.py` 가 하는 거리 제한·가림 검사가 없다.

거르는 조건 (사용자와 합의, 2026-11-12):
  a 가깝다  : A 의 3D 중앙점 ↔ 목적지 수평거리 <= dist_max (2.0 m)
              근거 = 로봇이 A 옆(약 1 m)에 서면 목적지에서 최대 3 m -> 성공 보장
  b 크다    : 마스크 면적 >= area_min (0.5 % = 40x40 px) — 더 작으면 사람도 못 알아본다
  (화면 가운데인지 · 잘렸는지는 **거르지 않는다**. 구석에 있어도 알아보면 단서가 된다.
   다만 테두리에 닿으면 3D 중앙점이 치우치므로 cropped 표시만 남긴다.)

같이 재는 값 (지금은 거르지 않고 기록만 — 나중에 문장 규칙에 쓴다):
  spread  : 3D 점들의 p90 거리. 크면 흩어진 것 ("the center of ..." 로 써야 함)
  n_inst  : 같은 질의로 잡힌 인스턴스 중 목적지 inst_r 안의 개수. 여러 개면 지목어가 필요

3D 위치는 **씬 메시 광선**으로 구한다 (Isaac depth 와 대조 검증: 80 점 중 96 % 가 0.3 m 안,
상관 0.90, 어긋난 2 점은 Isaac 의 max_depth=10 m 포화 — 메시 쪽이 더 멀리까지 잰다).

    python3 scripts/apptainer/noeun_landmark_sam.py --snaps logs/goal_snap,logs/goal_snap2 --out logs/landmark
"""
import argparse, gzip, json, math, re
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

W, H, HFOV_DEG = 640, 480, 68.7
FX = (W / 2) / math.tan(math.radians(HFOV_DEG) / 2)
STOP_RE = re.compile(r'\bstop(?:ping)?\s+(?:by|near|at|on|in|once you reach|when you reach)\s+(.+?)[.;]?$', re.I)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snaps', required=True, help='noeun_vlnpe_snap.py 결과 폴더 (쉼표로 여러 개)')
    ap.add_argument('--val', default='data/data/noeun_eval/labels/vln_pe_noeun_nav3d_rbq_v6_b1/raw_data/r2r/val_seen/val_seen.json.gz')
    ap.add_argument('--mesh', default='data/data/noeun_eval/navmesh/noeun_station/noeun_station.glb')
    ap.add_argument('--out', required=True)
    ap.add_argument('--det', default='IDEA-Research/grounding-dino-tiny')
    ap.add_argument('--sam', default='facebook/sam-vit-base')
    ap.add_argument('--box_thr', type=float, default=0.25)
    ap.add_argument('--dist_max', type=float, default=2.0, help='a 기준 — A 가 목적지에서 이 거리 안')
    ap.add_argument('--area_min', type=float, default=0.5, help='b 기준 — 마스크가 화면의 이 %% 이상')
    ap.add_argument('--inst_r', type=float, default=10.0, help='동종 인스턴스를 세는 반경')
    ap.add_argument('--max_cam_goal_m', type=float, default=3.0, help='목적지에서 이보다 먼 프레임은 쓰지 않는다')
    return ap.parse_args()


def landmark_phrase(text):
    for p in reversed(re.split(r'(?<=[.!?])\s+', text.strip())):
        m = STOP_RE.search(p.strip())
        if m:
            return re.sub(r'^(the|a|an)\s+', '', m.group(1).strip(), flags=re.I)
    return None


def ray_dirs(us, vs, c2w):
    d = np.stack([(us - (W - 1) / 2) / FX, (vs - (H - 1) / 2) / FX, np.ones_like(us)], axis=1)
    d = d @ np.asarray(c2w)[:3, :3].T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def main():
    a = parse()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    h2o = lambda q: np.array([q[0], -q[2], q[1]])
    eps = {str(e['episode_id']): e for e in json.load(gzip.open(a.val, 'rt'))['episodes']}

    by = {}
    for sd in a.snaps.split(','):
        sd = Path(sd.strip())
        if not (sd / 'snaps.jsonl').exists():
            continue
        for l in (sd / 'snaps.jsonl').open():
            r = json.loads(l); r['_dir'] = str(sd)
            by.setdefault(str(r['ep']), []).append(r)

    mesh = trimesh.load(a.mesh, force='mesh', process=False, skip_materials=True)
    RI = mesh.ray  # embreex 있으면 ray_pyembree (약 32 배 빠름). 순수 파이썬이면 마스크 광선이 사실상 불가
    print(f'[sam] 메시 {len(mesh.faces)} faces · 광선 {type(RI).__module__} · ep {len(by)}개', flush=True)

    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor)
    dp = AutoProcessor.from_pretrained(a.det)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(a.det).to('cuda').eval()
    spn = SamProcessor.from_pretrained(a.sam)
    sam = SamModel.from_pretrained(a.sam).to('cuda').eval()
    print('[sam] 검출기 · SAM 로드', flush=True)

    rows = []
    for ep in sorted(by, key=int):
        e = eps[ep]
        txt = e['instruction']['instruction_text'] if isinstance(e['instruction'], dict) else e['instruction']
        A = landmark_phrase(txt)
        gp = h2o(e['goals'][0]['position'])
        best = min(by[ep], key=lambda r: math.hypot(r['robot_pos'][0] - gp[0], r['robot_pos'][1] - gp[1]))
        cam_d = math.hypot(best['robot_pos'][0] - gp[0], best['robot_pos'][1] - gp[1])
        row = {'ep': ep, 'instruction': txt, 'landmark': A, 'frame': best['rgb'], 'dir': best['_dir'],
               'step': best['step'], 'cam_to_goal_m': round(cam_d, 2)}
        if A is None:
            row['verdict'] = 'A 추출 실패'; rows.append(row); print(f"  ep{ep:>3s} {row['verdict']}", flush=True); continue
        if cam_d > a.max_cam_goal_m:
            row['verdict'] = f'목적지 최근접 프레임이 {cam_d:.1f} m — 측정 불가'; rows.append(row); print(f"  ep{ep:>3s} {row['verdict']}", flush=True); continue

        img = Image.open(Path(best['_dir']) / best['rgb']).convert('RGB')
        inp = dp(images=img, text=A.rstrip('.') + '.', return_tensors='pt').to('cuda')
        with torch.no_grad():
            o = det(**inp)
        res = dp.post_process_grounded_object_detection(o, inp.input_ids, threshold=a.box_thr,
                                                        text_threshold=a.box_thr, target_sizes=[img.size[::-1]])[0]
        if len(res['scores']) == 0:
            row['verdict'] = '검출 없음 (환각 의심)'; rows.append(row); print(f"  ep{ep:>3s} {row['verdict']}", flush=True); continue

        boxes = [[float(v) for v in b] for b in res['boxes']]
        si = spn(img, input_boxes=[boxes], return_tensors='pt').to('cuda')
        with torch.no_grad():
            so = sam(**si, multimask_output=False)
        masks = spn.image_processor.post_process_masks(so.pred_masks.cpu(), si['original_sizes'].cpu(),
                                                       si['reshaped_input_sizes'].cpu())[0].numpy()[:, 0]
        c2w = np.asarray(best['c2w']); org = c2w[:3, 3]

        def mask_xyz(m):
            vs, us = np.nonzero(m)
            if len(us) == 0: return None, 0
            if len(us) > 600:
                k = np.random.RandomState(0).choice(len(us), 600, replace=False); us, vs = us[k], vs[k]
            d = ray_dirs(us.astype(float), vs.astype(float), c2w)
            loc, idx_r, _ = RI.intersects_location(np.repeat([org], len(d), axis=0), d, multiple_hits=False)
            return (loc, len(us)) if len(loc) else (None, len(us))

        # 가장 신뢰도 높은 인스턴스를 주 랜드마크로
        i0 = int(torch.argmax(res['scores']))
        pts, _ = mask_xyz(masks[i0])
        area_pct = float(masks[i0].sum()) / (W * H) * 100
        nlab, lab = cv2.connectedComponents(masks[i0].astype(np.uint8))
        sizes = [int((lab == k).sum()) for k in range(1, nlab)]
        blob = (max(sizes) / sum(sizes) * 100) if sizes else 0.0
        b = res['boxes'][i0]
        cropped = bool(b[0] < 5 or b[1] < 5 or b[2] > W - 5 or b[3] > H - 5)
        row.update(score=round(float(res['scores'][i0]), 3), area_pct=round(area_pct, 2),
                   blob_pct=round(blob, 1), cropped=cropped, n_det=len(boxes))
        if pts is None:
            row['verdict'] = '광선이 메시에 안 닿음'; rows.append(row); print(f"  ep{ep:>3s} {row['verdict']}", flush=True); continue
        pbar = np.median(pts, axis=0)
        d_goal = float(math.hypot(pbar[0] - gp[0], pbar[1] - gp[1]))
        spread = float(np.percentile(np.linalg.norm(pts - pbar, axis=1), 90))
        # 동종 인스턴스 개수
        n_inst = 0
        for j in range(len(boxes)):
            p2, _ = mask_xyz(masks[j])
            if p2 is None: continue
            c2 = np.median(p2, axis=0)
            if math.hypot(c2[0] - gp[0], c2[1] - gp[1]) <= a.inst_r: n_inst += 1
        ok_a = d_goal <= a.dist_max
        ok_b = area_pct >= a.area_min
        row.update(landmark_xyz=[round(float(t), 2) for t in pbar], dist_to_goal_m=round(d_goal, 2),
                   spread_m=round(spread, 2), n_inst=n_inst,
                   pass_a=ok_a, pass_b=ok_b, verdict='통과' if (ok_a and ok_b) else
                   ('멀다' if not ok_a else '작다'))
        rows.append(row)
        print(f"  ep{ep:>3s} '{A[:26]:26s}' d={d_goal:5.2f} area={area_pct:5.2f}% spread={spread:4.2f} n={n_inst} -> {row['verdict']}", flush=True)

    json.dump(rows, open(out / 'landmarks_sam.json', 'w'), ensure_ascii=False, indent=1)
    print(f"[sam] 저장 {out}/landmarks_sam.json ({len(rows)} ep)")


if __name__ == '__main__':
    main()
