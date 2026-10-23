"""VLM 에 보내는 요청(이미지 + 프롬프트)을 그대로 파일로 저장한다 (VLM 호출 없음). 그림 입력(작성자 B·C) 확인용.

    /usr/bin/python scripts/dataset_converters/instruction_labeling/label/viz_request.py --method b1 --set frames.overlay=arrow-v1 --set route.motion=none --split val_unseen --subset dev --n 6
출력: logs/instruction_labeling/viz_request/<label 이름>/ep_<id>/{NN_<label>.jpg, prompt.txt}  +  ep_<id>.jpg (이미지 모아 붙인 한 장)
"""
import argparse
import base64
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from annotate import REPO, select_subset  # noqa: E402
from e2e import build_request  # noqa: E402
from sources import episode_track, items  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--method', required=True)
ap.add_argument('--set', action='append', default=[])
ap.add_argument('--split', default='val_unseen')
ap.add_argument('--subset', default='dev')
ap.add_argument('--n', type=int, default=6, help='앞에서 N episode')
ap.add_argument('--out', default='logs/instruction_labeling/viz_request')
a = ap.parse_args()

cfg = config.load(a.method, a.set)
root = REPO / a.out / cfg['name']
for it in select_subset(items(a.split), a.subset)[:a.n]:
    content, inputs, _ = build_request(cfg, it['ep'], episode_track(it, cfg['frames']['rig']))
    d = root / f'ep_{it["ep"]["episode_id"]:06d}'
    d.mkdir(parents=True, exist_ok=True)
    texts, tiles, cap = [], [], ''
    for x in content:
        if x['type'] == 'text':
            texts.append(x['text'])
            cap = x['text']
            continue
        img = cv2.imdecode(np.frombuffer(base64.b64decode(x['image_url']['url'].split(',', 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        (d / f'{len(tiles) + 1:02d}.jpg').write_bytes(base64.b64decode(x['image_url']['url'].split(',', 1)[1]))
        t = cv2.resize(img, (int(img.shape[1] * 360 / img.shape[0]), 360))
        t = cv2.copyMakeBorder(t, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, cap.split(':')[0], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        tiles.append(t)
    (d / 'prompt.txt').write_text('\n'.join(texts))
    W = 4
    rows = []
    for i in range(0, len(tiles), W):
        row = tiles[i:i + W]
        row = [cv2.copyMakeBorder(t, 0, 0, 0, 480 - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) if t.shape[1] < 480 else t[:, :480] for t in row]
        row += [np.full_like(row[0], 255)] * (W - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(root / f'ep_{it["ep"]["episode_id"]:06d}.jpg'), np.vstack(rows))
    print(d, inputs.get('frame_labels'), inputs.get('arrow_deg'))
