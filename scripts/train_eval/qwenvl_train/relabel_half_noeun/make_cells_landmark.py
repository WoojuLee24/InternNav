#!/usr/bin/env python3
"""relabel_half_noeun_lm-v01 셀 생성기 — 노은역 랜드마크 경로 S2 학습 8 개 (2026-11-15 사용자 결정).

  데이터 : 랜드마크 경로 앞 400 · 1,200 개 (nav3d/noeun/64b_build_train_landmark.py, train_label noeun-lm<N>-<label>)
  라벨   : b1 · b1lm · post · postlm  (lm = 멈출 곳 랜드마크 줄 포함; 모두 구역 v2 · 역 단어 noeun-v1)
  학습   : relabel_half_noeun-v01 과 같음 — 공개 InternVLA-N1-System2 출발, r2r_60cm_15_15, 1 epoch, 평가 없음 (noeun_base.stage1).
           wait_mode 는 wait (랜드마크 데이터에는 사람이 없어 대기 프레임이 없다 -> 기본 로더와 같음).
  EXP_NAME 만 relabel_half_noeun_lm-v01/... 로 (예전 노은역 셀과 폴더를 나눔).

    python3 scripts/train_eval/qwenvl_train/relabel_half_noeun/make_cells_landmark.py
이미 있는 셀은 내용이 같을 때만 통과.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION_LM = "relabel_half_noeun_lm-v01"
TEMPLATE = '''"""노은역 랜드마크 S2 학습 ({version}) — 데이터 '{label}'. 학습만 (노은역 평가는 다른 노드).

make_cells_landmark.py 로 생성됨. 학습 설정은 noeun_base.stage1 과 같고 EXP_NAME 폴더만 {version}.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/{fname} --machine h200 --no-eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # noeun_base

import default_config as base  # noqa: E402
import noeun_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = noeun_base.stage1("{label}", "wait")
EXP_NAME = EXP_NAME.replace(noeun_base.VERSION + "/", "{version}/", 1)
eval_cfg = base.make_eval_cfg(PARAMS)
'''

CELLS = [f"noeun-lm{n}-{lab}" for n in (400, 1200) for lab in ("b1", "b1lm", "post", "postlm")]


def main():
    for label in CELLS:
        fname = f"s1.{label}.py"
        text = TEMPLATE.format(label=label, fname=fname, version=VERSION_LM)
        path = os.path.join(HERE, fname)
        if os.path.exists(path) and open(path).read() != text:
            sys.exit(f"{fname} exists with different content")
        open(path, "w").write(text)
        print(fname)


if __name__ == "__main__":
    main()
