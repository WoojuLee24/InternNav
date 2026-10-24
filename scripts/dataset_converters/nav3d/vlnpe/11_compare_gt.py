"""vln_pe 11 · 준비 — 릴리스 vln_ce GT 비교 (51 카메라 방식이 층 이동 표시를 읽는다).

연결 파일: 코드는 `common/compare_vlnce_gt.py` 에 있다. 이 파일은 거기에 아래 ARGS 를 앞에 붙여 실행만 한다
(뒤에 같은 인자를 주면 한 번만 받는 인자는 뒤 값이 이기고, 여러 번 받는 인자(--vlnce 등)는 더해진다).

사용 (저장소 루트):
    python3 scripts/dataset_converters/nav3d/vlnpe/11_compare_gt.py --scene 17DRP5sb8fy --recompute  (씬마다) · --aggregate
"""

import runpy
import sys
from pathlib import Path

TARGET = 'common/compare_vlnce_gt.py'
ARGS = []

if __name__ == '__main__':
    sys.argv = [sys.argv[0], *ARGS, *sys.argv[1:]]
    runpy.run_path(str(Path(__file__).resolve().parents[1] / TARGET), run_name='__main__')
