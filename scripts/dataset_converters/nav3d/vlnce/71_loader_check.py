"""vln_ce 71 · 검사 — 학습 로더로 nav3d wp · sg 읽기 (릴리스 한 씬 대조).

연결 파일: 코드는 `checks/loader_check.py` 에 있다. 이 파일은 거기에 아래 ARGS 를 앞에 붙여 실행만 한다
(뒤에 같은 인자를 주면 한 번만 받는 인자는 뒤 값이 이기고, 여러 번 받는 인자(--vlnce 등)는 더해진다).

사용 (저장소 루트):
    python3 scripts/dataset_converters/nav3d/vlnce/71_loader_check.py
"""

import runpy
import sys
from pathlib import Path

TARGET = 'checks/loader_check.py'
ARGS = [
    '--vlnce',
    '릴리스 vln_ce (대조)=CONTROL_CE=125,0,30',
    '--vlnce',
    'nav3d wp (MP3D)=data/InternData-N1-v0.5-mini/vln_ce_nav3d/traj_data/r2r/nav3d_wp=125,0,30',
    '--vlnce',
    'nav3d sg (MP3D)=data/InternData-N1-v0.5-mini/vln_ce_nav3d/traj_data/r2r/nav3d_sg=125,0,30',
    '--vlnpe',
    '릴리스 vln_pe (대조)=CONTROL_PE',
    '--out',
    'scripts/dataset_converters/nav3d/logs/checks/loader_check/summary_vlnce_nav3d.json',
]

if __name__ == '__main__':
    sys.argv = [sys.argv[0], *ARGS, *sys.argv[1:]]
    runpy.run_path(str(Path(__file__).resolve().parents[1] / TARGET), run_name='__main__')
