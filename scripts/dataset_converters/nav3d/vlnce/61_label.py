"""vln_ce 61 · 라벨 — b1 방법으로 instruction 생성 (VLM 서버 common/vlm_server.py 필요).

연결 파일: 코드는 `common/label_paths.py` 에 있다. 이 파일은 거기에 아래 ARGS 를 앞에 붙여 실행만 한다
(뒤에 같은 인자를 주면 한 번만 받는 인자는 뒤 값이 이기고, 여러 번 받는 인자(--vlnce 등)는 더해진다).

사용 (저장소 루트):
    python3 scripts/dataset_converters/nav3d/vlnce/61_label.py --limit 10
"""

import runpy
import sys
from pathlib import Path

TARGET = 'common/label_paths.py'
ARGS = ['--sources', 'nav3d_wp,nav3d_sg', '--method', 'b1']

if __name__ == '__main__':
    sys.argv = [sys.argv[0], *ARGS, *sys.argv[1:]]
    runpy.run_path(str(Path(__file__).resolve().parents[1] / TARGET), run_name='__main__')
