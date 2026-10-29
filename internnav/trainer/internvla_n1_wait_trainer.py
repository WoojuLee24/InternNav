"""WAIT 행동 trainer — S2 가 사람 앞에서 `WAIT` 를 내도록 학습 (노은역 hunav, action 4).

기본 ``internvla_n1_trainer.py`` · ``internvla_n1_lerobot_dataset.py`` · ``internvla_n1_argument.py`` 는 고치지 않는다.
BEV/provider trainer 와 같은 방식: ``--wait_*`` 플래그를 argv 에서 떼어 두고, 데이터셋 모듈의
``NavPixelGoalDataset`` 이름을 ``NavPixelGoalWaitDataset`` 로 바꾼 뒤 ``_base.train()``
(``make_supervised_data_module`` 이 호출 시점에 모듈 전역 이름으로 데이터셋을 만든다).

사용 (default_config.Params.wait_mode != 'none' 이면 runner 가 이 파일을 고른다):
    python internnav/trainer/internvla_n1_wait_trainer.py --wait_mode wait --wait_weight 5 --model_name_or_path ... (나머지는 기본 trainer 와 같음)
"""

import sys

import internnav.dataset.internvla_n1_lerobot_dataset as _data
import internnav.trainer.internvla_n1_trainer as _base
from internnav.dataset.internvla_n1_wait_lerobot_dataset import (
    NavPixelGoalWaitDataset,
    parse_wait_cli_args,
    set_pending_wait_settings,
)


def main():
    settings, remaining = parse_wait_cli_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    set_pending_wait_settings(settings)
    _data.NavPixelGoalDataset = NavPixelGoalWaitDataset
    print(f"[wait_trainer] {settings}", flush=True)
    _base.train()


if __name__ == '__main__':
    main()
