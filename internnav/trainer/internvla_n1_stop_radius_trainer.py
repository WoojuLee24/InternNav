"""반경 STOP 라벨 trainer — S2 가 목표 근처(stop_radius_m 안, 남은 행동이 전진뿐)에서 STOP 을 배우게 한다.

기본 ``internvla_n1_trainer.py`` · ``internvla_n1_lerobot_dataset.py`` 는 고치지 않는다. WAIT trainer 와 같은 방식:
데이터셋 모듈의 ``NavPixelGoalDataset`` 이름을 반경 STOP 데이터셋으로 바꾼 뒤 ``_base.train()``
(``make_supervised_data_module`` 이 호출 시점에 모듈 전역 이름으로 데이터셋을 만든다).
``--stop_radius_m`` 은 ``DataArguments.stop_radius_m`` (기본 0.0) 으로 그대로 파싱된다.

``--wait_mode`` 가 있으면 WAIT 설정을 떼어 두고 WAIT + 반경 STOP 데이터셋을, 없으면 반경 STOP 데이터셋만 쓴다
(``STOP_RADIUS_DATASETS``). default_config.Params.stop_radius_m > 0 이면 runner 가 이 파일을 고른다.

    python internnav/trainer/internvla_n1_stop_radius_trainer.py --stop_radius_m 1.0 [--wait_mode wait --wait_weight 5] ...
"""

import sys

import internnav.dataset.internvla_n1_lerobot_dataset as _data
import internnav.trainer.internvla_n1_trainer as _base
from internnav.dataset.internvla_n1_lerobot_dataset_stop_radius import STOP_RADIUS_DATASETS
from internnav.dataset.internvla_n1_wait_lerobot_dataset import parse_wait_cli_args, set_pending_wait_settings


def main():
    argv = sys.argv[1:]
    has_wait = any(a == "--wait_mode" or a.startswith("--wait_mode=") for a in argv)
    if has_wait:
        settings, argv = parse_wait_cli_args(argv)
        set_pending_wait_settings(settings)
        kind = "wait"
    else:
        settings = {}
        kind = "none"
    sys.argv = [sys.argv[0]] + argv
    _data.NavPixelGoalDataset = STOP_RADIUS_DATASETS[kind]
    print(f"[stop_radius_trainer] dataset={STOP_RADIUS_DATASETS[kind].__name__} wait={settings}", flush=True)
    _base.train()


if __name__ == '__main__':
    main()
