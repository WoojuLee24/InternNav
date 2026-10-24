"""H1 보행 컨트롤러 진단용 — `VlnMoveBySpeedController` 를 상속하고, 발밑 높이 입력의 자르기 폭만 인자로 바꾼다.

원본(`internnav/env/utils/internutopia_extension/controllers/h1_vln_move_by_speed_controller.py`)의 forward 는
발밑 96 점 높이 중 `바닥 ± 0.2 m` 밖을 **바닥 높이로 덮어쓴다** → 계단 두 단(≈0.34 m)부터는 평지로 보인다.
여기서는 원본 forward 소스를 그대로 가져와 그 두 줄의 0.2 만 `HEIGHT_CLIP_UP / HEIGHT_CLIP_DN` 으로 바꾼다
(소스 문자열이 바뀌면 assert 로 멈춘다 — 조용히 다른 코드를 돌리지 않게). 원본 파일은 고치지 않는다.

06r `--h1_height_clip <m>` 를 주면 env 설정의 컨트롤러 type 을 'Nav3dMoveBySpeedController' 로 바꿔 이 클래스를 쓴다.
"""

import inspect
import textwrap

from internnav.env.utils.internutopia_extension.controllers import h1_vln_move_by_speed_controller as _orig
from internutopia.core.robot.controller import BaseController

HEIGHT_CLIP_UP = 0.2   # 06r 이 실행 시 덮어쓴다
HEIGHT_CLIP_DN = 0.2

_A = 'heights[heights > floor_height + 0.2] = floor_height'
_B = 'heights[heights < floor_height - 0.2] = floor_height'


def _build_forward():
    src = textwrap.dedent(inspect.getsource(_orig.VlnMoveBySpeedController.forward))
    assert _A in src and _B in src, '원본 controller 의 높이 자르기 줄이 바뀌었다 — h1_tunable_controller 를 다시 맞출 것'
    src = src.replace(_A, 'heights[heights > floor_height + _nav3d_clip_up()] = floor_height')
    src = src.replace(_B, 'heights[heights < floor_height - _nav3d_clip_dn()] = floor_height')
    g = dict(vars(_orig))                         # 원본 모듈의 전역 (torch, np, quat_apply_yaw ...)
    g['_nav3d_clip_up'] = lambda: HEIGHT_CLIP_UP
    g['_nav3d_clip_dn'] = lambda: HEIGHT_CLIP_DN
    ns = {}
    exec(compile(src, '<nav3d h1_tunable forward>', 'exec'), g, ns)
    return ns['forward']


@BaseController.register('Nav3dMoveBySpeedController')
class Nav3dMoveBySpeedController(_orig.VlnMoveBySpeedController):
    forward = _build_forward()
