from typing import Optional

from internutopia.core.config.robot import ControllerCfg


class VlnMoveByFlashControllerCfg(ControllerCfg):
    type: Optional[str] = 'VlnMoveByFlashController'
    steps_per_action: int = 50
    forward_distance: float = 0.25
    rotation_angle: float = 15.0
    physics_frequency: int = 200
    robot_platform_size: Optional[float] = None  # robot platform radius in meters; None = auto from AABB
