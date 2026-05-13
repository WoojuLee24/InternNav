import math
import os
from typing import Any, Dict, List

import numpy as np
from internutopia.core.robot.articulation import ArticulationAction
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene

from internnav.evaluator.utils.path_plan import world_to_pixel

from ..configs.controllers.flash_controller import VlnMoveByFlashControllerCfg


@BaseController.register('VlnMoveByFlashCollisionController')
class VlnMoveByFlashCollisionController(BaseController):  # codespell:ignore
    """
    Discrete Controller, direct set robot world position to achieve teleport-type locomotion.
    This controller adds collision checking based on depth map from a top-down camera before each flash move.
    If there is an obstacle at the target position, the flash action will be aborted.
    a general controller adaptable to different type of robots.
    """

    def __init__(self, config: VlnMoveByFlashControllerCfg, robot: BaseRobot, scene: IScene) -> None:
        self._user_config = None
        self.current_steps = 0
        self.steps_per_action = config.steps_per_action if config.steps_per_action is not None else 200

        self.forward_distance = config.forward_distance if config.forward_distance is not None else 0.25
        self.rotation_angle = config.rotation_angle if config.rotation_angle is not None else 15.0  # in degrees
        self.physics_frequency = config.physics_frequency if config.physics_frequency is not None else 240

        self.forward_speed = self.forward_distance / self.steps_per_action * self.physics_frequency
        self.rotation_speed = np.deg2rad(
            self.rotation_angle / self.steps_per_action * self.physics_frequency
        )  # 200 is the physics_dt

        self.current_action = None
        self._footprint_radius = config.robot_platform_size  # None = compute lazily from AABB on first use
        self._collision_detected = False

        # BEV visualization via ZMQ (PUSH: controller connects, bridge binds)
        self._zmq_enabled = False
        self._traj_world = []
        self._collision_count = 0
        self._last_bev = None
        self._bev_save_dir = '/tmp/bev_debug'
        os.makedirs(self._bev_save_dir, exist_ok=True)
        try:
            import zmq
            self._zmq_ctx = zmq.Context.instance()
            self._zmq_sock = self._zmq_ctx.socket(zmq.PUSH)
            self._zmq_sock.setsockopt(zmq.SNDHWM, 1)
            self._zmq_sock.connect('tcp://localhost:5577')
            self._zmq_enabled = True
            print('[BEV VIS] ZMQ connected to tcp://localhost:5577')
        except Exception as e:
            print(f'[BEV VIS] ZMQ init failed: {e}')

        super().__init__(config=config, robot=robot, scene=scene)

    def get_new_position_and_rotation(self, robot_position, robot_rotation, action):
        """
        Calculate robot new state by previous state and action. The move should be based on the controller
        settings.
        Caution: the rotation need to reset pitch and roll to prevent robot falling. This may due to no
                    adjustment during the whole path and some rotation accumulated

        Args:
            robot_position (np.ndarray): Current world position of the robot, shape (3,), in [x, y, z] format.
            robot_rotation (np.ndarray): Current world orientation of the robot as a quaternion, shape (4,), in [x, y, z, w] format.
            action (int): Discrete action to apply:
                          - 0: no movement (stand still)
                          - 1: move forward
                          - 2: rotate left
                          - 3: rotate right

        Returns:
            Tuple[np.ndarray, np.ndarray]: The new robot position and rotation as (position, rotation),
                                           both in world frame.
        """
        from omni.isaac.core.utils.rotations import (
            euler_angles_to_quat,
            quat_to_euler_angles,
        )

        _, _, yaw = quat_to_euler_angles(robot_rotation)
        if action == 1:  # forward
            dx = self.forward_distance * math.cos(yaw)
            dy = self.forward_distance * math.sin(yaw)
            new_robot_position = robot_position + [dx, dy, 0]
            new_robot_rotation = robot_rotation
        elif action == 2:  # left
            new_robot_position = robot_position
            new_yaw = yaw + math.radians(self.rotation_angle)
            new_robot_rotation = euler_angles_to_quat(
                np.array([0.0, 0.0, new_yaw])
            )  # using 0 to prevent the robot from falling
        elif action == 3:  # right
            new_robot_position = robot_position
            new_yaw = yaw - math.radians(self.rotation_angle)
            new_robot_rotation = euler_angles_to_quat(np.array([0.0, 0.0, new_yaw]))
        else:
            new_robot_position = robot_position
            new_robot_rotation = robot_rotation

        return new_robot_position, new_robot_rotation

    def reset_robot_state(self, position, orientation):
        """
        Set robot state to the new position and orientation.

        Args:
            position, orientation: np.array, issac_robot.get_world_pose()
        """
        robot = self.robot.articulation
        robot._articulation.set_world_pose(position=position, orientation=orientation)
        robot._articulation.set_world_velocity(np.zeros(6))
        robot._articulation.set_joint_velocities(np.zeros(len(robot.dof_names)))
        robot._articulation.set_joint_positions(np.zeros(len(robot.dof_names)))
        robot._articulation.set_joint_efforts(np.zeros(len(robot.dof_names)))

    def get_map_info(self, topdown_global_map_camera):
        """
        Generate a binary free-space map from a top-down depth camera. Key function for collision checking.

        This function converts depth observations from a top-down global map camera
        into a 2D binary occupancy map, where free space is determined by height
        thresholds relative to the robot base.

        Args:
            topdown_global_map_camera: A top-down depth camera instance providing
                depth observations via `get_data()`.

        Returns:
            np.ndarray:
                A 2D binary map with the same spatial resolution as the input depth
                image. Values are:
                - 1: free space
                - 0: occupied or invalid space
        """

        max_height = 1.55 + 8
        data_info = topdown_global_map_camera.get_data()
        depth = np.array(data_info['depth'])
        flat_surface_mask = np.ones_like(depth, dtype=bool)
        base_height = self.robot.get_robot_base().get_world_pose()[0][2]
        foot_height = self.robot.get_ankle_height()
        min_height = base_height - foot_height + 0.05
        if self.robot.config.type == 'VLNH1Robot':
            depth_mask = ((depth >= min_height) & (depth < max_height)) | ((depth <= 0.5) & (depth > 0.02))
        elif self.robot.config.type == 'VLNAliengoRobot':
            depth_mask = (depth >= min_height) & (depth < max_height)
        free_map = np.zeros_like(depth, dtype=int)
        free_map[flat_surface_mask & depth_mask] = 1  # 1: free, 0: occupied

        import os
        if os.environ.get('DEBUGPY_ENABLE', '0') == '1':
            finite = depth[np.isfinite(depth)]
            camera_z = topdown_global_map_camera.get_world_pose()[0][2]
            robot_base_z = self.robot.get_robot_base().get_world_pose()[0][2]
            foot_height = self.robot.get_ankle_height()
            cx, cy = depth.shape[0] // 2, depth.shape[1] // 2
            print(f'[FREEMAP DEBUG] camera_Z={camera_z:.3f} robot_base_Z={robot_base_z:.3f} foot_height={foot_height:.3f}')
            print(f'[FREEMAP DEBUG] min_height={min_height:.3f} max_height={max_height:.3f} (base-foot={robot_base_z-foot_height:.3f})')
            print(f'[FREEMAP DEBUG] depth finite={len(finite)}/{depth.size} center={depth[cx,cy]:.4f}')
            print(f'[FREEMAP DEBUG] depth pct(10,25,50,75,90)={np.percentile(finite, [10,25,50,75,90]).round(3).tolist()}')
            print(f'[FREEMAP DEBUG] free={int((free_map==1).sum())} occupied={int((free_map==0).sum())}', flush=True)

        return free_map

    def check_collision(self, position, aperture=200) -> bool:
        """
        Check if there are any obstacles at the position.
        Generate a depth map based on a top down camera and check the position

        Return:
            bool: True if the position is already occupied
        """
        if self._footprint_radius is None:
            from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache
            bbox_cache = create_bbox_cache()
            prim_path = self.robot.articulation.prim.GetPath().pathString
            aabb = compute_aabb(bbox_cache, prim_path, include_children=True)
            self._footprint_radius = max(aabb[3] - aabb[0], aabb[4] - aabb[1]) / 2
            print(f'[COLLISION] AABB footprint_radius={self._footprint_radius:.3f}m')

        topdown_global_map_camera = self.robot.sensors['topdown_camera_500']
        free_map = self.get_map_info(topdown_global_map_camera)

        camera_pose = topdown_global_map_camera.get_world_pose()[0]
        width, height = topdown_global_map_camera.resolution
        pixels_per_meter = 10.0 / aperture * width
        robot_size = max(1, round(self._footprint_radius * pixels_per_meter))

        # Step 1: erase current robot footprint (robot body shows as occupied in free_map)
        cur_pos, _ = self.robot.articulation.get_world_pose()
        cur_px, cur_py = world_to_pixel(cur_pos, camera_pose, aperture, width, height)
        cur_px_int, cur_py_int = int(cur_px), int(cur_py)
        free_map[cur_px_int - robot_size : cur_px_int + robot_size,
                 cur_py_int - robot_size : cur_py_int + robot_size] = 1

        # Step 2: check target position with robot footprint
        px, py = world_to_pixel(position, camera_pose, aperture, width, height)
        px_int, py_int = int(px), int(py)
        sub_map = free_map[px_int - robot_size : px_int + robot_size,
                           py_int - robot_size : py_int + robot_size]

        occupied = int(np.sum(sub_map == 0))
        collision = occupied > 0
        if collision:
            print(f'[COLLISION CHECK] footprint={self._footprint_radius:.3f}m ({robot_size}px) occupied={occupied}/{sub_map.size}', flush=True)

        if self._zmq_enabled:
            self._publish_bev(free_map, position, camera_pose, aperture, width, height,
                              cur_px_int, cur_py_int, px_int, py_int, collision, robot_size)

        return collision  # 1 = free, so (any 0) = collision exists

    def _publish_bev(self, free_map, robot_world_pos, camera_pose, aperture, width, height,
                     cur_px, cur_py, robot_px, robot_py, collision=False, robot_size_px=3):
        import cv2
        import zmq

        # Build BGR visualization: free=light gray, occupied=black
        vis = np.zeros((free_map.shape[0], free_map.shape[1], 3), dtype=np.uint8)
        vis[free_map == 1] = [180, 180, 180]

        # Draw trajectory (orange dots)
        self._traj_world.append(robot_world_pos[:3].tolist())
        for wp in self._traj_world[-500:]:
            tpx, tpy = world_to_pixel(wp, camera_pose, aperture, width, height)
            tpx_i, tpy_i = int(tpx), int(tpy)
            if 0 <= tpx_i < free_map.shape[0] and 0 <= tpy_i < free_map.shape[1]:
                cv2.circle(vis, (tpy_i, tpx_i), 2, (0, 140, 255), -1)

        # Current robot position: cyan box (erased footprint area)
        cv2.rectangle(vis,
                      (cur_py - robot_size_px, cur_px - robot_size_px),
                      (cur_py + robot_size_px, cur_px + robot_size_px),
                      (255, 255, 0), 2)  # cyan
        cv2.circle(vis, (cur_py, cur_px), 3, (255, 255, 0), -1)

        # Target position: green (free) or red (collision)
        color = (0, 0, 255) if collision else (0, 255, 0)
        cv2.rectangle(vis,
                      (robot_py - robot_size_px, robot_px - robot_size_px),
                      (robot_py + robot_size_px, robot_px + robot_size_px),
                      color, 2)
        cv2.circle(vis, (robot_py, robot_px), 3, color, -1)

        if collision:
            # Highlight occupied pixels inside the target box in red
            r0 = robot_px - robot_size_px
            r1 = robot_px + robot_size_px
            c0 = robot_py - robot_size_px
            c1 = robot_py + robot_size_px
            region = free_map[r0:r1, c0:c1]
            occupied_mask = (region == 0)
            vis_region = vis[r0:r1, c0:c1]
            vis_region[occupied_mask] = [0, 0, 255]
            vis[r0:r1, c0:c1] = vis_region

        # Collision overlay: red border + text at bottom
        if collision:
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(vis, 'COLLISION', (w // 2 - 120, h - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)

        self._last_bev = vis.copy()
        _, jpeg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if os.environ.get('DEBUGPY_ENABLE') == '1':
            cv2.imwrite(f'{self._bev_save_dir}/latest.jpg', vis)
            if collision:
                self._collision_count += 1
                path = f'{self._bev_save_dir}/collision_{self._collision_count:04d}.jpg'
                cv2.imwrite(path, vis)
                print(f'[BEV VIS] Saved collision frame → {path}')

        try:
            self._zmq_sock.send(jpeg.tobytes(), zmq.NOBLOCK)
        except Exception:
            pass

    def forward(self, action: int) -> ArticulationAction:
        """
        Teleport robot by position, orientation and action

        Args:
            action: int
                    0. discrete action (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            ArticulationAction: joint signals to apply (nothing).
        """
        self._collision_detected = False
        # get robot new position
        positions, orientations = self.robot.articulation.get_world_pose()
        new_robot_position, new_robot_rotation = self.get_new_position_and_rotation(positions, orientations, action)

        # Check if there is a collision with obstacles. Abort the teleport if there is
        if action != 1 or not self.check_collision(new_robot_position):
            # set robot to new state
            self.reset_robot_state(new_robot_position, new_robot_rotation)
        else:
            self._collision_detected = True
            print("[FLASH CONTROLLER]: Collision detected, flash abort")

        # Dummy action to do nothing
        return ArticulationAction()

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Convert input action (in 1d array format) to joint signals to apply.

        Args:
            action (List | np.ndarray): 1-element 1d array containing
              0. discrete action (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            ArticulationAction: joint signals to apply.
        """
        assert len(action) == 1, 'action must contain 1 element'
        return self.forward(action=int(action[0]))

    def get_obs(self) -> Dict[str, Any]:
        return {
            'finished': True,
            'collision_detected': self._collision_detected,
        }
