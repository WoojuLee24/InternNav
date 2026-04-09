import argparse
import copy
import io
import json
import math
import threading
import time
from collections import deque
from enum import Enum

import numpy as np
import rclpy
import requests
from geometry_msgs.msg import Point, Pose, PoseStamped, PointStamped, Twist, TransformStamped
from tf2_ros import StaticTransformBroadcaster
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from visualization_msgs.msg import Marker
from PIL import Image as PIL_Image
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

frame_data = {}
frame_idx = 0
# user-specific
from controllers import Mpc_controller, PID_controller
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock
from transformation import Calibration


class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


# global variable
policy_init = True
mpc = None
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
http_idx = -1
first_running_time = 0.0
last_pixel_goal = None
last_s2_step = -1
manager = None
CLIENT_MODE = "sync"
CLIENT_OPT_FLAGS = {
    "kv_cache": False,
    "tensorrt": False,
    "quantization": False,
    "quant_method": "dynamic",
    "tensorrt_engine": "",
    "tf32": False,
    "vision_cache": False,
    "methods": [],
}
JPEG_QUALITY = 95
DEPTH_PNG_COMPRESS = 6
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None
calib = None

desired_v, desired_w = 0.0, 0.0
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()


def dual_sys_eval(image_bytes, depth_bytes, front_image_bytes, url='http://127.0.0.1:5802/eval_dual'):
    global policy_init, http_idx, first_running_time
    
    # [LOG] HTTP 요청 준비
    if manager:
        manager.get_logger().debug(f"[HTTP] Preparing request. idx: {http_idx + 1}, reset: {policy_init}")

    data = {
        "reset": policy_init,
        "idx": http_idx,
        "mode": CLIENT_MODE,
        "optimizations": CLIENT_OPT_FLAGS,
    }
    json_data = json.dumps(data)

    policy_init = False
    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
        'depth': ('depth_image', depth_bytes, 'image/png'),
    }
    start = time.time()
    
    try:
        response = requests.post(url, files=files, data={'json': json_data}, timeout=100)
    except Exception as e:
        if manager:
            manager.get_logger().error(f"[HTTP] Request Failed: {e}")
        return {}

    if response.status_code != 200:
        if manager:
            manager.get_logger().error(
                f"[HTTP] Bad status {response.status_code}. Body: {response.text[:200]}"
            )
        return {"status": "waiting"}

    # [LOG] 응답 확인 (print 대체)
    if manager:
        # 응답 내용은 너무 길 수 있으니 debug 레벨로, 상태 코드는 info로
        manager.get_logger().debug(f"[HTTP] Response text: {response.text}")
        
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    
    # [LOG] Latency 확인 (print 대체)
    latency = time.time() - start
    if manager:
        manager.get_logger().info(f"[HTTP] idx: {http_idx} | Latency: {latency:.4f}s")

    try:
        return response.json()
    except ValueError:
        if manager:
            manager.get_logger().error(f"[HTTP] Invalid JSON response: {response.text[:200]}")
        return {"status": "waiting"}


def control_thread():
    global desired_v, desired_w, manager
    
    # Manager 초기화 대기
    while manager is None:
        time.sleep(0.1)
    
    manager.get_logger().info("[Thread] Control Thread Started successfully.")

    while True:
        global current_control_mode
        if current_control_mode == ControlMode.MPC_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            
            if mpc is not None and manager is not None and odom is not None:
                local_mpc = mpc
                try:
                    opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                    v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]

                    desired_v, desired_w = v, w
                    manager.move(v, 0.0, w)
                    
                    # [LOG] MPC 제어 출력 (Throttle 1.0s: 로그 폭주 방지)
                    manager.get_logger().info(
                        f"[MPC] v: {v:.3f}, w: {w:.3f} | Odom: ({odom[0]:.2f}, {odom[1]:.2f})",
                        throttle_duration_sec=1.0
                    )
                except Exception as e:
                    manager.get_logger().error(f"[MPC] Solve Error: {e}")

        elif current_control_mode == ControlMode.PID_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0:
                    v = 0.0
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)
                
                # [LOG] PID 제어 출력 (Throttle 1.0s)
                manager.get_logger().info(
                    f"[PID] v: {v:.3f}, w: {w:.3f} | err_p: {e_p:.3f}, err_r: {e_r:.3f}",
                    throttle_duration_sec=1.0
                )

        time.sleep(0.1)


def planning_thread(frame_id="os_sensor", traj_width=0.05, subgoal_radius=0.2):
    global trajs_in_world, manager
    
    # Manager 초기화 대기
    while manager is None:
        time.sleep(0.1)
    
    manager.get_logger().info("[Thread] Planning Thread Started successfully.")

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.3
        time.sleep(0.05)

        if not manager.new_image_arrived:
            # [LOG] 이미지 대기 중 (너무 자주 찍히지 않게 5초에 한번)
            manager.get_logger().warn("[Plan] Waiting for new image...", throttle_duration_sec=5.0)
            time.sleep(0.01)
            continue
            
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        infer_rgb = copy.deepcopy(manager.rgb_image)
        infer_depth = copy.deepcopy(manager.depth_image)
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()
        
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        # time_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
                # time_diff = odom[0] - rgb_time
        # odom_time = manager.odom_timestamp
        odom_rw_lock.release_read()

        if odom_infer is not None and rgb_bytes is not None and depth_bytes is not None:
            global frame_data
            frame_data[http_idx] = {
                'infer_rgb': copy.deepcopy(infer_rgb),
                'infer_depth': copy.deepcopy(infer_depth),
                'infer_odom': copy.deepcopy(odom_infer),
            }
            if len(frame_data) > 100:
                del frame_data[min(frame_data.keys())]
            
            # [LOG] 추론 요청 직전
            manager.get_logger().debug(f"[Plan] Calling dual_sys_eval... Sync Diff: {min_diff:.5f}s")
            
            response = dual_sys_eval(rgb_bytes, depth_bytes, None)

            global current_control_mode
            traj_len = 0.0
            if 'trajectory' in response:
                trajectory = response['trajectory']
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
                
                # [LOG] Trajectory 수신 (print 대체)
                manager.get_logger().info(f"[Plan] Received Trajectory. Length: {traj_len:.4f}")
                
                for i, traj in enumerate(trajectory):
                    if i < 3:
                        continue
                    x_, y_, yaw_ = odom[0], odom[1], odom[2]

                    w_T_b = np.array(
                        [
                            [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                            [np.sin(yaw_), np.cos(yaw_), 0, y_],
                            [0.0, 0.0, 1.0, 0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    )
                    w_P = (w_T_b @ (np.array([traj[0], traj[1], 0.0, 1.0])).T)[:2]
                    trajs_in_world.append(w_P)
                trajs_in_world = np.array(trajs_in_world)
                
                # [LOG] Trajectory 업데이트 (print 대체)
                manager.get_logger().info(f"[Plan] Updated Trajs in World. Time: {time.time()}")

                manager.last_trajs_in_world = trajs_in_world

                # ── Trajectory 즉시 퍼블리시 (real-time, planning_thread) ──
                _stamp = manager.get_clock().now().to_msg()
                _traj_marker = Marker()
                _traj_marker.header = Header(stamp=_stamp, frame_id=frame_id)
                _traj_marker.ns = "trajectory"
                _traj_marker.id = 0
                _traj_marker.type = Marker.LINE_STRIP
                _traj_marker.action = Marker.ADD
                _traj_marker.scale.x = traj_width
                _traj_marker.color.r = 0.0
                _traj_marker.color.g = 1.0
                _traj_marker.color.b = 0.0
                _traj_marker.color.a = 1.0
                if frame_id == "os_sensor":
                    # body frame: raw model output (i >= 3 skip)
                    _traj_pts = [[float(t[0]), float(t[1])] for i, t in enumerate(trajectory) if i >= 3]
                else:
                    # world frame: w_T_b 변환 적용된 trajs_in_world
                    _traj_pts = [[float(pt[0]), float(pt[1])] for pt in trajs_in_world]
                for _x, _y in _traj_pts:
                    _p = Point()
                    _p.x = _x
                    _p.y = _y
                    _p.z = 0.0
                    _traj_marker.points.append(_p)
                manager.traj_pub.publish(_traj_marker)
                manager.get_logger().info(
                    f"[Plan][Traj] Published {len(_traj_pts)} pts in {frame_id}",
                    throttle_duration_sec=1.0,
                )

                if 'pixel_goal' in response:
                    manager.last_pixel_goal = response['pixel_goal']  # [row, col]
                    # ── Subgoal 즉시 퍼블리시 (real-time, planning_thread) ──
                    if calib is not None and infer_depth is not None:
                        _row, _col = int(response['pixel_goal'][0]), int(response['pixel_goal'][1])
                        _dh, _dw = infer_depth.shape
                        if 0 <= _row < _dh and 0 <= _col < _dw:
                            _z_sub = infer_depth[_row, _col]
                            if _z_sub > 0.1:
                                _fx, _fy = calib.f_u, calib.f_v
                                _cx, _cy = calib.c_u, calib.c_v
                                _x_cam = (_col - _cx) * _z_sub / _fx
                                _y_cam = (_row - _cy) * _z_sub / _fy
                                _pc_robot = calib.project_rect_to_velo(
                                    np.array([[_x_cam, _y_cam, _z_sub]]))[0]
                                if frame_id == "os_sensor":
                                    _sub_x, _sub_y = float(_pc_robot[0]), float(_pc_robot[1])
                                else:
                                    _ox, _oy, _oyaw = odom_infer
                                    _R2 = np.array([[np.cos(_oyaw), -np.sin(_oyaw)],
                                                    [np.sin(_oyaw),  np.cos(_oyaw)]])
                                    _xy_sub = _R2 @ _pc_robot[:2] + np.array([_ox, _oy])
                                    _sub_x, _sub_y = float(_xy_sub[0]), float(_xy_sub[1])
                                _sub_msg = Marker()
                                _sub_msg.header = Header(stamp=_stamp, frame_id=frame_id)
                                _sub_msg.ns = "subgoal"
                                _sub_msg.id = 0
                                _sub_msg.type = Marker.SPHERE
                                _sub_msg.action = Marker.ADD
                                _sub_msg.pose.position.x = _sub_x
                                _sub_msg.pose.position.y = _sub_y
                                _sub_msg.pose.position.z = float(_pc_robot[2])
                                _sub_msg.pose.orientation.w = 1.0
                                _sub_msg.scale.x = subgoal_radius
                                _sub_msg.scale.y = subgoal_radius
                                _sub_msg.scale.z = subgoal_radius
                                _sub_msg.color.r = 1.0
                                _sub_msg.color.g = 0.5
                                _sub_msg.color.b = 0.0
                                _sub_msg.color.a = 1.0
                                manager.subgoal_pub.publish(_sub_msg)

                mpc_rw_lock.acquire_write()
                global mpc
                if mpc is None:
                    mpc = Mpc_controller(np.array(trajs_in_world))
                    manager.get_logger().info("[Plan] MPC Controller Initialized")
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
                manager.request_cnt += 1
                mpc_rw_lock.release_write()
                current_control_mode = ControlMode.MPC_Mode
                
            else:
                # trajectory가 새로 생성되지 않으면 초기화
                manager.last_trajs_in_world = None
                manager.last_pixel_goal = None
                _stamp = manager.get_clock().now().to_msg()
                _del_traj = Marker()
                _del_traj.header = Header(stamp=_stamp, frame_id=frame_id)
                _del_traj.ns = "trajectory"
                _del_traj.id = 0
                _del_traj.action = Marker.DELETE
                manager.traj_pub.publish(_del_traj)
                _del_sub = Marker()
                _del_sub.header = Header(stamp=_stamp, frame_id=frame_id)
                _del_sub.ns = "subgoal"
                _del_sub.id = 0
                _del_sub.action = Marker.DELETE
                manager.subgoal_pub.publish(_del_sub)
                manager.get_logger().info("[Plan] No trajectory in response. Cleared.", throttle_duration_sec=2.0)

            if 'discrete_action' in response:
                actions = response['discrete_action']
                # [LOG] Discrete Action 수신
                manager.get_logger().info(f"[Plan] Received Discrete Actions: {actions}")

                if actions != [5] and actions != [9]:
                    manager.incremental_change_goal(actions)
                    current_control_mode = ControlMode.PID_Mode
        else:
            # [LOG] 데이터 부족으로 스킵 (print 대체)
            manager.get_logger().warn(
                f"[Plan] Skip. Odom: {odom_infer is not None}, RGB: {rgb_bytes is not None}, Depth: {depth_bytes is not None}",
                throttle_duration_sec=2.0
            )
            time.sleep(0.1)

        time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))


def build_occupancy_grid(pcloud_xy, stamp, frame_id, resolution=0.1, grid_size=100, center_x=0.0, center_y=0.0):
    """Build a ROS2 OccupancyGrid message from (N, 2) XY points.

    All points are treated as occupied (100). Unknown cells remain -1.
    Origin is set to the grid's bottom-left corner.
    """
    half = (grid_size * resolution) / 2.0
    xmin = center_x - half
    ymin = center_y - half

    grid = -np.ones((grid_size, grid_size), dtype=np.int16)

    if len(pcloud_xy) > 0:
        pts = np.asarray(pcloud_xy, dtype=np.float64)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) > 0:
            cx = np.floor((pts[:, 0] - xmin) / resolution).astype(np.int32)
            cy = np.floor((pts[:, 1] - ymin) / resolution).astype(np.int32)
            valid = (cx >= 0) & (cx < grid_size) & (cy >= 0) & (cy < grid_size)
            grid[cy[valid], cx[valid]] = 100

    msg = OccupancyGrid()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.info.resolution = float(resolution)
    msg.info.width = int(grid_size)
    msg.info.height = int(grid_size)
    msg.info.origin = Pose()
    msg.info.origin.position.x = float(xmin)
    msg.info.origin.position.y = float(ymin)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.reshape(-1).astype(np.int8).tolist()
    return msg


def visualize_thread(frame_id="os_sensor"):
    global manager, calib

    while manager is None or calib is None:
        time.sleep(0.1)

    manager.get_logger().info(f"[Thread] Visualize Thread Started successfully. OccGrid frame: {frame_id}")

    u_grid = None
    v_grid = None

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.1  # 10 Hz

        if not manager.new_vis_image_arrived:
            time.sleep(0.01)
            continue
        manager.new_vis_image_arrived = False

        # ── depth 복사 ────────────────────────────────────────
        rgb_depth_rw_lock.acquire_read()
        depth_image = copy.deepcopy(manager.depth_image)
        rgb_depth_rw_lock.release_read()

        if depth_image is None:
            time.sleep(0.05)
            continue

        # ── odom 확인 (frame 선택) ────────────────────────────
        odom_rw_lock.acquire_read()
        odom = manager.odom.copy() if manager.odom else None
        odom_rw_lock.release_read()

        # ── Depth → 3D (camera rect frame, meters) ───────────
        h, w = depth_image.shape
        if u_grid is None or u_grid.size != h * w:
            vg, ug = np.mgrid[0:h, 0:w]
            u_grid = ug.flatten()
            v_grid = vg.flatten()

        z = depth_image.flatten()
        valid = (z > 0.1) & (z < 10.0)
        z_v = z[valid]
        u_v = u_grid[valid]
        v_v = v_grid[valid]

        fx, fy = calib.f_u, calib.f_v
        cx_cam, cy_cam = calib.c_u, calib.c_v
        x_cam = (u_v - cx_cam) * z_v / fx
        y_cam = (v_v - cy_cam) * z_v / fy
        pcloud_cam = np.stack([x_cam, y_cam, z_v], axis=-1)  # (N, 3) camera rect coords

        # ── Camera rect → robot/velodyne frame ───────────────
        # project_rect_to_velo: camera(right,down,fwd) → robot(fwd,left,up)
        pcloud_robot = calib.project_rect_to_velo(pcloud_cam)  # (N, 3)

        # ── 높이 필터 (로봇 frame Z: 위쪽이 +) ───────────────
        z_r = pcloud_robot[:, 2]
        height_mask = (z_r > 0.05) & (z_r < 2.0)
        pcloud_filtered = pcloud_robot[height_mask]

        if len(pcloud_filtered) == 0:
            manager.get_logger().debug("[Vis] No valid obstacle points after height filter.", throttle_duration_sec=3.0)
            time.sleep(max(0.0, DESIRED_TIME - (time.time() - start_time)))
            continue

        # ── 좌표계 및 center 결정 ─────────────────────────────
        if frame_id == "camera_init" and odom is not None:
            x_, y_, yaw_ = odom
            cos_y, sin_y = np.cos(yaw_), np.sin(yaw_)
            R2 = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            xy_world = (R2 @ pcloud_filtered[:, :2].T).T + np.array([x_, y_])
            center_x, center_y = x_, y_
        else:
            xy_world = pcloud_filtered[:, :2]
            center_x, center_y = 0.0, 0.0

        stamp = manager.get_clock().now().to_msg()

        # ── OccupancyGrid 빌드 및 발행 ────────────────────────
        occ_msg = build_occupancy_grid(
            xy_world, stamp, frame_id,
            resolution=0.1, grid_size=100,
            center_x=center_x, center_y=center_y,
        )
        manager.occ_grid_pub.publish(occ_msg)

        manager.get_logger().debug(
            f"[Vis] OccGrid published. Frame: {frame_id} | Points: {len(xy_world)}",
            throttle_duration_sec=2.0,
        )

        # # ── Dummy Path (os_sensor frame, /internav/dummy_path) ──
        # dummy_path = Path()
        # dummy_path.header = Header(stamp=stamp, frame_id="os_sensor")
        # for i in range(10):
        #     ps = PoseStamped()
        #     ps.header = Header(stamp=stamp, frame_id="os_sensor")
        #     ps.pose.position.x = float(i) * 0.2
        #     ps.pose.position.y = 0.0
        #     ps.pose.position.z = 0.0
        #     ps.pose.orientation.w = 1.0
        #     dummy_path.poses.append(ps)
        # manager.dummy_path_pub.publish(dummy_path)
        # manager.get_logger().info(
        #     f"[Vis][DummyPath] frame={dummy_path.header.frame_id} "
        #     f"pts={len(dummy_path.poses)} "
        #     f"x=[{dummy_path.poses[0].pose.position.x:.2f} ~ {dummy_path.poses[-1].pose.position.x:.2f}] "
        #     f"y=[{dummy_path.poses[0].pose.position.y:.2f} ~ {dummy_path.poses[-1].pose.position.y:.2f}]",
        #     throttle_duration_sec=2.0,
        # )

        time.sleep(max(0.0, DESIRED_TIME - (time.time() - start_time)))


class Go2Manager(Node):
    def __init__(self, odom_topic='/odom_bridge'):
        super().__init__('go2_manager')
        
        # [LOG] 노드 시작 알림
        self.get_logger().info("Initializing Go2Manager Node...")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        rgb_down_sub = Subscriber(self, Image, "/camera/camera/color/image_raw")
        depth_down_sub = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw")

        self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 30, 0.5)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, qos_profile)
        self.get_logger().info(f"Subscribing to odometry topic: {odom_topic}")

        # static transform: map → camera_init (identity)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        _tf = TransformStamped()
        _tf.header.stamp = self.get_clock().now().to_msg()
        _tf.header.frame_id = 'map'
        _tf.child_frame_id = 'camera_init'
        _tf.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(_tf)
        self.get_logger().info("[TF] Published static transform: map -> camera_init (identity)")

        # publisher
        self.control_pub = self.create_publisher(Twist, '/cmd_vel_bridge', 5)
        self.occ_grid_pub = self.create_publisher(OccupancyGrid, '/internav/occupancy_grid', 5)
        self.traj_pub = self.create_publisher(Marker, '/internav/trajectory', 5)
        self.dummy_path_pub = self.create_publisher(Path, '/internav/dummy_path', 5)
        self.subgoal_pub = self.create_publisher(Marker, '/internav/subgoal', 5)

        # class member variable
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.rgb_forward_image = None
        self.rgb_forward_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.last_pixel_goal = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None
        
        # [LOG] 초기화 완료
        self.get_logger().info("Go2Manager Node Initialized Successfully.")

    def rgb_forward_callback(self, rgb_msg):
        # [LOG] 전방 카메라 수신 (디버깅용, 5초마다)
        self.get_logger().debug("Received Forward Image", throttle_duration_sec=5.0)
        
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_forward_image = raw_image
        image = PIL_Image.fromarray(self.rgb_forward_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG', quality=JPEG_QUALITY, optimize=False)
        image_bytes.seek(0)
        self.rgb_forward_bytes = image_bytes
        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        # [LOG] RGB-Depth 수신 확인 (2초마다)
        self.get_logger().debug("Received Synced RGB-Depth", throttle_duration_sec=2.0)
        
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_image = raw_image
        image = PIL_Image.fromarray(self.rgb_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG', quality=JPEG_QUALITY, optimize=False)
        image_bytes.seek(0)

        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        self.depth_image = raw_depth / 1000.0
        self.depth_image -= 0.0
        self.depth_image[np.where(self.depth_image < 0)] = 0
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth = PIL_Image.fromarray(depth)
        depth_bytes = io.BytesIO()
        depth.save(depth_bytes, format='PNG', compress_level=DEPTH_PNG_COMPRESS)
        depth_bytes.seek(0)

        rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = image_bytes

        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        self.last_rgb_time = self.rgb_time

        self.depth_bytes = depth_bytes
        self.depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1.0e9
        self.last_depth_time = self.depth_time

        rgb_depth_rw_lock.release_write()

        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def odom_callback(self, msg):
        self.odom_cnt += 1
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1.0e9
        self.odom_queue.append((odom_stamp, copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        odom_rw_lock.release_write()

        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()
            # [LOG] 첫 Odom 수신
            self.get_logger().info(f"[Odom] First Odom Received. Pose: ({self.odom[0]:.2f}, {self.odom[1]:.2f})")
            
        # [LOG] Odom 데이터 모니터링 (3초마다)
        self.get_logger().debug(
            f"[Odom] x:{self.odom[0]:.2f}, y:{self.odom[1]:.2f}, yaw:{yaw:.2f}", 
            throttle_duration_sec=3.0
        )

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            self.get_logger().error("homo_goal is None! Cannot change goal.")
            raise ValueError("Please initialize homo_goal before change it!")
        
        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal
        # [LOG] Goal 변경 확인
        self.get_logger().info(f"[Goal] Goal Updated incrementally. Action: {actions}")

    def move(self, vx, vy, vyaw):
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw

        self.control_pub.publish(request)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sync', choices=['sync', 'async'],
                        help='Execution mode. async is scaffold-only for now.')
    parser.add_argument('--kv-cache', action='store_true', help='Enable KV-cache optimization (scaffold flag).')
    parser.add_argument('--tensorrt', action='store_true', help='Enable TensorRT optimization (scaffold flag).')
    parser.add_argument('--quantization', action='store_true', help='Enable quantization optimization (scaffold flag).')
    parser.add_argument('--quant-method', type=str, default='dynamic', choices=['dynamic', 'static', 'qat'],
                        help='Quantization method tag for experiments.')
    parser.add_argument('--tensorrt-engine', type=str, default='',
                        help='TensorRT engine path tag for experiments.')
    parser.add_argument('--tf32', action='store_true', help='Enable TF32 mode tag for experiments.')
    parser.add_argument('--vision-cache', action='store_true', help='Enable vision-cache optimization (scaffold flag).')
    parser.add_argument('--jpeg-quality', type=int, default=95,
                        help='JPEG quality for rgb payload (1-100).')
    parser.add_argument('--depth-png-compress', type=int, default=6,
                        help='PNG compress level for depth payload (0-9).')
    parser.add_argument('--method', action='append', default=[],
                        help='Additional optimization method tag (repeatable, scaffold only).')
    parser.add_argument('--odom_topic', type=str, default='/gdq/msg/gdq_odom', help='ROS2 odometry topic name')
    parser.add_argument('--calib', type=str, default='/home/gdr/gd_vln/workspace/src/InternNav/scripts/realworld/calib/calib_scout.txt',
                        help='Path to calibration file (e.g. calib/calib_r64.txt)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Enable visualize_thread (OccupancyGrid)')
    parser.add_argument('--frame_id', type=str, default='os_sensor',
                        help='Frame ID for OccupancyGrid (e.g. os_sensor, camera_init)')
    parser.add_argument('--traj_width', type=float, default=0.1,
                        help='Trajectory marker line width in meters (default: 0.1)')
    parser.add_argument('--subgoal_radius', type=float, default=0.3,
                        help='Subgoal marker sphere radius in meters (default: 0.3)')
    args = parser.parse_args()

    if args.mode != 'sync':
        print(f"[Client] mode={args.mode} requested, but async path is not implemented yet. Falling back to sync.")
    CLIENT_MODE = 'sync'
    JPEG_QUALITY = max(1, min(100, int(args.jpeg_quality)))
    DEPTH_PNG_COMPRESS = max(0, min(9, int(args.depth_png_compress)))
    CLIENT_OPT_FLAGS = {
        "kv_cache": bool(args.kv_cache),
        "tensorrt": bool(args.tensorrt),
        "quantization": bool(args.quantization),
        "quant_method": str(args.quant_method),
        "tensorrt_engine": str(args.tensorrt_engine),
        "tf32": bool(args.tf32),
        "vision_cache": bool(args.vision_cache),
        "methods": list(args.method),
    }
    if any([args.kv_cache, args.tensorrt, args.quantization, args.vision_cache, len(args.method) > 0]):
        print(f"[Client] Optimization flags enabled (scaffold only): {CLIENT_OPT_FLAGS}")

    calib = Calibration(args.calib)

    dummy_odom = [0.0, 0.0, 0.0]
    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread, args=(args.frame_id, args.traj_width, args.subgoal_radius))
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True
    if args.visualize:
        visualize_thread_instance = threading.Thread(target=visualize_thread, args=(args.frame_id,))
        visualize_thread_instance.daemon = True
    rclpy.init()

    try:
        manager = Go2Manager(odom_topic=args.odom_topic)

        control_thread_instance.start()
        planning_thread_instance.start()
        if args.visualize:
            visualize_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        if manager:
            manager.get_logger().info("[Main] Keyboard Interrupt. Shutting down...")
    finally:
        if manager:
            manager.destroy_node()
        rclpy.shutdown()
