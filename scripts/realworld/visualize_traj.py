import queue
import threading
import time
import sys
import numpy as np
import os
import cv2
import ros2_numpy as rnp

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile,  QoSHistoryPolicy, QoSDurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Header, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from geometry_msgs.msg import Pose

from mmseg.apis import MMSegInferencer
from gdm_segmentation.transformation import transform_from_gps_and_imu, read_calibration, Calibration, rotz, roty, get_transform_between_poses

from mmengine.logging.history_buffer import HistoryBuffer
import torch
# Explicitly allow HistoryBuffer to be unpickled
torch.serialization.add_safe_globals([HistoryBuffer])


class GDMSegNode(Node):
    def __init__(self):
        super().__init__("gdm_seg_node")
        self.get_logger().info("gdm_seg_node started")

        # import debugpy
        # debugpy.listen(5678)
        # print("⏸ Waiting for debugger attach...")
        # debugpy.wait_for_client()

        user_qos_profile = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST, 
                                      reliability=ReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE)

        # -------------------------------
        # Declare parameters
        # -------------------------------
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw") # "/gdm/front_color_image" 
        self.declare_parameter("lidar_pointcloud_topic", "/ouster/points")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("odometry_topic", "/gdq/msg/gdq_odom") # /gdq/msg/gdq_odom' /gd/zed_odometry

        # calibration_filepath = self.declare_parameter("calibration_filepath", "/home/gdm/workspace/src/gdm_object_tracking/gdm_image_detection/gdm_image_detection/calib_scout.txt").value
        # calibration_filepath = self.declare_parameter("calibration_filepath", "/home/gdm/workspace/src/gdm_object_tracking/gdm_image_detection/gdm_image_detection/calib_r64.txt").value
        calibration_filepath = self.declare_parameter("calibration_filepath", "/home/gdr/workspace/src/gdm_object_tracking/gdm_image_detection/gdm_image_detection/calib_r64.txt").value

        self.voxel_leaf_size = self.declare_parameter("voxel_leaf_size", 0.00001).value
        self.debug = self.declare_parameter("debug", 'none').value 
        self.grid_size = self.declare_parameter("grid_size", 160).value 
        self.resolution = self.declare_parameter("resolution", 0.1).value
        self.mode = self.declare_parameter("mode", "trt").value # 'trt', 'torch'
        self.sensor = self.declare_parameter("sensor", "depth").value # 'lidar'
        self.grid_seg = self.declare_parameter("grid_seg", "grid_map").value # 'occ'
        self.weight = self.declare_parameter("weight", "none").value  

        # -------------------------------
        # Load models
        # -------------------------------
        if self.mode == 'trt':
            from mmdeploy.apis import build_task_processor
            from mmdeploy.utils import get_input_shape, load_config
            # model_cfg = "/home/gdm/workspace/src/gdm_object_tracking/mmdeploy/work_dir/mmseg_zoo/segformer_mit-b0_8xb1-160k_sideguide-1024x1024_from.city.py"
            # deploy_cfg = "/home/gdm/workspace/src/gdm_object_tracking/mmdeploy/configs/mmseg/segmentation_tensorrt_static-1024x1024.py"
            # backend_model = [f"/home/gdm/workspace/src/gdm_object_tracking/mmdeploy/work_dir/mmseg_zoo/segformer_mit-b0_8xb1-160k_sideguide-1024x1024_from.city_trt/end2end.engine"]

            model_cfg = "/home/gdr/workspace/src/gdm_object_tracking/mmdeploy/work_dir/mmseg_zoo/segformer_mit-b0_8xb1-160k_sideguide-1024x1024_from.city.py"
            deploy_cfg = "/home/gdr/workspace/src/gdm_object_tracking/mmdeploy/configs/mmseg/segmentation_tensorrt_static-1024x1024.py"
            backend_model = [f"/home/gdr/workspace/src/gdm_object_tracking/mmdeploy/work_dir/mmseg_zoo/segformer_mit-b0_8xb1-160k_sideguide-1024x1024_from.city_trt/end2end.engine"]

            deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

            self.task_processor = build_task_processor(
                                model_cfg=model_cfg,
                                deploy_cfg=deploy_cfg,
                                device="cuda"
                            )
            
            self.model = self.task_processor.build_backend_model(
                model_files=backend_model,
                device="cuda"
            )
            self.input_shape = get_input_shape(deploy_cfg)

        elif self.mode == 'torch':
            import torch
            from functools import partial
            torch.load = partial(torch.load, weights_only=False)
            self.get_logger().info(f'weight: {self.weight}' )
            model_path = self.weight[:-3] + 'py'
            self.model = MMSegInferencer(model=model_path,
                                        weights=self.weight,
                                        device='cuda')
        
        use_sim_time = self.get_parameter("use_sim_time").value
        image_topic = self.get_parameter("image_topic").value
        lidar_topic = self.get_parameter("lidar_pointcloud_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        odom_topic = self.get_parameter("odometry_topic").value

        # -------------------------------
        # 🕒 Clock 모드 표시
        # -------------------------------
        if use_sim_time:
            self.get_logger().info("🕒 [Clock Mode] use_sim_time=True → ROS /clock 기반 시간 사용")
        else:
            self.get_logger().info("🕒 [Clock Mode] use_sim_time=False → SystemClock (실제 시간) 사용")

        # -------------------------------
        # Topic 구독
        # -------------------------------
        self.get_logger().info("📡 Subscribing to:")
        self.get_logger().info(f"  Image       : {image_topic}")
        self.get_logger().info(f"  PointCloud2 : {lidar_topic}")
        self.get_logger().info(f"  Odometry    : {odom_topic}")

        # 최신 데이터 저장용 변수
        self.processing_queue = queue.Queue() # maxsize=2
        self.image_msg = None
        self.pcloud_msg = None
        self.odometry_msg = None

        # Mutex (thread-safe 보장)
        self.data_lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()  # ✅ 실제 OS-level thread 생성!

        # Subscriber 등록
        self.create_subscription(Image, image_topic, self.image_callback, user_qos_profile)
        self.create_subscription(PointCloud2, lidar_topic, self.lidar_callback, user_qos_profile)
        self.create_subscription(Image, depth_topic, self.depth_callback, user_qos_profile)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, user_qos_profile)

        # Publishter 등록
        self.pub_viz_seg = self.create_publisher(Image, '/gdm/viz_semantic_segmentation', qos_profile=user_qos_profile)
        self.pub_viz_projected = self.create_publisher(Image, '/gdm/viz_projected', qos_profile=user_qos_profile)
        
        # grid map publisher 선언
        if self.grid_seg == 'grid_map':
            self.grid_seg_pub = self.create_publisher(GridMap, "/gdm/grid_seg", 10)
        elif self.grid_seg == 'occ_map':
            self.grid_seg_pub = self.create_publisher(OccupancyGrid, "/gdm/grid_seg", 10)


        # 타이머: 주기적으로 데이터 동기화 확인 (0.1초마다)
        self.create_timer(0.1, self.check_sync_data)

        # 마지막 실행 시간 저장 (ROS Clock 기반)
        self.last_time = self.get_clock().now()

        # 기타
        self.bridge = CvBridge()
        self.calib = Calibration(calibration_filepath)
        self.u_grid, self.v_grid = None, None

        self.save_index = 0
        # Set up a directory to save images
        self.image_save_dir = "/home/gdr/workspace/src/gdm_object_tracking/gdm_segmentation/debug_images"  # Change this to the desired directory
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)
        self.get_logger().info("✅ GDMSegNode (ROS/System Clock 호환 버전) 초기화 완료")


        # -----------------------------
    # 각 센서 콜백
    # -----------------------------
    def image_callback(self, msg):
        with self.data_lock:
            self.image_msg = msg
            self.image_recv_time = self.get_clock().now()   

    def lidar_callback(self, msg):
        with self.data_lock:
            self.pcloud_msg = msg
            self.lidar_recv_time = self.get_clock().now()
        
    def depth_callback(self, msg):
        with self.data_lock:
            self.depth_msg = msg
            self.depth_recv_time = self.get_clock().now()

    def odom_callback(self, msg):
        self.get_logger().info('odom recv')
        with self.data_lock:
            self.odometry_msg = msg
            self.odom_recv_time = self.get_clock().now()


    # -----------------------------
    # 타이머 기반 동기화 체크
    # -----------------------------
    def check_sync_data(self):
        """3개 센서의 최근 메시지를 비교해 시간 차가 작으면 처리"""
        self.get_logger().info(f"[Thread] Sync: {threading.get_ident()} started")   if self.debug in ['assert', 'all'] else None

        with self.data_lock:
            # if self.image_msg is None or self.pcloud_msg is None or self.depth_msg is None or self.odometry_msg is None:
            if self.image_msg is None or self.pcloud_msg is None or self.depth_msg is None:
                return

            t_img  = self.image_recv_time
            t_pc   = self.lidar_recv_time
            t_depth = self.depth_recv_time
            # t_odom = self.odom_recv_time

            # 락 해제 후 Δt 계산
            delta_ip = abs((t_img - t_pc).nanoseconds) * 1e-9
            # delta_io = abs((t_img - t_odom).nanoseconds) * 1e-9
            # delta_po = abs((t_pc - t_odom).nanoseconds) * 1e-9
            delta_id = abs((t_img - t_depth).nanoseconds) * 1e-9

            # 현재 ROS Clock 기준 시간
            now = self.get_clock().now()
            dt = (now - self.last_time).nanoseconds * 1e-9
            self.last_time = now

            # if self.debug in ['time', 'all']:
            #     self.get_logger().info("🟢 [SYNC CHECK]")
            #     self.get_logger().info(f"   Δt(image-lidar): {delta_ip:.3f}s")
            #     self.get_logger().info(f"   Δt(image-odom) : {delta_io:.3f}s")
            #     self.get_logger().info(f"   Δt(lidar-odom) : {delta_po:.3f}s")

            # 0.1초 이내면 동기화된 프레임으로 간주
            # if delta_ip < 0.1 and delta_io < 0.1 and delta_po < 0.1 and delta_id < 0.1:
            if delta_ip < 0.1 and delta_id < 0.1:
                if self.debug in ['assert', 'all']:
                    self.get_logger().info("✅ [SYNCED FRAME FOUND]")
                    self.get_logger().info(f"   Callback interval: {dt:.3f}s\n")
                # self.process_synced_data(self.image_msg, self.pcloud_msg, self.odometry_msg)
                if self.processing_queue.qsize() > 1:
                    self.processing_queue.get_nowait()
                self.processing_queue.put((self.image_msg, self.pcloud_msg, self.depth_msg, self.odometry_msg))
        
    
    def worker_loop(self):
        self.get_logger().info(f"[Thread] worker_loop: {threading.get_ident()}")
        while rclpy.ok():
            image_msg, pcloud_msg, depth_msg, odometry_msg = self.processing_queue.get()
            self.get_logger().info(f"Processing on thread {threading.get_ident()}") if self.debug in ['assert', 'all'] else None
            time0 = time.time()

            # --------------------  
            # image preprocessing
            # --------------------
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8") 

            if self.debug == 'time':
                time1 = time.time()
                self.get_logger().info(f"inference time1: {(time1-time0) * 1000.0:.2f} ms")

            # if self.mode == 'torch':
            #     out = self.model([cv_image], return_vis=True)
            #     if self.debug == 'time':
            #         time2 = time.time()
            #         self.get_logger().info(f"inference time2: {(time2-time1) * 1000.0:.2f} ms")
            #     pred , vis = out['predictions'], out['visualization']
            #     self.pub_viz_seg.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
            # elif self.mode == 'trt':
            #     inputs, _ = self.task_processor.create_input(cv_image, self.input_shape)
            #     out = self.model.test_step(inputs)
            #     pred = out[0].pred_sem_seg.data.cpu().numpy()[0]

            #     if self.debug == 'time':
            #         time2 = time.time()
            #         self.get_logger().info(f"inference time2: {(time2-time1) * 1000.0:.2f} ms")
            #     elif self.debug == 'trt':
            #         timestamp = self.get_clock().now().to_msg().sec  # Get the ROS time in seconds
            #         timestamp_str = str(timestamp)
            #         filename = os.path.join(self.image_save_dir, f"image_{timestamp_str}.png")
            #         predname = os.path.join(self.image_save_dir, f"pred_{timestamp_str}.png")

            #         # Save the image as a PNG file
            #         cv2.imwrite(filename, cv_image)
            #         self.task_processor.visualize(image=cv_image,
            #                                             model=self.model,
            #                                             result = out[0],
            #                                             window_name='tensorRT vis',
            #                                             output_file=predname)
            time2 = time.time()

            if self.sensor == 'depth':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                pcloud, u, v = self.depth_to_pcloud(depth_image)
                in_image_idx = (u >= 0) & (u < cv_image.shape[1]) & (v >= 0) & (v < cv_image.shape[0]) # & valid_pc
                valid_pc = (pcloud[:, 2]) > 0
                pcloud = pcloud[in_image_idx]

            elif self.sensor == 'lidar':
                # pcloud projection
                pcloud, pcloud_projected = self.project_pcloud(pcloud_msg)
                # filtering pcloud
                # u = safe_int_cast(pcloud_projected[:, 0])
                # v = safe_int_cast(pcloud_projected[:, 1])
                # valid_pc = (pcloud[:, 0] > 0) 
                # in_image_idx = (u >= 0) & (u < cv_image.shape[1]) & (v >= 0) & (v < cv_image.shape[0]) & valid_pc
                valid_pc = np.isfinite(pcloud).all(axis=1)
                pcloud = pcloud[valid_pc]

            if self.debug in ['all']:
                self.get_logger().info(f"Raw pcloud points: {len(pcloud)}")
            elif self.debug == 'time':
                time3 = time.time()
                self.get_logger().info(f"inference time3: {(time3-time2) * 1000.0:.2f} ms")

            # pcloud = pcloud[in_image_idx]
            # class_to_occ = np.array([1, 0, 0, 0, 0, 1, 0], dtype=np.int8)   
            # pred_per_point = pred[v[in_image_idx], u[in_image_idx]]
            # occ_per_point = class_to_occ[pred_per_point]
            z_values = pcloud[:, 2]
            occ_per_point = np.zeros(len(pcloud), dtype=np.int8)
            obstacle_mask = (z_values > 0.1)
            occ_per_point[obstacle_mask] = 1

            if self.debug in ['all']:
                self.get_logger().info(f"--- Frame Debug ---")
                self.get_logger().info(f"Points in front of camera (valid_pc): {np.sum(valid_pc)}")
                self.get_logger().info(f"Points inside image bounds (in_image_idx): {np.sum(in_image_idx)}")

                # if np.sum(in_image_idx) == 0:
                #     self.get_logger().error("ZERO points left after filtering! Check your calibration or depth values.")
                # else:
                #     unique_preds = np.unique(pred_per_point)
                #     occ_count = np.sum(occ_per_point == 1)
                #     self.get_logger().info(f"Unique classes found: {unique_preds}")
                #     self.get_logger().info(f"Occupied points to be mapped: {occ_count}")

            # self.debug_visualize_occmap(pcloud, occ_per_point)
            # Generating grid map
            if self.grid_seg == 'grid_map':
                grid_msg = points_occ_to_gridmap(
                    pcloud,
                    occ_per_point,
                    stamp_ros=self.get_clock().now().to_msg(),
                    frame_id="os_sensor",
                    resolution=self.resolution,
                    width=self.grid_size, height=self.grid_size,
                    # zlim=(-1.0, 1.0)  # 필요하면
                )
            elif self.grid_seg == 'occ_map':
                grid_msg = points_occ_to_occmap(
                    pcloud,
                    occ_per_point,
                    stamp_ros=self.get_clock().now().to_msg(),
                    frame_id="os_sensor",
                    resolution=self.resolution,
                    width=self.grid_size, height=self.grid_size,
                    # zlim=(-1.0, 1.0)  # 필요하면
                )
            
            if self.debug == 'time':
                time4 = time.time()
                self.get_logger().info(f"inference time4: {(time4-time3) * 1000.0:.2f} ms")

            self.grid_seg_pub.publish(grid_msg)

            if self.debug in ['calib']:
                # save
                save_root = '/home/gdr/workspace/calib_images'
                base_name = f"{self.save_index:06d}"
                self.save_index += 1

                # 이미지 저장
                img_path = os.path.join(save_root, base_name + ".png")
                cv2.imwrite(img_path, cv_image)

                # 포인트클라우드 PCD 저장 (XYZ)
                pcd_path = os.path.join(save_root, base_name + ".pcd")
                save_pcd_ascii(pcloud_msg, pcd_path)

                self.get_logger().info(
                    f"Saved image & pcloud: {base_name}.png / {base_name}.pcd "
                )

                ## pcloud project for debug ##
                pcloud, pcloud_projected = self.project_pcloud(pcloud_msg)
                # filtering pcloud
                u = safe_int_cast(pcloud_projected[:, 0])
                v = safe_int_cast(pcloud_projected[:, 1])
                calib_image = cv_image.copy()

                valid_pc = (pcloud[:, 0] > 0) 
                in_image_idx = (u >= 0) & (u < calib_image.shape[1]) & (v >= 0) & (v < calib_image.shape[0]) # & valid_pc

                pts_v = v[in_image_idx]
                pts_u = u[in_image_idx]

                calib_image[pts_v, pts_u] = (0, 255, 255) 
                cv2.imshow("Debug Visualization (calib image)", calib_image)
                cv2.waitKey(1)

                colors_bgr = calib_image[pts_v, pts_u].astype(np.uint8)  # (N,3) BGR
                colors_rgb = colors_bgr[:, ::-1]  # RGB로 변환
                xyz = pcloud[in_image_idx, :3].astype(np.float32)
                pcd_col_path = os.path.join(save_root, base_name + "_colored.pcd")
                save_pcd_ascii_xyzrgb(xyz, colors_rgb, pcd_col_path)
                self.get_logger().info(f"Saved colored PCD: {base_name}_colored.pcd")


            elif self.debug == 'save':
                timestamp = self.get_clock().now().to_msg().sec  # Get the ROS time in seconds
                timestamp_str = str(timestamp)
                filename = os.path.join(self.image_save_dir, f"image_{timestamp_str}.png")
                predname = os.path.join(self.image_save_dir, f"pred_{timestamp_str}.png")

                # Save the image as a PNG file
                cv2.imwrite(filename, cv_image)
                cv2.imwrite(predname, vis)


    def project_pcloud(self, pcloud_msg):
        pcloud_tp = rnp.numpify(pcloud_msg)
        pcloud_np = get_xyzi_points(pcloud_tp)
        pcloud_np = pcloud_np.reshape(-1, 3)
        # pcloud = downsample_pointcloud(pcloud_np, voxel_size=self.voxel_leaf_size)      # downsampling mode             
        pcloud = pcloud_np  # no downsampling
        pcloud_projected = self.calib.project_velo_to_image(pcloud)
        return pcloud, pcloud_projected


    def depth_to_pcloud(self, depth_image):
        """
        Converts a depth image into a 3D point cloud (N, 3) 
        using the pinhole camera model.
        """
        # These should be set from your CameraInfo topic
        fx, fy = self.calib.f_u, self.calib.f_v # self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.calib.c_u, self.calib.c_v # self.intrinsics['cx'], self.intrinsics['cy']

        # Create coordinate grid
        h, w = depth_image.shape
        # v, u = np.mgrid[0:h, 0:w]
        
        # # Flatten for vectorization
        # u = u.flatten()
        # v = v.flatten()
        # z = depth_image.flatten() * 0.001 # Convert mm to meters if 16UC1
        if self.u_grid is None:
            self.v_grid, self.u_grid = np.mgrid[0:h, 0:w]
            self.u_grid = self.u_grid.flatten()
            self.v_grid = self.v_grid.flatten()
            
        z = depth_image.flatten() * 0.001
        # Filter out zero depth (invalid)
        valid = (z > 0.1) & (z < 20) # Filter noise
        z, u, v = z[valid], self.u_grid[valid], self.v_grid[valid]
        
        # Back-projection formula: X = (u-cx)*Z/fx , Y = (v-cy)*Z/fy
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Create (N, 3) array: [X, Y, Z]
        # Note: In camera coordinates, Z is forward, X is right, Y is down
        pcloud = np.stack((x, y, z), axis=-1)
        pcloud_projected = self.calib.project_rect_to_velo(pcloud)
        if self.debug in ['all']:
            self.get_logger().info(f"Pcloud sample (x,y,z): {pcloud[0] if len(pcloud)>0 else 'N/A'}")
            self.get_logger().info(f"Points after project_rect_to_velo: {len(pcloud_projected)}")
        return pcloud_projected, u, v


def save_pcd_ascii(pcloud_msg, filepath: str):
    """
    points: (N, 3) numpy array, XYZ only
    filepath: .pcd filename
    """ 
    pcloud_tp = rnp.numpify(pcloud_msg)
    pcloud_np = get_xyzi_points(pcloud_tp)
    points = pcloud_np.reshape(-1, 3)
    n_points = points.shape[0]

    header = f"""# .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z
    SIZE 4 4 4
    TYPE F F F
    COUNT 1 1 1
    WIDTH {n_points}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {n_points}
    DATA ascii
    """
    with open(filepath, "w") as f:
        f.write(header)
        # xyz 한 줄씩
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")


def save_pcd_ascii_xyzrgb(xyz: np.ndarray, rgb: np.ndarray, path: str):
    assert xyz.shape[0] == rgb.shape[0]
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z r g b\n")
        f.write("SIZE 4 4 4 1 1 1\n")
        f.write("TYPE F F F U U U\n")
        f.write("COUNT 1 1 1 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def safe_int_cast(arr):
    arr = np.where(np.isfinite(arr), arr, -1)  # NaN/Inf → -1
    return arr.astype(np.int32)


def get_xyzi_points(cloud_array, remove_nans=True, dtype=np.float32):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
    a 3xN matrix.
    '''
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z'] # - 1.5
    # points[..., 3] = cloud_array['intensity'] 
    return points


def points_occ_to_occmap(
    pcloud_xyz: np.ndarray,     # (N,3) in os_sensor
    occ_per_point: np.ndarray,  # (N,) values {0,1,-1}
    *,
    stamp_ros,
    frame_id: str = "os_sensor",
    resolution: float = 0.10,
    width: int = 20,
    height: int = 20,
    center_x: float = 0.0,
    center_y: float = 0.0,
    zlim=None,                  # optional
):
    pts = pcloud_xyz.astype(np.float64)
    occ = occ_per_point.astype(np.int16)

    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    occ = occ[finite]

    if zlim is not None:
        zmin, zmax = zlim
        keep = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        pts = pts[keep]
        occ = occ[keep]

    half_w = (width * resolution) / 2.0
    half_h = (height * resolution) / 2.0
    xmin = center_x - half_w
    ymin = center_y - half_h

    grid = -np.ones((height, width), dtype=np.int16)  # unknown(-1)

    if pts.shape[0] > 0:
        cx = np.floor((pts[:, 0] - xmin) / resolution).astype(np.int32)
        cy = np.floor((pts[:, 1] - ymin) / resolution).astype(np.int32)
        valid = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)

        cx, cy, occ = cx[valid], cy[valid], occ[valid]

        # free(0) 
        free_idx = np.where(occ == 0)[0]
        grid[cy[free_idx], cx[free_idx]] = 0

        # occupied(1) 
        occ_idx = np.where(occ == 1)[0]
        grid[cy[occ_idx], cx[occ_idx]] = 100


    msg = OccupancyGrid()
    msg.header = Header(stamp=stamp_ros, frame_id=frame_id)
    msg.info.resolution = float(resolution)
    msg.info.width = int(width)
    msg.info.height = int(height)

    msg.info.origin = Pose()
    msg.info.origin.position.x = float(xmin)
    msg.info.origin.position.y = float(ymin)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0

    msg.data = grid.reshape(-1).astype(np.int8).tolist()
    return msg

def points_occ_to_gridmap(
    pcloud_xyz: np.ndarray,
    occ_per_point: np.ndarray,
    *,
    stamp_ros,
    frame_id: str = "os_sensor",
    resolution: float = 0.10,
    width: int = 20,
    height: int = 20,
    center_x: float = 0.0,
    center_y: float = 0.0,
    zlim=None,
):
    # 1. 필터링 (기존 동일)
    pts = pcloud_xyz.astype(np.float64)
    occ = occ_per_point.astype(np.int16)
    finite = np.isfinite(pts).all(axis=1)
    pts, occ = pts[finite], occ[finite]

    if zlim is not None:
        zmin, zmax = zlim
        keep = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        pts, occ = pts[keep], occ[keep]

    # 2. Grid 데이터 생성
    # GridMap은 기본적으로 NaN을 'Unknown'으로 인식합니다. -1.0 대신 np.nan 권장.
    grid = np.full((height, width), np.nan, dtype=np.float32)

    half_w_m = (width * resolution) / 2.0
    half_h_m = (height * resolution) / 2.0
    
    # GridMap 좌표계: center_x/y는 지도의 중심. 
    # 인덱스 (0,0)은 +x, +y 방향(Top-Left)에 해당함.
    # 계산 편의를 위해 min 좌표 계산
    xmin = center_x - half_w_m
    ymin = center_y - half_h_m

    # coordinate system should be debugged
    # grid map coordinate system is os_sensor
    if pts.shape[0] > 0: 
        cx = height - 1 - np.floor((pts[:, 0] - xmin) / resolution).astype(np.int32)
        cy = width - 1 - np.floor((pts[:, 1] - ymin) / resolution).astype(np.int32)

        valid = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
        cx, cy, occ = cx[valid], cy[valid], occ[valid]

        # 데이터 채우기 (0.0: Free, 1.0: Occupied 권장)
        # GridMap 플러그인 기본 설정은 0~1 사이인 경우가 많음
        grid[cy, cx] = np.where(occ == 1, 1.0, 0.0)

    # 3. GridMap 메시지 생성
    msg = GridMap()
    msg.header = Header(stamp=stamp_ros, frame_id=frame_id)
    msg.info.resolution = float(resolution)
    msg.info.length_x = float(width * resolution)
    msg.info.length_y = float(height * resolution)
    msg.info.pose.position.x = float(center_x)
    msg.info.pose.position.y = float(center_y)
    msg.info.pose.orientation.w = 1.0

    # 4. Layer 및 Data (중요 수정 구간)
    layer_name = "elevation" # RViz2 기본 플러그인이 보통 'elevation'을 먼저 찾음
    msg.layers.append(layer_name)
    data_array = Float32MultiArray()
    
    # GridMap 라이브러리 표준에 따른 Dimension 설정
    # Dim 0: x축 (row), Dim 1: y축 (column)
    dim0 = MultiArrayDimension(label="column_index", size=width, stride=width * height)
    dim1 = MultiArrayDimension(label="row_index", size=height, stride=height)

    data_array.layout.dim = [dim0, dim1]
    data_array.layout.data_offset = 0

    # [수정 핵심] GridMap 시각화 플러그인은 데이터를 Row-major로 받지만 
    # 내부적으로 좌표계가 뒤집혀 있을 수 있음. .flatten() 사용.
    data_array.data = grid.flatten().tolist()

    msg.data.append(data_array)
    msg.outer_start_index = 0
    msg.inner_start_index = 0

    return msg


def main(args=None):
    rclpy.init(args=args)
    node = GDMSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

