import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import threading
import time

class RosRealSense:
    def __init__(self, rgb_topic='/camera/color/image_raw', depth_topic='/camera/aligned_depth_to_color/image_raw'):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('internnav_cam_bridge')
        self.rgb_sub = self.node.create_subscription(Image, rgb_topic, self._rgb_callback, 10)
        self.depth_sub = self.node.create_subscription(Image, depth_topic, self._depth_callback, 10)
        
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_ts = None
        self.lock = threading.Lock()
        
        # Start background spinning
        self.thread = threading.Thread(target=lambda: rclpy.spin(self.node), daemon=True)
        self.thread.start()

    def _rgb_callback(self, msg):
        with self.lock:
            # Manual conversion if cv_bridge is missing
            self.latest_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            # ROS is usually RGB, InternNav expects RGB (based on your code converting BGR to RGB)
            self.latest_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _depth_callback(self, msg):
        with self.lock:
            # Depth is usually 16-bit mm (uint16)
            depth_uint16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            self.latest_depth = depth_uint16.astype(np.float32) / 1000.0  # Convert to meters

    def get_observation(self, timeout_ms=2000):
        start_time = time.time()
        while time.time() - start_time < (timeout_ms / 1000.0):
            with self.lock:
                if self.latest_rgb is not None and self.latest_depth is not None:
                    return {
                        "rgb": self.latest_rgb.copy(),
                        "depth": self.latest_depth.copy(),
                        "timestamp_s": self.latest_ts
                    }
            time.sleep(0.01)
        raise RuntimeError("Timeout waiting for ROS camera topics. Check 'ros2 topic list'.")

    def start(self): pass
    def stop(self): rclpy.shutdown()
