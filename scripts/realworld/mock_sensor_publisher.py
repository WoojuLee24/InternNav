#!/usr/bin/env python3
"""
Mock publisher that simulates camera RGB, depth, and odometry data.
Used for testing the InternVLA client without real robot hardware.
"""

import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image as ImageMsg
from nav_msgs.msg import Odometry


class MockSensorPublisher(Node):
    def __init__(self, rgb_topic, depth_topic, odom_topic, width=640, height=480, camera_fps=40, odom_fps=60):
        super().__init__('mock_sensor_publisher')
        
        self.width = width
        self.height = height
        self.camera_fps = camera_fps
        self.odom_fps = odom_fps
        self.camera_dt = 1.0 / camera_fps
        self.odom_dt = 1.0 / odom_fps
        
        qos = QoSProfile(depth=10)
        
        self.rgb_pub = self.create_publisher(ImageMsg, rgb_topic, qos)
        self.depth_pub = self.create_publisher(ImageMsg, depth_topic, qos)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, qos)
        
        self.camera_frame_id = 0
        self.odom_frame_id = 0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        # Pre-allocate image data for speed
        self.rgb_data = np.full((height, width, 3), 128, dtype=np.uint8).tobytes()
        self.depth_data = np.full((height, width), 3000, dtype=np.uint16).tobytes()
        
        # Pre-create message templates
        self.rgb_msg = ImageMsg()
        self.rgb_msg.height = height
        self.rgb_msg.width = width
        self.rgb_msg.encoding = 'rgb8'
        self.rgb_msg.is_bigendian = False
        self.rgb_msg.step = width * 3
        
        self.depth_msg = ImageMsg()
        self.depth_msg.height = height
        self.depth_msg.width = width
        self.depth_msg.encoding = '16UC1'
        self.depth_msg.is_bigendian = False
        self.depth_msg.step = width * 2
        
        self.last_camera_time = time.time()
        self.last_odom_time = time.time()
        
        self.get_logger().info(f"Mock publisher started: RGB={rgb_topic}, Depth={depth_topic}, Odom={odom_topic}")
        self.get_logger().info(f"Resolution: {width}x{height}, Camera FPS: {camera_fps}, Odom FPS: {odom_fps}")
    
    def publish_rgb(self):
        self.rgb_msg.header.stamp = self.get_clock().now().to_msg()
        self.rgb_msg.header.frame_id = 'camera_link'
        self.rgb_msg.data = self.rgb_data
        self.rgb_pub.publish(self.rgb_msg)
    
    def publish_depth(self):
        self.depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_msg.header.frame_id = 'camera_link'
        self.depth_msg.data = self.depth_data
        self.depth_pub.publish(self.depth_msg)
    
    def publish_odom(self):
        self.x += 0.002 * math.cos(self.yaw)
        self.y += 0.002 * math.sin(self.yaw)
        self.yaw += 0.04
        
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = self.euler_to_quaternion(0, 0, self.yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.twist.twist.linear.x = 0.1
        msg.twist.twist.angular.z = 0.1
        self.odom_pub.publish(msg)
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw
    
    def spin_once(self):
        current_time = time.time()
        
        if current_time - self.last_camera_time >= self.camera_dt:
            self.publish_rgb()
            self.publish_depth()
            self.camera_frame_id += 1
            self.last_camera_time = current_time
        
        if current_time - self.last_odom_time >= self.odom_dt:
            self.publish_odom()
            self.odom_frame_id += 1
            self.last_odom_time = current_time


def main():
    parser = argparse.ArgumentParser(description='Mock sensor publisher for testing')
    parser.add_argument('--rgb_topic', type=str, default='/camera/camera/color/image_raw')
    parser.add_argument('--depth_topic', type=str, default='/camera/camera/aligned_depth_to_color/image_raw')
    parser.add_argument('--odom_topic', type=str, default='/gdq/msg/gdq_odom')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--camera_fps', type=int, default=40)
    parser.add_argument('--odom_fps', type=int, default=60)
    args = parser.parse_args()
    
    rclpy.init()
    node = MockSensorPublisher(
        args.rgb_topic, args.depth_topic, args.odom_topic,
        args.width, args.height, args.camera_fps, args.odom_fps
    )
    
    print("=" * 60)
    print("MOCK SENSOR PUBLISHER (Optimized)")
    print("=" * 60)
    print(f"Camera FPS: {args.camera_fps} Hz (RGB + Depth)")
    print(f"Odometry FPS: {args.odom_fps} Hz")
    print("=" * 60)
    
    loop_time = 0.001  # 1000 Hz loop
    try:
        while rclpy.ok():
            start = time.time()
            rclpy.spin_once(node, timeout_sec=0)
            node.spin_once()
            elapsed = time.time() - start
            sleep_time = loop_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
