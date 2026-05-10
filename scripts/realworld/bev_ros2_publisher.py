#!/usr/bin/env python3
"""
BEV map ROS2 publisher bridge.

Receives annotated BEV image (JPEG) from Isaac Sim (Python 3.11) via ZMQ,
and republishes as sensor_msgs/Image on /bev/map for rviz2.

Usage (Terminal 2, after starting eval):
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3.12 scripts/realworld/bev_ros2_publisher.py
"""
import sys
sys.path.insert(0, '/opt/ros/jazzy/lib/python3.12/site-packages')

import numpy as np
import zmq
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2


ZMQ_ADDR = 'tcp://localhost:5577'
ROS_TOPIC = '/bev/map'
PUBLISH_HZ = 10


class BevRos2Publisher(Node):
    def __init__(self):
        super().__init__('bev_ros2_publisher')
        self.pub = self.create_publisher(Image, ROS_TOPIC, 1)

        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVHWM, 1)   # keep only latest frame
        self.sock.bind(ZMQ_ADDR)

        self.create_timer(1.0 / PUBLISH_HZ, self._timer_cb)
        self.get_logger().info(f'Listening on {ZMQ_ADDR}, publishing {ROS_TOPIC}')

    def _timer_cb(self):
        try:
            data = self.sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            return

        nparr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.height, msg.width = img_rgb.shape[:2]
        msg.encoding = 'rgb8'
        msg.step = msg.width * 3
        msg.data = img_rgb.tobytes()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = BevRos2Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
