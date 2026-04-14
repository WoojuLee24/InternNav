#!/usr/bin/env python3

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class CmdVelToOdomBridge(Node):
    def __init__(self):
        super().__init__("cmdvel_to_odom_bridge")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )

        self.create_subscription(Twist, "/gdq/msg/cmd_vel", self._cmd_cb, qos)
        self.odom_pub = self.create_publisher(Odometry, "/gdq/msg/gdq_odom", qos)

        self.vx = 0.0
        self.wz = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_t = time.time()

        self.create_timer(0.02, self._tick)
        self.get_logger().info("cmd_vel -> odom bridge started")
        self.get_logger().info("sub: /gdq/msg/cmd_vel, pub: /gdq/msg/gdq_odom")

    def _cmd_cb(self, msg: Twist):
        self.vx = float(msg.linear.x)
        self.wz = float(msg.angular.z)

    def _tick(self):
        now = time.time()
        dt = max(1e-4, now - self.last_t)
        self.last_t = now

        self.yaw += self.wz * dt
        self.x += self.vx * math.cos(self.yaw) * dt
        self.y += self.vx * math.sin(self.yaw) * dt

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(self.yaw * 0.5)
        msg.pose.pose.orientation.w = math.cos(self.yaw * 0.5)
        msg.twist.twist.linear.x = self.vx
        msg.twist.twist.angular.z = self.wz
        self.odom_pub.publish(msg)


def main():
    rclpy.init()
    node = CmdVelToOdomBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
