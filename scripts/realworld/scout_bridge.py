#!/usr/bin/env python3

import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class ScoutBridge(Node):
    def __init__(self):
        super().__init__('scout_bridge')

        # If user passes --ros-args -p use_sim_time:=true, ROS may already declare it.
        try:
            self.declare_parameter('use_sim_time', False)
        except ParameterAlreadyDeclaredException:
            pass

        # Topics
        self.declare_parameter('odom_in', '/gdq/msg/gdq_odom')
        self.declare_parameter('odom_out', '/odom_bridge')
        self.declare_parameter('cmd_in', '/cmd_vel_bridge')
        self.declare_parameter('cmd_out', '/gdq/msg/cmd_vel')

        # Safety
        self.declare_parameter('deadman_timeout_sec', 0.5)
        self.declare_parameter('publish_stop_on_timeout', True)
        self.declare_parameter('max_linear_x', 0.6)
        self.declare_parameter('max_angular_z', 0.8)

        odom_in = str(self.get_parameter('odom_in').value)
        odom_out = str(self.get_parameter('odom_out').value)
        cmd_in = str(self.get_parameter('cmd_in').value)
        cmd_out = str(self.get_parameter('cmd_out').value)

        self.deadman_timeout_sec = float(self.get_parameter('deadman_timeout_sec').value)
        self.publish_stop_on_timeout = bool(self.get_parameter('publish_stop_on_timeout').value)
        self.max_linear_x = float(self.get_parameter('max_linear_x').value)
        self.max_angular_z = float(self.get_parameter('max_angular_z').value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.odom_pub = self.create_publisher(Odometry, odom_out, qos)
        self.cmd_pub = self.create_publisher(Twist, cmd_out, qos)

        self.create_subscription(Odometry, odom_in, self._odom_cb, qos)
        self.create_subscription(Twist, cmd_in, self._cmd_cb, qos)

        self._last_cmd_time = None
        self._last_cmd = Twist()
        self._stopped_sent = False

        self.timer = self.create_timer(0.05, self._tick)

        self.get_logger().info(f'odom: {odom_in} -> {odom_out}')
        self.get_logger().info(f'cmd:  {cmd_in} -> {cmd_out}')
        self.get_logger().info(
            f'deadman_timeout_sec={self.deadman_timeout_sec} publish_stop_on_timeout={self.publish_stop_on_timeout}'
        )

    def _odom_cb(self, msg: Odometry):
        self.odom_pub.publish(msg)

    def _cmd_cb(self, msg: Twist):
        out = Twist()
        out.linear.x = max(min(float(msg.linear.x), self.max_linear_x), -self.max_linear_x)
        out.linear.y = 0.0
        out.linear.z = 0.0
        out.angular.x = 0.0
        out.angular.y = 0.0
        out.angular.z = max(min(float(msg.angular.z), self.max_angular_z), -self.max_angular_z)

        self._last_cmd = out
        self._last_cmd_time = time.time()
        self._stopped_sent = False
        self.cmd_pub.publish(out)

    def _tick(self):
        if not self.publish_stop_on_timeout:
            return

        if self._last_cmd_time is None:
            return

        if self._stopped_sent:
            return

        if (time.time() - self._last_cmd_time) < self.deadman_timeout_sec:
            return

        stop = Twist()
        self.cmd_pub.publish(stop)
        self._stopped_sent = True
        self.get_logger().warn('cmd_vel timed out -> publishing STOP', throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = ScoutBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
