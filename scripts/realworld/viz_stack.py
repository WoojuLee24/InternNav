#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class VizStack(Node):
    def __init__(self):
        super().__init__("viz_stack")

        # If user passes --ros-args -p use_sim_time:=true, ROS may already declare it.
        try:
            self.declare_parameter("use_sim_time", True)
        except ParameterAlreadyDeclaredException:
            pass

        # I/O
        self.declare_parameter("odom_in", "/gdq/msg/gdq_odom")
        self.declare_parameter("odom_out", "/odom_bridge")
        self.declare_parameter("cmd_vel_in", "/cmd_vel_bridge")

        # Visualization frames
        self.declare_parameter("map_frame", "map")
        # Use LiDAR pose TF so point cloud + path share the same origin.
        self.declare_parameter("path_frame", "os_sensor_on_map")

        self.declare_parameter("max_path_points", 5000)
        self.declare_parameter("path_hz", 20.0)

        # Some bags publish TF for `os_sensor_on_map`, while point clouds use frame_id `os_sensor`.
        self.declare_parameter("alias_os_sensor", True)
        self.declare_parameter("os_sensor_on_map_frame", "os_sensor_on_map")
        self.declare_parameter("os_sensor_frame", "os_sensor")

        self.odom_in = self.get_parameter("odom_in").value
        self.odom_out = self.get_parameter("odom_out").value
        self.cmd_vel_in = self.get_parameter("cmd_vel_in").value

        self.map_frame = self.get_parameter("map_frame").value
        self.path_frame = self.get_parameter("path_frame").value

        self.max_path_points = int(self.get_parameter("max_path_points").value)
        self.path_hz = float(self.get_parameter("path_hz").value)

        self.alias_os_sensor = bool(self.get_parameter("alias_os_sensor").value)
        self.os_sensor_on_map_frame = self.get_parameter("os_sensor_on_map_frame").value
        self.os_sensor_frame = self.get_parameter("os_sensor_frame").value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_static = StaticTransformBroadcaster(self)
        if self.alias_os_sensor:
            self._publish_static_alias_tf()

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, self.odom_out, 20)
        self.path_pub = self.create_publisher(Path, "/visualization/path", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization/cmd_vel", 10)

        # Subscriptions
        self.create_subscription(Odometry, self.odom_in, self._odom_in_cb, 50)
        self.create_subscription(Twist, self.cmd_vel_in, self._cmd_cb, 10)

        # Path state
        self.path = Path()
        self.path.header.frame_id = self.map_frame
        self._last_clock_ns = None

        # Publish path at fixed rate (TF-based)
        period = 1.0 / self.path_hz if self.path_hz > 0.0 else 0.05
        self.timer = self.create_timer(period, self._path_timer_cb)

        self.get_logger().info(
            f"viz_stack: odom_in={self.odom_in} -> odom_out={self.odom_out}, cmd_vel_in={self.cmd_vel_in}"
        )
        self.get_logger().info(
            f"viz_stack: path from TF {self.map_frame}->{self.path_frame} (RViz Fixed Frame: {self.map_frame})"
        )
        self.get_logger().info("Publishing: /visualization/path, /visualization/cmd_vel")

    def _publish_static_alias_tf(self):
        t = TransformStamped()
        t.header.stamp.sec = 0
        t.header.stamp.nanosec = 0
        t.header.frame_id = self.os_sensor_on_map_frame
        t.child_frame_id = self.os_sensor_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_static.sendTransform(t)
        self.get_logger().info(f"Published tf_static alias: {self.os_sensor_on_map_frame}->{self.os_sensor_frame}")

    def _odom_in_cb(self, msg: Odometry):
        # Keep InternNav client compatibility (it expects /odom_bridge).
        self.odom_pub.publish(msg)

    def _maybe_reset_on_time_jump(self):
        now_ns = self.get_clock().now().nanoseconds
        if self._last_clock_ns is not None and now_ns + 50_000_000 < self._last_clock_ns:
            # Bag restarted or /clock jumped backwards -> clear path so it overlays correctly.
            self.path.poses.clear()
            self.get_logger().warn("/clock jumped backwards: clearing /visualization/path")
        self._last_clock_ns = now_ns

    def _path_timer_cb(self):
        self._maybe_reset_on_time_jump()

        # Use latest available TF (Time(0) in tf2 terms)
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.path_frame, Time())
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed {self.map_frame}->{self.path_frame}: {e}",
                throttle_duration_sec=2.0,
            )
            return

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.map_frame
        pose.pose.position.x = float(tf.transform.translation.x)
        pose.pose.position.y = float(tf.transform.translation.y)
        pose.pose.position.z = float(tf.transform.translation.z)
        pose.pose.orientation = tf.transform.rotation

        self.path.poses.append(pose)
        if len(self.path.poses) > self.max_path_points:
            del self.path.poses[: len(self.path.poses) - self.max_path_points]

        self.path.header.stamp = pose.header.stamp
        self.path.header.frame_id = self.map_frame
        self.path_pub.publish(self.path)

    def _cmd_cb(self, msg: Twist):
        v = float(msg.linear.x)
        w = float(msg.angular.z)

        stamp = self.get_clock().now().to_msg()

        arrow = Marker()
        arrow.header.stamp = stamp
        # Anchor cmd_vel marker to the same TF tree as the point cloud.
        arrow.header.frame_id = self.path_frame
        arrow.ns = "cmd_vel"
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x = 0.4
        arrow.scale.y = 0.08
        arrow.scale.z = 0.08
        arrow.color.r = 1.0
        arrow.color.g = 0.6
        arrow.color.b = 0.0
        arrow.color.a = 1.0

        p1 = Point(x=0.0, y=0.0, z=0.2)
        p2 = Point(x=max(min(v, 2.0), -2.0) * 0.8, y=0.0, z=0.2)
        arrow.points = [p1, p2]

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = self.path_frame
        text.ns = "cmd_vel"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.scale.z = 0.25
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.pose.position.x = 0.0
        text.pose.position.y = 0.6
        text.pose.position.z = 0.4
        text.text = f"v: {v:.2f} m/s\nw: {w:.2f} rad/s"

        arr = MarkerArray()
        arr.markers = [arrow, text]
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = VizStack()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
