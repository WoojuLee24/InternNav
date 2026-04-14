#!/usr/bin/env python3

import json
import math
import sys
from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node
from rclpy.time import Time


def _now_sec(node: Node) -> float:
    return node.get_clock().now().nanoseconds / 1e9


class TrialEval(Node):
    def __init__(self):
        super().__init__('trial_eval')

        # If user passes --ros-args -p use_sim_time:=true, ROS may already declare it.
        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass

        # Topics
        self.declare_parameter('odom_topic', '/odom_bridge')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_bridge')

        # Trial logic
        self.declare_parameter('timeout_sec', 30.0)
        # If require_motion is true, the trial "arms" only after motion is observed.
        # This avoids immediate failure when the system starts already stopped.
        self.declare_parameter('require_motion', True)
        self.declare_parameter('arm_timeout_sec', 10.0)
        self.declare_parameter('min_move_dist', 0.30)

        # Motion detection (arming)
        self.declare_parameter('motion_v_thresh', 0.08)
        self.declare_parameter('motion_w_thresh', 0.08)
        self.declare_parameter('motion_hold_sec', 0.25)

        # Stop detection (either cmd_vel or odom.twist)
        self.declare_parameter('stop_v_thresh', 0.05)
        self.declare_parameter('stop_w_thresh', 0.05)
        self.declare_parameter('stop_hold_sec', 2.0)

        # Distance target (optional)
        # If target_distance < 0, distance check is disabled.
        self.declare_parameter('target_distance', -1.0)
        self.declare_parameter('distance_tolerance', 0.50)

        # Output
        self.declare_parameter('json_out', '')

        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)

        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        self.require_motion = bool(self.get_parameter('require_motion').value)
        self.arm_timeout_sec = float(self.get_parameter('arm_timeout_sec').value)
        self.min_move_dist = float(self.get_parameter('min_move_dist').value)
        self.stop_v_thresh = float(self.get_parameter('stop_v_thresh').value)
        self.stop_w_thresh = float(self.get_parameter('stop_w_thresh').value)
        self.stop_hold_sec = float(self.get_parameter('stop_hold_sec').value)
        self.target_distance = float(self.get_parameter('target_distance').value)
        self.distance_tolerance = float(self.get_parameter('distance_tolerance').value)
        self.json_out = str(self.get_parameter('json_out').value)

        self.motion_v_thresh = float(self.get_parameter('motion_v_thresh').value)
        self.motion_w_thresh = float(self.get_parameter('motion_w_thresh').value)
        self.motion_hold_sec = float(self.get_parameter('motion_hold_sec').value)

        # State
        # Timing uses node clock (sim time when enabled).
        # Initialize t0 on first incoming message (sim time can be 0 until /clock starts).
        self.t0 = None
        self.armed = not self.require_motion
        self.start_t = None
        self.last_t = None
        self.start_xy = None
        self.last_xy = None
        self.path_len = 0.0
        self.last_odom_twist = None  # (v, w)

        self.cmd_hist = deque()  # (t, v, w)
        self.odom_twist_hist = deque()  # (t, v, w)

        self.stopped_t = None
        self.completed = False

        self.motion_first_t = None
        self.motion_seen = False

        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 50)
        self.create_subscription(Twist, self.cmd_vel_topic, self._cmd_cb, 50)
        self.timer = self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f"trial_eval: odom={self.odom_topic} cmd_vel={self.cmd_vel_topic} timeout={self.timeout_sec:.1f}s"
        )

    def _odom_cb(self, msg: Odometry):
        t = _now_sec(self)
        # When use_sim_time is enabled, ROS time can be 0 until /clock starts.
        if t <= 1.0:
            return
        self.last_t = t

        if self.t0 is None:
            self.t0 = t

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        if self.last_xy is not None:
            dx = x - self.last_xy[0]
            dy = y - self.last_xy[1]
            if self.armed:
                self.path_len += math.hypot(dx, dy)
        self.last_xy = (x, y)

        v = float(msg.twist.twist.linear.x)
        w = float(msg.twist.twist.angular.z)
        self.last_odom_twist = (v, w)
        self._push_hist(self.odom_twist_hist, t, v, w)

        self._update_motion(t, v, w)

    def _cmd_cb(self, msg: Twist):
        t = _now_sec(self)
        if t <= 1.0:
            return
        v = float(msg.linear.x)
        w = float(msg.angular.z)
        self._push_hist(self.cmd_hist, t, v, w)

        if self.t0 is None:
            self.t0 = t

        self._update_motion(t, v, w)

    def _update_motion(self, t: float, v: float, w: float):
        moving = abs(v) >= self.motion_v_thresh or abs(w) >= self.motion_w_thresh
        if not moving:
            return
        if self.motion_first_t is None:
            self.motion_first_t = t
        if (t - self.motion_first_t) >= self.motion_hold_sec:
            self.motion_seen = True

    def _push_hist(self, hist: deque, t: float, v: float, w: float):
        hist.append((t, v, w))
        cutoff = t - self.stop_hold_sec
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def _is_stopped(self) -> bool:
        # Prefer cmd_vel if available; fall back to odom twist if not.
        for hist in (self.cmd_hist, self.odom_twist_hist):
            if not hist:
                continue
            dt = hist[-1][0] - hist[0][0]
            if dt < self.stop_hold_sec * 0.9:
                continue
            if all(abs(v) <= self.stop_v_thresh and abs(w) <= self.stop_w_thresh for _, v, w in hist):
                return True
        return False

    def _tick(self):
        if self.completed:
            return

        # Need odom pose and a valid clock.
        if self.last_t is None or self.last_xy is None or self.t0 is None:
            return

        # Arm on motion.
        if not self.armed:
            if (self.last_t - self.t0) >= self.arm_timeout_sec:
                result = {
                    'success': False,
                    'reasons': ['never_started'],
                    'elapsed_sec': float(self.last_t - self.t0),
                    'displacement_m': 0.0,
                    'path_length_m': 0.0,
                    'start_xy': None,
                    'end_xy': {'x': float(self.last_xy[0]), 'y': float(self.last_xy[1])},
                    'stopped': None,
                    'stopped_at_sec': None,
                    'odom_topic': self.odom_topic,
                    'cmd_vel_topic': self.cmd_vel_topic,
                }
                self.get_logger().error(json.dumps(result, sort_keys=True))
                self.completed = True
                rclpy.shutdown()
                raise SystemExit(2)

            if not self.motion_seen:
                return

            # Start trial now.
            self.armed = True
            self.start_t = self.last_t
            self.start_xy = self.last_xy
            self.path_len = 0.0
            self.cmd_hist.clear()
            self.odom_twist_hist.clear()
            self.stopped_t = None
            self.get_logger().info(
                f"armed at t={self.start_t:.3f} start_xy=({self.start_xy[0]:.3f},{self.start_xy[1]:.3f})"
            )
            return

        if self.start_t is None or self.start_xy is None:
            # Should not happen, but keep safe.
            self.start_t = self.last_t
            self.start_xy = self.last_xy

        elapsed = self.last_t - self.start_t
        dx = self.last_xy[0] - self.start_xy[0]
        dy = self.last_xy[1] - self.start_xy[1]
        disp = math.hypot(dx, dy)

        stopped = self._is_stopped()
        if stopped and self.stopped_t is None:
            self.stopped_t = self.last_t

        timed_out = elapsed >= self.timeout_sec

        # Decide completion
        if stopped or timed_out:
            success = True
            reasons = []

            if disp < self.min_move_dist:
                success = False
                reasons.append(f"no_motion(d={disp:.2f}m)")

            if self.target_distance >= 0.0:
                lo = self.target_distance - self.distance_tolerance
                hi = self.target_distance + self.distance_tolerance
                if not (lo <= disp <= hi):
                    success = False
                    reasons.append(
                        f"distance_out_of_range(d={disp:.2f}m target={self.target_distance:.2f}±{self.distance_tolerance:.2f})"
                    )

            if not stopped:
                success = False
                reasons.append("no_stop")

            if timed_out and not stopped:
                reasons.append("timeout")

            result = {
                'success': success,
                'reasons': reasons,
                'elapsed_sec': float(elapsed),
                'displacement_m': float(disp),
                'path_length_m': float(self.path_len),
                'start_xy': {'x': float(self.start_xy[0]), 'y': float(self.start_xy[1])},
                'end_xy': {'x': float(self.last_xy[0]), 'y': float(self.last_xy[1])},
                'stopped': bool(stopped),
                'stopped_at_sec': (float(self.stopped_t) if self.stopped_t is not None else None),
                'odom_topic': self.odom_topic,
                'cmd_vel_topic': self.cmd_vel_topic,
            }

            line = json.dumps(result, sort_keys=True)
            if success:
                self.get_logger().info(line)
            else:
                self.get_logger().error(line)

            if self.json_out:
                try:
                    with open(self.json_out, 'w', encoding='utf-8') as f:
                        f.write(line + "\n")
                    self.get_logger().info(f"wrote {self.json_out}")
                except Exception as e:
                    self.get_logger().error(f"failed to write {self.json_out}: {e}")

            self.completed = True
            # Exit code: 0=success, 2=failure
            rclpy.shutdown()
            raise SystemExit(0 if success else 2)


def main():
    rclpy.init()
    node = TrialEval()
    try:
        rclpy.spin(node)
    except SystemExit as e:
        raise e
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
