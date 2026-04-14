#!/usr/bin/env python3
"""Simple test client that publishes cmd_vel at maximum rate."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import time
from message_filters import Subscriber, ApproximateTimeSynchronizer

class SimpleClient(Node):
    def __init__(self):
        super().__init__('simple_client')
        
        self.pub = self.create_publisher(Twist, '/cmd_vel_bridge', 10)
        self.odom = None
        
        rgb_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 30, 0.1)
        self.sync.registerCallback(self.callback)
        
        self.create_subscription(Odometry, '/gdq/msg/gdq_odom', self.odom_callback, 10)
        
        self.cmd_count = 0
        self.start = time.time()
        self.get_logger().info('Simple client started')
    
    def callback(self, rgb, depth):
        pass
    
    def odom_callback(self, msg):
        import math
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
    
    def publish_cmd(self):
        if self.odom is None:
            return
        
        cmd = Twist()
        cmd.linear.x = 0.3
        cmd.angular.z = 0.2
        self.pub.publish(cmd)
        self.cmd_count += 1
        
        if self.cmd_count % 1000 == 0:
            hz = self.cmd_count / (time.time() - self.start)
            self.get_logger().info(f'cmd_vel Hz: {hz:.0f}')

def main():
    rclpy.init()
    node = SimpleClient()
    
    rate = node.create_rate(100)
    while rclpy.ok():
        rclpy.spin_once(node)
        node.publish_cmd()
        rate.sleep()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
