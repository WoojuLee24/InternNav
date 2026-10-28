"""sim 순간이동(/sim/teleport_pose) + 사람 재생 시각(/sim/hunav_t0) 확인용: 사용 python3 teleport_probe.py <x> <y> <바닥 z> <yaw rad> <t0> <out.png> (gd_vln 컨테이너)."""
import math, sys, time
import numpy as np, rclpy, cv2
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
rclpy.init(); n = Node('probe'); st = {}
n.create_subscription(Odometry, '/gdq/msg/gdq_odom', lambda m: st.__setitem__('odom', m.pose.pose.position), 10)
n.create_subscription(Image, '/camera/camera/color/image_raw', lambda m: st.__setitem__('img', m), qos_profile_sensor_data)
tp = n.create_publisher(Pose, '/sim/teleport_pose', 1); t0 = n.create_publisher(Float64, '/sim/hunav_t0', 1)
def spin(s):
    e = time.time() + s
    while time.time() < e: rclpy.spin_once(n, timeout_sec=0.05)
while tp.get_subscription_count() == 0: spin(0.2)
x, y, z, yaw = map(float, sys.argv[1:5])
p = Pose(); p.position.x, p.position.y, p.position.z = x, y, z + 0.55; p.orientation.z, p.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
for _ in range(2): tp.publish(p); spin(4.0)
print('odom after teleport', st.get('odom'))
t0.publish(Float64(data=float(sys.argv[5]))); spin(4.0)
m = st['img']; a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
cv2.imwrite(sys.argv[6], cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR) if m.encoding.startswith('rgb') else a[..., :3])
print('img', m.width, m.height, m.encoding, '->', sys.argv[6]); print('odom', st.get('odom'))
