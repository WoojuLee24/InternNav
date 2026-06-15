import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
import numpy as np
import time

class PointCloudHzMonitor(Node):
    def __init__(self):
        super().__init__('pointcloud_hz_monitor',
                         automatically_declare_parameters_from_overrides=True)

        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        ])

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        self.sub = self.create_subscription(
            PointCloud2, '/ouster/points', self.callback, qos
        )

        self.header_stamps = []
        self.wall_times = []
        self.count = 0

    def callback(self, msg):
        wall_now = time.monotonic()
        stamp_now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.wall_times.append(wall_now)
        self.header_stamps.append(stamp_now)
        self.count += 1

        if len(self.wall_times) < 2:
            return

        # Wall-clock Hz: 실제 메시지 수신 속도 (bag play 속도 반영)
        wall_diffs = np.diff(self.wall_times[-11:])
        wall_hz = 1.0 / np.mean(wall_diffs)

        # Header-stamp Hz: 원본 센서 타임스탬프 기준 (녹화 당시 데이터 품질 반영)
        stamp_diff = stamp_now - self.header_stamps[-2]
        if stamp_diff > 0:
            stamp_instant_hz = 1.0 / stamp_diff
        else:
            stamp_instant_hz = float('nan')

        valid_stamp_diffs = np.diff(self.header_stamps[-11:])
        valid_stamp_diffs = valid_stamp_diffs[valid_stamp_diffs > 0]
        stamp_avg_hz = 1.0 / np.mean(valid_stamp_diffs) if len(valid_stamp_diffs) > 0 else float('nan')

        # 큰 gap 경고 (0.2s 이상 = 5hz 미만)
        gap_warn = " [GAP!]" if (stamp_diff > 0 and stamp_diff > 0.15) else ""

        print(f"[{self.count:4d}] "
              f"wall: {wall_hz:5.1f}Hz | "
              f"stamp_inst: {stamp_instant_hz:5.1f}Hz | "
              f"stamp_avg: {stamp_avg_hz:5.1f}Hz"
              f"{gap_warn}")

def main():
    rclpy.init()
    node = PointCloudHzMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        wall_arr = np.array(node.wall_times)
        stamp_arr = np.array(node.header_stamps)
        print(f"\n=== 최종 통계 ===")
        print(f"총 메시지 수: {node.count}")

        if len(wall_arr) > 1:
            wall_diffs = np.diff(wall_arr)
            print(f"[Wall-clock] Mean: {1/np.mean(wall_diffs):.2f}Hz  "
                  f"Median: {1/np.median(wall_diffs):.2f}Hz")

        if len(stamp_arr) > 1:
            stamp_diffs = np.diff(stamp_arr)
            valid = stamp_diffs[stamp_diffs > 0]
            if len(valid) > 0:
                print(f"[Header stamp] Mean: {1/np.mean(valid):.2f}Hz  "
                      f"Median: {1/np.median(valid):.2f}Hz  "
                      f"Min interval: {valid.min()*1000:.1f}ms  "
                      f"Max interval: {valid.max()*1000:.1f}ms")
                gaps = valid[valid > 0.15]
                if len(gaps) > 0:
                    print(f"  → Gap (>150ms) 발생 횟수: {len(gaps)}  "
                          f"(전체의 {100*len(gaps)/len(valid):.1f}%)")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
