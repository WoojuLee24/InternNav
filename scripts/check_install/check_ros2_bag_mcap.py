from mcap.reader import make_reader
import numpy as np

mcap_path = '/datasets/bags/260317_scout_noeun/my_camera_bag_20260317_080412/my_camera_bag_20260317_080412_0.mcap'

log_times = []
publish_times = []

with open(mcap_path, 'rb') as f:
    reader = make_reader(f)
    for schema, channel, message in reader.iter_messages(topics=['/ouster/points']):
        log_times.append(message.log_time)
        publish_times.append(message.publish_time)

log_times = np.array(log_times)
publish_times = np.array(publish_times)

print("=== log_time 기준 ===")
diffs = np.diff(log_times) / 1e9
print(f"Mean Hz: {1/np.mean(diffs):.2f}")

print("\n=== publish_time 기준 ===")
diffs2 = np.diff(publish_times) / 1e9
print(f"Mean Hz: {1/np.mean(diffs2):.2f}")

# 이상한 간격 출력
print("\n=== 100ms 이상 간격 구간 ===")
big_gaps = np.where(diffs > 0.15)[0]
print(f"큰 gap 수: {len(big_gaps)}")
for i in big_gaps[:5]:
    print(f"  idx {i}: {diffs[i]*1000:.1f} ms")
