"""
Dataloader bottleneck profiler for NavDP training.

Usage:
    python scripts/train/base_train/profile_dataloader.py \
        --model-name navdp_1gpu \
        --num-workers 4 \
        --num-batches 20

Patches load_image / load_depth / load_pointcloud / process_data_parquet
so each I/O call is timed independently. No double-processing.
"""
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './src/diffusion-policy')

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='navdp_1gpu')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--num-batches', type=int, default=20)
parser.add_argument('--prefetch-factor', type=int, default=None)
parser.add_argument('--world-size', type=int, default=1,
                    help='Simulate this many GPUs via DistributedSampler (use 8 for 1node)')
args = parser.parse_args()

# ── config ────────────────────────────────────────────────────────────────────
from scripts.train.base_train.configs import (
    navdp_1gpu_exp_cfg, navdp_1node_exp_cfg, navdp_h200_1gpu_exp_cfg,
)
cfg_map = {
    'navdp_1gpu':      navdp_1gpu_exp_cfg,
    'navdp_1node':     navdp_1node_exp_cfg,
    'navdp_h200_1gpu': navdp_h200_1gpu_exp_cfg,
}
config = cfg_map[args.model_name]
config.il.num_workers = args.num_workers

print(f"\n{'='*60}")
print(f"  NavDP DataLoader bottleneck profiler")
print(f"  model      : {args.model_name}")
print(f"  root_dir   : {config.il.root_dir}")
print(f"  batch_size : {config.il.batch_size}")
print(f"  num_workers: {args.num_workers}")
print(f"  world_size : {args.world_size}  (DistributedSampler num_replicas)")
print(f"  scene_scale: {config.il.scene_scale}")
print(f"  num_batches: {args.num_batches}")
print(f"{'='*60}\n")

# ── patch low-level I/O methods ───────────────────────────────────────────────
# We patch at the class level so the patches are visible inside worker processes
# (workers fork after class definition).

import internnav.dataset.navdp_lerobot_dataset as _mod

_times = {
    'ply':     [],   # o3d.io.read_point_cloud
    'png_rgb': [],   # load_image (RGB)
    'png_dep': [],   # load_depth (depth PNG)
    'parquet': [],   # pd.read_parquet
}

_orig_load_image       = _mod.NavDP_Base_Datset.load_image
_orig_load_depth       = _mod.NavDP_Base_Datset.load_depth
_orig_load_pc          = _mod.NavDP_Base_Datset.load_pointcloud
_orig_proc_parquet     = _mod.NavDP_Base_Datset.process_data_parquet
_orig_proc_obstacle    = _mod.NavDP_Base_Datset.process_obstacle_points

def _timed_load_image(self, url):
    t = time.perf_counter()
    r = _orig_load_image(self, url)
    _times['png_rgb'].append(time.perf_counter() - t)
    return r

def _timed_load_depth(self, url):
    t = time.perf_counter()
    r = _orig_load_depth(self, url)
    _times['png_dep'].append(time.perf_counter() - t)
    return r

def _timed_load_pc(self, url):
    t = time.perf_counter()
    r = _orig_load_pc(self, url)
    _times['ply'].append(time.perf_counter() - t)
    return r

def _timed_proc_parquet(self, index):
    t = time.perf_counter()
    r = _orig_proc_parquet(self, index)
    _times['parquet'].append(time.perf_counter() - t)
    return r

def _timed_proc_obstacle(self, index):
    path = self.trajectory_afford_path[index]
    if path in self._ply_cache:
        return self._ply_cache[path], None
    npz_path = path.replace('pointcloud.ply', 'pointcloud_obstacle.npz')
    if _mod.os.path.isfile(npz_path):
        t = time.perf_counter()
        pts = _mod.np.load(npz_path)['obstacle_points']
        _times['ply'].append(time.perf_counter() - t)
        self._ply_cache[path] = pts
        return pts, None
    return _orig_proc_obstacle(self, index)  # fallback (also adds to _ply_cache)

_mod.NavDP_Base_Datset.load_image           = _timed_load_image
_mod.NavDP_Base_Datset.load_depth           = _timed_load_depth
_mod.NavDP_Base_Datset.load_pointcloud      = _timed_load_pc
_mod.NavDP_Base_Datset.process_data_parquet = _timed_proc_parquet
_mod.NavDP_Base_Datset.process_obstacle_points = _timed_proc_obstacle

# ── build dataset & loader ────────────────────────────────────────────────────
from internnav.dataset.navdp_lerobot_dataset import NavDP_Base_Datset, navdp_collate_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

print("Building dataset …")
t0 = time.perf_counter()
dataset = NavDP_Base_Datset(
    config.il.root_dir,
    config.il.dataset_navdp,
    config.il.memory_size,
    config.il.predict_size,
    config.il.batch_size,
    config.il.image_size,
    config.il.scene_scale,
    pixel_channel=config.il.pixel_channel,
    preload=config.il.preload,
    random_digit=config.il.random_digit,
    prior_sample=config.il.prior_sample,
)
print(f"Dataset built in {time.perf_counter()-t0:.1f}s  |  {len(dataset)} samples\n")

# prefetch_factor: match trainer logic (config override → fallback 2)
cfg_pf = getattr(config.il, 'prefetch_factor', None)
prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else (cfg_pf if cfg_pf is not None else 2)
cfg_pw = getattr(config.il, 'persistent_workers', None)

def make_loader(nw, pin_memory, world_size=args.world_size, rank=0, shuffle=True):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=shuffle, seed=1234)
    pw = cfg_pw if cfg_pw is not None else (nw > 0)
    return DataLoader(
        dataset,
        batch_size=config.il.batch_size,
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=navdp_collate_fn,
        prefetch_factor=prefetch_factor if nw > 0 else None,
        persistent_workers=pw,
    )

# ── Phase 1: per-call I/O timing — call low-level methods directly ─────────────
# DataLoader workers have separate memory; __getitem__ has pre-existing data issues
# in some episodes. Time the I/O methods directly on the first N unique paths.
IO_SAMPLES = 50
print(f"Phase 1: per-call I/O timing (direct method calls, {IO_SAMPLES} samples) …")
unique_ply     = list(dict.fromkeys(dataset.trajectory_afford_path))[:IO_SAMPLES]
unique_parquet = list(dict.fromkeys(dataset.trajectory_data_dir))[:IO_SAMPLES]
unique_rgb     = [dataset.trajectory_rgb_path[i][0] for i in range(min(IO_SAMPLES, len(dataset)))]
unique_depth   = [dataset.trajectory_depth_path[i][0] for i in range(min(IO_SAMPLES, len(dataset)))]

for path in unique_ply:
    # clear cache so each call hits disk
    dataset._ply_cache.pop(path, None)
    dataset.process_obstacle_points(dataset.trajectory_afford_path.index(path))
for path in unique_parquet:
    dataset._parquet_cache.pop(path, None)
    dataset.process_data_parquet(dataset.trajectory_data_dir.index(path))
for path in unique_rgb:
    dataset.load_image(path)
for path in unique_depth:
    dataset.load_depth(path)
print(f"  done ({IO_SAMPLES} samples per method)", flush=True)

# ── Phase 2: throughput with requested num_workers ────────────────────────────
nw = args.num_workers
print(f"\nPhase 2: throughput timing (num_workers={nw}, prefetch_factor={prefetch_factor}, {args.num_batches} batches) …\n")
loader = make_loader(nw=nw, pin_memory=True, shuffle=True)
batch_times = []
loader_iter = iter(loader)
done = 0
while done < args.num_batches:
    t0 = time.perf_counter()
    try:
        _ = next(loader_iter)
        batch_times.append(time.perf_counter() - t0)
        done += 1
        print(f"  batch {done:>3}/{args.num_batches}  {batch_times[-1]*1000:.0f} ms", flush=True)
    except StopIteration:
        break
    except Exception:
        pass  # skip bad samples

# ── report ────────────────────────────────────────────────────────────────────
def stats(label, arr):
    if not arr:
        print(f"  {label:<22}  n/a")
        return
    a = np.array(arr) * 1000  # → ms
    print(f"  {label:<22}  n={len(a):>6}  "
          f"mean={a.mean():6.1f}ms  "
          f"p50={np.median(a):6.1f}ms  "
          f"p95={np.percentile(a,95):6.1f}ms  "
          f"max={a.max():6.1f}ms  "
          f"total={a.sum()/1000:5.1f}s")

print(f"\n{'='*65}")
print(f"  Per-call I/O timing")
print(f"{'='*65}")
stats("PLY read (open3d)",   _times['ply'])
stats("PNG read RGB",        _times['png_rgb'])
stats("PNG read depth",      _times['png_dep'])
stats("parquet read",        _times['parquet'])

# per-sample totals
n = min(len(_times['ply']), len(_times['parquet']))
if n:
    # images per sample = png_rgb calls / parquet calls
    imgs_per_sample = len(_times['png_rgb']) / max(len(_times['parquet']), 1)
    print(f"\n  ~{imgs_per_sample:.1f} RGB image reads per sample")

print(f"\n{'='*65}")
print(f"  Batch-level timing  (batch_size={config.il.batch_size}, workers={nw})")
print(f"{'='*65}")
stats("batch wall time",     batch_times)

avg_ms = np.mean(batch_times) * 1000 if batch_times else 0
print(f"\n  Throughput : {config.il.batch_size / np.mean(batch_times):.1f} samples/s")
print(f"  If GPU step < {avg_ms:.0f} ms → DataLoader IS the bottleneck")
print(f"  If GPU step > {avg_ms:.0f} ms → GPU compute / DDP sync is the bottleneck")
print(f"{'='*65}\n")
