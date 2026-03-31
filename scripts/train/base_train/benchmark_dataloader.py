"""
DataLoader 병목 측정 스크립트 — I/O vs CPU (CubicSpline) 분리 타이밍

실행:
    python scripts/train/base_train/benchmark_dataloader.py \\
        --model-name navdp_ablation_1node \\
        --num-workers 4 \\
        --batch-size 256 \\
        --n-batches 20 \\
        --use-npy-cache True \\
        --scene-scale 0.01

출력 (per batch):
    [Batch] throughput=XXX samples/s | t_batch=Xs
    [Section] io=Xs/sample  cpu=Xs/sample  other=Xs/sample
"""
import os
import sys
import time
import statistics
import argparse

sys.path.append('./src/diffusion-policy')

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train.base_train.configs import (
    navdp_exp_cfg,
    navdp_1gpu_exp_cfg,
    navdp_1node_exp_cfg,
    navdp_ablation_1node_exp_cfg,
)
from internnav.dataset.navdp_lerobot_dataset import NavDP_Base_Datset, navdp_collate_fn
from internnav.dataset.navdp_webdataset import NavDP_WebDataset


# -----------------------------------------------------------------------
# Timed wrapper — monkey-patches process_image, process_depth, process_actions
# -----------------------------------------------------------------------

class TimedNavDP(NavDP_Base_Datset):
    """
    NavDP_Base_Datset 서브클래스.
    process_image / process_depth (I/O) 와 process_actions (CPU) 를
    각각 타이밍하여 worker 프로세스에서 집계.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 공유 누산기 (worker 프로세스 내 local)
        self._t_io = 0.0
        self._t_cpu = 0.0
        self._t_total = 0.0
        self._n_items = 0

        # --- wrap process_image ---
        _orig_img = self.process_image.__func__  # unbound
        def _timed_image(self_inner, path):
            t = time.perf_counter()
            r = _orig_img(self_inner, path)
            self._t_io += time.perf_counter() - t
            return r
        import types
        self.process_image = types.MethodType(_timed_image, self)

        # --- wrap process_depth ---
        _orig_dep = self.process_depth.__func__
        def _timed_depth(self_inner, path):
            t = time.perf_counter()
            r = _orig_dep(self_inner, path)
            self._t_io += time.perf_counter() - t
            return r
        self.process_depth = types.MethodType(_timed_depth, self)

        # --- wrap process_actions (CubicSpline 포함) ---
        _orig_act = self.process_actions.__func__
        def _timed_actions(self_inner, *a, **kw):
            t = time.perf_counter()
            r = _orig_act(self_inner, *a, **kw)
            self._t_cpu += time.perf_counter() - t
            return r
        self.process_actions = types.MethodType(_timed_actions, self)

    def __getitem__(self, index):
        t0 = time.perf_counter()
        result = super().__getitem__(index)
        self._t_total += time.perf_counter() - t0
        self._n_items += 1

        # 100 samples마다 한 번 출력
        if self._n_items % 100 == 0:
            n = self._n_items
            io_ms  = self._t_io   / n * 1000
            cpu_ms = self._t_cpu  / n * 1000
            tot_ms = self._t_total / n * 1000
            oth_ms = tot_ms - io_ms - cpu_ms
            print(
                f'[TimedNavDP] pid={os.getpid()} n={n}'
                f' | io={io_ms:.1f}ms  cpu(CubicSpline)={cpu_ms:.1f}ms'
                f'  other={oth_ms:.1f}ms  total={tot_ms:.1f}ms/sample'
            )
        return result


# -----------------------------------------------------------------------
# Config helper
# -----------------------------------------------------------------------

SUPPORTED_CFG = {
    'navdp':                navdp_exp_cfg,
    'navdp_1gpu':           navdp_1gpu_exp_cfg,
    'navdp_1node':          navdp_1node_exp_cfg,
    'navdp_ablation_1node': navdp_ablation_1node_exp_cfg,
}


def build_dataset(args, exp_cfg):
    il = exp_cfg.il

    # overrides
    if args.scene_scale is not None:
        il.scene_scale = args.scene_scale
    if args.use_npy_cache is not None:
        il.use_npy_cache = args.use_npy_cache

    use_webdataset = args.use_webdataset
    shard_dir = args.webdataset_shard_dir

    if use_webdataset:
        if not shard_dir:
            raise ValueError('--webdataset-shard-dir required when --use-webdataset True')
        return NavDP_WebDataset(
            il.root_dir,
            shard_dir,
            il.dataset_navdp,
            il.memory_size,
            il.predict_size,
            il.batch_size,
            il.image_size,
            il.scene_scale,
            pixel_channel=il.pixel_channel,
            random_digit=il.random_digit,
            prior_sample=il.prior_sample,
        )
    else:
        # TimedNavDP wraps NavDP_Base_Datset for I/O vs CPU breakdown
        return TimedNavDP(
            il.root_dir,
            il.dataset_navdp,
            il.memory_size,
            il.predict_size,
            il.batch_size,
            il.image_size,
            il.scene_scale,
            pixel_channel=il.pixel_channel,
            preload=getattr(il, 'preload', False),
            random_digit=il.random_digit,
            prior_sample=il.prior_sample,
            use_npy_cache=getattr(il, 'use_npy_cache', True),
        )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='navdp_ablation_1node')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--n-batches', type=int, default=30,
                        help='측정할 배치 수 (초반 warmup 제외)')
    parser.add_argument('--warmup-batches', type=int, default=5,
                        help='측정에서 제외할 warmup 배치 수')
    parser.add_argument('--scene-scale', type=float, default=None)
    parser.add_argument('--use-npy-cache', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use-webdataset', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--webdataset-shard-dir', default=None)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--persistent-workers', type=lambda x: x.lower() == 'true', default=True)
    args = parser.parse_args()

    exp_cfg = SUPPORTED_CFG[args.model_name]
    if args.batch_size:
        exp_cfg.il.batch_size = args.batch_size

    print('=' * 60)
    print(f'  model        : {args.model_name}')
    print(f'  num_workers  : {args.num_workers}')
    print(f'  batch_size   : {exp_cfg.il.batch_size}')
    print(f'  scene_scale  : {args.scene_scale or exp_cfg.il.scene_scale}')
    print(f'  use_npy_cache: {args.use_npy_cache}')
    print(f'  use_webdataset: {args.use_webdataset}')
    print(f'  n_batches    : {args.n_batches}  (warmup={args.warmup_batches})')
    print('=' * 60)

    dataset = build_dataset(args, exp_cfg)
    print(f'Dataset size (×50 replicated): {len(dataset):,}')

    loader = DataLoader(
        dataset,
        batch_size=exp_cfg.il.batch_size,
        num_workers=args.num_workers,
        collate_fn=navdp_collate_fn,
        shuffle=False,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=False,
    )

    batch_times = []
    total_batches = args.warmup_batches + args.n_batches

    print(f'\nRunning {total_batches} batches ({args.warmup_batches} warmup) ...\n')

    loader_iter = iter(loader)
    for i in range(total_batches):
        t0 = time.perf_counter()
        batch = next(loader_iter)
        elapsed = time.perf_counter() - t0

        samples_in_batch = batch[0].shape[0]
        throughput = samples_in_batch / elapsed

        if i < args.warmup_batches:
            status = '[warmup]'
        else:
            batch_times.append(elapsed)
            status = f'[{i - args.warmup_batches + 1:3d}/{args.n_batches}]'

        print(
            f'{status} batch={i+1}  t={elapsed:.3f}s  '
            f'throughput={throughput:.1f} samples/s  '
            f'batch_samples={samples_in_batch}'
        )

    # ---- summary ----
    if batch_times:
        batch_size = exp_cfg.il.batch_size
        mean_t   = statistics.mean(batch_times)
        median_t = statistics.median(batch_times)
        stdev_t  = statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0
        p95_t    = sorted(batch_times)[int(len(batch_times) * 0.95)]

        mean_tp   = batch_size / mean_t
        median_tp = batch_size / median_t

        print()
        print('=' * 60)
        print('  BENCHMARK SUMMARY')
        print(f'  batch_size    : {batch_size}')
        print(f'  num_workers   : {args.num_workers}')
        print(f'  use_npy_cache : {args.use_npy_cache}')
        print(f'  use_webdataset: {args.use_webdataset}')
        print()
        print(f'  Batch time  mean   : {mean_t*1000:.1f} ms')
        print(f'  Batch time  median : {median_t*1000:.1f} ms')
        print(f'  Batch time  stdev  : {stdev_t*1000:.1f} ms')
        print(f'  Batch time  p95    : {p95_t*1000:.1f} ms')
        print()
        print(f'  Throughput  mean   : {mean_tp:.1f} samples/s')
        print(f'  Throughput  median : {median_tp:.1f} samples/s')
        print('=' * 60)
        print()
        print('  [해석]')
        print('  io_ms >> cpu_ms  → I/O 병목  → WebDataset 효과 있음')
        print('  cpu_ms >> io_ms  → CPU 병목  → WebDataset 효과 없음, CubicSpline 최적화 필요')
        print('  (io_ms, cpu_ms 수치는 worker 프로세스 stdout에 출력됨)')
        print('=' * 60)


if __name__ == '__main__':
    main()
