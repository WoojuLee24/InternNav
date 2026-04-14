"""
Performance Profiling Module for InternVLA Real-World Navigation

This module provides comprehensive profiling and benchmarking for the dual-system navigation pipeline.
It tracks timing for each module and outputs performance metrics to help identify bottlenecks.

Usage:
    from performance_profiler import PerformanceProfiler, get_profiler
    
    # Get singleton profiler
    profiler = get_profiler()
    
    # Or create new instance
    profiler = PerformanceProfiler()
    
    # Mark timing points
    profiler.start("s2_preprocess")
    # ... do work ...
    profiler.end("s2_preprocess")
    
    # Get metrics
    stats = profiler.get_stats()
    print(stats)
"""

import time
import threading
import json
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TimingRecord:
    """Record of a single timing measurement."""
    name: str
    duration_ms: float
    timestamp: float
    thread_id: int
    iteration: int


@dataclass 
class ModuleStats:
    """Statistics for a single module."""
    name: str
    count: int
    total_ms: float
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    p95_ms: float
    p99_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "std_ms": self.std_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms
        }


@dataclass
class PipelineStats:
    """Statistics for the complete pipeline."""
    total_iterations: int
    total_time_ms: float
    mean_time_ms: float
    frequency_hz: float
    throughput_fps: float
    
    def to_dict(self) -> Dict:
        return {
            "total_iterations": self.total_iterations,
            "total_time_ms": self.total_time_ms,
            "mean_time_ms": self.mean_time_ms,
            "frequency_hz": self.frequency_hz,
            "throughput_fps": self.throughput_fps
        }


class PerformanceProfiler:
    """
    Comprehensive performance profiler for the dual-system navigation pipeline.
    
    Tracks timing for all modules and computes statistics including:
    - Per-module timing (mean, median, std, p95, p99)
    - Pipeline frequency (Hz)
    - Module dependencies and call patterns
    
    Thread-safe for use in multi-threaded environments.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single profiler instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._records: Dict[str, List[float]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}
        self._start_times_per_thread: Dict[int, Dict[str, float]] = defaultdict(dict)
        self._iteration_records: List[Dict[str, float]] = []
        self._current_iteration: Dict[str, float] = {}
        self._iteration_lock = threading.Lock()
        self._timing_lock = threading.Lock()
        
        self._iteration_count = 0
        self._start_time = time.time()
        self._last_iteration_end = self._start_time
        
        # Module definitions with expected ranges
        self._module_info = {
            # System 2 (VLM) modules
            "s2_preprocess": {"expected_ms": 5, "description": "Image preprocessing for S2"},
            "s2_prompt_build": {"expected_ms": 1, "description": "Prompt template construction"},
            "s2_processor": {"expected_ms": 10, "description": "HuggingFace processor call"},
            "s2_model_forward": {"expected_ms": 300, "description": "VLM forward pass (main bottleneck)"},
            "s2_token_decode": {"expected_ms": 5, "description": "Token decoding"},
            "s2_latent_extract": {"expected_ms": 20, "description": "Latent plan extraction"},
            "s2_total": {"expected_ms": 350, "description": "Total S2 time"},
            
            # System 1 (DiT) modules  
            "s1_preprocess": {"expected_ms": 5, "description": "Image preprocessing for S1"},
            "s1_rgb_encoding": {"expected_ms": 10, "description": "RGB feature extraction"},
            "s1_diffusion": {"expected_ms": 30, "description": "Diffusion process"},
            "s1_postprocess": {"expected_ms": 5, "description": "Action extraction"},
            "s1_total": {"expected_ms": 50, "description": "Total S1 time"},
            
            # Pipeline modules
            "http_request_parse": {"expected_ms": 2, "description": "HTTP request parsing"},
            "image_decode": {"expected_ms": 5, "description": "Image decoding"},
            "network_transfer": {"expected_ms": 20, "description": "Network transfer time"},
            "total_pipeline": {"expected_ms": 400, "description": "Total end-to-end pipeline"},
        }
        
        # Performance targets (from optimization goals)
        self._targets = {
            "s2_total": 100,  # Target: 100ms for S2 (from ~350ms)
            "s1_total": 10,   # Target: 10ms for S1 (from ~50ms)
            "total_pipeline": 20,  # Target: 20ms = 50Hz (from ~400ms)
        }
    
    def start(self, module_name: str, thread_specific: bool = True) -> None:
        """
        Start timing for a module.
        
        Args:
            module_name: Name of the module to time
            thread_specific: If True, tracks timing per-thread for multi-threading
        """
        current_time = time.time()
        
        with self._timing_lock:
            if thread_specific:
                thread_id = threading.current_thread().ident
                if thread_id not in self._start_times_per_thread:
                    self._start_times_per_thread[thread_id] = {}
                self._start_times_per_thread[thread_id][module_name] = current_time
            else:
                self._start_times[module_name] = current_time
    
    def end(self, module_name: str, thread_specific: bool = True) -> Optional[float]:
        """
        End timing for a module and record the duration.
        
        Args:
            module_name: Name of the module being timed
            thread_specific: If True, uses per-thread timing
            
        Returns:
            Duration in milliseconds, or None if start was not called
        """
        current_time = time.time()
        
        with self._timing_lock:
            start_time = None
            
            if thread_specific:
                thread_id = threading.current_thread().ident
                if thread_id in self._start_times_per_thread:
                    start_time = self._start_times_per_thread[thread_id].pop(module_name, None)
            else:
                start_time = self._start_times.pop(module_name, None)
            
            if start_time is not None:
                duration_ms = (current_time - start_time) * 1000
                self._records[module_name].append(duration_ms)
                
                # Also record in current iteration for pipeline timing
                with self._iteration_lock:
                    self._current_iteration[module_name] = duration_ms
                
                return duration_ms
            
            return None
    
    def mark(self, module_name: str, duration_ms: float) -> None:
        """
        Mark a timing value directly (for cases where timing is done externally).
        
        Args:
            module_name: Name of the module
            duration_ms: Duration in milliseconds
        """
        with self._timing_lock:
            self._records[module_name].append(duration_ms)
            
            with self._iteration_lock:
                self._current_iteration[module_name] = duration_ms
    
    def start_iteration(self) -> int:
        """Start a new pipeline iteration."""
        with self._iteration_lock:
            self._current_iteration = {}
            self._iteration_count += 1
            return self._iteration_count
    
    def end_iteration(self) -> Dict[str, float]:
        """End current iteration and record timings."""
        with self._iteration_lock:
            iteration_end = time.time()
            
            # Calculate total pipeline time
            if self._last_iteration_end > 0:
                pipeline_duration_ms = (iteration_end - self._last_iteration_end) * 1000
                self._current_iteration["pipeline_interval_ms"] = pipeline_duration_ms
            
            # Record iteration
            record = {
                "iteration": self._iteration_count,
                "timestamp": iteration_end,
                **self._current_iteration
            }
            self._iteration_records.append(record)
            
            self._last_iteration_end = iteration_end
            
            # Return copy of iteration data
            return dict(self._current_iteration)
    
    def get_module_stats(self, module_name: str) -> Optional[ModuleStats]:
        """Get statistics for a specific module."""
        with self._timing_lock:
            if module_name not in self._records or len(self._records[module_name]) == 0:
                return None
            
            values = self._records[module_name]
            
            # Calculate statistics
            mean_ms = statistics.mean(values)
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            return ModuleStats(
                name=module_name,
                count=n,
                total_ms=sum(values),
                mean_ms=mean_ms,
                median_ms=statistics.median(values),
                min_ms=min(values),
                max_ms=max(values),
                std_ms=statistics.stdev(values) if n > 1 else 0.0,
                p95_ms=sorted_values[int(n * 0.95)] if n >= 20 else mean_ms,
                p99_ms=sorted_values[int(n * 0.99)] if n >= 100 else mean_ms
            )
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all modules."""
        stats = {
            "profiler_info": {
                "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
                "uptime_seconds": time.time() - self._start_time,
                "total_iterations": self._iteration_count,
            },
            "module_stats": {},
            "pipeline_stats": {},
            "targets": self._targets,
            "achievement": {}
        }
        
        # Module statistics
        for module_name in self._records:
            module_stats = self.get_module_stats(module_name)
            if module_stats:
                stats["module_stats"][module_name] = module_stats.to_dict()
        
        # Pipeline statistics
        if self._iteration_count > 0:
            pipeline_times = [
                r.get("pipeline_interval_ms", 0) 
                for r in self._iteration_records 
                if "pipeline_interval_ms" in r
            ]
            
            if pipeline_times:
                stats["pipeline_stats"] = PipelineStats(
                    total_iterations=self._iteration_count,
                    total_time_ms=sum(pipeline_times),
                    mean_time_ms=statistics.mean(pipeline_times),
                    frequency_hz=1000.0 / statistics.mean(pipeline_times) if pipeline_times else 0,
                    throughput_fps=1000.0 / statistics.mean(pipeline_times) if pipeline_times else 0
                ).to_dict()
        
        # Achievement vs targets
        for module_name, target_ms in self._targets.items():
            if module_name in stats["module_stats"]:
                actual_ms = stats["module_stats"][module_name]["mean_ms"]
                achievement_pct = (target_ms / actual_ms * 100) if actual_ms > 0 else 0
                stats["achievement"][module_name] = {
                    "target_ms": target_ms,
                    "actual_ms": actual_ms,
                    "achievement_pct": achievement_pct,
                    "meets_target": actual_ms <= target_ms
                }
        
        return stats
    
    def print_summary(self) -> None:
        """Print a formatted summary of performance statistics."""
        stats = self.get_all_stats()
        
        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILING SUMMARY")
        print("=" * 80)
        
        # Profiler info
        print(f"\nProfiler Info:")
        print(f"  Uptime: {stats['profiler_info']['uptime_seconds']:.1f}s")
        print(f"  Total Iterations: {stats['profiler_info']['total_iterations']}")
        
        # Pipeline stats
        if stats["pipeline_stats"]:
            ps = stats["pipeline_stats"]
            print(f"\nPipeline Performance:")
            print(f"  Mean Time: {ps['mean_time_ms']:.2f}ms")
            print(f"  Frequency: {ps['frequency_hz']:.2f}Hz")
            print(f"  Target: 40-50Hz (20-25ms)")
        
        # Module breakdown
        print(f"\n{'Module':<25} {'Count':>8} {'Mean':>10} {'Median':>10} {'P95':>10} {'Max':>10}")
        print("-" * 85)
        
        # Sort by importance
        priority_order = [
            "total_pipeline", "s2_total", "s1_total", 
            "s2_model_forward", "s1_diffusion",
            "http_request_parse", "network_transfer"
        ]
        
        shown_modules = set()
        for module_name in priority_order:
            if module_name in stats["module_stats"] and module_name not in shown_modules:
                ms = stats["module_stats"][module_name]
                info = self._module_info.get(module_name, {})
                desc = info.get("description", module_name)
                print(f"{desc:<25} {ms['count']:>8} {ms['mean_ms']:>9.2f}ms {ms['median_ms']:>9.2f}ms {ms['p95_ms']:>9.2f}ms {ms['max_ms']:>9.2f}ms")
                shown_modules.add(module_name)
        
        # Achievement vs targets
        print(f"\n{'Target Achievement:':<40}")
        print("-" * 50)
        for module_name, data in stats.get("achievement", {}).items():
            status = "✓" if data["meets_target"] else "✗"
            print(f"  {status} {module_name}: {data['actual_ms']:.1f}ms (target: {data['target_ms']}ms, {data['achievement_pct']:.1f}%)")
        
        print("\n" + "=" * 80)
    
    def export_json(self, filepath: str) -> None:
        """Export statistics to JSON file."""
        stats = self.get_all_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Performance stats exported to {filepath}")
    
    def reset(self) -> None:
        """Reset all profiling data."""
        with self._timing_lock:
            self._records.clear()
            self._start_times.clear()
            self._start_times_per_thread.clear()
            
        with self._iteration_lock:
            self._iteration_records.clear()
            self._current_iteration.clear()
            
        self._iteration_count = 0
        self._start_time = time.time()
        self._last_iteration_end = self._start_time
    
    def get_current_hz(self, window_size: int = 10) -> float:
        """Get current estimated Hz based on recent iterations."""
        with self._iteration_lock:
            if len(self._iteration_records) < 2:
                return 0.0
            
            recent = self._iteration_records[-window_size:]
            intervals = [r.get("pipeline_interval_ms", 0) for r in recent if "pipeline_interval_ms" in r]
            
            if not intervals or sum(intervals) == 0:
                return 0.0
            
            mean_interval = statistics.mean(intervals)
            return 1000.0 / mean_interval if mean_interval > 0 else 0.0


def get_profiler() -> PerformanceProfiler:
    """Get the singleton profiler instance."""
    return PerformanceProfiler()


class ProfilerContext:
    """Context manager for easy profiling."""
    
    def __init__(self, module_name: str, profiler: Optional[PerformanceProfiler] = None, 
                 verbose: bool = False):
        self.module_name = module_name
        self.profiler = profiler or get_profiler()
        self.verbose = verbose
        self.duration_ms = 0.0
    
    def __enter__(self):
        self.profiler.start(self.module_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = self.profiler.end(self.module_name)
        if self.verbose and self.duration_ms is not None:
            print(f"[{self.module_name}] {self.duration_ms:.2f}ms")


# Decorator for profiling functions
def profiled(module_name: Optional[str] = None, profiler: Optional[PerformanceProfiler] = None):
    """Decorator to profile a function."""
    def decorator(func):
        name = module_name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            p = profiler or get_profiler()
            p.start(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                p.end(name)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# Global profiler instance
_profiler = None

def init_profiler() -> PerformanceProfiler:
    """Initialize the global profiler."""
    global _profiler
    _profiler = PerformanceProfiler()
    return _profiler


if __name__ == "__main__":
    # Test the profiler
    profiler = PerformanceProfiler()
    
    # Simulate some profiling
    profiler.start("test_module_1")
    time.sleep(0.01)  # 10ms
    profiler.end("test_module_1")
    
    profiler.start("test_module_2")
    time.sleep(0.005)  # 5ms
    profiler.end("test_module_2")
    
    # Print summary
    profiler.print_summary()
