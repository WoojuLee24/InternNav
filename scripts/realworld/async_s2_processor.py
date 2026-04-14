"""
Async S2 Processing for InternVLA-N1 Dual-System Navigation

Allows S1 (controller) to run at high frequency (50+ Hz)
while S2 (planner) runs asynchronously in background thread.

Features:
- Double buffering for S2 outputs
- Non-blocking S1 execution
- Thread-safe communication
- Configurable update rates
"""

import copy
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch


@dataclass
class S2Command:
    """Command for S2 worker thread."""
    rgb: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    instruction: str
    intrinsic: np.ndarray
    look_down: bool = False
    priority: int = 0


@dataclass 
class S2Result:
    """Result from S2 processing."""
    output_action: Optional[List[int]] = None
    output_latent: Optional[torch.Tensor] = None
    output_pixel: Optional[List[int]] = None
    pixel_goal_rgb: Optional[np.ndarray] = None
    pixel_goal_depth: Optional[np.ndarray] = None
    llm_output: str = ""
    episode_idx: int = 0
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    valid: bool = True


class AsyncS2Processor:
    """
    Asynchronous S2 (planner) processor with double buffering.
    
    Runs S2 in background thread while S1 executes synchronously
    at high frequency using the latest available S2 result.
    """
    
    def __init__(
        self,
        step_func: Callable,
        device: torch.device,
        s2_update_hz: float = 2.0,
        buffer_size: int = 2
    ):
        """
        Initialize async S2 processor.
        
        Args:
            step_func: S2 step function (agent.step_s2)
            device: CUDA device
            s2_update_hz: Target S2 update frequency (Hz)
            buffer_size: Number of result buffers
        """
        self.step_func = step_func
        self.device = device
        self.s2_update_hz = s2_update_hz
        self.s2_period = 1.0 / s2_update_hz
        
        self.command_queue: queue.Queue[S2Command] = queue.Queue(maxsize=2)
        self.result_buffers: List[Optional[S2Result]] = [None] * buffer_size
        self.active_buffer = 0
        self.lock = threading.RLock()
        
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        
        self.stats = {
            's2_updates': 0,
            's2_avg_latency_ms': 0,
            's2_total_time_ms': 0,
            'queue_size': 0,
            'dropped_commands': 0
        }
    
    def start(self):
        """Start the S2 worker thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"[AsyncS2] Started (target: {self.s2_update_hz} Hz)")
    
    def stop(self):
        """Stop the S2 worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
        print("[AsyncS2] Stopped")
    
    def request_s2(self, command: S2Command):
        """
        Request S2 processing (non-blocking).
        
        Args:
            command: S2Command with processing parameters
        """
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            self.stats['dropped_commands'] += 1
            try:
                self.command_queue.get_nowait()
                self.command_queue.put_nowait(command)
            except queue.Full:
                pass
    
    def get_latest_result(self) -> Optional[S2Result]:
        """
        Get the latest S2 result (non-blocking).
        
        Returns:
            Latest S2Result or None if no result available
        """
        with self.lock:
            return self.result_buffers[self.active_buffer]
    
    def _worker_loop(self):
        """Main worker loop running in background thread."""
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
                
                t_start = time.time()
                action, latent, pixel = self.step_func(
                    command.rgb, command.depth, command.pose,
                    command.instruction, command.intrinsic, command.look_down
                )
                latency_ms = (time.time() - t_start) * 1000
                
                result = S2Result(
                    output_action=action,
                    output_latent=latent,
                    output_pixel=pixel,
                    llm_output=getattr(self.step_func, 'llm_output', ''),
                    episode_idx=command.pose[0, 0] if isinstance(command.pose, np.ndarray) else 0,
                    timestamp=time.time(),
                    latency_ms=latency_ms,
                    valid=True
                )
                
                with self.lock:
                    next_buffer = (self.active_buffer + 1) % len(self.result_buffers)
                    self.result_buffers[next_buffer] = result
                    self.active_buffer = next_buffer
                    
                    self.stats['s2_updates'] += 1
                    self.stats['s2_total_time_ms'] += latency_ms
                    self.stats['s2_avg_latency_ms'] = (
                        self.stats['s2_total_time_ms'] / self.stats['s2_updates']
                    )
                    self.stats['queue_size'] = self.command_queue.qsize()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncS2] Error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self.lock:
            stats = copy.deepcopy(self.stats)
            stats['s2_current_hz'] = (
                1000.0 / stats['s2_avg_latency_ms'] if stats['s2_avg_latency_ms'] > 0 else 0
            )
            return stats


class DualBufferS2:
    """
    Double-buffered S2 output for seamless S1/S2 coordination.
    
    Provides atomic swap between buffers to avoid race conditions
    when S1 reads S2 output while S2 is updating.
    """
    
    def __init__(self):
        self.buffers: List[Optional[S2Result]] = [None, None]
        self.write_idx = 0
        self.read_idx = 1
        self.lock = threading.RLock()
        self.generation = 0
    
    def write(self, result: S2Result):
        """Write new result (called from S2 thread)."""
        with self.lock:
            self.buffers[self.write_idx] = result
            self.write_idx, self.read_idx = self.read_idx, self.write_idx
            self.generation += 1
    
    def read(self) -> Optional[S2Result]:
        """Read latest result (called from S1 thread)."""
        with self.lock:
            return self.buffers[self.read_idx]
    
    def peek(self) -> Optional[S2Result]:
        """Peek at result without locking."""
        return self.buffers[self.read_idx]


class RateLimitedS2Requester:
    """
    Rate-limited S2 requester that respects target frequency.
    
    Ensures S2 is not called more frequently than specified,
    improving GPU utilization and reducing unnecessary computation.
    """
    
    def __init__(self, target_hz: float = 2.0, burst_allowance: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            target_hz: Target S2 frequency
            burst_allowance: Allow burst of N requests beyond target rate
        """
        self.target_period = 1.0 / target_hz
        self.burst_allowance = burst_allowance
        self.burst_count = 0
        self.last_request_time = 0.0
        self.lock = threading.Lock()
    
    def should_request(self, force: bool = False) -> bool:
        """
        Check if S2 request should be made.
        
        Args:
            force: Force request regardless of rate limit
            
        Returns:
            True if request should proceed
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            
            if force:
                self.last_request_time = now
                return True
            
            if elapsed >= self.target_period:
                if self.burst_count > 0:
                    self.burst_count -= 1
                self.last_request_time = now
                return True
            
            if self.burst_count < self.burst_allowance:
                self.burst_count += 1
                self.last_request_time = now
                return True
            
            return False
    
    def reset(self):
        """Reset rate limiter state."""
        with self.lock:
            self.last_request_time = 0.0
            self.burst_count = 0


def create_async_s2_processor(
    agent_step_func: Callable,
    device: torch.device,
    target_hz: float = 2.0
) -> Tuple[AsyncS2Processor, RateLimitedS2Requester]:
    """
    Factory function to create async S2 processor with rate limiter.
    
    Args:
        agent_step_func: Agent's S2 step function
        device: CUDA device
        target_hz: Target S2 update frequency
        
    Returns:
        Tuple of (AsyncS2Processor, RateLimitedS2Requester)
    """
    processor = AsyncS2Processor(
        step_func=agent_step_func,
        device=device,
        s2_update_hz=target_hz
    )
    
    rate_limiter = RateLimitedS2Requester(
        target_hz=target_hz,
        burst_allowance=1
    )
    
    return processor, rate_limiter
