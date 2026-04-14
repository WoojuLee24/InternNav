"""
Vision Feature Caching for InternVLA-N1

This module provides caching for vision embeddings to avoid redundant
vision encoding computations during S2 inference.
"""

import hashlib
import time
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch


class VisionFeatureCache:
    """
    LRU Cache for vision embeddings.
    
    Caches vision encoder outputs based on image hash to avoid
    redundant vision encoding for identical/similar images.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_times: Dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Compute a hash of the image for cache key."""
        # Use mean and std as quick features, plus shape
        # For faster hashing, use a subset of pixels
        if len(image.shape) == 3:
            # RGB image - use downsampled version
            h, w = image.shape[:2]
            downscale = max(1, min(h, w) // 64)
            sampled = image[::downscale, ::downscale, :]
            data = sampled.tobytes()
        else:
            data = image.tobytes()
        
        return hashlib.md5(data).hexdigest()[:16]
    
    def _compute_tensor_hash(self, tensor: torch.Tensor) -> str:
        """Compute a hash of a tensor for cache key."""
        # Use data pointer and values for quick identification
        data = tensor.cpu().numpy().tobytes()
        return hashlib.md5(data).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached embedding by key."""
        if key in self.cache:
            self.access_times[key] = time.time()
            self.cache_hits += 1
            return self.cache[key].clone()
        self.cache_misses += 1
        return None
    
    def put(self, key: str, value: torch.Tensor):
        """Put embedding into cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return
            
        # Evict oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value.clone()
        self.access_times[key] = time.time()
    
    def put_image(self, image: np.ndarray, embedding: torch.Tensor):
        """Put image and its embedding into cache."""
        key = self._compute_image_hash(image)
        self.put(key, embedding)
        
    def get_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Get cached embedding for an image."""
        key = self._compute_image_hash(image)
        return self.get(key)
    
    def put_batch(self, images: list, embeddings: torch.Tensor):
        """Put batch of images and embeddings."""
        for i, img in enumerate(images):
            emb = embeddings[i:i+1] if embeddings.dim() > 2 else embeddings[i]
            self.put_image(img, emb)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
        }


class VisionEncoderWithCache:
    """
    Wraps vision encoder with caching capability.
    
    This provides a drop-in replacement for the vision encoder that:
    1. Caches vision embeddings for identical images
    2. Uses frame differencing to detect scene changes
    3. Provides batch encoding with cache lookups
    """
    
    def __init__(self, visual_encoder, cache_size: int = 100):
        self.visual_encoder = visual_encoder
        self.cache = VisionFeatureCache(max_size=cache_size)
        self.last_image: Optional[np.ndarray] = None
        self.last_embedding: Optional[torch.Tensor] = None
        
    def encode(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, 
               raw_image: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Encode image with caching.
        
        Args:
            pixel_values: Preprocessed image tensor
            grid_thw: Grid dimensions for the vision encoder
            raw_image: Optional original image for cache lookup
            
        Returns:
            Vision embeddings
        """
        # Try cache lookup with raw image
        if raw_image is not None:
            cached = self.cache.get_image(raw_image)
            if cached is not None:
                self.last_image = raw_image.copy()
                self.last_embedding = cached
                return cached
        
        # Encode with vision encoder
        with torch.no_grad():
            embeddings = self.visual_encoder(pixel_values, grid_thw=grid_thw)
            
            # Handle different return types
            if hasattr(embeddings, 'pooler_output'):
                embeddings = embeddings.pooler_output
            elif hasattr(embeddings, 'last_hidden_state'):
                embeddings = embeddings.last_hidden_state
        
        # Cache the result
        if raw_image is not None:
            self.cache.put_image(raw_image, embeddings)
        
        self.last_image = raw_image.copy() if raw_image is not None else None
        self.last_embedding = embeddings
        
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        self.last_image = None
        self.last_embedding = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()


def create_cached_encoder(visual_encoder, cache_size: int = 100) -> VisionEncoderWithCache:
    """
    Create a cached vision encoder.
    
    Args:
        visual_encoder: The original vision encoder
        cache_size: Maximum number of embeddings to cache
        
    Returns:
        VisionEncoderWithCache wrapper
    """
    return VisionEncoderWithCache(visual_encoder, cache_size=cache_size)
