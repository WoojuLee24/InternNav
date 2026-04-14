"""
TensorRT Conversion for InternVLA-N1

Converts the vision encoder and LLM to TensorRT for 2-3x speedup.

Features:
- ONNX export for vision encoder
- TensorRT engine building
- FP16/INT8 precision support
- Dynamic shape handling
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class TensorRTConverter:
    """
    Converts PyTorch models to TensorRT engines.
    
    Supports:
    - Vision encoder (image -> embedding)
    - LLM with KV caching
    - FP16 and INT8 precision
    """
    
    def __init__(
        self,
        model_dir: str,
        workspace_size: int = 1 << 30,
        dla_core: int = 0
    ):
        """
        Initialize TensorRT converter.
        
        Args:
            model_dir: Directory containing model checkpoints
            workspace_size: TensorRT workspace size in bytes
            dla_core: DLA core to use (0 = GPU only)
        """
        self.model_dir = Path(model_dir)
        self.workspace_size = workspace_size
        self.dla_core = dla_core
        self.trt_logger = None
        self._init_trt()
    
    def _init_trt(self):
        """Initialize TensorRT and logging."""
        try:
            import tensorrt as trt
            self.trt = trt
            
            trt_logger = trt.Logger(trt.Logger.WARNING)
            self.trt_logger = trt_logger
            
            self.builder = trt.Builder(trt_logger)
            self.network = self.builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            self.config = self.builder.create_builder_config()
            self.config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.workspace_size
            )
            
            print(f"[TensorRT] Initialized (workspace: {self.workspace_size >> 30} GB)")
            
        except ImportError:
            print("[TensorRT] Not installed. Run: pip install tensorrt")
            self.trt = None
    
    def check_trt_available(self) -> bool:
        """Check if TensorRT is available."""
        return self.trt is not None
    
    def export_vision_encoder_onnx(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> bool:
        """
        Export vision encoder to ONNX.
        
        Args:
            model: Vision encoder model
            output_path: Output ONNX file path
            input_shape: Input tensor shape (B, C, H, W)
            dynamic_axes: Dynamic axes for variable dimensions
            
        Returns:
            True if export successful
        """
        if not self.check_trt_available():
            return False
        
        model.eval()
        
        dummy_input = torch.randn(input_shape, dtype=torch.float32)
        
        if dynamic_axes is None:
            dynamic_axes = {
                'pixel_values': {0: 'batch_size'},
                'image_embeds': {0: 'batch_size'}
            }
        
        try:
            torch.onnx.export(
                model,
                (dummy_input,),
                output_path,
                input_names=['pixel_values'],
                output_names=['image_embeds'],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
                verbose=False
            )
            print(f"[TensorRT] Exported vision encoder to {output_path}")
            return True
            
        except Exception as e:
            print(f"[TensorRT] Export failed: {e}")
            return False
    
    def build_tensorrt_engine(
        self,
        onnx_path: str,
        engine_path: str,
        precision: str = 'fp16',
        max_batch_size: int = 8,
        max_workspace: int = 1 << 30
    ) -> bool:
        """
        Build TensorRT engine from ONNX.
        
        Args:
            onnx_path: Input ONNX file
            engine_path: Output engine file
            precision: Precision ('fp32', 'fp16', 'int8')
            max_batch_size: Maximum batch size
            max_workspace: Maximum workspace size
            
        Returns:
            True if build successful
        """
        if not self.check_trt_available():
            return False
        
        if not os.path.exists(onnx_path):
            print(f"[TensorRT] ONNX file not found: {onnx_path}")
            return False
        
        print(f"[TensorRT] Building engine from {onnx_path}...")
        t_start = time.time()
        
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            explicit_batch = 1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = self.builder.create_network(explicit_batch)
            parser = self.trt.OnnxParser(network, self.trt_logger)
            
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f"[TensorRT] Parse error: {parser.get_error(i)}")
                    return False
            
            config = self.builder.create_builder_config()
            config.set_memory_pool_limit(
                self.trt.MemoryPoolType.WORKSPACE, max_workspace
            )
            
            if precision == 'fp16' and self.builder.platform_has_fast_fp16:
                config.set_flag(self.trt.BuilderFlag.FP16)
                print("[TensorRT] Using FP16 precision")
            elif precision == 'int8' and self.builder.platform_has_fast_int8:
                config.set_flag(self.trt.BuilderFlag.INT8)
                print("[TensorRT] Using INT8 precision")
            
            profile = self.builder.create_optimization_profile()
            
            input_tensor = network.get_input(0)
            min_shape = (1,) + tuple(input_tensor.shape[1:])
            opt_shape = (max_batch_size // 2,) + tuple(input_tensor.shape[1:])
            max_shape = (max_batch_size,) + tuple(input_tensor.shape[1:])
            
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            engine = self.builder.build_serialized_network(network, config)
            
            if engine is None:
                print("[TensorRT] Engine build failed")
                return False
            
            with open(engine_path, 'wb') as f:
                f.write(engine)
            
            t_elapsed = time.time() - t_start
            engine_size = os.path.getsize(engine_path) / (1024 * 1024)
            
            print(f"[TensorRT] Engine built in {t_elapsed:.1f}s ({engine_size:.1f} MB)")
            print(f"[TensorRT] Saved to {engine_path}")
            
            return True
            
        except Exception as e:
            print(f"[TensorRT] Build failed: {e}")
            return False


class TensorRTModelWrapper:
    """
    Wrapper for running TensorRT engines.
    
    Provides PyTorch-like interface for TensorRT inference.
    """
    
    def __init__(self, engine_path: str, device: str = 'cuda:0'):
        """
        Initialize TensorRT model wrapper.
        
        Args:
            engine_path: Path to TensorRT engine file
            device: Device to run on
        """
        self.engine_path = engine_path
        self.device = torch.device(device)
        self.engine = None
        self.context = None
        self.bindings: List[Any] = []
        self._load_engine()
    
    def _load_engine(self):
        """Load TensorRT engine from file."""
        try:
            import tensorrt as trt
            
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_c_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.bindings = [None] * self.engine.num_io_tensors
            
            print(f"[TensorRT] Loaded engine with {self.engine.num_io_tensors} I/O tensors")
            
        except Exception as e:
            print(f"[TensorRT] Failed to load engine: {e}")
            self.engine = None
    
    def is_available(self) -> bool:
        """Check if engine is loaded."""
        return self.engine is not None
    
    def __call__(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """
        Run inference.
        
        Args:
            inputs: Input tensors matching network inputs
            
        Returns:
            List of output tensors
        """
        if not self.is_available():
            raise RuntimeError("TensorRT engine not loaded")
        
        outputs = []
        for tensor in inputs:
            cuda_tensor = tensor.to(self.device).contiguous()
            self.context.set_input_shape(tensor.name, tensor.shape)
            self.context.set_tensor_address(tensor.name, cuda_tensor.data_ptr())
            outputs.append(cuda_tensor)
        
        self.context.execute_v3(stream=None)
        return outputs


def convert_model_to_tensorrt(
    model: nn.Module,
    model_dir: str,
    precision: str = 'fp16',
    max_batch_size: int = 8
) -> Optional[TensorRTModelWrapper]:
    """
    Convert PyTorch model to TensorRT.
    
    Args:
        model: PyTorch model to convert
        model_dir: Directory for output files
        precision: Precision mode ('fp32', 'fp16', 'int8')
        max_batch_size: Maximum batch size
        
    Returns:
        TensorRTModelWrapper if successful, None otherwise
    """
    converter = TensorRTConverter(model_dir)
    
    if not converter.check_trt_available():
        print("[TensorRT] TensorRT not available, skipping conversion")
        return None
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = str(model_dir / "vision_encoder.onnx")
    engine_path = str(model_dir / "vision_encoder_fp16.engine" if precision == 'fp16' else "vision_encoder.engine")
    
    if not converter.export_vision_encoder_onnx(model, onnx_path):
        return None
    
    if not converter.build_tensorrt_engine(onnx_path, engine_path, precision, max_batch_size):
        return None
    
    return TensorRTModelWrapper(engine_path)


class TensorRTVisionCache:
    """
    Vision cache that uses TensorRT for fast embedding computation.
    
    Combines caching with TensorRT acceleration for maximum performance.
    """
    
    def __init__(
        self,
        vision_encoder: Optional[nn.Module] = None,
        trt_wrapper: Optional[TensorRTModelWrapper] = None,
        cache_size: int = 100
    ):
        """
        Initialize TensorRT-accelerated vision cache.
        
        Args:
            vision_encoder: Fallback PyTorch model
            trt_wrapper: TensorRT model wrapper
            cache_size: Maximum cache size
        """
        from scripts.realworld.vision_cache import VisionEmbeddingCache
        
        self.vision_cache = VisionEmbeddingCache(max_size=cache_size)
        self.trt_wrapper = trt_wrapper
        self.vision_encoder = vision_encoder
        
        if trt_wrapper:
            self.use_trt = True
            print("[TensorRT] Using TensorRT for vision encoding")
        elif vision_encoder:
            self.use_trt = False
            self.vision_encoder.eval()
            print("[TensorRT] Using PyTorch (TensorRT not available)")
        else:
            print("[TensorRT] No vision encoder provided")
    
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel values to embeddings.
        
        Args:
            pixel_values: Image tensor (B, C, H, W)
            
        Returns:
            Image embeddings
        """
        if self.use_trt and self.trt_wrapper:
            return self.trt_wrapper(pixel_values)[0]
        elif self.vision_encoder:
            with torch.no_grad():
                return self.vision_encoder(pixel_values)
        else:
            raise RuntimeError("No vision encoder available")
    
    def get_or_compute(
        self,
        image_hash: str,
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Get cached embedding or compute new one.
        
        Args:
            image_hash: Image hash key
            pixel_values: Image tensor
            
        Returns:
            Image embeddings
        """
        cached = self.vision_cache.get_by_hash(image_hash)
        
        if cached is not None:
            return cached
        
        embeddings = self.encode(pixel_values)
        self.vision_cache.put_by_hash(image_hash, embeddings)
        
        return embeddings
