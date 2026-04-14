"""
INT8 Quantization for InternVLA-N1

Post-training quantization for 1.5-2x speedup with minimal accuracy loss.

Features:
- Dynamic quantization (fastest, no calibration)
- Static quantization (PTQ with calibration)
- QAT (Quantization-Aware Training) support
- Per-layer scaling factors
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.quantization import (
        quantize_dynamic,
        prepare,
        convert,
        QConfig,
        default_dynamic_qconfig,
        per_channel_dynamic_qconfig,
    )
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("[Quantization] torch.quantization not available in this PyTorch version")


class DynamicQuantizer:
    """
    Dynamic quantization wrapper.
    
    Quantizes weights to int8 dynamically while keeping activations in fp32.
    No calibration data required.
    """
    
    def __init__(self, dtype: Any = None):
        """
        Initialize dynamic quantizer.
        
        Args:
            dtype: Target dtype for quantized weights
        """
        if dtype is None and QUANTIZATION_AVAILABLE:
            dtype = torch.qint8
        self.dtype = dtype
        self.quantized_model = None
        self.available = QUANTIZATION_AVAILABLE
    
    def quantize(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization to model.
        
        Args:
            model: PyTorch model to quantize
            
        Returns:
            Quantized model
        """
        if not self.available:
            print("[Quantization] Dynamic quantization not available in this PyTorch version")
            return model
        
        print("[Quantization] Applying dynamic INT8 quantization...")
        
        quantized = quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.LSTMCell, nn.GRUCell, nn.GRU},
            dtype=self.dtype
        )
        
        self.quantized_model = quantized
        print("[Quantization] Dynamic quantization complete")
        
        return quantized
    
    def get_model(self) -> Optional[nn.Module]:
        """Get quantized model."""
        return self.quantized_model


class StaticQuantizer:
    """
    Static quantization with calibration.
    
    Requires calibration dataset for better accuracy.
    """
    
    def __init__(
        self,
        qconfig: Optional[Any] = None,
        example_inputs: Optional[Tuple[torch.Tensor, ...]] = None
    ):
        """
        Initialize static quantizer.
        
        Args:
            qconfig: Quantization configuration
            example_inputs: Example inputs for tracing
        """
        if not QUANTIZATION_AVAILABLE:
            print("[Quantization] Static quantization not available")
            self.qconfig = None
        else:
            self.qconfig = qconfig or torch.quantization.get_default_qconfig('fbgemm')
        self.example_inputs = example_inputs
        self.quantized_model = None
        self.prepared_model = None
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for static quantization.
        
        Args:
            model: PyTorch model
            
        Returns:
            Prepared model
        """
        if not QUANTIZATION_AVAILABLE:
            return model
            
        model.eval()
        
        if self.example_inputs:
            self.prepared_model = prepare(model, self.qconfig, example_inputs=self.example_inputs)
        else:
            self.prepared_model = prepare(model, self.qconfig)
        
        print("[Quantization] Model prepared for static quantization")
        return self.prepared_model
    
    def calibrate(self, dataloader, num_batches: int = 100):
        """
        Calibrate model with representative dataset.
        
        Args:
            dataloader: Calibration data loader
            num_batches: Number of batches to calibrate
        """
        if self.prepared_model is None:
            raise RuntimeError("Model not prepared. Call prepare_model first.")
        
        print(f"[Quantization] Calibrating with {num_batches} batches...")
        
        self.prepared_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = [b.to(next(self.prepared_model.parameters()).device) 
                              for b in batch if torch.is_tensor(b)]
                    self.prepared_model(*inputs)
                elif torch.is_tensor(batch):
                    self.prepared_model(batch.to(self.prepared_model.device))
        
        print("[Quantization] Calibration complete")
    
    def convert(self) -> nn.Module:
        """
        Convert prepared model to quantized model.
        
        Returns:
            Quantized model
        """
        if self.prepared_model is None:
            raise RuntimeError("Model not prepared. Call prepare_model first.")
        
        self.quantized_model = convert(self.prepared_model)
        print("[Quantization] Model converted to INT8")
        
        return self.quantized_model
    
    def quantize_full(
        self,
        model: nn.Module,
        dataloader,
        num_batches: int = 100
    ) -> nn.Module:
        """
        Full static quantization pipeline.
        
        Args:
            model: PyTorch model
            dataloader: Calibration data loader
            num_batches: Number of calibration batches
            
        Returns:
            Quantized model
        """
        self.prepare_model(model)
        self.calibrate(dataloader, num_batches)
        return self.convert()


class QuantizedVisionEncoder:
    """
    Quantized vision encoder with INT8 acceleration.
    
    Provides methods for quantizing vision encoders with
    automatic fallback to fp16/fp32.
    """
    
    def __init__(self, model: Optional[nn.Module] = None, device: str = 'cuda:0'):
        """
        Initialize quantized vision encoder.
        
        Args:
            model: Vision encoder model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.quantized = False
        self.quantized_model = None
    
    def quantize_dynamic(self) -> nn.Module:
        """Apply dynamic quantization."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        quantizer = DynamicQuantizer()
        self.quantized_model = quantizer.quantize(self.model)
        self.quantized = True
        
        return self.quantized_model
    
    def quantize_static(
        self,
        calibration_inputs: List[torch.Tensor],
        batch_size: int = 8
    ) -> nn.Module:
        """
        Apply static quantization with calibration.
        
        Args:
            calibration_inputs: List of calibration tensors
            batch_size: Batch size for calibration
            
        Returns:
            Quantized model
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        class DummyDataset:
            def __init__(self, inputs, batch_size):
                self.inputs = inputs
                self.batch_size = batch_size
            
            def __iter__(self):
                for i in range(0, len(self.inputs), self.batch_size):
                    batch = self.inputs[i:i + self.batch_size]
                    yield torch.stack(batch) if isinstance(batch[0], torch.Tensor) else batch
        
        quantizer = StaticQuantizer()
        self.quantized_model = quantizer.quantize_full(
            self.model, 
            DummyDataset(calibration_inputs, batch_size),
            num_batches=len(calibration_inputs) // batch_size
        )
        self.quantized = True
        
        return self.quantized_model
    
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            pixel_values: Image tensor (B, C, H, W)
            
        Returns:
            Embeddings
        """
        model = self.quantized_model if self.quantized else self.model
        
        if model is None:
            raise RuntimeError("No model available")
        
        model.eval()
        
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device)
            embeddings = model(pixel_values)
        
        return embeddings


class QuantizationAwareTrainer:
    """
    Quantization-aware training wrapper.
    
    Simulates quantization effects during training for better accuracy.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda:0',
        qconfig: Optional[QConfig] = None
    ):
        """
        Initialize QAT trainer.
        
        Args:
            model: Model to train
            device: Training device
            qconfig: Quantization config
        """
        self.model = model
        self.device = device
        self.qconfig = qconfig or torch.quantization.get_default_qat_qconfig('fbgemm')
        self.quantized_model = None
    
    def prepare_qat(self) -> nn.Module:
        """
        Prepare model for QAT.
        
        Returns:
            Model ready for QAT training
        """
        self.model.train()
        
        self.quantized_model = prepare(
            self.model.to(self.device),
            self.qconfig,
            example_inputs=(torch.randn(1, 3, 224, 224).to(self.device),),
            inplace=True
        )
        
        print("[QAT] Model prepared for quantization-aware training")
        return self.quantized_model
    
    def finetune(
        self,
        train_loader,
        num_epochs: int = 1,
        lr: float = 1e-5
    ) -> nn.Module:
        """
        Finetune with quantization awareness.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            lr: Learning rate
            
        Returns:
            Trained quantized model
        """
        if self.quantized_model is None:
            self.prepare_qat()
        
        optimizer = torch.optim.Adam(self.quantized_model.parameters(), lr=lr)
        
        self.quantized_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if len(batch) > 1 else inputs
                else:
                    inputs = batch.to(self.device)
                    targets = inputs
                
                optimizer.zero_grad()
                outputs = self.quantized_model(inputs)
                
                if isinstance(outputs, tuple):
                    loss = F.mse_loss(outputs[0], targets)
                else:
                    loss = F.mse_loss(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"[QAT] Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return self.convert_qat()
    
    def convert_qat(self) -> nn.Module:
        """
        Convert QAT model to quantized model.
        
        Returns:
            Quantized model
        """
        self.quantized_model.eval()
        quantized = convert(self.quantized_model)
        
        print("[QAT] Model converted to quantized format")
        return quantized


def quantize_for_deployment(
    model: nn.Module,
    method: str = 'dynamic',
    calibration_data: Optional[List[torch.Tensor]] = None,
    output_path: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Quantize model for deployment.
    
    Args:
        model: PyTorch model
        method: Quantization method ('dynamic', 'static', 'qat')
        calibration_data: Calibration data for static/qat
        output_path: Path to save quantized model
        
    Returns:
        Tuple of (quantized_model, metadata)
    """
    if not QUANTIZATION_AVAILABLE:
        print(f"[Quantization] Skipping {method} quantization (not available in this PyTorch)")
        return model, {'method': 'none', 'quantized': False}
    
    print(f"[Quantization] Applying {method} quantization...")
    
    metadata = {
        'method': method,
        'original_state_dict_keys': len(model.state_dict()),
        'quantized': True
    }
    
    if method == 'dynamic':
        quantizer = DynamicQuantizer()
        quantized_model = quantizer.quantize(model)
        
    elif method == 'static':
        if calibration_data is None:
            print("[Quantization] Warning: No calibration data, using dynamic instead")
            method = 'dynamic'
            quantizer = DynamicQuantizer()
            quantized_model = quantizer.quantize(model)
        else:
            quantizer = QuantizedVisionEncoder(model)
            quantized_model = quantizer.quantize_static(calibration_data)
    
    elif method == 'qat':
        quantizer = QuantizationAwareTrainer(model)
        quantized_model = quantizer.prepare_qat()
        metadata['needs_training'] = True
    
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'metadata': metadata
        }, output_path)
        print(f"[Quantization] Saved to {output_path}")
    
    return quantized_model, metadata


def benchmark_quantization(
    fp32_model: nn.Module,
    int8_model: nn.Module,
    test_inputs: List[torch.Tensor],
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark quantized vs non-quantized models.
    
    Args:
        fp32_model: Original FP32 model
        int8_model: Quantized INT8 model
        test_inputs: Test input tensors
        num_runs: Number of benchmark iterations
        
    Returns:
        Dictionary with benchmark results
    """
    device = next(fp32_model.parameters()).device
    
    fp32_model.eval()
    int8_model.eval()
    
    results = {'fp32': [], 'int8': []}
    
    with torch.no_grad():
        for _ in range(10):
            _ = fp32_model(test_inputs[0].to(device))
            _ = int8_model(test_inputs[0].to(device))
        
        for i in range(num_runs):
            for inp in test_inputs:
                inp = inp.to(device)
                
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                
                t0.record()
                _ = fp32_model(inp)
                t1.record()
                torch.cuda.synchronize()
                results['fp32'].append(t0.elapsed_time(t1))
                
                t0.record()
                _ = int8_model(inp)
                t1.record()
                torch.cuda.synchronize()
                results['int8'].append(t0.elapsed_time(t1))
    
    avg_fp32 = sum(results['fp32']) / len(results['fp32'])
    avg_int8 = sum(results['int8']) / len(results['int8'])
    speedup = avg_fp32 / avg_int8 if avg_int8 > 0 else 1.0
    
    return {
        'fp32_avg_ms': avg_fp32,
        'int8_avg_ms': avg_int8,
        'speedup': speedup,
        'fp32_hz': 1000 / avg_fp32 if avg_fp32 > 0 else 0,
        'int8_hz': 1000 / avg_int8 if avg_int8 > 0 else 0
    }
