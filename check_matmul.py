import torch
import time

def test_size(n, dtype=torch.float16):
    try:
        a = torch.randn(n, n, device='cuda', dtype=dtype)
        b = torch.randn(n, n, device='cuda', dtype=dtype)
        torch.cuda.synchronize()
        
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"✅ [Size {n:>5}x{n:<5}] {str(dtype):<15} : 성공 ({ (end-start)*1000:.2f}ms)")
        return True
    except Exception as e:
        print(f"❌ [Size {n:>5}x{n:<5}] {str(dtype):<15} : 실패 ({str(e)[:40]}...)")
        return False

print(f"GPU: {torch.cuda.get_device_name(0)} / PyTorch: {torch.__version__}\n")

sizes = [128, 512, 1024, 2048, 4096]
for s in sizes:
    # FP16 테스트
    test_size(s, torch.float16)
    # BF16 테스트 (Blackwell 최적화 포맷)
    test_size(s, torch.bfloat16)
