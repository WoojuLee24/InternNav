import torch
import sys

def diagnose():
    print(f"--- 1. 시스템 정보 ---")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"GPU 장치명: {torch.cuda.get_device_name(0)}")
    print(f"현재 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print(f"\n--- 2. 단순 메모리 할당 테스트 ---")
    try:
        # 10GB 할당 시도
        x = torch.empty(1024**3 * 10 // 4, dtype=torch.float32, device='cuda')
        print("✅ 10GB 연속 할당 성공")
        del x
    except Exception as e:
        print(f"❌ 메모리 할당 실패: {e}")

    print(f"\n--- 3. 정밀도별 연산 테스트 ---")
    size = 4096
    
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        try:
            print(f"[{dtype}] 행렬 곱셈 시도 중...", end=" ")
            a = torch.randn(size, size, device='cuda', dtype=dtype)
            b = torch.randn(size, size, device='cuda', dtype=dtype)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            print("✅ 성공")
        except Exception as e:
            print(f"❌ 실패: {str(e)[:50]}...")

    print(f"\n--- 4. cuBLAS MatMul 연산 테스트 (에러 발생 지점) ---")
    try:
        # cuBLASLt를 트리거하기 위해 큰 행렬 곱셈 수행
        size = 4096
        a = torch.randn(size, size, device='cuda', dtype=torch.float16)
        b = torch.randn(size, size, device='cuda', dtype=torch.float16)
        
        print(f"행렬 곱셈({size}x{size}) 시도 중...")
        for i in range(5):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print("✅ cuBLAS 연산 성공 (라이브러리 정상)")
    except Exception as e:
        print(f"❌ cuBLAS 연산 실패: {e}")
        if "CUBLAS_STATUS_NOT_INITIALIZED" in str(e):
            print("\n💡 진단 결과: 라이브러리 초기화 실패.")
            print("이것은 메모리 파편화이거나, 현재 CUDA 버전이 Blackwell GPU의 특정 명령어를 지원하지 않을 때 발생합니다.")

if __name__ == "__main__":
    diagnose()
