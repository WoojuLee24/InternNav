# Qwen2.5-VL `patch_embed` Conv3d 병목 — 근본 원인 분석

_작성 2026-08-24. 측정 환경: H200 141GB, torch 2.9.0+cu130, cudnn 91300, cuda 13.0, transformers 4.51.0_

## 한 줄 요약

`Qwen2_5_VisionPatchEmbed`의 `nn.Conv3d`는 bf16 학습에서 **stage1 micro-step의 58%**를 먹고
weight gradient에 **14% 오차**를 만든다. 둘의 원인은 하나다 — PyTorch가 저정밀 3D conv를
cuDNN에 보내지 않아 `SlowDilated3d` fallback으로 떨어지고, 그 fallback이 **배치 차원을
하나씩 루프로 돌기** 때문이다. 이 모델은 패치 17,240개를 배치 차원에 넣는다.

## 문제의 코드 (transformers 4.51.0)

```python
class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size=14, temporal_patch_size=2, in_channels=3, embed_dim=1152):
        kernel_size = [temporal_patch_size, patch_size, patch_size]          # [2, 14, 14]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, 3, 2, 14, 14)                 # (N, 3, 2, 14, 14)
        return self.proj(hidden_states).view(-1, self.embed_dim)             # 출력 1x1x1
```

`stride == kernel_size`이고 입력이 커널과 정확히 같은 크기라 **슬라이딩할 자리가 없다.**
즉 정의상 dense linear map이다: `out[n,e] = Σ_d x[n,d] · w[e,d]` (d = 3·2·14·14 = 1176).

가장 어이없는 점: 입력은 **이미 `(N, 1176)` flat**인데 `.view()`로 5D로 만들고,
fallback의 `vol2col`이 그걸 **다시 `(N, 1176)`으로 편다.** 비트 동일한 레이아웃을 복사하는 데
1,154 ms를 쓴다.

## 측정 1 — 얼마나 느리고 부정확한가

stage1 micro-batch(bs4) 기준 N = 34,480 패치:

```
conv3d  fwd 1256.97 ms   fwd+bwd 2885.6 ms     0.08 TFLOPS   wgrad rel_err(L2) 1.399e-01
gemm    fwd    0.156 ms  fwd+bwd    1.216 ms  665.4 TFLOPS   wgrad rel_err(L2) 1.661e-03
                                              => 2,373배      => 84배 더 정확
forward 출력: torch.equal = True (max_abs_diff = 0.000e+00)
```

0.1 TFLOP짜리 연산을 peak의 1/12,000로 돌리고 있었다.
wgrad 오차는 fp64 정확해(`go.double().T @ x.double()`) 기준. **conv가 틀린 것이고 GEMM이 맞다.**

## 측정 2 — 어떤 backend가 선택되는가 (`torch._C._select_conv_backend`)

| dtype | C | memory format | 선택 backend | cudnn_acceptable | conv fwd |
|---|---|---|---|---|---|
| **bf16** | **3** | **contiguous (실제 경로)** | **SlowDilated3d** | **False** | **992 ms** |
| bf16 | 8 / 16 | contiguous | SlowDilated3d | False | 1516 / 1529 ms |
| fp16 | 3 | contiguous | SlowDilated3d | True | 1099 ms |
| fp32 | 3 | contiguous | **Cudnn** | True | **3.12 ms** |
| bf16 | 3 | **channels_last_3d** | **Cudnn** | — | **4.36 ms** |
| bf16 | 3 | contiguous, `torch.cudnn_convolution` 강제 | (우회) | — | **5.17 ms** |
| bf16 | — | **GEMM** | — | — | **0.156 ms** |

프로파일에 뜨는 커널: `at::native::vol2col_kernel<c10::BFloat16>` 1153.69 ms + `aten::slow_conv_dilated3d`.

## 결론: cuDNN / CUDA / PyTorch 버전 문제가 아니다

1. **cuDNN은 할 수 있다.** `torch.cudnn_convolution`을 직접 호출하면 bf16 contiguous에서도
   성공하고 5.17 ms다. → cuDNN 능력/버전 문제 아님.
2. **원인은 PyTorch의 디스패치 정책.** `channels_last_3d`로 바꾸면 bf16에서도 `Cudnn`이 선택된다.
   cuDNN의 텐서코어 3D conv 커널이 NDHWC를 요구하므로, PyTorch는 **저정밀(bf16/fp16) 3D conv를
   contiguous(NCDHW)로 받으면 cuDNN을 쓰지 않는다.** `cudnn_acceptable`이 bf16에서 False인 것도 일관.
3. **`in_channels` 정렬은 무관.** C를 3 → 8 → 16으로 바꿔도 계속 `SlowDilated3d`다.
   (초기 가설 "C=3이 8의 배수가 아니라서"는 **반증됨**.)

## 왜 그렇게까지 느린가 — 배치 루프

`SlowDilated3d`는 배치 차원을 순차 루프로 돈다. 배치를 270배 늘려도 **요소당 비용이 일정**:

```
      N    conv fwd      per-N |   gemm fwd      per-N
     64      4.28 ms     66.9 us |    0.483 ms    7.551 us
    256     16.97 ms     66.3 us |    0.484 ms    1.890 us
   1024     67.24 ms     65.7 us |    0.487 ms    0.476 us
   4096    317.13 ms     77.4 us |    0.505 ms    0.123 us
  17240   1273.47 ms     73.9 us |    0.135 ms    0.008 us
```

conv는 per-N이 66~77 µs로 평평(= 병렬화 없음), GEMM은 7.55 → 0.008 µs로 940배 개선(= 정상 스케일링).
이 모델은 패치를 배치 차원에 넣으므로 루프가 17,240회 돌고, 각 회차는 1176→1280 GEMV
하나(3 MFLOP)라 순전히 커널 런치 지연 바운드. 17,240 × 66 µs ≈ 1.14 s = 관측값.

## 같은 루프가 14% 오차도 만든다

wgrad는 그 루프에서 `grad_weight += ...`를 17,240회 **순차 누적**하고 `grad_weight`는 bf16이다.
즉 bf16 덧셈 17,240번. dtype별 오차가 이를 확증한다:

| dtype | conv wgrad err | gemm wgrad err | 비율 |
|---|---|---|---|
| bf16 | 1.400e-01 | 1.663e-03 | 84x |
| fp16 | 1.904e-02 | 2.088e-04 | 91x |
| fp32 | 2.935e-04 | 1.051e-06 | 279x |

conv 오차가 bf16 → fp16에서 **7.4배** 줄어드는데 이는 mantissa 차이(8bit vs 11bit → 8배)와 일치.
**누산기가 fp32가 아니라 입력 dtype 그대로**임이 확정. cuBLAS GEMM은 텐서코어가 fp32로 누산한다.

## 세 mode 비교 (실제 클래스로 실측, 2026-08-24)

`internnav/model/basemodel/internvla_n1/patch_embed_impl.py`의 세 구현.
backend: `conv` -> `SlowDilated3d`, `channels_last` -> `Cudnn` (`_select_conv_backend` 조회).

| N=34,480 (stage1 bs4) | fwd+bwd | equal(conv) | weight contiguous |
|---|---|---|---|
| `conv` | 2554.99 ms | True (기준) | True |
| **`gemm`** | **1.05 ms (2,433x)** | **True**, maxdiff 0 | True |
| `channels_last` | 6.34 ms (403x) | False, maxdiff 1.56e-2 | True |

| N=137,920 (stage2 bs16) | fwd+bwd |
|---|---|
| `conv` | 10,217.73 ms |
| **`gemm`** | **4.02 ms (2,541x)** |
| `channels_last` | 24.71 ms (413x) |

세 경우 모두 `proj.weight`가 contiguous로 유지된다 -> ZeRO-2 flatten 안전.
(`channels_last`는 **입력만** 변환하도록 구현했다. Parameter 레이아웃을 바꾸면 ZeRO와 충돌 위험.)

### 결정적 차이: `channels_last`는 "기존 동작 보존"이 아니다

`verify_patch_embed_impl.py` 전체 모델 검증 결과:

```
gemm           : visual() torch.equal=True (maxdiff 0), loss 비트 동일,
                 bf16 반올림 초과 파라미터 = 1개 (visual.patch_embed.proj.weight)  -> PASS
channels_last  : visual() torch.equal=False (maxdiff 9.766e-04),
                 loss 11.89615536 -> 11.89618111 (달라짐),
                 grad 0/729 비트 동일, bf16 반올림 초과 = 372개                     -> 대조군용
```

cuDNN 커널의 누적 순서가 달라 forward부터 바뀌고 그것이 LLM 전체로 전파된다.
**보존이 목표면 `gemm`을 써야 한다** — 동시에 `channels_last`보다 6배 빠르다.
`channels_last`의 가치는 "conv 구현이 틀린 게 아니라 라우팅이 문제였다"를 확인하는 교차검증뿐이다.

## 실제 runner 실측 — 최종 (2026-08-24, 8xH200, 30 step, --no-eval, 같은 날 순차)

| | `conv` (기존) | `gemm` | 배수 |
|---|---|---|---|
| **stage1** (full finetune, bs4 x ga4) | **19.17 s/it** | **9.02 s/it** | **2.13x** |
| **stage2** (VLM freeze, bs16 x ga1) | **10.19 s/it** | **5.18 s/it** | **1.97x** |

| | s1 conv | s1 gemm | s2 conv | s2 gemm |
|---|---|---|---|---|
| train_runtime | 591.79 s | 289.84 s | 348.27 s | 200.14 s |
| samples/s | 6.49 | 13.25 | 11.03 | 19.19 |
| train_loss(30step) | 1.3759 | 1.3817 | 0.4463 | 0.4765 |

**기준선 교차검증 통과**: stage1 conv 19.17 == 실제 로그 단독 실행 19.2 s/it,
stage2 conv 10.19 == 완료된 run 전체 평균 10.25 s/it.

환산: stage1 261h(10.9일) -> **123h(5.1일)**. stage2 3.5일 -> **약 1.8일**.

### 30-step loss 차이는 신호가 아니다

stage2는 `patch_embed` gradient가 **아예 없는데도** 0.4463 vs 0.4765로 벌어졌다
(S1 랜덤 초기화 + 데이터 순서 무작위성). stage1의 1.3759 vs 1.3817(0.42%)은 그보다 작다.
즉 이 규모의 편차는 run 간 무작위성이며, 동등성 판단에 쓸 수 없다.
쓰려면 같은 seed로 conv를 두 번 돌려 편차 바닥을 먼저 재야 한다.

### 예측 이력 — 외삽은 실패, end-to-end는 성공

| | 예측 | 실측 | |
|---|---|---|---|
| stage1 | 2.32x (합성 배치 end-to-end 측정) | **2.13x** | 맞음 |
| stage2 | 6.8x (부품 측정 외삽) | **1.97x** | 3.5배 과대추정 |

**교훈: 고립된 부품 벤치마크를 실제 step에 곱하지 말 것.** 반드시 end-to-end로 측정한다.

## 처방

`internnav/model/basemodel/internvla_n1/linear_patch_embed.py` 의 `LinearPatchEmbed`:
`forward`만 override해 `proj.weight`를 `reshape()`(view)로 2D로 보고 GEMM 한 번.
파라미터 객체·이름(`visual.patch_embed.proj.weight`)·shape가 유지되므로
**state_dict / checkpoint / ZeRO 파티셔닝 / resume 무영향.**
config flag `fast_patch_embed` (default False)로 게이팅, train+eval 대칭 적용.

기각한 대안:
- **fp32로 patch_embed만**: 3.12 ms(conv) — GEMM보다 20배 느리고, forward가 더 이상
  현재와 비트 동일하지 않게 된다. fp32 GEMM(2.72 ms)조차 fp32 conv보다 빠르고 정확하므로
  conv는 어떤 dtype에서도 정답이 아니다.
- **channels_last_3d**: 4.36 ms — GEMM보다 28배 느리고, 매 step 입력 전치 비용 + 메모리 포맷 파급.
- **cuDNN 강제 호출**: 5.17 ms — 33배 느림.

## 파급 범위 — stage2가 stage1보다 더 심하다

`Qwen2_5_VisionPatchEmbed` 실제 모듈로 N 스케일링 재측정 (no_grad, 유휴 GPU 단독):

```
       N label            |  conv fwd(no_grad)     per-N |  gemm fwd | conv/gemm
    8620 1 sample         |           364.12 ms    42.2 us |   0.056 ms |     6559x
   17240 bs2              |          1351.53 ms    78.4 us |   0.677 ms |     1996x
   34480 stage1 bs4       |          1699.70 ms    49.3 us |   0.222 ms |     7648x
   68960 bs8              |          4643.20 ms    67.3 us |   0.367 ms |    12638x
  137920 stage2 bs16      |          8752.30 ms    63.5 us |   1.816 ms |     4818x
```

per-N은 42~78 µs로 노이즈가 크지만(런치 바운드) 전체적으로 선형.

| | batch | 패치/micro-step | patch_embed | step time | 비중 | gradient 오차 |
|---|---|---|---|---|---|---|
| **stage1** (`tune_mm_vision=True`) | bs4 × ga4 | 34,480 | 2.56 s (fwd+bwd) | 4.40 s (micro) | **58%** | **14%** |
| **stage2** (VLM 전체 freeze) | bs16 × ga1 | 137,920 | 8.75 s (fwd only) | **10.25 s 실측** | **70~85%** | **없음** |

stage2 교차검증: patch_embed 8.75 s + ViT 32블록 fwd ~0.96 s + LLM fwd ~1.3 s ≈ 11 s
vs 실제 로그 10.25 s/it → 앞뒤가 맞는다 (isolated 측정이 다소 pessimistic).

### 정정 (2026-08-24): stage2 실제 이득은 6.8배가 아니라 **1.97배**

위 "patch_embed 8.75 s = step의 70~85%"는 **고립 벤치마크를 실제 step에 곱한 외삽이었고 틀렸다.**
실제 runner로 30 step씩 동일 조건(같은 머신·같은 시점) 측정:

```
                정상상태 s/it   warmup포함   train_runtime   samples/s   train_loss
conv (기존)        10.19         11.61        348.27 s        11.03       0.4463
gemm               5.18          6.67         200.14 s        19.19       0.4765
배수               1.97x         1.74x        1.74x           1.74x
```

`conv` 30-step 값 10.19 s/it이 완료된 run의 전체 평균 10.25 s/it과 일치 -> 기준선 신뢰 가능.
역산하면 `patch_embed`는 stage2 step의 **약 49%**(5.01 s)였고 70~85%가 아니었다.
샘플당 패치 수 가정(8,620개)이 실제보다 높았던 것으로 보인다 —
`pixel_goal_only=True`에서 history가 8장 미만인 샘플이 섞이면 평균이 내려간다.

**교훈: 부품 측정을 전체로 외삽하면 과대추정된다.** stage1의 2.32배는 합성 배치로 end-to-end
측정한 값이라 유효했지만, stage2의 6.8배는 외삽이었고 실측으로 무너졌다.

실용 환산: 3.5일 -> 약 1.8일. 2배도 충분히 큰 이득이고 stage2는 VLM freeze라 위험이 0이다.

(30-step loss 차이 0.4463 vs 0.4765는 S1이 랜덤 초기화되고 데이터 순서에 무작위성이 있는
run 간 편차다. stage2는 patch_embed gradient가 아예 없고 forward가 비트 동일하므로
patch_embed 때문일 수 없다. 확인하려면 같은 seed로 conv를 두 번 돌려 편차를 봐야 한다.)

stage2가 stage1보다 비중이 큰 이유: (1) bs16이라 micro-step당 패치가 4배, (2) VLM freeze로
다른 backward 비용이 없어 희석되지 않음.

**freeze한 run은 gradient를 계산하지 않으므로 학습 품질은 멀쩡하고 순수하게 시간만 낭비했다.**
즉 stage2 계열은 A1 적용의 위험이 0이고 이득이 더 크다 — "동작 불변" 제약을 완전히 만족한다.

`data-vol2/checkpoints/` 아래 `baseline` / `batch_size` / `batch_size2` / `bev` / `dagger` 계열
전부 해당. 나아가 이 repo 한정 문제가 아니라 **bf16으로 Qwen2-VL / Qwen2.5-VL을 파인튜닝하는
모든 코드**가 같은 비용을 냈다 (upstream `train_system2.sh` / `train_dual_system.sh` 포함).

## "버전 때문 아닌가?" — 전부 닫음

| 가설 | 검증 | 결과 |
|---|---|---|
| transformers 버전 차이 | 4.49.0 / 4.51.0 / 4.57.1의 `Qwen2_5_VisionPatchEmbed` 소스 비교 | **바이트 단위 동일.** 한 번도 수정된 적 없고 최신 버전에서도 안 고쳐짐 |
| repo가 다른 버전을 씀 | `requirements/{internvla_n1,model_requirements,habitat_requirements}.txt` | 셋 다 `transformers==4.51.0` 핀, 설치본과 일치. 이 브랜치에서 핀 변경 없음 |
| torch 버전 회귀 | torch 2.4.0 wheel의 `torch/backends/cudnn/__init__.py` | `CUDNN_TENSOR_DTYPES = {half, float, double}` — 2.9.0과 동일, **bf16은 원래부터 없음** |
| bf16이 이 repo의 선택 | upstream `scripts/train/qwenvl_train/train_system2.sh:46` | `--bf16` 이 원저자 설정. repo가 넣은 게 아님 |

(정밀도: `CUDNN_TENSOR_DTYPES`는 Python 헬퍼 `is_acceptable`이 쓰는 상수다. 실제 conv 디스패치는
C++ `Convolution.cpp`이고 추가 조건이 있다 — fp16은 `is_acceptable=True`인데도 `SlowDilated3d`로
간다(memory format 조건). C++ 경로가 2.4에서도 동일했는지는 다른 torch를 설치할 수 없어 미검증.
단 Python 상수가 불변인 점과 일관된다.)

## fp32는 답이 아니다

LLM MLP up-proj 대표 GEMM (9048×3584×18944) 측정:

| | 시간 | 처리량 |
|---|---|---|
| bf16 (현재) | 3.40 ms | 361.8 TFLOPS |
| **fp32 (tf32 off = 이 환경 기본)** | **46.61 ms** | **26.4 TFLOPS** |
| fp32 (tf32 on) | 4.48 ms | 274.2 TFLOPS |

이 환경은 `matmul.allow_tf32=False`. 전체 fp32는 LLM이 **13.7배** 느려지고 메모리도 2배
(파라미터만 28GB)라 불가. tf32를 켜면 1.3배로 줄지만 tf32는 mantissa 10bit라 정밀도 이점도 사라진다.
`patch_embed`만 fp32(3.12 ms)로 돌리는 건 가능하나 GEMM(0.156 ms)이 20배 빠르고
**forward 비트 동일**이라는 이점을 잃는다. fp32에서도 GEMM(2.72 ms) > conv(3.12 ms)라
conv는 어느 dtype에서도 최선이 아니다.

## 왜 원저자들은 몰랐나 — 규모에 가려졌다

upstream은 `batch 2/GPU × 64 GPU`. micro-step당 패치가 2 × 8,620 = 17,240개로 여기(bs4, 34,480)의
절반이고 GPU는 8배 많다. 같은 48,974 step이지만 절대 시간이 하루 남짓이라 **7B VLM 학습이
하루 걸리는 건 전혀 이상해 보이지 않는다.** 4배 낭비가 있어도 프로파일할 동기가 없다.
여기서 드러난 이유는 GPU가 8배 적어 같은 낭비가 10.9일로 증폭됐기 때문.

그리고 이건 InternNav 코드가 아니라 **HuggingFace transformers의 Qwen2.5-VL 공식 모델링 파일**이다.
bf16으로 Qwen2-VL/2.5-VL을 파인튜닝하는 모든 코드가 같은 비용을 낸다.

## wgrad 오차의 실제 크기 — 실제 checkpoint + 실제 이미지

합성 랜덤이 아니라 진짜 forward/backward에서 hook으로 `grad_output`을 붙잡아 측정
(`scratchpad/wgrad_real.py`, ckpt = `data-vol2/checkpoints/InternVLA-N1-System2`,
이미지 = r2r/17DRP5sb8fy 실제 프레임, bs2 → 17,240 패치):

```
실제 loss = 9.342174
실제 grad_output: |mean|=6.628e-05  max=1.689e-01   상쇄지표 0.0054

                  L2 상대오차      방향 일치도(cosine)
conv3d (현재)     1.3293e-01       0.991135
gemm   (제안)     1.6655e-03       0.999999
```

**13.3% L2 오차는 실재한다** (합성 14%와 일치, 인공물 아님). 상관구조를 바꿔가며 확인한 결과
상쇄가 없는 조건에서는 오히려 92%까지 나빠진다 — 같은 부호 항의 순차 bf16 누적은
naive summation catastrophe라 오차가 √N이 아니라 N에 비례한다.

**그러나 cosine 0.9911 = 방향이 약 7.6도만 틀어진다.**

### 7.6도가 큰가 — 기준선 측정 (`scratchpad/grad_noise.py`)

실제 ckpt로 서로 다른 4개 mini-batch(다른 scene/episode/프롬프트)를 돌려 두 각도를 비교:

```
(A) conv 오차각 (같은 데이터, conv vs fp64 정확해)
    7.52 / 5.80 / 8.52 / 7.43 도                        평균  7.32도

(B) mini-batch 노이즈각 (데이터만 다른 정확해끼리)
    76.24 / 92.10 / 77.23 / 95.63 / 76.52 / 90.67 도    평균 84.73도
    cosine: 0.238 / -0.037 / 0.221 / -0.098 / 0.233 / -0.012
=> 노이즈가 conv 오차보다 11.6배 큼
```

**mini-batch만 바꿔도 이 레이어의 gradient 방향이 85도 움직인다.** 일부 쌍은 cosine이 음수로
거의 반대 방향을 가리킨다. SGD가 원래 다루는 신호가 이 정도로 요동치므로 conv의 7.3도는
**11.6배 작은 교란**이고, "학습이 잘못되고 있다"고 부를 수준이 아니다.

남은 단서: mini-batch 노이즈는 무작위라 step을 거듭하면 평균되지만, conv 오차가 무작위인지
**체계적 편향**인지는 미구분. 완전히 체계적이면 √N로 줄지 않는다. 다만 그 최악의 경우에도
0.018% 파라미터 하나에 7.3도이고 Adam이 파라미터별로 정규화한다.

### 결론: A1을 하는 이유는 속도 하나다

- 영향 대상은 `visual.patch_embed.proj.weight` **단 하나** (1.5M / 8.3B = 0.018%)
- LoRA·vision tower freeze 등 **대부분의 학습은 이 레이어를 아예 학습시키지 않는다**
  (stage2가 그 예 → gradient 영향 0, 속도 이득만)
- Qwen2-VL/2.5-VL full finetune이 실제로 잘 된다는 사실이 "치명적이지 않다"는 최강 증거
- **"전세계가 잘못 학습한다"는 과장이었다.** 정확히는 "모두가 불필요한 속도 비용을 냈고,
  그 중 vision tower를 학습시킨 경우엔 한 레이어의 gradient 방향이 7.6도 틀어졌다"

파라미터 수(0.018%)와 시간 비중(58%)이 무관한 이유: `patch_embed`의 FLOPs는 103.8 GFLOP =
micro-step 전체의 **0.015%**다. 즉 0.015%의 일을 58%의 시간에 하고 있다(효율 격차 ~3,900배).
gradient 정확도는 파라미터 수가, 속도는 실행 방식이 결정한다.

## 미해결

- 14% gradient 오차가 최종 학습 성능에 미치는 영향은 **측정하지 않았다.** forward/loss는 정확하고
  틀리는 것은 `visual.patch_embed.proj.weight`(1.5M / 8.3B = 0.018%)의 업데이트 방향뿐이다.
  mini-batch 샘플링 노이즈가 이미 훨씬 크므로 영향이 작을 가능성이 있으나 미검증.
- 이 정책이 과거 torch 버전에서도 동일했는지는 미검증 (구버전 테스트 안 함).
  단 "능력 부재"가 아니라 "디스패치 정책"임은 확인됨.

## 재현 스크립트

`scripts/train_eval/qwenvl_train/baseline/verify_fast_patch_embed.py` (V1/V2 등가성 검증, repo).
분석용 벤치마크는 scratchpad: `bench_{gc,parts,imgs,vit2,patchembed}.py`, `why_conv.py`,
`backend_pick.py`, `batchloop.py`, `wgrad_ref.py`.
