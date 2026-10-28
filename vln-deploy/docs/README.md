# vln-deploy/docs

InternVLA-N1(DualVLN)을 경량화해서 배포하는 절차와 결과. **지금 쓰는 문서는 아래 두 개**다.

| 문서 | 내용 |
|---|---|
| [command.md](command.md) | **재현 명령.** 환경 빌드 → 경량화(S1 TensorRT FP16, S2 GGUF Q8_0) → 검증(데이터셋 open-loop, Isaac Sim closed-loop, RGB-D 프레임) → Jetson Thor. 맨 위 "한 번에 실행"이 러너 한 줄 |
| [Report.html](Report.html) | 2026-10-27 RTX 5090 재현 결과. 용어 설명, 단계별 표·차트, 이전 결과와 비교 |

옛 문서 `q4_0812/` (08월 Q4 경로, eval 번들·Isaac eval config 기준, **사용 안 함**):
[command_q4_0812.md](q4_0812/command_q4_0812.md) · [WORKFLOW_q4_0812.md](q4_0812/WORKFLOW_q4_0812.md) · [COMPARISON_q4_0812.md](q4_0812/COMPARISON_q4_0812.md) · [issue_260818.md](q4_0812/issue_260818.md) (S2 look-down 해상도 이슈 기록)

## 무엇이 어디에 있나

| 경로 (vln-deploy 기준) | 내용 |
|---|---|
| `verify/build_and_verify.sh` | 2장·3장을 한 번에 (x86은 docker, Thor는 장치에서) |
| `verify/verify_rgbd.py` | RGB-D 프레임 폴더로 S2·S1·S2+S1 실행, 두 장치 결과 비교 |
| `quantization_kit/tools/build_s1_trt.sh` | S1 → ONNX → TensorRT FP16 엔진 |
| `quantization_kit/tools/build_s2_gguf.sh` | S2 → GGUF Q8_0 + mmproj → `quantization_kit/out/gguf/` (중간 파일은 끝나면 삭제) |
| `quantization_kit/tools/fetch.sh` | git에 없는 외부 자산 받기: `llama`(upstream + patch) · `ckpt`(원본 체크포인트) · `data`(검증 6 장면) |
| `quantization_kit/tools/setup_venv.sh` | 경량화 venv 만들기 (x86: `requirements_x86.txt` 고정 버전, Thor: SBSA torch) |
| `quantization_kit/tools/llama.cpp-vln.patch` | llama.cpp upstream `389ff61d7` 위의 우리 수정 (tools/mtmd 8 파일) |
| `quantization_kit/verify_s2_gguf.py` | S2 replay (GGUF / HF) 와 비교 |
| `quantization_kit/verify_s1s2_full.py` | S2 출력 → S1 (PyTorch / TRT) 와 최종 명령 비교 |
| `quantization_kit/verify_trt_repro.py` | S1 TRT vs PyTorch 단독 비교 |
| `quantization_kit/{onnx,trt_engines,gguf}_repro/` | 경량화 결과물 (배포 기본 구성이 쓰는 파일) |
| `quantization_kit/RUN.md`, `deploy/README.md`, `../BUNDLE.md` | Thor 원본 번들(Q4) 실행 안내 |

git에 올리는 것과 올리지 않는 것(다시 얻는 방법 포함): [command.md 5장](command.md#5-git에-올리는-것--올리지-않는-것).

설계 배경과 전체 측정값: [`scripts/sim/report.md`](../../scripts/sim/report.md). sim 실행·검증 명령: [`scripts/sim/command.md`](../../scripts/sim/command.md).
