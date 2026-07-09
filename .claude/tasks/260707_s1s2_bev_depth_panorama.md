## 앞으로 할 작업

### 목표: S2, S1에서 각각 BEV 이미지, Depth 이미지, panorama이미지 provider와 이미지에 맞는 encoder 구현
### 이미지 provider 
- BEV 이미지는 depth 이미지 활용하여 생성
- Depth 이미지는 depth 이미지 / occupancy grid map 이미지로 생성
- Panorama 이미지는 panorama 이미지 / 60,120 FoV로 나눈 이미지로 생성
### 이미지에 맞는 encoder
- 각 이미지에 맞는 in_channel 
- 각 이미지에 맞는 encoder arhitecture
### 참고: GT / estimated depth 사용 가능
### 확인: 명령어 실행 및 정상 작동 확인

## 개발 계획

### 핵심 방향

- branch feature/bev_v0.1로부터 새로운 branch feature/input_v0.1를 생성한다. 
- Provider는 `VisualInputProvider`, `DepthImageProvider`, `PanoramaImageProvider`처럼 입력 종류별 클래스로 나누지 않고, 하나의 통합 `ImageProvider`로 설계한다.
- `ImageProvider`는 config를 받아 S1/S2에 필요한 이미지 타입을 생성한다.
- 기존 FPV/default 경로는 완전 no-op으로 유지한다.
- 기존 구현을 크게 갈아엎지 않고, 현재 `internnav/model/utils/visual_input_provider.py`의 BEV 생성 로직을 일반화한다.
- 새 기능은 config/argument로 선택 가능해야 하며, 기존 config에 필드가 없어도 기존 동작이 유지되어야 한다.

### 1. 현재 구조 확인

- 기존 S1 BEV training 경로:
  - `internnav/model/basemodel/internvla_n1/internvla_n1_bev_provider.py`
  - `internnav/model/utils/visual_input_provider.py`
- 기존 S1 encoder 경로:
  - `internnav/model/basemodel/internvla_n1/internvla_n1.py`
  - `rgb_model`, `memory_encoder`, `rgb_resampler`, `cond_projector` 사용
- trainer argument/config 경로:
  - `internnav/trainer/internvla_n1_argument.py`
  - `scripts/train_eval/qwenvl_train/default_config.py`
- 먼저 현재 batch에 다음 입력이 있는지 확인한다:
  - `traj_images`
  - `traj_depths`
  - `traj_tdmaps`
  - `traj_cam_heights`
  - `traj_cam_pitch_1`, `traj_cam_pitch_2`
  - panorama image/depth 관련 key

### 2. 통합 ImageProvider 설계

`ImageProvider`는 하나의 클래스/모듈에서 다음 기능을 config 기반으로 처리한다.

- S1용 이미지 생성
- S2용 extra image 생성
- BEV/depth/panorama 변환
- GT depth와 estimated depth 선택
- debug image 저장

예상 API:

```python
@dataclass
class ImageProviderOutput:
    images: Optional[torch.Tensor] = None
    depths: Optional[torch.Tensor] = None
    pil_images: Optional[list] = None
    features: Optional[torch.Tensor] = None

class ImageProvider:
    def get_s1_input(self, rgb, depth=None, tdmap=None, panorama=None, **meta) -> ImageProviderOutput:
        ...

    def get_s2_extra(self, rgb=None, depth=None, tdmap=None, panorama=None, **meta) -> list:
        ...
```

Config 예시:

```python
s1_image_type = "fpv"       # fpv|bev|depth|panorama
s1_image_mode = "replace"   # replace|concat|none
s2_image_type = "fpv"       # fpv|bev|depth|panorama
s2_image_mode = "append"    # append|replace|none
bev_image_type = "rgb"      # rgb|occ|occ.binary|occ.prob|occ.dist|occ.dist.sep
depth_image_type = "raw"    # raw|normalized|colormap|occ
panorama_mode = "full"      # full|crop_60|crop_120
depth_source = "gt"         # gt|dav2|udv2
```

### 3. ImageProvider 내부 변환 규칙

#### BEV

- 입력: RGB + depth
- depth는 GT 또는 estimated depth 사용 가능
- 기존 `BEVProcessor` 로직을 재사용한다.
- 출력:
  - `rgb`: color BEV image
  - `occ`: occupancy grid image
  - `occ.binary`, `occ.prob`, `occ.dist`, `occ.dist.sep` 지원 유지

#### Depth

- 입력: depth 또는 RGB
- `depth_source=gt`이면 batch의 depth 사용
- `depth_source=dav2|udv2`이면 RGB에서 estimated depth 생성
- 출력 옵션:
  - `raw`: 1-channel depth tensor
  - `normalized`: 0-1 normalized depth image
  - `colormap`: 3-channel visualization image
  - `occ`: depth 기반 occupancy grid map image

#### Panorama

- 입력: panorama image가 batch에 있으면 사용
- batch에 panorama가 없으면 dataset loader 확장이 선행 작업
- 출력 옵션:
  - `full`: panorama 원본 사용
  - `crop_60`: 60도 FoV crop 여러 장 생성
  - `crop_120`: 120도 FoV crop 여러 장 생성
- S1에서는 crop들을 T dimension에 이어 붙이는 방식을 우선 사용한다.
- S2에서는 crop들을 PIL image list로 append하는 방식을 우선 사용한다.

### 4. Encoder 설계

입력 이미지 종류별 encoder를 완전히 별도 구현하기 전에, adapter 방식으로 시작한다.

- 3-channel 입력:
  - 기존 `rgb_model` 그대로 사용
- 1-channel depth 입력:
  - `1ch -> 3ch` repeat 또는 learnable `Conv2d(1, 3, kernel_size=1)` adapter
- occupancy multi-channel 입력:
  - channel 수가 3이면 기존 `rgb_model` 사용
  - channel 수가 다르면 `Conv2d(C, 3, kernel_size=1)` adapter
- panorama 입력:
  - 1차 구현은 panorama/crop 이미지를 기존 image stack으로 넣고 `rgb_model` 재사용
  - 성능 개선이 필요하면 이후 panorama 전용 encoder branch 추가

예상 구조:

```python
class ImageInputAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3):
        ...

    def forward(self, x):
        ...
```

`InternVLAN1ForCausalLM`의 기존 forward 본문을 크게 수정하지 않기 위해, S1 image tensor가 `rgb_model`에 들어가기 직전에 adapter를 적용한다.

### 5. S1 적용 계획

- training path에서 `traj_images`, `traj_depths`, `traj_tdmaps`, panorama 관련 입력을 `ImageProvider.get_s1_input()`에 넘긴다.
- provider output은 기본적으로 `[B, T, H, W, C]` 형태를 유지한다.
- `s1_image_mode=none` 또는 `s1_image_type=fpv`이면 기존 path와 동일해야 한다.
- `replace`이면 기존 FPV frame을 provider output으로 교체한다.
- `concat`이면 T dimension으로 이어 붙인다.
- T가 증가하는 경우 `video_frame_num`, `traj_depths`, mask 처리도 같이 맞춘다.

### 6. S2 적용 계획

- S2 prompt에 추가되는 이미지는 `ImageProvider.get_s2_extra()`에서 PIL image list로 반환한다.
- 1차 구현은 기존 prompt text/token 흐름을 유지하고 이미지만 append한다.
- S2 image config는 S1과 분리한다.
- 예:
  - S1: `bev`
  - S2: `panorama crop_120`
  - S1: `depth normalized`
  - S2: `bev occ`

### 7. Config/Argument 추가 계획

- `internnav/trainer/internvla_n1_argument.py`에 optional field만 추가한다.
- `scripts/train_eval/qwenvl_train/default_config.py`에도 같은 값을 전달한다.
- 기존 `bev_mode`는 바로 제거하지 않고 compatibility layer로 유지한다.
- 기존 BEV config는 새 config로 매핑한다.

예상 신규 argument:

```python
s1_image_type: str = "fpv"
s1_image_mode: str = "none"
s2_image_type: str = "fpv"
s2_image_mode: str = "none"
bev_image_type: str = "rgb"
depth_image_type: str = "raw"
panorama_mode: str = "full"
depth_source: str = "gt"
image_provider_debug: bool = False
```

### 8. 구현 순서

1. 현재 data batch key와 panorama availability 확인
2. `ImageProviderOutput`, `ImageProvider` 추가
3. 기존 `BEVProcessor`를 `ImageProvider` 내부에서 재사용하도록 연결
4. depth image 생성 함수 추가
5. panorama full/crop 생성 함수 추가
6. S1 training path에 `ImageProvider.get_s1_input()` 연결
7. S2 extra image path에 `ImageProvider.get_s2_extra()` 연결
8. `ImageInputAdapter` 추가
9. config/argument 추가 및 기존 `bev_mode` compatibility 처리
10. debug image 저장 기능 추가
11. unit/smoke test 추가
12. runner 명령으로 정상 작동 확인

### 9. 검증 계획

#### Unit/smoke 검증

- FPV no-op:
  - `s1_image_type=fpv`, `s1_image_mode=none`일 때 기존 tensor shape와 path 유지
- BEV:
  - RGB BEV output shape/dtype/range 확인
  - occupancy output shape/dtype/range 확인
- Depth:
  - raw/normalized/colormap output 확인
- Panorama:
  - full/crop_60/crop_120 output 개수와 shape 확인
- Encoder adapter:
  - 1-channel, 3-channel, multi-channel 입력이 기존 encoder 앞에서 정상 변환되는지 확인

#### 실행 검증

기준 명령:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/bev/base_s1.fpv_s2.fpv_rgb_gt.py --machine 5090
```

추가 검증 config를 만든 뒤 다음 조합을 확인한다.

- S1 FPV / S2 FPV baseline
- S1 BEV / S2 FPV
- S1 depth / S2 FPV
- S1 panorama / S2 panorama
- S1 BEV / S2 panorama

### 10. 주요 리스크

- panorama image가 현재 trainer/evaluator batch에 없을 수 있다. 없으면 dataset loader 확장이 먼저 필요하다.
- S1에서 T dimension을 늘리는 concat mode는 `video_frame_num`, loss mask, depth tensor shape를 함께 맞춰야 한다.
- 1-channel depth를 바로 encoder에 넣으면 기존 `rgb_model`과 shape가 맞지 않는다. adapter가 필요하다.
- estimated depth 모델 DAV2/UDV2는 checkpoint와 GPU memory 영향을 받는다.
- S2 image append는 prompt/image token 개수와 processor 입력 구조를 함께 확인해야 한다.

### 11. 완료 조건

- 통합 `ImageProvider`로 FPV/BEV/depth/panorama를 선택 가능하다.
- S1/S2가 서로 독립적으로 image type과 mode를 선택할 수 있다.
- 기존 FPV/default config가 깨지지 않는다.
- BEV/depth/panorama 각각 debug image를 저장해서 육안 확인 가능하다.
- 최소 smoke/unit test가 통과한다.
- 기준 runner 명령 또는 축소 실행 명령이 정상 작동한다.

