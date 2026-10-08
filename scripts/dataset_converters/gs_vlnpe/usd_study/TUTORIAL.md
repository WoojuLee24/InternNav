# USD 실습 시작하기

**목표**: "USD 파일을 읽어봤다"에서 **"USD를 직접 써봤다"**로 넘어가기.

USD는 3D 장면을 적어둔 문서다. 파워포인트 파일처럼 안에 물건들이 들어 있고,
**포토샵처럼 여러 장을 겹쳐서** 하나의 결과를 만든다. 이 두 가지가 전부다.

이 저장소의 USD·USDZ는 **전부 밖에서 받은 것**이고 우리 코드는 읽기만 한다
(`Layer.CreateNew` 같은 게 한 군데도 없다). 그래서 **원본 위에 덧칠 파일을 얹는 것**이
우리가 값을 바꿀 수 있는 유일한 방법이다.

---

## 0단계 · 눈으로 보기 (5분)

### 먼저 GUI를 쓴다

이 컨테이너에는 **Isaac Sim GUI가 이미 깔려 있고 `DISPLAY=:0`도 있다.**

```
/isaac-sim/isaac-sim.sh
```

여기에 USD를 열면 **Stage 창**(물건 트리) · **Property 창**(값) · **Layer 창**(겹친 파일과
어느 레이어가 이겼는지)이 다 있다. 둘러보기는 이쪽이 압도적으로 낫다.

> **`usdview`를 따로 깔지 말 것.** 바이너리도 없고 PySide6·PyOpenGL도 없어서 소스 빌드를 해야
> 하는데, 이 컨테이너의 `pxr`은 **Isaac이 쓰는 바로 그 빌드**다(`_isaac_sim/kit/python/.../pxr`).
> 여기에 다른 USD를 얹으면 Isaac이 깨질 수 있다. (`pip install usd-core`로는 usdview가 오지도
> 않는다 — 그 wheel은 headless 전용이다.) 실제로 이 환경은 pxr 이중화에 민감하다 —
> `pxr`을 `SimulationApp`보다 먼저 import하면 omni 확장이 전부 죽는 걸 겪었다.

### 터미널에서 빠르게 볼 때는 `usd_peek.py`

GUI는 뜨는 데 수십 초 걸리고 GPU를 쓴다. **값 하나만 빨리 보거나, 스크립트로 확인할 때**는
이쪽이 낫다(~2초).

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/usd_peek.py data/scene_data/mp3d_pe/17DRP5sb8fy/matterport_mesh/bed1a77d92d64f5cbbaaae4feed64ec1/fixed.usd
```

파일이 뭐라고 선언하는지(위쪽 축·단위·기본 물건)와 안에 뭐가 몇 개 있는지가 나온다.
안을 더 보려면 `--tree --depth 4`.

| | GUI | `usd_peek` |
|---|---|---|
| 둘러보기 | **훨씬 낫다** | 답답하다 |
| 뜨는 시간 | 수십 초 + GPU | ~2초 |
| 스크립트로 확인 | 안 됨 | **됨** |
| 깨진 sublayer 경고 | 찾아봐야 함 | **크게 찍어준다** |

**여기서 할 일**: 나중에 값을 바꿔볼 **물건 경로 하나를 골라 적어둔다**
(`/Root/...`로 시작하는 줄 중 `[Mesh]`인 것 아무거나).

## 1단계 · 덧칠 파일을 손으로 한 장 쓴다 (30분) ← **여기가 핵심**

개념은 이 한 단계로 끝난다. 나머지는 응용이다.

`logs/gs-vlnpe/usd_study/my/first.usda`를 만들고 아래를 넣는다.
(이미 만들어 둔 것이 있으니 열어서 고쳐도 된다.)

```
#usda 1.0
(
    # 원본의 "파일 전체 설정"은 아래 깔아도 안 올라온다 — 직접 적는다 (아래 설명)
    defaultPrim = "Root"
    upAxis = "Z"

    subLayers = [
        @../../../../data/scene_data/mp3d_pe/17DRP5sb8fy/matterport_mesh/bed1a77d92d64f5cbbaaae4feed64ec1/fixed.usd@
    ]
)

over "Root"
{
    over "isaacsim_bed1a77d92d64f5cbbaaae4feed64ec1"
    {
        over "geometry"
        {
            over "isaacsim_bed1a77d92d64f5cbbaaae4feed64ec1"
            {
                over "chunk000_group000_sub002"
                {
                    token visibility = "invisible"
                }
            }
        }
    }
}
```

읽는 법:

| 줄 | 뜻 |
|---|---|
| `subLayers = [ @...@ ]` | **"원본을 내 밑에 깔아라"** |
| `defaultPrim = "Root"` | 남이 이 파일을 가져다 쓸 때 기본으로 딸려올 물건 |
| `upAxis = "Z"` | 어느 축이 위쪽인가 — **원본에서 안 올라오므로 직접 적는다** |
| `over "..."` | "그 물건을 **덮어쓴다**" (없으면 새로 만들지 않고 원본 것을 가리킨다) |
| `token visibility = "invisible"` | 바꿀 값 |

`over`가 계층대로 중첩되는 게 처음엔 낯설지만, 그냥 **폴더 경로를 괄호로 쓴 것**이다.

### 꼭 알아야 할 규칙 하나 — 「파일 전체 설정」은 안 올라온다

아래에 깐 원본에서 **물건(prim)과 그 값은 올라오지만, 파일 전체 설정은 안 올라온다.**
실측으로 확인한 것이다:

| | 원본이 적어둔 값 | 덧칠 파일이 아무것도 안 적었을 때 읽히는 값 |
|---|:--:|:--:|
| 물건 수 | 121 | **121** (올라옴) |
| `upAxis` | Z | **Y** (USD 기본값) |
| `metersPerUnit` | 1.0 | **0.01** (USD 기본값) |
| `defaultPrim` | 있음 | **없음** |

물건 수가 멀쩡해서 잘 된 줄 알기 쉬운데, 축과 단위는 조용히 기본값으로 돌아간다.
그래서 이 세 가지는 **원본 것을 보고 덧칠 파일에 직접 적어야** 한다.

- `defaultPrim`을 빼먹으면 → 렌더할 때 **빈 화면**이 나온다
- `upAxis`/`metersPerUnit`을 빼먹으면 → 파일을 열어 크기·방향을 재는 쪽이 틀린 값을 본다

원본이 뭐라고 적어놨는지는 `usd_peek.py`로 원본을 열어 확인하면 된다.

### 확인

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/usd_peek.py logs/gs-vlnpe/usd_study/my/first.usda --prim /Root/isaacsim_bed1a77d92d64f5cbbaaae4feed64ec1/geometry/isaacsim_bed1a77d92d64f5cbbaaae4feed64ec1/chunk000_group000_sub002 --attr visibility
```

이렇게 나오면 성공이다:

```
지금 값  'invisible'
이 값을 주장하는 파일 (맨 위가 이긴 값):
  0. 'invisible'   from  logs/gs-vlnpe/usd_study/my/first.usda  ← 이긴 값
```

같은 질문을 **원본 파일에** 하면 `'inherited'`가 나온다. **원본은 안 바뀌었다.**

### 여기서 꼭 해볼 것 3가지

1. **물건 경로를 다른 것으로 바꿔본다** — 경로를 틀리면 조용히 아무 일도 안 일어난다.
   이게 가장 흔한 실수다(에러가 안 난다).
2. **값을 `"inherited"`로 되돌려본다** — 덮어쓰기를 끄는 것도 덮어쓰기다.
3. **`subLayers` 경로를 일부러 틀려본다** — 어떻게 실패하는지 보면 다음에 빨리 찾는다.

   **주의**: 위의 `--prim ... --attr visibility` 명령으로는 **차이를 못 느낀다.** 값이 여전히
   `'invisible'`로 나온다. 내가 적어둔 값은 **내 파일 자신에게서** 오기 때문에, 원본이 안 깔려도
   그대로 읽히기 때문이다. (실제로 이걸로 헷갈린 사례가 있어서 여기 적어둔다.)

   틀린 걸 확인하려면 **`--prim` 없이** 보면 된다:

   ```
   /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/usd_peek.py logs/gs-vlnpe/usd_study/my/first.usda
   ```

   | | 경로 맞을 때 | 경로 틀릴 때 |
   |---|---|---|
   | 물건 수 | **124** | **0** |
   | 위쪽 축 | Z | Y (기본값) |
   | 겹쳐진 파일 | 원본이 보임 | 내 파일만 |

   `usd_peek`은 이 경우 **맨 위에 경고를 크게 찍는다**(`--prim`을 줘도 찍는다).
   USD 자체는 경고 한 줄만 내고 그냥 넘어가므로(`skipping`) 놓치기 쉽다.

---

## 2단계 · 그 덧칠로 실제 렌더한다 (1시간)

만든 파일이 진짜로 그림을 바꾸는지 본다.

### 커맨드는 2개다 — 순서가 중요하다

**왜 2개인가**: 같은 원본을 두 번 렌더해도 픽셀이 조금 흔들린다(사진을 두 번 찍는 것과 같다).
그래서 **"그냥 두면 얼마나 흔들리나"를 먼저 재고**, 그다음 "원본 vs 덧칠"을 비교한다.
앞의 것이 **대조군**이고 `--control` 플래그로 돌린다. 이 단계를 빼먹으면 멀쩡한 것을 실패로
판정한다(실제로 한 번 그랬다).

| | 무엇을 비교 | 플래그 |
|:--:|---|---|
| ① 대조군 (**먼저**) | 원본 ↔ **원본** | `--control` |
| ② 시험 | 원본 ↔ **덧칠 레이어** | (없음) |

②는 ①이 남긴 숫자를 읽어서 판정한다. ①을 안 돌리면 **"대조군 없음 — 판정 보류"**가 나온다.

### 경로는 전부 인자로 바꿀 수 있다

아래 커맨드는 기본 경로를 **일부러 다 적어놓은** 것이다. 다른 씬·다른 자산으로 옮기려면
그 줄만 고치면 된다. 인자를 생략하면 같은 기본값이 쓰인다.

실행하면 맨 앞에 **어떤 파일을 읽고 어디에 쓰는지** 그대로 찍힌다:

```
[s6b] 읽는 파일:
   OK   원본 usdz     data/GS_USDZ/Subway/noeun_station_collision.usdz   (2,336,320,706 B)
   OK   씬 정보        .../apply_real/scene_meta/noeun_station_mid.json   (1,566 B)
   OK   카메라 경로     .../apply_real/paths/noeun_station_mid_random.json   (746,191 B)
[s6b] 쓰는 곳:
        덧칠 레이어     logs/gs-vlnpe/usd_study/s6_use_layers/b_noeun/override_visibility.usda
        결과          logs/gs-vlnpe/usd_study/s6_use_layers/b_noeun/s6_b_control.json
```

`없음`이 뜨면 그 경로가 틀린 것이다.

### B · 노은역 usdz (런타임 파이썬 조작을 레이어로)

**① 대조군 먼저**

```
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s6_use_layers.py --mode b --control --n_frames 4 --usdz data/GS_USDZ/Subway/noeun_station_collision.usdz --scene_meta scripts/dataset_converters/gs_vlnpe/apply_real/scene_meta/noeun_station_mid.json --paths_json scripts/dataset_converters/gs_vlnpe/apply_real/paths/noeun_station_mid_random.json --camera d455_nominal --out_dir logs/gs-vlnpe/usd_study/s6_use_layers/b_noeun --log_dir logs/gs-vlnpe/usd_study
```

**② 그다음 시험** (`--control`만 뺀다)

```
timeout --signal=KILL 3600 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s6_use_layers.py --mode b --n_frames 4 --usdz data/GS_USDZ/Subway/noeun_station_collision.usdz --scene_meta scripts/dataset_converters/gs_vlnpe/apply_real/scene_meta/noeun_station_mid.json --paths_json scripts/dataset_converters/gs_vlnpe/apply_real/paths/noeun_station_mid_random.json --camera d455_nominal --out_dir logs/gs-vlnpe/usd_study/s6_use_layers/b_noeun --log_dir logs/gs-vlnpe/usd_study
```

| 인자 | 뜻 |
|---|---|
| `--usdz` | 원본 usdz. 같은 폴더의 `subway_car.usdz` 등으로 바꿔볼 수 있다 |
| `--scene_meta` | 바닥 높이·범위가 든 json |
| `--paths_json` | 카메라가 지나갈 경로가 든 json |
| `--camera` | 렌즈 설정 (`d455_nominal` / `d455_30m`) |
| `--out_dir` | 결과가 쌓일 폴더 |
| `--apply_real_dir` | 위 두 json을 찾을 뿌리 폴더 (개별 지정 대신 이것만 줘도 된다) |

### A · vln_pe 씬 (텍스처 경로를 레이어로)

**① 대조군 먼저**

```
timeout --signal=KILL 2400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s6_use_layers.py --mode a --control --scene 17DRP5sb8fy --dataset vln_pe --n_frames 4 --base_variant isaacsim --usd_root data/scene_data/mp3d_pe --data_root data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --rtx_ambient 10.0 --film_iso 70 --out_dir logs/gs-vlnpe/usd_study/s6_use_layers/a_17DRP5sb8fy --log_dir logs/gs-vlnpe/usd_study
```

**② 그다음 시험**

```
timeout --signal=KILL 2400 /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s6_use_layers.py --mode a --scene 17DRP5sb8fy --dataset vln_pe --n_frames 4 --base_variant isaacsim --usd_root data/scene_data/mp3d_pe --data_root data/InternData-N1-v0.5-mini/vln_pe/traj_data/r2r --rtx_ambient 10.0 --film_iso 70 --out_dir logs/gs-vlnpe/usd_study/s6_use_layers/a_17DRP5sb8fy --log_dir logs/gs-vlnpe/usd_study
```

| 인자 | 뜻 |
|---|---|
| `--scene` | 씬 이름. **생략하면 90개 중 무작위**로 고른다 |
| `--base_variant` | 어느 USD 위에 덧칠할지 — `isaacsim`(기본, 렌더가 쓰는 것) / `fixed` / `fixed_docker` / `isaacsim_non_metric` |
| `--usd_root` | 씬 USD가 있는 뿌리 폴더 |
| `--data_root` | 궤적·실제 rgb가 있는 폴더 (비교 대상) |
| `--out_dir` | 결과가 쌓일 폴더 |

### 무엇을 보나

②의 마지막 줄들이 답이다.

```
[s6b] test: 평균 차이 0.6862 · 최대 123
[s6b] 대조군(이전 실행) 평균 차이 0.6862      <- 이 둘이 비슷하면 "같다"
[s6b] PASS
```

**시험 ≈ 대조군이면 덧칠이 기존 방식과 같은 그림을 낸 것이다.**
시험이 대조군보다 훨씬 크면 덧칠이 뭔가를 바꾼 것이다.

리포트: `logs/gs-vlnpe/usd_study/s6_use_layers/b_noeun/report.html`
(A는 `a_<씬이름>/report.html`)

### 참고 — 한 프로세스에서 씬은 2번까지만 올릴 수 있다

3번째로 씬을 올리면 죽는다
(`AnnotatorRegistryError: Annotator rgb is not attached to any render products`, 실측).
대조군을 **따로 실행**하는 이유가 이것이다. 한 번에 다 하려고 하지 말 것.

---

## 3단계 · 쓸모 있는 것을 하나 만든다 (반나절)

2단계까지 하면 개념은 다 익힌 것이다. 이제 실제로 파이프라인에 붙인다.

**지금 막혀 있는 지점**: 렌더러가 원본 USD를 **자기가 glob으로 찾는다**
(`04_render_obs_isaac.py`의 `find_scene_usd()`). 덧칠 파일을 옆에 놔둬도 무시한다.

그래서 필요한 것은 **"이 파일을 써라"라고 알려줄 optional 인자 하나**다.

- 이름 예: `--usd_override <경로>`
- **기본값 `None` = 지금과 100% 같은 동작** (프로젝트 가이드라인의 guard clause + 위임)
- `build_renderer(...)`가 이미 `usd_path`를 인자로 받으므로, `find_scene_usd()`의 결과를
  대체하기만 하면 된다

이걸 붙이면 이런 것들이 가능해진다:

| 만들 것 | 얻는 것 |
|---|---|
| 텍스처 경로 덧칠 | 컨테이너 밖에 심링크를 안 만들어도 됨 (S6-A가 이미 증명) |
| 노은역 visibility 덧칠 | 코드가 아니라 파일로 (S6-B가 이미 증명) |
| 조명 설정 덧칠 | `--light`/`--rtx_ambient`가 파일이 되어 diff·공유·재현 가능 |

---

## 4단계 · 조건을 바꿔가며 눈으로 확인한다 (1시간)

만든 것을 쓰는 단계다. 뷰어를 띄운 채 **조명·텍스처를 바꿔 같은 장면이 어떻게 달라지는지** 본다.

### 터미널 3개 — 폴더 이름을 세 곳에서 **똑같이** 쓴다

아래는 `logs/gs-vlnpe/usd_study/s7_run`으로 통일한 것이다. **변수를 쓰지 않고 경로를 그대로
적는다** — 변수를 쓰면 따옴표 이스케이프가 꼬이고, 창과 제어 파일이 다른 폴더를 가리키는 실수가 난다.

**① 렌더** — `--hold`로 카메라를 고정해야 조건 변화만 보인다

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s7_live_view.py --usd data/scene_data/mp3d_pe/17DRP5sb8fy/matterport_mesh/bed1a77d92d64f5cbbaaae4feed64ec1/isaacsim_bed1a77d92d64f5cbbaaae4feed64ec1.usd --poses gt --scene 17DRP5sb8fy --dataset vln_pe --light ambient_only --rtx_ambient 10.0 --film_iso 70 --hold --start_frame 20 --out_dir logs/gs-vlnpe/usd_study/s7_run
```

**② 창** — `--dir`가 위 `--out_dir`와 같아야 한다

```
/workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/gs_vlnpe/usd_study/s7_watch.py --dir logs/gs-vlnpe/usd_study/s7_run
```

**③ 조건 바꾸기** — 홑따옴표로 감싸면 그대로 들어간다

체커보드로 교체:
```
echo '{"light":"ambient_only","ambient":10.0,"iso":70.0,"albedo":null,"roughness":null,"colorspace":null,"frame":20,"hold":true,"fps":4.0,"texture":"logs/gs-vlnpe/usd_study/s7_run/assets/checker.png","texture_scale":null,"texture_rotate":null,"texture_translate":null}' > logs/gs-vlnpe/usd_study/s7_run/control.json
```

타일 4배 + 30도 회전:
```
echo '{"light":"ambient_only","ambient":10.0,"iso":70.0,"albedo":null,"roughness":null,"colorspace":null,"frame":20,"hold":true,"fps":4.0,"texture":"logs/gs-vlnpe/usd_study/s7_run/assets/checker.png","texture_scale":4,"texture_rotate":30,"texture_translate":null}' > logs/gs-vlnpe/usd_study/s7_run/control.json
```

원본으로 복귀:
```
echo '{"light":"ambient_only","ambient":10.0,"iso":70.0,"albedo":null,"roughness":null,"colorspace":null,"frame":20,"hold":true,"fps":4.0,"texture":null,"texture_scale":null,"texture_rotate":null,"texture_translate":null}' > logs/gs-vlnpe/usd_study/s7_run/control.json
```

조명만 바꾸기:
```
echo '{"light":"three_light","ambient":10.0,"iso":70.0,"albedo":null,"roughness":null,"colorspace":null,"frame":20,"hold":true,"fps":4.0,"texture":null,"texture_scale":null,"texture_rotate":null,"texture_translate":null}' > logs/gs-vlnpe/usd_study/s7_run/control.json
```

### 반영이 안 될 때 — 이 세 가지가 원인이다 (전부 실제로 겪음)

**① 렌더러가 옛 코드로 돌고 있다.**
파이썬 프로세스는 **시작할 때의 코드를 들고 간다.** 스크립트를 고쳐도 이미 돌던 프로세스엔
반영되지 않는다. `light`는 먹는데 `texture`는 무시되면 거의 이 경우다.

    확인:  grep -c "재질/텍스처 적용" logs/gs-vlnpe/usd_study/s7_run/live_view.log
           텍스처를 바꿨는데 0이면 옛 코드다 (구버전은 "재질 적용:"으로 찍는다)
    해결:  Ctrl+C 로 끄고 ① 을 다시 실행

**② 창이 다른 폴더를 보고 있다.**
렌더러의 `--out_dir`, 창의 `--dir`, `echo` 대상 경로가 **셋 다 같아야** 한다.
여러 번 띄우다 보면 폴더가 갈린다.

    확인:  ls -la --time-style=+%H:%M:%S logs/gs-vlnpe/usd_study/*/live_latest.jpg
           지금 시각으로 갱신되는 폴더가 살아 있는 렌더러다

**③ 여러 줄을 연달아 넣었다.**
렌더러는 프레임마다 한 번 읽으므로 **마지막 것만 남는다.** 세 줄을 붙여 넣으면 중간 상태가
스쳐 지나가 "변화 없음"으로 보인다. **한 줄씩 넣고 10초쯤 기다린다.**

    확인:  tail -f logs/gs-vlnpe/usd_study/s7_run/live_view.log
           `제어 변경: [...]` 과 `재질/텍스처 적용: {...}` 이 찍히면 받은 것이다

밝기만 숫자로 보려면:

```
/workspace/isaaclab/_isaac_sim/python.sh -c "import cv2; im=cv2.imread('logs/gs-vlnpe/usd_study/s7_run/live_latest.jpg'); print(im[im.shape[0]-270:, :480].mean())"
```

원본 ≈ 125, 체커보드 ≈ 147 이다.

### 순서대로 해볼 것

| | 바꿀 것 | 무엇을 보나 |
|:--:|---|---|
| 1 | `"light":"dome"` → `"three_light"` | 조명 레시피에 따라 밝기 분포가 어떻게 달라지나 |
| 2 | `"iso":130` | 노출만 올렸을 때 |
| 3 | `"colorspace":"raw"` | **텍스처 색 해석**이 틀리면 얼마나 어긋나나 |
| 4 | `"texture"` 에 `.../s7_run/assets/checker.png` | 텍스처가 **어디에 어떻게 붙는지** (체커보드는 자동 생성됨) |
| 5 | `"texture_scale":4,"texture_rotate":30` | UV 타일링·회전 |
| 6 | 전부 `null`로 | **원본으로 정확히 돌아오나** |

6번이 검증이다. 화면 밝기가 처음 값으로 돌아오면 성공이다(실측 차이 0.38 = 노이즈 수준).

### 참고 — 실측 변화폭 (씬1 vln_pe, frame 20)

```
텍스처 교체      39.5   ← 가장 큼
colorspace raw   21.6
dome             12.7
iso 130          12.5
three_light       9.2
원본 복귀         0.38  ← 노이즈 수준
```

### 막히면

- 화면에 `control.json ERROR`가 뜨면 json이 깨진 것이다 — 이전 상태를 유지하니 다시 쓰면 된다
- 화면 위 글자에 현재 조건이 항상 찍힌다(`light=... iso=... texture=...`) — 스냅샷만 봐도 설정을 안다
- 창에서 `s`를 누르면 그 순간 화면이 저장된다

---

## 막히면 볼 것

| 궁금한 것 | 어디 |
|---|---|
| 조건을 실시간으로 바꾸는 법 | 위 4단계 · `reports.md`의 「실행 중에 조명·텍스처 조건 바꾸기」 |
| 모르는 단어 | `reports.md`의 「용어」 절 |
| 파일 4개가 왜 있나 / 누가 뭘 쓰나 | S1 리포트의 「전체 그림」 |
| 덧칠이 실제로 통하나 (렌더 증거) | S6 리포트 2종 |
| 내가 만든 파일이 왜 안 먹나 | `usd_peek.py --prim ... --attr ...` 로 "이긴 값"을 확인 |

## 자주 하는 실수

- **`defaultPrim`을 빠뜨린다** → 검사할 땐 멀쩡한데 **렌더하면 빈 화면**이 나온다.
  덧칠 파일을 렌더에 쓸 거면 반드시 넣는다(원본 것을 그대로 복사).
- **물건 경로를 틀린다** → 에러 없이 조용히 무시된다. `usd_peek`으로 확인하는 습관.
- **`subLayers` 경로를 틀린다** → USD는 경고 한 줄 내고 넘어가고, **내가 적은 값은 그대로 읽힌다.**
  값만 보면 멀쩡해 보인다. **물건 수가 0인지**로 판단한다(`usd_peek`이 경고도 찍어준다).
- **원본을 고친다** → 하지 않는다. 덧칠 파일만 만든다.
  (원본이 안 바뀌었는지는 `sha256sum`으로 확인할 수 있다.)
