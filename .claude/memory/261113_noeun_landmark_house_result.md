# 노은역 랜드마크 지도 → `.house` 생성 (2026-11-13)

## 왜 했나

평가셋 40개 중 **21개가 못 쓸 라벨**이었다. 목적지를 아무 데나 찍고 "거기 멈출 만한 게 있나"를
나중에 찾았기 때문이다. 대상이 목적지에서 16 m 밖에 있는 경우도 있었다.

근본 원인은 `stop_cue: null` 이다. MP3D 는 `.house`(사람이 붙인 물체 주석)에서 멈출 대상을 고르는데,
노은역은 GS 스캔이라 `.house` 가 없어서 31B VLM 이 멈출 대상을 **지어냈다**.

→ **`.house` 를 만들어 주면 기존 MP3D 파이프라인을 한 줄도 안 고치고 그대로 쓴다.**

## 용어

| 말 | 뜻 |
|---|---|
| 랜드마크 | "저기서 멈춰라"로 지목되는 물건. 바닥·벽은 어디에나 있어 못 쓴다 |
| 대표 이름 | 묶이는 기준. `.house` 의 `raw_name` |
| 다른 이름 | 같은 물체를 부르는 다른 말. 문장 만들 때 고른다 |
| 공통 class | 여러 종류를 묶는 윗단계 이름. `.house` 의 `mpcat40_name` |
| 목적지 후보 | 랜드마크 옆 0.6~2 m 의 "서는 자리". 이게 에피소드의 끝점이 된다 |
| CRITICAL 구멍 | `gs`=렌더에 바닥이 비침 · `mesh`=로봇이 빠짐 |

## 순서를 뒤집었다

```
전: 목적지 아무 데나  →  멈출 대상 찾기      →  16 m 밖에 있기도
후: 물체 먼저 찾기    →  그 옆 2 m 를 목적지  →  설계상 3 m 안
```

로봇이 랜드마크 옆 약 1 m 에 서므로 **목적지에서 최대 3 m** = SR@3m 충족.

## 결과

| 단계 | 수 |
|---|---|
| 씬 스캔 | 167곳 × 4방향 = 666장 |
| Gemma 이름 | 531종류 |
| GroundingDINO 통과 어휘 | 163 → 대표 100 |
| 검출 | 4,998건 |
| 랜드마크 | **812** (77종류) |
| 목적지 후보 (기둥·표지판·자동차 제외) | **1,412** (필요 1,280) |
| 목적지–랜드마크 거리 | 중앙 1.31 m · **최대 2.00 m** |

## 새 코드

| 파일 | 하는 일 |
|---|---|
| `noeun_scene_scan.py` | 격자로 역 안 사진 찍기 (`--make_spots` 는 habitat, 렌더는 isaac) |
| `69_scene_landmarks.py` | Gemma 이름 → 어휘 검증 → DINO+SAM 검출 → 3D 묶기 |
| `70_make_house.py` | 랜드마크 → `.house`. PCA 로 축·반지름. `load_objects` 로 되읽어 검증 |
| `71_landmark_viz.py` | 단계별 시각화 (NanumGothic, 글자 17/14/12) |
| `72_landmark_tool_data.py` | 웹 도구용 데이터 (사진 506장 data URI, 13 MB) |
| `73_goal_candidates.py` | 구멍 거르고 목적지 후보 뽑기 |
| `corrections.py` | 사람이 고친 내용을 70·73 이 **같이** 읽는다 |
| `21b_refine_paths_parallel.py` | 24 샤드 병렬 refine (순차와 bit-identical 확인) |
| `noeun_goal_snap_fast.py` | 목표로 순간이동 후 1프레임 — 7분 → **7초** (60배) |

## `corrections.json` — 사람이 고친 것을 한 파일로

70 과 73 이 **같은 파일**을 읽는다. 따로 넘기면 한쪽만 고쳐지는 사고가 난다.

```json
{"exclude_types": ["pillar"],            // 머리 명사 묶음째
 "exclude_names": ["white sign"],        // 종류 하나 (green sign 은 남음)
 "exclude_ids":   [372],                 // 랜드마크 하나 (landmarks.json 순서, 0부터)
 "rename":     {"digital sign": "exit sign"},
 "rename_ids": {"501": "ticket gate"},
 "aliases":    {"exit sign": ["station sign"]},
 "alias_ids":  {"501": ["gate"]},
 "common":     {"exit sign": "sign"}}
```

적용 순서는 **이름 먼저, 빼기 나중**. 화면에 보이는 이름이 곧 거르는 대상이기 때문이다.

이름을 바꿔도 3D 상자는 **원래 이름의 검출 점들**로 계산한다 (`orig_name` 보관).

## 웹 도구 (`scripts/apptainer/tools/`)

| 파일 | 쓰임 |
|---|---|
| `landmark_report.html` | 단계별 리포트 + **종류별 사진 갤러리**. 제외·이름·묶음을 여기서 고친다. db 자동 저장 |
| `landmark_editor.html` | 지도에서 랜드마크 확인·수정 |

`logs/` 는 gitignore 라 도구 원본만 여기에 둔다. 데이터(`data.json`)는 72 로 다시 만든다.

## 확인한 것

- `.house` 되읽기: `load_objects` 가 506개 읽음 (기대 506)
- 공통 class: `exit/white/green/blue sign` → `.house` 의 mpcat40 `sign` 57개로 합쳐짐
- 메시 광선 ≈ Isaac depth: **96.2 %가 0.3 m 안** (Isaac depth 는 `max_depth=10` 으로 정규화돼 있어 ×10 필요)
- Gemma vs DINO+SAM 거리: 좌표 1000 배율 보정 후 **중앙 차이 0.09 m** (22개 중 19개가 0.5 m 안)
- `load_holes(path)` 기본값 = 기존 동작 (mesh 58개) · `('mesh','gs')` 로 197개

## 겪은 함정

- `apptainer.conf` 의 `mount tmp = yes` 때문에 `%post` 에서 `/tmp` 가 호스트로 가려진다 → `%files` 는 `/opt` 로
- Isaac 렌더는 **GPU 0 에서만** 된다 (GPU 1 은 `No device could be created`). GPU 1 은 Gemma 용
- `pkill -f <패턴>` 이 **자기 셸을 죽인다** (exit 144) → `pgrep` 으로 자기 제외
- 학습 `reference_path`/`goals` 는 **habitat 좌표** — 그대로 쓰면 계단 판정이 틀린다 (57 % → 실제 15.8 %)
- GroundingDINO 텍스트 **256 토큰 한계** → 어휘를 700자 단위로 쪼갠다
- 단수화 버그: `tiles` → `til` 이라 `white tiles` 가 제외를 빠져나갔다. `es` 는 sh/ch/s/x/z 뒤에서만 떼야 한다
- `grounding-dino-tiny` 는 큰 기둥을 `trash`(0.44), 바닥을 `station map` 으로 봤다 → **base 로 교체**하니 사라짐
- 아티팩트 뷰어는 `<a download>` 를 막는다 → `downloads` capability 필요

## 사람이 정한 것

- 격자 5 m · 어휘 통과 rate ≥ 0.8 · score ≥ 0.35 · 2회 관측
- 구멍은 **CRITICAL 만** (minor 1,993곳까지 빼면 역을 못 쓴다)
- **기둥·표지판·자동차·차양·화살표 제외** (297개, 37 %).
  물체를 못 찾아서가 아니라 **흰 기둥이 103개라 말로 지목이 안 되기 때문**
- 랜드마크당 목적지 **3자리** (2자리면 942개로 1,280에 모자람)

## 다음

1. 리포트 갤러리에서 사람 검수 → `corrections.json` 확정
2. `70_make_house.py --corrections` 로 `.house` 최종본
3. `73` 으로 목적지 확정 → 경로 생성 (`22_select_paths --hole_kinds mesh,gs`)
4. Gemma-4-31B 로 문장 생성 (이번엔 `stop_cue` 가 채워진다)
5. 학습 400 / 1,200 두 셀 → 평가 80
