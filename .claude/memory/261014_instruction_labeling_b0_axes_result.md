# instruction_labeling — b0 축별 ablation (dev, 문장 지표) (2026-10-14 KST)

## 구조 변경 (먼저)
- 방법 = `configs/methods/<method>.yaml` 의 그룹별 인자 조합. 그룹: frames / route / delivery / style / model / gen.
- 텍스트·표 자원은 버전 키로: `configs/route.yaml`(words, start_turn_rate), `configs/style.yaml`(template·규칙 줄·정지 표현),
  `configs/delivery/<name>.yaml`, `configs/model/<name>.yaml`. 내용째 gen_hash 에 들어간다.
- 실행: `annotate.py --method b0 [--set <group>.<arg>=<value> ...]`, 결과 이름 자동 suffix (예: `b0_slight-omit_fewshot-3`).
  b1 이름은 b0 기반 변형이 SR 로 검증된 뒤에만 `methods/b1.yaml` 로 만든다 (2026-10-14 결정).
- 리팩터 검증: b0 요청(이미지 포함) 303/303, 2stage writer 프롬프트 303/303 이 이전과 글자 단위 동일 → 기존 산출물은 새 hash 로 이전
  (`migrated_from_hash` 기록). `b0_2stage-v1` → `b0_delivery-2stage-v1`.

## 축 정의
| --set | 뜻 | 겨냥한 b0 문제 |
|---|---|---|
| route.start_turn=gtrate-v1 | 시작 회전을 GT train 비율로만 말함 (30-90°: 14 %, 90-135°: 23 %, 135°+: 25 %) | 시작 회전 항상 언급 |
| route.slight=omit | 60° 미만 회전은 경로 단어에서 뺌 ("bear slightly" 없음) | bear slightly 40 %, 회전 과다 |
| style.stop=gtdist | 마지막 표현을 GT 비율로 지정 (stop 55 / wait 32 / stand 3 / 없음 10) | stop 100 % |
| style.detail=plain-v1 | 흔한 명사로, 색·재질 형용사 자제 | 형용사·GT 에 없는 명사 과다 |
| style.structure=room-v1 | 방·문을 나가고 지나고 들어가는 것 위주, 회전은 필요할 때만 | 방 단위 표현 부족 |
| style.fewshot=3 | GT train 문장 3개를 문체 예시로 (train scene 만, 같은 trajectory 제외) | 문체 전반 |

## 결과 (dev 303 ep; 같은 설정 재실행 잡음: CIDEr ±0.005, BLEU-4 ±0.002, METEOR ±0.001, goal_mismatch ±1.4 %p)
| 지표 | GT | b0 | start_turn-gtrate | **slight-omit** | stop-gtdist | detail-plain | structure-room | **fewshot-3** |
|---|---|---|---|---|---|---|---|---|
| BLEU-4 | .133 | .111 | .110 | **.121** | .113 | **.127** | .116 | **.121** |
| METEOR | .202 | .201 | .192 | .195 | .197 | .191 | .202 | **.203** |
| ROUGE-L | .383 | .366 | .363 | .369 | .363 | .362 | .367 | **.372** |
| CIDEr | .328 | .235 | .231 | **.277** | .227 | .245 | .226 | .233 |
| noun_precision | .428 | .323 | .305 | .307 | .324 | **.380** | .328 | .323 |
| goal_mismatch % | 35 | 50 | 56 | 54 | 53 | 54 | **46** | 55 |
| unique_ratio | 1.0 | .83 | .84 | .83 | .93 | **.68** | .81 | **.97** |
| content_vocab | 462 | 207 | 211 | 216 | 211 | **125** | 201 | 220 |
| sibling BLEU-4 | .13 | .85 | .81 | .84 | .78 | .89 | .86 | **.74** |
| 첫 단어 turn % | 16 | 68 | 14 (walk 74) | 64 | 67 | 66 | 67 | 65 |

- **slight-omit**: CIDEr +0.042 (잡음의 8 배). "bear slightly" 를 그대로 옮겨 쓰던 것이 사라짐. 첫 회전 오류 6.9→3.3 %.
- **fewshot-3**: BLEU-4·METEOR·ROUGE 소폭↑, 다양성 크게↑ (고유 97 %, 형제 유사도 0.74).
- **detail-plain-v1**: 명사 정확도↑(0.38)지만 **어휘 붕괴** (내용어 125, 고유 68 %, 문장 짧아짐) — 그대로는 못 씀. 더 약한 v2 필요.
- **start_turn-gtrate-v1**: 과보정 (첫 단어 walk 74 % vs GT 36 %), 지표 변화 없거나 하락 (METEOR −0.009, goal +6).
- **stop-gtdist**: 정지 표현 분포는 GT 와 같아지고 다양성↑, n-gram 지표는 약간↓ (CIDEr −0.009).
- **structure-room-v1**: goal_mismatch 만 −3.6 %p, 나머지 변화 없음 (회전 언급 수 그대로 → 지시를 거의 안 따름).

## 조합 (+ 이전 label: v278 = `val_unseen_v278_reproduce` (Kemal v278 재현, 수정·seed 없음), v278fix = v278 + key frame/hfov79/검은 프레임 수정, seed)
| 지표 | GT | v278 | v278fix | b0 | slight-omit | +fewshot-3 | +stop-gtdist | +structure-room |
|---|---|---|---|---|---|---|---|---|
| BLEU-4 | .133 | .096 | .102 | .111 | .121 | **.126** | .119 | .125 |
| METEOR | .202 | .174 | .180 | .201 | .195 | .201 | .197 | **.203** |
| ROUGE-L | .383 | .332 | .343 | .366 | .369 | **.378** | .372 | **.378** |
| CIDEr | .328 | .170 | .174 | .235 | .277 | **.283** | .267 | .279 |
| noun_precision | .428 | .238 | .246 | .323 | .307 | .313 | .313 | **.329** |
| goal_mismatch % | 35 | 61 | 61 | **50** | 54 | 57 | 60 | 55 |
| unique_ratio | 1.0 | .93 | .97 | .83 | .83 | .98 | .99 | .99 |
| sibling BLEU-4 | .13 | .75 | .73 | .85 | .84 | .72 | .67 | .68 |
| 알려진 SR (mini, s1 1 epoch, label 학습→GT 평가) | 42.5 (GT) | — | 31.9 | — | — | — | — | — |

- **b1 후보: `b0_slight-omit_fewshot-3`** — CIDEr 0.235→0.283 (사람 상한 0.328 의 86 %, b0 72 %, v278fix 53 %), BLEU-4·ROUGE 최고, 다양성 크게 개선.
- v278 → v278fix 의 수정(key frame·hfov·검은 프레임)은 문장 지표를 거의 안 바꿨다 (CIDEr .170→.174).
- 남은 문제: **goal_mismatch 가 b0 보다 나빠짐** (50→57 %). 도착 지점 묘사는 텍스트 규칙으로 안 고쳐진다 → 입력 이미지(도착 좌우 시점 등) 쪽 과제.
- 문장 지표 기준일 뿐이다. b1 확정은 follower SR / 2x2 SR 로 한다.

## frames.select=keyframes-goalside-v1 (도착 지점 좌·우 90° 시점 2장 추가) — 2026-10-14
- 정의 `configs/frames.yaml`, 렌더 `render/goal_views.py` (trajectory 단위, 끝 pose 재생 오차 < 1 cm 검사), dev 101 traj.
  캡션 "destination, looking to your left/right" (vln-s2-pipeline v4 와 같음).
- 구조 변경으로 frames 정의가 cfg 에 들어가 gen_hash 가 바뀜 → 기존 dev 산출물 hash 갱신 (요청 동일: b0 303/303, 2stage 303/303).

| 지표 | GT | v278fix | b0 | b0 + goalside | b0_slight-omit_fewshot-3 | + goalside |
|---|---|---|---|---|---|---|
| CIDEr | .328 | .174 | .235 | .231 | **.283** | .265 |
| BLEU-4 | .133 | .102 | .111 | .112 | **.126** | .121 |
| METEOR | .202 | .180 | .201 | .201 | .201 | .200 |
| goal_mismatch % | 35 | 61 | **50** | 50 | 57 | 55 |
| noun_precision | .428 | .246 | **.323** | .315 | .313 | .309 |

- **효과 없음**: b0 에선 전부 잡음 범위, 조합에선 goal_mismatch −2.3 %p(잡음 1.4 의 1.6 배)지만 CIDEr −0.018 로 손해. 채택 안 함.
- 원인 진단 (2stage 장면 요약의 랜드마크 목록 이용): GT 마지막 문장의 명사가
  도착 프레임 묘사에 있는 경우 28 %, 어느 key frame 에든 있는 경우 41 %, GT 마지막 문장에 명사가 없는 경우 14 %.
  → 도착 물체는 대부분 우리 프레임/VLM 묘사에 **보이지 않거나 다른 이름으로** 잡힌다. 좌·우 시점으로는 안 풀린다.
  goal_mismatch 는 단어 일치라 동의어(couch/sofa)도 불일치로 센다 — GT 끼리도 35 %.

## 정지 표현 유도 (분포 강제 대신) — 2026-10-14
- `style.stop=prefer-stop-v1`: "stop 권장, wait 나 장소로 끝내도 됨" / `prefer-stop-v2`: 중립 표현 ("stop 이 흔하지만 wait·장소로 끝내기도 자연스럽다, 맞는 것을 골라라").
  두 값 모두 품질검사의 정지 단어 필수 조건을 끈다 (`style.yaml stop_no_require`).
- 결과: gemma 는 **wait 를 한 번도 쓰지 않는다** (v1 stop 96 / none 4 %, v2 stop 92 / none 8 %; GT stop 55 / wait 29 / none 13).
  v2 는 goal_mismatch 를 줄였다 (b0 49.8→45.2, 조합 57.4→52.5; 잡음 ±1.4) — 장소로 끝내는 문장이 GT 도착 표현과 더 겹침. CIDEr 는 조합에서 −0.012.
- 구조 수정: cfg 에 **선택된 정지 표현 줄만** 넣도록 바꿈 → style.yaml 에 키를 추가해도 기존 방법의 gen_hash 불변 (확인: b0 218a4421fb2c 유지).
  이 수정으로 hash 가 한 번 바뀌어 완료된 폴더를 `label/rehash.py` 로 갱신 (요청 동일 확인 후). b0/train 은 실행 중이라 queue 에 rehash 를 다음 job 으로 넣음.

## 방 단위 표현·입력 프레임 옵션 — 2026-10-14 12시 KST
새 옵션 (모두 인자, 꺼짐이 기본 -> b0 hash aba44a796514 유지):
- `route.rooms=house-v1`: MP3D `.house` 방 바닥 다각형으로 reference_path 가 지나는 방 순서를 구해 경로 단어 뒤에 한 줄
  ("Rooms along the path (from scene annotation): bathroom -> bedroom -> hallway -> bedroom. You stop in the bedroom.").
  dev 101 traj: 점 coverage 평균 93 %, 방 1~5개. traj 104 는 GT "bathroom → bedroom → hall → bedroom" 과 정확히 일치.
- `frames.select=keyframes-startsurround-v1`: 출발 지점 좌·우·뒤 시점 3장 (`render/extra_views.py`, goal 시점도 같은 스크립트로 일반화).
- `frames.keyframe_deg=0` (처음 이름 frames.turn_deg; 결과 이름에 "." 이 들어가 셀 파일명 규칙과 충돌해 바꿈): 모든 waypoint 프레임 (기존 렌더 그대로, 코드 변경 없음).
- `frames.extra_rigs=[125cm_30deg]`: 30° 내려다본 이미지 추가. 이미지 상한은 프레임 수 × rig 수로 계산하도록 수정.
- `style.stop=mix-v1`: 같은 trajectory 의 형제 3문장에 stop / wait / 없음을 하나씩 (순서는 trajectory 마다 섞음).
- hash 규칙: 값이 None 인 옵션은 hash 에서 빠진다 (새 옵션을 꺼진 상태로 추가해도 기존 결과 hash 불변). 모든 산출물 rehash 완료.

결과 (dev 303, C = b0_slight-omit_fewshot-3; 잡음 CIDEr ±0.005, METEOR ±0.001, goal ±1.4 %p):
| 지표 | GT | b0 | C | C+rooms | C+startsurround(SS) | C+모든wp(WP) | C+LD30 | C+SS+WP | **C+SS+WP+LD30** |
|---|---|---|---|---|---|---|---|---|---|
| BLEU-4 | .133 | .111 | .126 | .126 | .130 | .127 | .124 | .129 | **.133** |
| METEOR | .202 | .201 | .201 | .201 | .205 | .207 | .202 | .207 | **.209** |
| ROUGE-L | .383 | .366 | .378 | .384 | .381 | .386 | .378 | .384 | **.387** |
| CIDEr | .328 | .235 | .283 | .269 | .297 | .292 | .279 | .296 | **.308** |
| goal_mismatch % | 35 | 50 | 57 | 62 | 58 | 56 | 55 | 56 | 54 |
| "enter" % | 22 | 5 | 5 | 13 | 7 | 12 | 10 | 9 | 6 |
| "hallway" % | 18 | 27 | 33 | **66** | 26 | 31 | 30 | — | — |

- **최고: C + startsurround + 모든 waypoint + LD30** — CIDEr 0.308 (사람 상한의 94 %), BLEU-4 = 사람, METEOR·ROUGE 는 사람 상한 이상.
- rooms-house-v1: 방 이름은 쓰게 하지만 "hallway" 를 과하게 씀 (66 % vs GT 18 %; MP3D 의 작은 복도 region 까지 다 나열) → 조합에서 CIDEr −0.013. 방 순서를 다듬은 v2 가 필요 (복도 생략·짧은 통과 제거 강화 등).
- "exit/leave" 는 어떤 옵션으로도 거의 안 나온다 (GT 16 % / 5 %).
- b0 의 단일 축: startsurround CIDEr +0.013, 모든 waypoint +0.016 / METEOR +0.007, LD30 goal_mismatch −4.9 %p.

## 전체 split 생성 완료 (node1_norepeat queue, 2026-10-14 10:58~11:42 KST)
- b0 / b0_slight-omit / b0_slight-omit_fewshot-3 / b0_delivery-2stage-v1 × {val_unseen 1,839, train 10,683}.
- train 10,683 → **10,684 로 수정**: jh4fc5c5qoQ 의 lerobot 82/85 는 지시문이 같지만 다른 trajectory (raw 5258/7415). 매핑을 build_relabel_dataset 과 같은 규칙(같은 문장이면 미사용 후보 순서대로)으로 바꾸고 (`label/sources.py`), 4 label 모두 7415 생성 + 잘못 묶였던 5258 재생성 (slight-omit, fewshot-3).

## InternNav 학습·평가 준비 (2026-10-14 13시 KST)
- 4 label (b0, b0_slight-omit, b0_slight-omit_fewshot-3, b0_delivery-2stage-v1) × train/val_unseen 을 `data-vol2/relabel_half/labels/` 에 복사.
- `build_relabel_dataset.py --splits train,val_unseen` (새 옵션, 기본값은 기존 3 split) 로 `vln_ce_<L>` 빌드·평가 yaml 생성.
- `relabel_half/make_cells.py` 로 s1 셀 3종 × 4 label. node2: b0, b0_delivery-2stage-v1 / node3: b0_slight-omit_fewshot-3, b0_slight-omit.
- 비교 기준 (같은 설정, 기존 보고서): GT→GT 42.5, v278fix→GT 31.9, v278fix_fewshot-3→GT 33.6.
- 2026-10-14 15시 KST: C+WP (`b0_slight-omit_fewshot-3_keyframe_deg-0`), C+SS (`b0_slight-omit_fewshot-3_select-keyframes-startsurround-v1`) 전체 생성
  (train 10,684 / val_unseen 1,839, 예외 0) → relabel_half 셀 → node4. LD30 은 제외 (사용자 결정; 프롬프트가 이미지 수를 프레임 수로 세고 rig 구분 캡션이 없는 결함도 있음).
