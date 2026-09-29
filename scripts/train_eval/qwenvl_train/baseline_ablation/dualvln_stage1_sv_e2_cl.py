"""`baseline/dualvln_stage1_full.py` + **scalevln 추가** + patch_embed channels_last, 2.0 epoch.

## 왜 이 실험인가

upstream `train_system2.sh:30` 은 scalevln 을 **주석 처리**한 채 배포됐다:

    vln_datasets=r2r_...,rxr_60cm_30_30 #,scalevln_125cm_0_30,scalevln_60cm_30_30

그 스크립트를 그대로 따른 우리 stage1 은 SR 44.5 로, 목표(wo-dagger 55.4)에 10.9점 못 미쳤다.
GitHub 이슈 #300 에서 같은 증상이 보고됐고(SR 41.8), 메인테이너 kellyiss 가
"55.1 의 효과는 scalevln 을 쓴 것으로 보인다. 추가해서 시도해 보라"고 답했다.
보고자가 scalevln 을 넣자 41.8 -> 48.3 (+6.5점) 으로 올랐다. 이슈는 미해결로 닫혔다.

우리 쪽 조사에서 코드/데이터/설정은 전부 배제됐다:
  - main 브랜치와 학습 레이블 209,741개 전 항목 일치, preprocess_qwen_2_visual 무변경
  - stop_weight 도 main 의 하드코딩 *5 와 동일, chat_template md5 동일
  - upstream argv parity 통과 (verify_params_stage1.py)
  - patch_embed: gemm/channels_last 가 loss 5,402 step 평균 차이 0.0009 (난수 수준)
즉 **scalevln 누락이 남은 유일한 유력 후보**다.

## 데이터 (실측)

    r2r×4 + rxr×4 (현재)        3,134,432 샘플
    scalevln×2 (추가)           2,901,136 샘플   (125cm_0_30 단독 1,450,568 실측 × 2)
    합계 10 setting             6,035,568 샘플   (1.93배)   1 epoch = 47,153 step

scalevln 은 등록된 3개 중 2개만 쓴다 — `scalevln_125cm_0_45` 는 데이터에
`pose.125cm_45deg` 컬럼이 없어 사용 불가(upstream 주석이 2개만 나열한 이유).

부수 효과: scalevln 은 4스텝(60도) 회전 비율이 43.0% 로 r2r/rxr(32.3%)보다 높다.
합치면 37.1%. 우리 모델이 짧은 회전을 남발하고(출력 41%) 공개 모델이 긴 회전을 내는(87%)
차이를 데이터 쪽에서 밀어주는 방향이다.

## epoch 선택: 2.0

우리 실측 기준:
    loss 0.30 -> 0.98 epoch,  loss 0.20 -> 1.19 epoch,  2.00 epoch -> loss 0.038 (과적합, SR 44.5)
메인테이너가 밝힌 정상 수렴 지표는 **최종 loss 0.2~0.3**.
이슈 보고자는 11.5M 노출(우리 데이터 기준 1.91 epoch) / 최종 loss 0.18 로 48.3 을 얻었다.

**2.0 epoch (94,306 step, 약 9.8일)** 을 택했다. 우리 이전 run 은 2.0 epoch 에서 과적합했지만
그때는 데이터가 3.13M 이었고 지금은 1.93배 다양하다. 이슈 보고자도 1.91 epoch 에서
loss 0.18 로 건강했고 48.3(그들의 최고)을 얻었다. ep1 과 짝을 이뤄 epoch 수의 영향을 직접 측정한다.

주의: ep1 과 ep2 는 **중첩 실험이 아니다**. cosine LR 이 각각 47k/94k 에 맞춰 감쇠하므로
ep2 의 47k 지점 체크포인트는 ep1 의 최종 모델과 다르다(LR 이 안 식었다).

## patch_embed = channels_last

torch 2.9.x 회귀(pytorch#174051)로 bf16 Conv3d 가 cuDNN 대신 ATen fallback 으로 떨어져
2.1배 느리고 wgrad 오차가 1.901e-01 이 된다. channels_last 는 입력 레이아웃만 바꿔
cuDNN 경로를 복구하며, **torch 2.10 의 원본 conv 와 forward/wgrad 가 비트 단위로 동일**하다
(입력 고정 후 torch.equal 확인). 즉 원저자 환경에 가장 가깝다.
실측 9.00~9.04 s/it (conv 19.17, gemm 9.01~9.03).

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-eval
"""

import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_QWENVL = os.path.dirname(_HERE)
sys.path.insert(0, _QWENVL)                            # import default_config
sys.path.insert(0, os.path.join(_QWENVL, "baseline"))  # import 부모 config (eval.py 는 이 dir 를 안 넣어준다)

import default_config as base  # noqa: E402
import dualvln_stage1_full as full  # noqa: E402

TRAIN_MACHINE = full.TRAIN_MACHINE  # 재-export 필수 (빼면 조용히 mini 데이터셋으로 떨어진다)

# 출력 디렉토리 이름에 'internvla-n1-system2' 를 반드시 포함시킨다:
#   internvla_n1_trainer.py 가 model_name_or_path 에 이 substring 이 있는지로 모델 클래스를
#   고른다. 없으면 조용히 Qwen2VL(2.5 도 아님)로 로드되어 stage2 가 망가진다.
EXP_NAME = "baseline_ablation/dualvln_stage1_sv_ep2_internvla-n1-system2"
assert "internvla-n1-system2" in EXP_NAME

PARAMS = replace(
    full.PARAMS,
    # scalevln 2개를 기존 8개 뒤에 추가 (upstream 주석의 선행 쉼표가 '누적'을 뜻한다)
    vln_datasets=full.PARAMS.vln_datasets + ",scalevln_125cm_0_30,scalevln_60cm_30_30",
    num_train_epochs=2.0,
    # 4,000 step 간격으로 곡선을 그린다. eval 은 1.5~2h 이므로 다른 노드에서 병렬로 돌리면
    # 학습이 끝나기 전에 추세가 보인다. 중간 ckpt 는 LR 이 덜 식어 최종 모델보다 과소평가된다는
    # 점에 주의 — 추세 확인용이고 절대 성능이 아니다.
    save_interval_steps=4000,
    save_total_limit=25,
    patch_embed_impl="channels_last",
    # eval: 모델 rank GPU 0-3, habitat GL 렌더 GPU 4-7 로 분리 (libnvidia-eglcore SIGABRT 회피)
    eval_nproc=4,
    eval_render_gpu_offset=4,
)
eval_cfg = base.make_eval_cfg(PARAMS)
