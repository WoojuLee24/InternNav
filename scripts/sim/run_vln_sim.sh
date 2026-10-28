#!/bin/bash
# InternVLA-N1 sim 연동 런처. tmux 세션 VLN-SIM 한 화면을 분할해 sim / server / bridge / web / gpu pane 을 띄운다.
# 호스트에서 실행한다(권장). gd_vln 컨테이너(internnav-habitat) 안에서도 되지만 docker 가 없어 sim 과 S1 TRT·S2 GGUF 는 못 띄운다.
#   호스트   : sim(isaac-sim 컨테이너)·모델 스택은 호스트에서, bridge/web/(single 서버)는 docker exec 로 gd_vln 컨테이너에서 실행
#   컨테이너 : sim 은 따로 띄워야 하고, single·`-b hf --s1 hf`(run_dual.sh)만 직접 띄운다. 그 밖에는 :5801 에 떠 있는 스택을 쓴다
#
# 사용: scripts/sim/run_vln_sim.sh [옵션] ["<instruction>"]
#       scripts/sim/run_vln_sim.sh stop [--gdm]   tmux 세션 + sim 프로세스 + 모델 스택 + bridge/web 정지 (--gdm: gdm 도)
#       scripts/sim/run_vln_sim.sh instr "<instruction>"   실행 중인 bridge 에 instruction 전달 ("" = 정지). web 입력과 같다
#   instruction 기본값은 빈 문자열: 터미널(instr) 이나 web 에서 줄 때까지 로봇은 정지해 있다
#   -m, --mode a|b         a = pixel_goal : S2 pixel goal → map goal → gdm planner   (gdm 쪽 ./lpp-sim-travmap)
#                          b = trajectory : S1 trajectory → MPC → /gdq/msg/cmd_vel   (gdm 쪽 ./lpp-sim-vln)
#                          (pixel_goal|trajectory 도 받는다. 기본 b)
#   -b, --backend single|hf|gguf   (기본 gguf)
#                          single : vln_server.py 단일 서버(HF, 동기). --quant 적용
#                          hf     : S2 HF(--quant) + S1(--s1) 분리 + dual_server
#                          gguf   : S2 GGUF Q8_0 + S1(--s1) 분리 + dual_server   (경량 기본 구성, report.md §0)
#       --s1 trt|hf        hf/gguf backend 의 S1 (기본 trt = TensorRT FP16 10 step). trt·gguf 는 dualvln 전용
#       --async | --sync   dual_server 실행 방식 (기본 async)
#       --model dualvln|navdp|wodagger   (기본 dualvln)
#   -q, --quant bf16|nf4|int8          single / hf 의 S2 양자화 (기본 bf16)
#       --map NAME|PATH|none           web_viz 배경 지도: scripts/sim/maps/<NAME>.{png,json} (noeun = noeun_station,
#                                      etri = etri_indoor, 기본 etri_indoor) 또는 .png/.json 을 뺀 경로
#       --sim SCENE|none   sim_docker/run_rbq10_etri_sim.sh <SCENE> (indoor|1f|outdoor|noeun|worldcup).
#                          기본은 --map 에서 정한다(noeun → noeun, etri_indoor → indoor, 그 밖 → none).
#                          sim 이 이미 돌고 있으면 새로 띄우지 않는다
#       --hunav            HuNavSim 보행자 (GUIDEDOG_HUNAV=1, = run_rbq10_*_dynamic_sim.sh)
#       --lookdown         30° look-down 카메라를 S2 "↓" 입력으로 사용 (sim 에도 GUIDEDOG_LOOKDOWN_CAM=1). LOOKDOWN=1 과 같다
#       --dry              goal/cmd_vel 발행 없이 계산·시각화만
#       --gdm              gdm 도 띄운다 (기본 off): 별도 터미널 창에서 Mode A ./lpp-sim-travmap, Mode B ./lpp-sim-vln 를
#                          sim /clock 이 나온 뒤 실행. 이때 stop 칸은 `stop --gdm` 이 되어 gdm 도 끈다
#   예) 노은역 Mode A, 경량 기본(GGUF Q8_0 + S1 TRT, async):
#         scripts/sim/run_vln_sim.sh -m a --map noeun "Walk straight and stop at the ticket gate."
#       Mode B, HF bf16 + S1 TRT, sync:   scripts/sim/run_vln_sim.sh -m b -b hf --sync --map noeun
#   기존 형식 {pixel_goal|trajectory} {dualvln|navdp|wodagger} [bf16|nf4|int8] "<instruction>" [dry] 은
#   -b single 로 그대로 동작한다.
#   gdm 은 기본으로 따로 띄운다: gdm_docker/run.sh → Mode A ./lpp-sim-travmap, Mode B ./lpp-sim-vln (--gdm 이면 자동)
#   웹 모니터: http://<host>:8090 (web pane), rviz2: 호스트에서 scripts/sim/rviz/open_rviz.sh
#   pane 을 닫아도 컨테이너 안 프로세스(sim·스택)는 남는다 → 끌 때 `scripts/sim/run_vln_sim.sh stop`
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"   # InternNav
GD=${GD_CONTAINER:-internnav-habitat}
GDM_CTR=${GDM_CONTAINER:-gdm_docker-gdm-1}

MODE=trajectory; BACKEND=gguf; S1=trt; DUAL=async; MODEL=dualvln; QUANT=bf16
MAP=etri_indoor; LOOKDOWN=${LOOKDOWN:-0}; DRY=false; INSTR=""
SIM=auto; HUNAV=0; GDM=0
SIM_DIR=$(cd "$ROOT/../../../.." && pwd)/sim_docker   # gd_project/modules/sim_docker (호스트 경로)
S=VLN-SIM
IN_CTR=$([[ -f /.dockerenv ]] && echo 1 || echo 0)
# 컨테이너 쪽 잔여 프로세스 (이전 bridge 가 남아 있으면 goal/cmd_vel 을 이중 발행한다)
GD_PROCS="^python3 scripts/sim/(vln_bridge_node|web_viz|dual_server|vln_server)\.py"

if [[ "${1:-}" == stop ]]; then
    if [[ $IN_CTR == 1 ]]; then
        pkill -f "$GD_PROCS" || true
        bash "$ROOT/scripts/sim/run_dual.sh" stop
        echo "[run_vln_sim] 컨테이너 쪽만 정지했다. sim·S1 TRT/S2 GGUF 컨테이너는 호스트에서 run_vln_sim.sh stop"
    else
        docker exec isaac-sim pkill -INT -f run_rbq10_sim.py 2>/dev/null && echo "[run_vln_sim] sim 정지 요청" || true
        "$ROOT/scripts/sim/run_vln_stack.sh" stop
        docker exec $GD pkill -f "$GD_PROCS" 2>/dev/null || true
        if [[ "${2:-}" == --gdm ]]; then
            docker exec $GDM_CTR pkill -f '[V]LN_SIM_GDM' 2>/dev/null || true   # /clock 을 기다리던 gdm 창 (tmux 세션 전 단계)
            docker exec $GDM_CTR tmux kill-session -t GDM-LPP 2>/dev/null && echo "[run_vln_sim] gdm 정지" || true
        fi
        echo "[run_vln_sim] stopped"
    fi
    tmux kill-session -t $S 2>/dev/null || true   # 마지막에: stop pane 자신도 이 세션 안에 있다
    exit 0
fi

if [[ "${1:-}" == instr ]]; then
    PUB=". scripts/sim/setup_ros_ws.sh && timeout 15 ros2 topic pub --once -w 1 /vln/instruction std_msgs/msg/String $(printf %q "{data: $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${2:-}")}")"
    if [[ $IN_CTR == 1 ]]; then cd "$ROOT" && bash -c "$PUB"; else docker exec -w /ws/src/InternNav $GD bash -c "$PUB"; fi
    exit 0
fi

if [[ "${1:-}" == pixel_goal || "${1:-}" == trajectory ]]; then   # 기존 positional 형식
    MODE=$1; MODEL=${2:?model dualvln|navdp|wodagger}; QUANT=${3:-bf16}; INSTR=${4:-$INSTR}
    [[ "${5:-}" == dry ]] && DRY=true
    BACKEND=single
else
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--mode)    MODE=$2; shift 2 ;;
            -b|--backend) BACKEND=$2; shift 2 ;;
            --s1)         S1=$2; shift 2 ;;
            --async)      DUAL=async; shift ;;
            --sync)       DUAL=sync; shift ;;
            --model)      MODEL=$2; shift 2 ;;
            -q|--quant)   QUANT=$2; shift 2 ;;
            --map)        MAP=$2; shift 2 ;;
            --lookdown)   LOOKDOWN=1; shift ;;
            --dry)        DRY=true; shift ;;
            --sim)        SIM=$2; shift 2 ;;
            --hunav)      HUNAV=1; shift ;;
            --gdm)        GDM=1; shift ;;
            -h|--help)    sed -n '2,/^set -e/p' "$0" | sed '$d'; exit 0 ;;
            -*)           echo "unknown option $1" >&2; exit 2 ;;
            *)            INSTR=$1; shift ;;
        esac
    done
fi

case "$MODE" in
    a|A|pixel_goal) MODE=pixel_goal ;;
    b|B|trajectory) MODE=trajectory ;;
    *) echo "unknown mode $MODE (a|b)" >&2; exit 2 ;;
esac
case "$MODEL" in
    dualvln) CKPT=/checkpoints/InternVLA-N1-DualVLN ;;
    navdp)   CKPT=/checkpoints/InternVLA-N1-w-NavDP ;;
    wodagger) CKPT=/checkpoints/InternVLA-N1-wo-dagger ;;  # 이전 공식 릴리스 "InternVLA-N1" (legacy config -> quant.legacy_config)
    *) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
case "$QUANT" in
    bf16|nf4|int8) ;;
    *) echo "unknown quant $QUANT" >&2; exit 2 ;;
esac
case "$BACKEND" in
    single|hf|gguf) ;;
    *) echo "unknown backend $BACKEND (single|hf|gguf)" >&2; exit 2 ;;
esac
case "$S1" in
    trt|hf) ;;
    *) echo "unknown s1 $S1 (trt|hf)" >&2; exit 2 ;;
esac
if [[ $BACKEND != single && $MODEL != dualvln && ( $S1 == trt || $BACKEND == gguf ) ]]; then
    echo "S1 TRT 엔진과 S2 GGUF 는 DualVLN 전용이다 → $MODEL 은 -b hf --s1 hf 또는 -b single" >&2; exit 2
fi
case "$MAP" in
    none|"") MAP_ARG="-p \"map:=''\"" ;;
    noeun)   MAP_ARG="-p map:=scripts/sim/maps/noeun_station" ;;
    etri)    MAP_ARG="-p map:=scripts/sim/maps/etri_indoor" ;;
    */*)     MAP_ARG="-p map:=$MAP" ;;
    *)       MAP_ARG="-p map:=scripts/sim/maps/$MAP" ;;
esac
case "$SIM" in
    auto) case "$MAP" in
              noeun|noeun_station) SIM=noeun ;;
              etri|etri_indoor)    SIM=indoor ;;
              *)                   SIM=none ;;
          esac ;;
    none|indoor|1f|outdoor|noeun|worldcup) ;;
    *) echo "unknown sim $SIM (indoor|1f|outdoor|noeun|worldcup|none)" >&2; exit 2 ;;
esac
GDM_HINT=$([[ $MODE == pixel_goal ]] && echo "./lpp-sim-travmap (Mode A)" || echo "./lpp-sim-vln (Mode B)")

if [[ $IN_CTR == 0 ]]; then
    command -v tmux >/dev/null || { echo "호스트에 tmux 가 없다" >&2; exit 1; }
    docker ps --format '{{.Names}}' | grep -qx "$GD" || {
        echo "gd_vln 컨테이너 $GD 가 없다 → 먼저 InternNav/docker/run-torch2.9.0-cuda13.0-habitat-jazzy-5090.sh" >&2; exit 1; }
fi
# gd_vln 컨테이너 안에서 돌릴 명령 → pane 에 칠 문자열 (호스트면 docker exec 로 감싼다)
in_gd() {
    if [[ $IN_CTR == 1 ]]; then echo "cd $ROOT && $1"
    else echo "docker exec -it -w /ws/src/InternNav $GD bash -c $(printf %q "$1")"; fi
}

INSTR_ARG=$([[ -n "$INSTR" ]] && echo "-p instruction:=\"$INSTR\"" || echo "")
LD_ARG=$([[ "$LOOKDOWN" == 1 ]] && echo "-p lookdown_ns:=/camera_lookdown/camera" || echo "")
TAG=$([[ $BACKEND == single ]] && echo "${MODEL}_${QUANT}" || echo "${BACKEND}_s1${S1}_${DUAL}")
LOG=scripts/sim/verify/out/$(date +%y%m%d_%H%M%S)_${MODE}_${TAG}   # InternNav 기준 상대경로 (컨테이너에서 생성)
ROS=". scripts/sim/setup_ros_ws.sh"
WAIT_SERVER="until curl -sf localhost:5801/health >/dev/null; do echo waiting for server :5801...; sleep 5; done"

# --- sim pane (호스트) ---
if [[ $SIM == none ]]; then
    SIM_CMD="echo '[sim] --sim none: sim 은 따로 띄운다'"
elif [[ $IN_CTR == 1 ]]; then
    SIM_CMD="echo '[sim] 컨테이너 안에는 docker 가 없다 → 호스트에서 sim_docker/run_rbq10_etri_sim.sh $SIM'"
else
    SIM_ENV="GUIDEDOG_HUNAV=$HUNAV"
    [[ $LOOKDOWN == 1 ]] && SIM_ENV+=" GUIDEDOG_LOOKDOWN_CAM=1"
    # 노은역은 nurec.usdz+mesh.obj 가 없고 splat+충돌 mesh 를 합친 collision.usdz 만 있다 (학습 데이터와 같은 파일)
    [[ $SIM == noeun ]] && SIM_ENV+=" GUIDEDOG_BACKGROUND_USD=${GUIDEDOG_BACKGROUND_USD:-/isaac-sim/workspace/NoeunStation/noeun_station_collision.usdz}"
    SIM_CMD="if docker exec isaac-sim pgrep -f run_rbq10_sim.py >/dev/null 2>&1; then echo '[sim] isaac-sim 에서 run_rbq10_sim.py 가 이미 실행 중 → 그대로 쓴다 (다른 씬이면 run_vln_sim.sh stop 후 재실행)'; else cd $SIM_DIR && $SIM_ENV ./run_rbq10_etri_sim.sh $SIM; fi"
fi

# --- server pane ---
if [[ $BACKEND == single ]]; then
    SERVER_CMD=$(in_gd "mkdir -p $LOG && python3 scripts/sim/vln_server.py --model_path $CKPT --quant $QUANT --log_dir $LOG/server 2>&1 | tee $LOG/server.log")
    [[ $IN_CTR == 0 ]] && SERVER_CMD="$ROOT/scripts/sim/run_vln_stack.sh stop; $SERVER_CMD"   # GPU 를 비운다
elif [[ $IN_CTR == 0 ]]; then
    SERVER_CMD="S1=$S1 S2=$BACKEND S2_QUANT=$QUANT $ROOT/scripts/sim/run_vln_stack.sh start $CKPT $DUAL && tail -F $ROOT/scripts/sim/verify/out/live/stack_{s2,s1,dual}.log"
elif [[ $BACKEND == hf && $S1 == hf ]]; then   # docker 없이 컨테이너 안에서 띄울 수 있는 유일한 분리 구성
    SERVER_CMD=$(in_gd "bash scripts/sim/run_dual.sh start $CKPT $DUAL 0 0 $QUANT && tail -F scripts/sim/verify/out/live/dual_server.log")
elif curl -sf localhost:5801/health >/dev/null; then
    echo "[run_vln_sim] WARN: 컨테이너에서는 $BACKEND/S1 $S1 스택을 띄울 수 없어 :5801 에 이미 떠 있는 서버를 쓴다: $(curl -s localhost:5801/health)"
    SERVER_CMD=$(in_gd "curl -s localhost:5801/health; echo; tail -F scripts/sim/verify/out/live/stack_dual.log")
else
    echo "[run_vln_sim] -b $BACKEND --s1 $S1 은 docker 가 필요하다 → 호스트에서 실행" >&2
    exit 1
fi

# --- bridge / web / gpu pane (gd_vln 컨테이너) ---
BRIDGE_CMD=$(in_gd "mkdir -p $LOG && $ROS && $WAIT_SERVER; python3 scripts/sim/vln_bridge_node.py --ros-args -p use_sim_time:=true -p mode:=$MODE -p dry_run:=$DRY -p log_dir:=$LOG $INSTR_ARG $LD_ARG 2>&1 | tee $LOG/bridge.log")
WEB_CMD=$(in_gd "mkdir -p $LOG && $ROS && python3 scripts/sim/web_viz.py --ros-args -p use_sim_time:=true -p port:=8090 $MAP_ARG 2>&1 | tee $LOG/web_viz.log")
GPU_CMD=$(in_gd "mkdir -p $LOG && nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 5 | tee $LOG/gpu.csv")

# 이전 실행의 bridge/web/server 정리 (단일 서버·run_dual 은 같은 :5801 을 쓴다)
if [[ $IN_CTR == 1 ]]; then pkill -f "$GD_PROCS" || true
else docker exec $GD pkill -f "^python3 scripts/sim/(vln_bridge_node|web_viz|vln_server)\.py" || true; fi

# 한 화면 분할: 왼쪽 위 sim / 오른쪽 위 server / 왼쪽 아래 bridge / 오른쪽 아래 web, 맨 아래 gpu(좁게)
tmux kill-session -t $S 2>/dev/null || true
tmux new-session -d -s $S -n vln -x "$(tput cols 2>/dev/null || echo 200)" -y "$(tput lines 2>/dev/null || echo 50)"
tmux set -t $S mouse on
tmux set -t $S pane-border-status top
tmux set -t $S pane-border-format " #{pane_index}: #{pane_title} "
P_SIM=$(tmux display -p -t $S:vln '#{pane_id}')
P_GPU=$(tmux split-window -v -l 6 -t "$P_SIM" -P -F '#{pane_id}')
P_BRIDGE=$(tmux split-window -v -t "$P_SIM" -P -F '#{pane_id}')
P_SERVER=$(tmux split-window -h -t "$P_SIM" -P -F '#{pane_id}')
P_WEB=$(tmux split-window -h -t "$P_BRIDGE" -P -F '#{pane_id}')
P_STOP=$(tmux split-window -h -t "$P_GPU" -P -F '#{pane_id}')
for x in "SIM:sim ($SIM)" "SERVER:server ($BACKEND$([[ $BACKEND != single ]] && echo " s1=$S1 $DUAL"))" "BRIDGE:bridge ($MODE)" "WEB:web :8090 (map $MAP)" "GPU:gpu" "STOP:stop (Enter = 전체 종료)"; do
    v=P_${x%%:*}; tmux select-pane -t "${!v}" -T "${x#*:}"
done
tmux send-keys -t "$P_SIM" "$SIM_CMD" C-m
tmux send-keys -t "$P_SERVER" "$SERVER_CMD" C-m
tmux send-keys -t "$P_BRIDGE" "$BRIDGE_CMD" C-m
tmux send-keys -t "$P_WEB" "$WEB_CMD" C-m
tmux send-keys -t "$P_GPU" "$GPU_CMD" C-m
tmux send-keys -t "$P_STOP" "$ROOT/scripts/sim/run_vln_sim.sh stop$([[ $GDM == 1 ]] && echo " --gdm")"   # Enter 없이 입력만
tmux select-pane -t "$P_BRIDGE"
echo "[run_vln_sim] mode=$MODE backend=$BACKEND$([[ $BACKEND != single ]] && echo " s1=$S1 $DUAL") sim=$SIM map=$MAP  gdm: $GDM_HINT"
echo "[run_vln_sim] logs: InternNav/$LOG   web: http://$(hostname -I | awk '{print $1}'):8090  rviz: scripts/sim/rviz/open_rviz.sh (host)   정지: scripts/sim/run_vln_sim.sh stop"
echo "[run_vln_sim] instruction: $([[ -n "$INSTR" ]] && echo "\"$INSTR\"" || echo "없음(정지 대기)") → web 또는 scripts/sim/run_vln_sim.sh instr \"<instruction>\""
# --gdm: 같은 화면 분할이 아니라 별도 터미널 창 (gdm 은 컨테이너 안에 자체 tmux 세션 GDM-LPP 를 만든다)
if [[ $GDM == 1 && $IN_CTR == 0 ]]; then
    GDM_SCRIPT=$([[ $MODE == pixel_goal ]] && echo lpp-sim-travmap || echo lpp-sim-vln)
    # VLN_SIM_GDM: 이 창을 찾아 끄는 표식. 이전 실행이 /clock 을 기다리며 남아 있으면 gdm 이 두 벌 뜬다 → 먼저 끈다.
    # 이전 세션 정리는 기다린 뒤(실행 직전)에 한다
    docker exec $GDM_CTR pkill -f '[V]LN_SIM_GDM' 2>/dev/null || true
    GDM_CMD="docker exec -it $GDM_CTR bash -lc ': VLN_SIM_GDM; until timeout 3 ros2 topic echo --once /clock >/dev/null 2>&1; do echo waiting for sim /clock...; sleep 2; done; tmux kill-session -t GDM-LPP 2>/dev/null; ./$GDM_SCRIPT'"
    if [[ -n "${DISPLAY:-}" ]] && command -v gnome-terminal >/dev/null; then
        gnome-terminal --title "GDM ($GDM_SCRIPT)" -- bash -c "$GDM_CMD" && echo "[run_vln_sim] gdm: 새 터미널 창에서 ./$GDM_SCRIPT"
    else
        tmux new-window -d -t $S -n gdm "$GDM_CMD"; echo "[run_vln_sim] gdm: DISPLAY 가 없어 tmux 창 '$S:gdm' 에서 ./$GDM_SCRIPT"
    fi
fi
if [[ -n "${TMUX:-}" ]]; then tmux switch-client -t $S; else tmux attach -t $S; fi
