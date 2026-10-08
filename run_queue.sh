#!/bin/bash
# run_queue.sh

# 큐 파일을 인자로 받는다 (생략 시 기존 동작인 ./queue.txt).
#   bash run_queue.sh /home/irteam/data-vol2/queues/node1.txt
#
# 로그는 **큐 파일과 같은 디렉토리**에 큐 이름을 따라 저장된다:
#   /.../queues/node1.txt -> /.../queues/run_queue_node1.log
#   /.../queues/status_node1.txt (한 줄 요약 — 노드별 현황을 한눈에 보기 위한 것)
# 큐가 공유 스토리지(data-vol2)에 있으므로 어느 노드에서든 전체 진행을 볼 수 있다:
#   tail -n2 /home/irteam/data-vol2/queues/status_node*.txt
QUEUE_FILE="${1:-queue.txt}"
QDIR="$(cd "$(dirname "$QUEUE_FILE")" && pwd)"
QNAME="$(basename "$QUEUE_FILE" .txt)"
LOG_FILE="${2:-$QDIR/run_queue_$QNAME.log}"
STATUS_FILE="$QDIR/status_$QNAME.txt"
# 명령이 이 시간 안에 0이 아닌 코드로 죽으면 그만큼 대기한 뒤 다음 루프로 간다.
FAIL_BACKOFF="${FAIL_BACKOFF:-60}"

# 한 줄 요약 갱신. 로그는 길어서 훑기 어려우므로 현재 상태만 따로 남긴다.
status() {
    printf '%s | host=%s pid=%s | %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" "$(hostname -s)" "$$" "$*" > "$STATUS_FILE" 2>/dev/null || true
}

if [ ! -f "$QUEUE_FILE" ]; then
    echo "큐 파일이 없다: $QUEUE_FILE" >&2
    exit 1
fi

# 같은 큐 파일에 러너가 둘 이상 붙으면 서로 다른 라인을 동시에 집어 GPU/포트가 충돌한다
# (실제로 2026-09-11 에 eval 3건이 이렇게 날아갔다). 큐 파일당 하나만 허용한다.
LOCK="/tmp/run_queue_$(basename "$QUEUE_FILE" .txt).lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "이미 이 큐에 러너가 붙어 있다: $QUEUE_FILE (lock: $LOCK)" >&2
    exit 1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "===== run_queue.sh started (PID=$$) queue=$QUEUE_FILE log=$LOG_FILE ====="
status "STARTED  queue=$QUEUE_FILE"

while true; do
    # 줄 번호로 가져와서 sed 특수문자 문제 회피
    LINE_NUM=$(grep -n -m1 '^\s*[^#]' "$QUEUE_FILE" 2>/dev/null | cut -d: -f1)

    if [ -z "$LINE_NUM" ]; then
        printf "\r[$(date '+%Y-%m-%d %H:%M:%S')] Queue empty, waiting 30s..."
        status "IDLE  (큐에 실행할 job 없음)"
        sleep 30
        continue
    fi

    NEXT=$(sed -n "${LINE_NUM}p" "$QUEUE_FILE")

    # 남은 non-comment 라인 수 확인
    REMAINING=$(grep -c '^\s*[^#]' "$QUEUE_FILE" 2>/dev/null || echo 0)

    log "========================================"
    status "RUNNING  (남은 job $REMAINING)  $NEXT"
    if [ "$REMAINING" -gt 1 ]; then
        log "Running: $NEXT"
        log "========================================"

        # 줄 번호 기반으로 "# " 를 앞에 붙이기만 한다.
        # 명령 내용을 sed 식에 넣으면 안 된다 - 명령에 '|' 가 들어가면
        # (예: STAGE1_CKPT=$(ls -dt ... | head -1) ...) 구분자가 깨져
        #   sed: -e expression #1, char 127: unknown option to `s'
        # 로 실패하고, 줄이 comment out 되지 않아 같은 명령을 무한 재실행한다.
        sed -i "${LINE_NUM}s/^/# /" "$QUEUE_FILE"
    else
        log "Running (last, repeating): $NEXT"
        log "========================================"
        # 마지막 라인은 done 처리 없이 반복 실행
    fi

    START_TS=$(date +%s)
    eval "$NEXT" 2>&1 | tee -a "$LOG_FILE"
    # 파이프라인이라 $? 는 tee 의 것이다. job 자체의 종료코드는 PIPESTATUS[0].
    RC=${PIPESTATUS[0]}
    ELAPSED=$(( $(date +%s) - START_TS ))

    status "FINISHED rc=$RC ${ELAPSED}s (남은 job $((REMAINING-1)))  $NEXT"
    log "Finished (rc=$RC, ${ELAPSED}s): $NEXT"
    log "========================================"

    # 즉시 실패(설정/데이터 오류 등)한 명령을 곧바로 다시 돌리면 로그가 폭주한다.
    # 마지막 라인은 comment out 되지 않고 반복되므로 특히 위험하다.
    if [ "$RC" -ne 0 ] && [ "$ELAPSED" -lt "$FAIL_BACKOFF" ]; then
        log "Failed in ${ELAPSED}s (<${FAIL_BACKOFF}s) -> ${FAIL_BACKOFF}s 대기 후 계속"
        sleep "$FAIL_BACKOFF"
    fi
done
