#!/bin/bash
# run_queue.sh

# 큐 파일을 인자로 받는다 (생략 시 기존 동작인 ./queue.txt).
#   bash run_queue.sh /home/irteam/data-vol2/queues/node1.txt
# 로그는 큐 파일 이름을 따라 자동 분리된다 -> node1.txt 면 run_queue_node1.log.
# (노드별 큐가 data-vol2 공유 스토리지에 있으므로 로그를 나누지 않으면 서로 덮어쓴다.)
QUEUE_FILE="${1:-queue.txt}"
LOG_FILE="${2:-run_queue_$(basename "$QUEUE_FILE" .txt).log}"

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

while true; do
    # 줄 번호로 가져와서 sed 특수문자 문제 회피
    LINE_NUM=$(grep -n -m1 '^\s*[^#]' "$QUEUE_FILE" 2>/dev/null | cut -d: -f1)

    if [ -z "$LINE_NUM" ]; then
        printf "\r[$(date '+%Y-%m-%d %H:%M:%S')] Queue empty, waiting 30s..."
        sleep 30
        continue
    fi

    NEXT=$(sed -n "${LINE_NUM}p" "$QUEUE_FILE")

    # 남은 non-comment 라인 수 확인
    REMAINING=$(grep -c '^\s*[^#]' "$QUEUE_FILE" 2>/dev/null || echo 0)

    log "========================================"
    if [ "$REMAINING" -gt 1 ]; then
        log "Running: $NEXT"
        log "========================================"

        # 줄 번호 기반으로 comment out (특수문자 안전)
        sed -i "${LINE_NUM}s|.*|# ${NEXT}|" "$QUEUE_FILE"
    else
        log "Running (last, repeating): $NEXT"
        log "========================================"
        # 마지막 라인은 done 처리 없이 반복 실행
    fi

    eval "$NEXT" 2>&1 | tee -a "$LOG_FILE"

    log "Finished: $NEXT"
    log "========================================"
done
