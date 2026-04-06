#!/bin/bash
# run_queue.sh

while true; do
    # queue.txt 첫 줄 읽기
    NEXT=$(head -1 queue.txt 2>/dev/null)
    
    if [ -z "$NEXT" ]; then
        sleep 30
        continue
    fi
    
    # 첫 줄 실행 후 queue.txt에서 제거
    echo "Running: $NEXT"
    bash "$NEXT"
    sed -i '1d' queue.txt
done