#!/bin/bash
# Appends a session end marker to .claude/session-log.md
# Called by the Stop hook whenever Claude finishes responding

LOG_FILE="$(git rev-parse --show-toplevel 2>/dev/null)/.claude/session-log.md"
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
DATE=$(date '+%Y-%m-%d %H:%M')

mkdir -p "$(dirname "$LOG_FILE")"
echo "[$DATE] branch=$BRANCH" >> "$LOG_FILE"
