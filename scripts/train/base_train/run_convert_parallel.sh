#!/usr/bin/env bash
# Run convert_parquet_to_npz.py and convert_ply_to_npy.py in parallel
# across all top-level subdirectories of InternData-N1/vln_n1/traj_data,
# then verify all converted files match their sources.
#
# Usage:
#   bash scripts/train/base_train/run_convert_parallel.sh
#   bash scripts/train/base_train/run_convert_parallel.sh --overwrite

set -euo pipefail

ROOT_DIR="/home/irteam/data-vol1/InternData-N1/vln_n1/traj_data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/tmp/convert_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

EXTRA_ARGS="${*}"  # e.g. --overwrite (verify-only is handled internally)

# ── helper: wait for a list of pids, return number of failures ──────────────
wait_pids() {
    local label="$1"; shift
    local -n _pids=$1; shift
    local -n _names=$1
    local failed=0
    for i in "${!_pids[@]}"; do
        if wait "${_pids[$i]}"; then
            :
        else
            echo "FAILED $label job: ${_names[$i]} (pid=${_pids[$i]})"
            failed=$((failed + 1))
        fi
    done
    echo "$failed"
}

# ── phase 1: conversion ──────────────────────────────────────────────────────
echo "=== [1/2] Conversion: $ROOT_DIR ==="
echo "Logs: $LOG_DIR"
echo ""

subdirs=()
pids_pq=()
pids_ply=()

for subdir in "$ROOT_DIR"/*/; do
    name=$(basename "$subdir")
    subdirs+=("$name")

    python "$SCRIPT_DIR/convert_parquet_to_npz.py" \
        --root-dir "$subdir" $EXTRA_ARGS \
        >"$LOG_DIR/${name}_parquet.log" 2>&1 &
    pids_pq+=($!)

    python "$SCRIPT_DIR/convert_ply_to_npy.py" \
        --root-dir "$subdir" $EXTRA_ARGS \
        >"$LOG_DIR/${name}_ply.log" 2>&1 &
    pids_ply+=($!)

    echo "  Launched $name  pq_pid=${pids_pq[-1]}  ply_pid=${pids_ply[-1]}"
done

echo ""
echo "Waiting for parquet conversion jobs..."
conv_fail=0
for i in "${!pids_pq[@]}"; do
    if wait "${pids_pq[$i]}"; then :; else
        echo "  FAILED parquet: ${subdirs[$i]} (pid=${pids_pq[$i]})"
        conv_fail=$((conv_fail + 1))
    fi
done

echo "Waiting for ply conversion jobs..."
for i in "${!pids_ply[@]}"; do
    if wait "${pids_ply[$i]}"; then :; else
        echo "  FAILED ply: ${subdirs[$i]} (pid=${pids_ply[$i]})"
        conv_fail=$((conv_fail + 1))
    fi
done

echo ""
echo "=== Conversion done. Failed jobs: $conv_fail ==="
echo ""
echo "--- Parquet conversion summary ---"
grep -h "^Done\|^Verification" "$LOG_DIR"/*_parquet.log || true
echo ""
echo "--- PLY conversion summary ---"
grep -h "^Done\|^Verification" "$LOG_DIR"/*_ply.log || true

# ── phase 2: verification ────────────────────────────────────────────────────
echo ""
echo "=== [2/2] Verification ==="
echo ""

pids_vpq=()
pids_vply=()

for subdir in "$ROOT_DIR"/*/; do
    name=$(basename "$subdir")

    python "$SCRIPT_DIR/convert_parquet_to_npz.py" \
        --root-dir "$subdir" --verify-only \
        >"$LOG_DIR/${name}_parquet_verify.log" 2>&1 &
    pids_vpq+=($!)

    python "$SCRIPT_DIR/convert_ply_to_npy.py" \
        --root-dir "$subdir" --verify-only \
        >"$LOG_DIR/${name}_ply_verify.log" 2>&1 &
    pids_vply+=($!)

    echo "  Verifying $name  pq_pid=${pids_vpq[-1]}  ply_pid=${pids_vply[-1]}"
done

echo ""
echo "Waiting for parquet verification jobs..."
verify_fail=0
for i in "${!pids_vpq[@]}"; do
    if wait "${pids_vpq[$i]}"; then :; else
        echo "  FAILED verify parquet: ${subdirs[$i]}"
        verify_fail=$((verify_fail + 1))
    fi
done

echo "Waiting for ply verification jobs..."
for i in "${!pids_vply[@]}"; do
    if wait "${pids_vply[$i]}"; then :; else
        echo "  FAILED verify ply: ${subdirs[$i]}"
        verify_fail=$((verify_fail + 1))
    fi
done

echo ""
echo "=== Verification done. Failed jobs: $verify_fail ==="
echo ""
echo "--- Parquet verification summary ---"
grep -h "^Verification" "$LOG_DIR"/*_parquet_verify.log || true
echo ""
echo "--- PLY verification summary ---"
grep -h "^Verification" "$LOG_DIR"/*_ply_verify.log || true

# ── final result ─────────────────────────────────────────────────────────────
echo ""
total_fail=$((conv_fail + verify_fail))
if [ "$total_fail" -gt 0 ]; then
    echo "RESULT: FAILED (conv_fail=$conv_fail  verify_fail=$verify_fail)"
    echo "Check logs in $LOG_DIR for details."
    exit 1
else
    echo "RESULT: SUCCESS — all files converted and verified."
fi
