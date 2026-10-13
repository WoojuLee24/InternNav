#!/bin/bash
# Copy a dataset directory between local/mounted filesystems, in parallel and resumable.
#
# Usage:
#   bash copy_dataset.sh SRC DST [JOBS]
#
#   JOBS    Number of files copied at the same time (default 16)
#
# Same result as `rsync -a SRC/ DST/`: the *contents* of SRC end up under DST.
# If interrupted, just run the exact same command again - finished files are
# skipped by size, and a half-copied file continues from where it stopped.
#
# Example:
#   bash copy_dataset.sh /home/irteam/data-vol2/InternData-N1-v0.5-mini /home/irteam/data-vol1/InternData-N1-v0.5-mini
#
# Verify afterwards (no output means identical, ignore the leading "./" line):
#   rsync -an --checksum --itemize-changes SRC/ DST/

SRC=${1%/}
DST=${2%/}
JOBS=${3:-16}

if [ -z "$SRC" ] || [ -z "$DST" ]; then
    sed -n '2,18p' "$0"
    exit 1
fi

if [ ! -d "$SRC" ]; then
    echo "SRC is not a directory: $SRC"
    exit 1
fi

echo "Copying $SRC -> $DST with $JOBS parallel jobs..."

# find -printf '%P\n'         paths relative to SRC (e.g. vln_ce/traj_data/r2r/x.tar.gz)
# rsync --relative "SRC/./F"  the /./ marks where the relative part starts, so the
#                             directory structure is recreated under DST without cd
# --inplace --partial         leave a short file behind on interrupt, so it can be resumed
# --append-verify             continue from the end of that file, checksumming what is there
find "$SRC" -type f -printf '%P\n' | \
  xargs -P "$JOBS" -I FILE rsync -a --relative --inplace --partial --append-verify "$SRC/./FILE" "$DST/"

echo "All done. Verify with:"
echo "  rsync -an --checksum --itemize-changes $SRC/ $DST/"
