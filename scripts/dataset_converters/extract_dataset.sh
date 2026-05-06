#!/bin/bash
# Extract all tar.gz files in InternData-N1-mini (traj_data + scene_data)
#
# Usage:
#   bash extract_dataset.sh [DATA_DIR] [JOBS] [--delete]
#
#   --delete    Delete each tar.gz after successful extraction

DATA_DIR=${1:-"/home/irteam/git/InternNav/data/InternData-N1"}
JOBS=${2:-4}
DELETE=0

for arg in "$@"; do
    case $arg in
        --delete) DELETE=1 ;;
    esac
done

echo "Extracting all tar.gz files in $DATA_DIR using $JOBS parallel jobs..."
[ $DELETE -eq 1 ] && echo "Mode: delete tar.gz after extraction"

find "$DATA_DIR" -name "*.tar.gz" ! -path "*/.cache/*" | \
  xargs -P "$JOBS" -I{} bash -c '
    archive="{}"
    delete='"$DELETE"'
    dest_dir="$(dirname "$archive")"

    # Detect the top-level directory inside the archive
    top_entry=$(tar -tzf "$archive" 2>/dev/null | head -1 | cut -d/ -f1)

    if [ -n "$top_entry" ] && [ -d "$dest_dir/$top_entry" ]; then
      echo "SKIP (already extracted): $archive"
    else
      echo "Extracting: $archive -> $dest_dir/"
      if tar -xzf "$archive" -C "$dest_dir"; then
        echo "Done: $archive"
        if [ "$delete" -eq 1 ]; then
          rm -f "$archive"
          echo "Deleted: $archive"
        fi
      else
        echo "FAILED: $archive"
      fi
    fi
  '

echo "All done."
