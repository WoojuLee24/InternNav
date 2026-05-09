#!/usr/bin/env bash
# Post-process Gate 3k results: copy from container → host, run parser, print LaTeX rows.
set -e

echo "=== Copying Gate 3k results from container ==="
docker cp vlnav_internvla_server:/tmp/gate3k_ema /tmp/gate3k_ema
echo "Copied to /tmp/gate3k_ema"

echo ""
echo "=== Parsing results ==="
python3 /home/kemal/VLNav/VLNav/workspaces/model/InternNav/scripts/viz/parse_gate3k.py

echo ""
echo "=== Regenerating frontier figure ==="
cd /home/kemal/VLNav/VLNav/workspaces/model/InternNav
python3 scripts/viz/plot_frontier.py
