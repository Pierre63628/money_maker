#!/usr/bin/env bash
# Incremental refresh: extends the store with what has happened since the last run.
#
# Why it is needed even though the loop completes history over REST:
#   - bars: REST serves 1500 points per request, so a few weeks are recoverable, no more;
#   - metrics: the public endpoints only keep ~30 days. Once the store falls further behind than that,
#     the 90-day z-score can no longer be rebuilt and the loop starts warning about stale metrics.
# Weekly keeps a wide margin. Minutes, not hours, because every downloader is incremental.
set -euo pipefail

exec 9> /tmp/money_maker_download.lock   # shared: never two downloaders at once
flock -n 9 || { echo "another download is already running, skipping"; exit 0; }

APP_DIR="${APP_DIR:-/opt/money_maker}"
PY="$APP_DIR/.venv/bin/python"
cd "$APP_DIR"

echo "=== $(date -u +%FT%TZ) incremental refresh"
# --extend appends to every stored series; --skip-existing would leave them frozen at the bootstrap date
$PY scripts/download_data.py --universe crypto --intervals 1h --workers 4 --extend
$PY scripts/download_metrics.py --top-n 50 --start 2022-01-01 --workers 2 --archive-workers 6 \
    --purge-raw --resample 1h
echo "=== $(date -u +%FT%TZ) done; store is $(du -sh var/data | cut -f1)"
