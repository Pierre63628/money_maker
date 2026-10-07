#!/usr/bin/env bash
# Download everything the strategy needs, straight from Binance (faster on a VPS than copying it up).
# Roughly 2 GB and 1-2 hours on a decent link. Safe to re-run: existing files are skipped.
set -euo pipefail

# Refuse to run twice: two downloaders competing for memory is how the first attempt died.
exec 9> /tmp/money_maker_download.lock   # shared: never two downloaders at once
flock -n 9 || { echo "another download is already running" >&2; exit 1; }

APP_DIR="${APP_DIR:-/opt/money_maker}"
PY="$APP_DIR/.venv/bin/python"
cd "$APP_DIR"

# Worker counts are tuned for a 2 vCPU / 4 GB VPS: the metrics downloader holds a full symbol in
# memory while merging, and 8 of those at once got the process OOM-killed on a 3.8 GB box.
BARS_WORKERS="${BARS_WORKERS:-6}"
METRIC_WORKERS="${METRIC_WORKERS:-3}"
ARCHIVE_WORKERS="${ARCHIVE_WORKERS:-8}"

if ! swapon --show | grep -q .; then
    echo "warning: no swap configured; a memory spike will kill this process" >&2
fi

echo "== perpetual bars (1h) for the whole crypto universe"
$PY scripts/download_data.py --universe crypto --intervals 1h --workers "$BARS_WORKERS" --skip-existing

echo "== repairing gaps via REST"
$PY scripts/repair_data.py --interval 1h --all --no-funding

echo "== open interest and positioning (hourly storage, raw cache purged)"
$PY scripts/download_metrics.py --top-n 50 --start 2022-01-01 --workers "$METRIC_WORKERS" \
    --archive-workers "$ARCHIVE_WORKERS" --purge-raw --resample 1h

echo "== done"
du -sh var/data
