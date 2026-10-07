#!/usr/bin/env bash
# Push the code (never the secrets, never the data) from this workstation to the VPS.
#
#   bash deploy/sync.sh root@1.2.3.4            # first install
#   bash deploy/sync.sh root@1.2.3.4 --restart  # push an update and restart the loops
#
set -euo pipefail

TARGET="${1:?usage: sync.sh user@host [--restart]}"
APP_DIR="${APP_DIR:-/opt/money_maker}"
RESTART="${2:-}"

EXCLUDES=(--exclude '.venv/' --exclude '__pycache__/' --exclude '.git/' --exclude '.pytest_cache/'
          --exclude 'var/' --exclude '*.pyc')

ssh "$TARGET" "sudo mkdir -p $APP_DIR && sudo chown \$(whoami) $APP_DIR"

# macOS ships openrsync, which rejects several GNU rsync options; fall back to tar over ssh.
if ! rsync -az --delete "${EXCLUDES[@]}" ./ "$TARGET:$APP_DIR/" 2>/dev/null; then
    echo "rsync unavailable or incompatible, using tar over ssh"
    tar czf - --exclude .venv --exclude __pycache__ --exclude .git --exclude .pytest_cache \
        --exclude var --exclude '*.pyc' . | ssh "$TARGET" "tar xzf - -C $APP_DIR"
fi

echo "code synced to $TARGET:$APP_DIR (var/ and secrets untouched)"

if [ "$RESTART" = "--restart" ]; then
    ssh "$TARGET" "sudo install -m 644 $APP_DIR/deploy/*.service /etc/systemd/system/ && \
        sudo systemctl daemon-reload && \
        sudo systemctl restart quant-testnet 2>/dev/null || true; \
        sudo systemctl restart quant-paper 2>/dev/null || true; \
        systemctl is-active quant-testnet quant-paper || true"
fi
