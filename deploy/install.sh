#!/usr/bin/env bash
# Prepare a fresh Debian/Ubuntu VPS to run the trading loops. Idempotent: safe to re-run.
#
#   sudo bash deploy/install.sh
#
set -euo pipefail

APP_USER="${APP_USER:-ubuntu}"
APP_DIR="${APP_DIR:-/opt/money_maker}"

echo "== packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# chrony matters: Binance rejects signed requests whose timestamp drifts outside recvWindow
apt-get install -y -qq python3 python3-venv python3-dev build-essential chrony rsync curl jq logrotate
systemctl enable --now chrony

echo "== user and directories"
# On a single-purpose box the login user owns everything: fewer sudo dances, same blast radius.
id -u "$APP_USER" &>/dev/null || useradd --create-home "$APP_USER"
mkdir -p "$APP_DIR" "$APP_DIR/var/secrets" "$APP_DIR/var/logs"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod 700 "$APP_DIR/var/secrets"

echo "== python environment"
if [ ! -d "$APP_DIR/.venv" ]; then
    python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
if [ -f "$APP_DIR/pyproject.toml" ]; then
    "$APP_DIR/.venv/bin/pip" install --quiet -e "$APP_DIR[dev]"
    echo "   $("$APP_DIR/.venv/bin/python" -V) with the project installed"
else
    echo "   pyproject.toml not found yet: run deploy/sync.sh from the workstation first"
fi

echo "== log rotation"
cat > /etc/logrotate.d/money_maker <<'ROT'
/opt/money_maker/var/logs/*.log /opt/money_maker/var/paper/*/events.jsonl {
    weekly
    rotate 8
    compress
    missingok
    notifempty
    copytruncate
}
ROT

echo "== systemd units"
for unit in "$APP_DIR"/deploy/*.service; do
    [ -e "$unit" ] || continue
    install -m 644 "$unit" /etc/systemd/system/
    echo "   installed $(basename "$unit")"
done
systemctl daemon-reload

echo "== systemctl without sudo for the trading units"
cat > /etc/polkit-1/rules.d/50-quant.rules <<RULE
// Lets $APP_USER start/stop/restart the trading services without sudo.
polkit.addRule(function(action, subject) {
    if (action.id == "org.freedesktop.systemd1.manage-units" && subject.user == "$APP_USER") {
        var unit = action.lookup("unit");
        if (unit == "quant-testnet.service" || unit == "quant-paper.service") {
            return polkit.Result.YES;
        }
    }
});
RULE
systemctl restart polkit

cat <<'NEXT'

done. Remaining steps:
  1. put the API credentials on the box (never commit them):
       tee /opt/money_maker/var/secrets/binance_testnet.json >/dev/null <<'JSON'
       {"api_key": "...", "api_secret": "..."}
       JSON
       chmod 600 /opt/money_maker/var/secrets/binance_testnet.json
  2. fetch the market data:   bash /opt/money_maker/deploy/bootstrap_data.sh
  3. check the connection:    /opt/money_maker/.venv/bin/python /opt/money_maker/scripts/testnet_check.py
  4. start the loop:          sudo systemctl enable quant-testnet && systemctl start quant-testnet
  5. watch it:                journalctl -u quant-testnet -f
NEXT
