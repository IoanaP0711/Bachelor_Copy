#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
SURICATA_CONFIG="${SURICATA_CONFIG:-/etc/suricata/suricata.yaml}"
SURICATA_INTERFACE="${SURICATA_INTERFACE:-auto}"
SURICATA_EVE_PATH="${SURICATA_EVE_PATH:-/var/log/suricata/eve.json}"


detect_interface() {
  USB_IFACE="$(ip -br addr | awk '$1 ~ /^enx/ && $2 != "DOWN" {print $1; exit}')"
  WIFI_IFACE="$(ip -br addr | awk '$1 ~ /^wlp/ && $2 != "DOWN" {print $1; exit}')"
  DEFAULT_IFACE="$(ip route | awk '/default/ {print $5; exit}')"

  if [ -n "$USB_IFACE" ]; then
    echo "$USB_IFACE"
  elif [ -n "$WIFI_IFACE" ]; then
    echo "$WIFI_IFACE"
  else
    echo "$DEFAULT_IFACE"
  fi
}

if [ "$SURICATA_INTERFACE" = "auto" ]; then
  SURICATA_INTERFACE="$(detect_interface)"
fi

DASHBOARD_IP="$(ip -4 addr show "$SURICATA_INTERFACE" | awk '/inet / {print $2}' | cut -d/ -f1 | head -n 1)"

echo "Starting Bachelor IDS..."
echo "Project: $PROJECT_DIR"
echo "Interface: $SURICATA_INTERFACE"
echo "Dashboard local: http://127.0.0.1:$APP_PORT"
echo "Dashboard LAN/VLAN: http://$DASHBOARD_IP:$APP_PORT"

echo ""
echo "Stopping old Suricata instances..."
sudo systemctl stop suricata 2>/dev/null || true
sudo pkill -9 suricata 2>/dev/null || true
sudo rm -f /var/run/suricata.pid /run/suricata.pid

echo "Starting Suricata..."
sudo suricata -D --af-packet -c "$SURICATA_CONFIG" -i "$SURICATA_INTERFACE"

echo "Starting FastAPI..."
uvicorn src.realtime.server:app --host "$APP_HOST" --port "$APP_PORT" &
SERVER_PID=$!

sleep 3

echo "Starting bridge..."
sudo tail -n 0 -f "$SURICATA_EVE_PATH" | python3 src/realtime/run_live_suricata.py &
BRIDGE_PID=$!

echo ""
echo "Running."
echo "Open:"
echo "  http://127.0.0.1:$APP_PORT"
echo "  http://$DASHBOARD_IP:$APP_PORT"
echo ""
echo "Press CTRL+C to stop server and bridge."

trap "echo 'Stopping...'; kill $SERVER_PID $BRIDGE_PID 2>/dev/null || true; exit" INT TERM

wait