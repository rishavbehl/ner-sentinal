#!/usr/bin/env bash
# NER Logistics Sentinel — one command to run everything.
#
#   ./run.sh                  dashboard on http://localhost:8000
#   ./run.sh --https          also serve TLS on :8443  (REQUIRED for phone GPS)
#   ./run.sh --live           pull real weather from Open-Meteo
#   ./run.sh --live --https   both
set -e
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
PORT=${PORT:-8000}
HTTPS_PORT=${HTTPS_PORT:-8443}
USE_HTTPS=0

for arg in "$@"; do
  case "$arg" in
    --https) USE_HTTPS=1 ;;
    --live)  export SENTINEL_LIVE_WEATHER=1 ;;
    --no-sim-fleet) export SENTINEL_SIM_FLEET=0 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

echo "──────────────────────────────────────────────────────────────"
echo "  NER LOGISTICS SENTINEL"
echo "──────────────────────────────────────────────────────────────"

# ---- preflight: fail with a useful message, not a stack trace -------------
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "  ERROR: '$PY' not found."
  echo "  Install Python 3.9+ (https://www.python.org/downloads/) and re-run."
  exit 1
fi

MISSING=$("$PY" - <<'PYEOF'
mods = {"sklearn": "scikit-learn", "pandas": "pandas", "numpy": "numpy",
        "networkx": "networkx", "joblib": "joblib",
        "fastapi": "fastapi", "starlette": "starlette", "uvicorn": "uvicorn"}
missing = []
for m, pkg in mods.items():
    try:
        __import__(m)
    except ImportError:
        missing.append(pkg)
print(" ".join(missing))
PYEOF
)

if [ -n "$MISSING" ]; then
  echo "  Missing Python packages: $MISSING"
  echo
  echo "  Install them with:"
  echo "      $PY -m pip install $MISSING"
  echo
  echo "  (add --break-system-packages if pip refuses on a system Python,"
  echo "   or use a venv:  $PY -m venv .venv && source .venv/bin/activate)"
  exit 1
fi

if [ ! -f data/artifacts/risk_clf.joblib ] || [ ! -f data/sentinel.db ]; then
  echo "  First run — building dataset and training models (~2 minutes)…"
  echo
  "$PY" scripts/pipeline.py
  echo
fi

# ---- local IP, so the phone URLs are copy-pasteable ----------------------
IP=$(ipconfig getifaddr en0 2>/dev/null \
     || ipconfig getifaddr en1 2>/dev/null \
     || hostname -I 2>/dev/null | awk '{print $1}' \
     || echo "YOUR-IP")

if [ "$USE_HTTPS" = "1" ]; then
  mkdir -p data/certs
  if [ ! -f data/certs/cert.pem ] || [ ! -f data/certs/key.pem ]; then
    echo "  Generating a self-signed certificate (needed for phone GPS)…"
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
      -keyout data/certs/key.pem -out data/certs/cert.pem \
      -subj "/CN=ner-sentinel" \
      -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${IP}" \
      >/dev/null 2>&1 || \
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
      -keyout data/certs/key.pem -out data/certs/cert.pem \
      -subj "/CN=ner-sentinel" >/dev/null 2>&1
    echo "  Certificate written to data/certs/"
    echo
  fi
  echo "  Dashboard      https://localhost:$HTTPS_PORT"
  echo "  API Docs (UI)  https://localhost:$HTTPS_PORT/docs        <- interactive Swagger"
  echo "  Driver (GPS)   https://$IP:$HTTPS_PORT/track     <- open on the phone"
  echo "  Field reporter https://$IP:$HTTPS_PORT/field"
  echo
  echo "  Your phone will warn about the certificate. Tap Advanced -> Proceed."
  echo "  That warning is expected: the cert is self-signed, not broken."
  echo "──────────────────────────────────────────────────────────────"
  exec "$PY" -m uvicorn backend.app:app --host 0.0.0.0 --port "$HTTPS_PORT" \
       --ssl-keyfile data/certs/key.pem --ssl-certfile data/certs/cert.pem
else
  echo "  Dashboard      http://localhost:$PORT"
  echo "  API Docs (UI)  http://localhost:$PORT/docs        <- interactive Swagger"
  echo "  Field reporter http://localhost:$PORT/field"
  echo "  Driver (GPS)   http://localhost:$PORT/track"
  echo "  API index      http://localhost:$PORT/api"
  echo
  echo "  On the same wifi, the phone can reach   http://$IP:$PORT"
  echo "  NOTE: phone GPS needs HTTPS — use ./run.sh --https for that."
  echo "  Ctrl-C to stop"
  echo "──────────────────────────────────────────────────────────────"
  exec "$PY" -m uvicorn backend.app:app --host 0.0.0.0 --port "$PORT"
fi
