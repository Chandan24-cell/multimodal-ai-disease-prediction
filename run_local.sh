#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

if [[ ! -d "$ROOT_DIR/venv" ]]; then
  python3 -m venv "$ROOT_DIR/venv"
fi

source "$ROOT_DIR/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/backend/requirements.txt"

cd "$ROOT_DIR/frontend"
npm install
npm run build
cd "$ROOT_DIR"

# Compose provides the local authenticated MongoDB instance. Load credentials
# from .env without printing them, then point the host-run backend at localhost.
if command -v docker >/dev/null 2>&1; then
  docker compose -f docker-compose.yml -f docker-compose.host.yml up -d mongodb
  if [[ -n "${MONGO_ROOT_USER:-}" && -n "${MONGO_ROOT_PASSWORD:-}" ]]; then
    export MONGODB_URI="mongodb://${MONGO_ROOT_USER}:${MONGO_ROOT_PASSWORD}@localhost:27017/multimodal_healthcare?authSource=admin"
  fi
fi

export PYTHONPATH="$ROOT_DIR/backend${PYTHONPATH:+:$PYTHONPATH}"
echo "Application: http://localhost:8000"
echo "Health:      http://localhost:8000/api/health"
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
