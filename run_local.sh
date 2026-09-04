#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: Docker CLI is not installed or is not on PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Error: Docker is not running. Start Docker Desktop and run this script again." >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Error: .env is missing. Copy .env.example to .env and set secure values first." >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1; then
  port_owner="$(lsof -nP -iTCP:8000 -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true)"
  if [[ -n "$port_owner" ]]; then
    echo "Error: port 8000 is already in use by process $port_owner:" >&2
    ps -p "$port_owner" -o pid=,comm=,args= >&2 || true
    exit 1
  fi
fi

echo "Starting Multimodal Healthcare AI..."
docker compose down --remove-orphans >/dev/null 2>&1 || true

build_log="$(mktemp)"
trap 'rm -f "$build_log"' EXIT
if ! docker compose build >"$build_log" 2>&1; then
  echo "Error: Docker build failed." >&2
  if grep -qi "frontend" "$build_log"; then
    echo "Frontend build error:" >&2
  fi
  cat "$build_log" >&2
  exit 1
fi

docker compose up -d

echo "Waiting for MongoDB to become healthy..."
for attempt in {1..30}; do
  mongo_health="$(docker inspect -f '{{.State.Health.Status}}' healthcare_mongodb 2>/dev/null || true)"
  if [[ "$mongo_health" == "healthy" ]]; then
    break
  fi
  if [[ "$mongo_health" == "unhealthy" || "$attempt" -eq 30 ]]; then
    echo "Error: MongoDB failed its healthcheck. Recent MongoDB logs:" >&2
    docker compose logs --tail=100 mongodb >&2 || true
    exit 1
  fi
  sleep 2
done

echo "Waiting for the backend to start..."
for attempt in {1..30}; do
  if curl -fsS http://localhost:8000/api/health >/dev/null 2>&1; then
    echo "Backend is running"
    break
  fi
  if [[ "$attempt" -eq 30 ]]; then
    echo "Error: backend did not become healthy." >&2
    docker compose ps >&2 || true
    docker compose logs --tail=100 mongodb backend >&2 || true
    exit 1
  fi
  sleep 2
done

echo
echo "========================================="
echo "Application is running"
echo "========================================="
echo "Frontend UI:        http://localhost"
echo "Backend API Docs:   http://localhost:8000/api/docs"
echo "Health check:       http://localhost:8000/api/health"
echo "MongoDB:            localhost:27017 (internal Compose network)"
echo "========================================="
echo
echo "To stop: docker compose down"
echo "To view logs: docker compose logs -f"