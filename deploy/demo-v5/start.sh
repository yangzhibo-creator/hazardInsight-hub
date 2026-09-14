#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v docker >/dev/null || { echo 'Install Docker Desktop / Docker Engine with Compose first.'; exit 1; }
docker info >/dev/null
docker compose version
if ! docker image inspect nuclear-hazard-demo:demo-v5 >/dev/null 2>&1 || ! docker image inspect nuclear-hazard-clustering:demo-v5 >/dev/null 2>&1; then
  if command -v sha256sum >/dev/null; then sha256sum -c images.sha256; else shasum -a 256 -c images.sha256; fi
  docker load -i hazard-demo-v5.tar
fi
mkdir -p data/jobs data/engine
test -f data/.runtime-model.json || printf '{}' > data/.runtime-model.json
test -f data/.rules-imported.json || printf '[]' > data/.rules-imported.json
export DEMO_PORT="${DEMO_PORT:-3001}"
docker compose up -d --wait --wait-timeout 600
echo "Demo is ready: http://localhost:$DEMO_PORT"
if command -v open >/dev/null; then open "http://localhost:$DEMO_PORT"; elif command -v xdg-open >/dev/null; then xdg-open "http://localhost:$DEMO_PORT"; fi
