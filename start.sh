#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "CloudTier starting"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker missing"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon not running"
  exit 1
fi

docker compose up --build --scale consumer=3 --scale migrator=2

