#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "CloudTier stopping"
docker compose down

