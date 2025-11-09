#!/bin/bash

echo "=========================================="
echo "  Stopping ICMS Services"
echo "=========================================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Determine which Docker Compose command to use
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose is not available"
    exit 1
fi

# Function to execute docker compose commands safely
run_compose() {
    if [[ "$COMPOSE_CMD" == "docker compose" ]]; then
        docker compose "$@"
    else
        $COMPOSE_CMD "$@"
    fi
}

# Stop all containers
echo "🛑 Stopping all ICMS containers..."
run_compose down

echo ""
echo "✅ All ICMS services stopped successfully!"
echo ""
echo "To start again, run: ./start.sh"
echo "To remove all data, run: $COMPOSE_CMD down -v"
echo ""