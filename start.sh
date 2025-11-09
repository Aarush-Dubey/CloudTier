#!/bin/bash

echo "=========================================="
echo "  ICMS - Intelligent Cloud Management"
echo "=========================================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# Function to start Docker Desktop on macOS
start_docker_mac() {
    echo "🐳 Starting Docker Desktop on macOS..."
    open -a Docker
    echo "⏳ Waiting for Docker to start..."
    
    # Wait up to 60 seconds for Docker to be ready
    local count=0
    while ! docker info >/dev/null 2>&1; do
        sleep 2
        count=$((count + 2))
        if [ $count -ge 60 ]; then
            echo "❌ Docker failed to start within 60 seconds"
            return 1
        fi
        echo "   Still waiting... ($count/60 seconds)"
    done
    
    echo "✅ Docker is ready!"
    return 0
}

# Function to start Docker on Linux
start_docker_linux() {
    echo "🐳 Starting Docker service on Linux..."
    
    # Try systemctl first (most modern Linux distributions)
    if command -v systemctl &> /dev/null; then
        sudo systemctl start docker
        sleep 3
        if docker info >/dev/null 2>&1; then
            echo "✅ Docker started successfully!"
            return 0
        fi
    fi
    
    # Try service command (older systems)
    if command -v service &> /dev/null; then
        sudo service docker start
        sleep 3
        if docker info >/dev/null 2>&1; then
            echo "✅ Docker started successfully!"
            return 0
        fi
    fi
    
    echo "⚠️  Could not start Docker automatically"
    return 1
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo ""
    echo "Installation links:"
    echo "  • macOS:   https://docs.docker.com/desktop/install/mac-install/"
    echo "  • Windows: https://docs.docker.com/desktop/install/windows-install/"
    echo "  • Linux:   https://docs.docker.com/engine/install/"
    exit 1
fi

# Check if Docker daemon is running, if not try to start it
if ! docker info >/dev/null 2>&1; then
    echo "⚠️  Docker daemon is not running. Attempting to start..."
    
    # Detect OS and start Docker accordingly
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        start_docker_mac
        if [ $? -ne 0 ]; then
            echo "❌ Please start Docker Desktop manually and try again"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        start_docker_linux
        if [ $? -ne 0 ]; then
            echo "❌ Please start Docker manually:"
            echo "   sudo systemctl start docker"
            exit 1
        fi
    else
        echo "❌ Please start Docker Desktop manually and try again"
        exit 1
    fi
fi

echo "✅ Docker is running"
echo ""

# Determine which Docker Compose command to use
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo "✅ Using: docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
    echo "✅ Using: docker compose"
else
    echo "❌ Docker Compose is not available"
    echo "   Docker Compose should come with Docker Desktop"
    echo "   For Linux, install: sudo apt-get install docker-compose-plugin"
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
echo ""

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ ERROR: docker-compose.yml not found!"
    echo "   Please ensure all files are in the same directory"
    echo ""
    echo "Current directory contents:"
    ls -la
    exit 1
fi

# Check disk space requirements
echo "💾 Checking disk space..."
REQUIRED_SPACE_GB=10

if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Get available space in GB
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        AVAILABLE_SPACE=$(df -g . | awk 'NR==2 {print $4}')
    else
        # Linux
        AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    fi
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE_GB" ]; then
        echo "⚠️  WARNING: Low disk space!"
        echo "   Available: ${AVAILABLE_SPACE}GB"
        echo "   Required:  ${REQUIRED_SPACE_GB}GB"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Setup cancelled. Please free up disk space and try again."
            exit 1
        fi
    else
        echo "   ✅ Sufficient disk space: ${AVAILABLE_SPACE}GB available"
    fi
else
    echo "   ⚠️  Disk space check not available on this OS"
fi

# Check Docker memory allocation
echo ""
echo "🧠 Checking Docker resources..."
DOCKER_MEM=$(docker info --format '{{.MemTotal}}' 2>/dev/null)
if [ -n "$DOCKER_MEM" ]; then
    # Convert bytes to GB
    DOCKER_MEM_GB=$((DOCKER_MEM / 1024 / 1024 / 1024))
    
    if [ "$DOCKER_MEM_GB" -lt 4 ]; then
        echo "⚠️  WARNING: Docker has less than 4GB RAM allocated"
        echo "   Current: ${DOCKER_MEM_GB}GB"
        echo "   Recommended: 4GB or more"
        echo ""
        echo "   To increase:"
        echo "   • macOS/Windows: Docker Desktop > Settings > Resources > Memory"
        echo "   • Linux: Edit /etc/docker/daemon.json"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "❌ Setup cancelled. Please increase Docker memory and try again."
            exit 1
        fi
    else
        echo "   ✅ Docker memory: ${DOCKER_MEM_GB}GB allocated"
    fi
else
    echo "   ⚠️  Could not detect Docker memory allocation"
fi
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p templates static

# Check if ML model files exist
if [ ! -f "model/reads_model_h8.pth" ] || [ ! -f "model/writes_model_h8.pth" ]; then
    echo "⚠️  Warning: ML model files not found!"
    echo "Creating dummy model files for demonstration..."
    
    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required to generate model files"
        echo "   Install Python 3 or provide reads_model_h8.pth and writes_model_h8.pth"
        exit 1
    fi
    
    # Create dummy PyTorch model files
    python3 - <<'EOF'
try:
    import torch
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(12, 64, batch_first=True, num_layers=2, dropout=0.2)
            self.fc = nn.Linear(64, 1)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    model = DummyModel()
    torch.save(model.state_dict(), 'reads_model_h8.pth')
    torch.save(model.state_dict(), 'writes_model_h8.pth')
    print("✅ Dummy model files created")
except ImportError:
    print("⚠️  PyTorch not installed locally, models will be created in Docker container")
except Exception as e:
    print(f"⚠️  Could not create models: {e}")
    print("   Models will be created in Docker container if needed")
EOF
fi

# Stop any existing containers (if running)
echo ""
echo "🛑 Checking for existing containers..."
if [ $(run_compose ps -q | wc -l) -gt 0 ]; then
    echo "   Stopping existing containers..."
    run_compose down 2>/dev/null
else
    echo "   No existing containers found"
fi

# Build and start containers
echo ""
echo "🏗️  Building Docker images..."
echo "   This will create all necessary containers (first run: 5-10 minutes)"
echo ""
run_compose build

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed! Check the errors above."
    echo "   Common fixes:"
    echo "   • Ensure Docker has enough memory (4GB+)"
    echo "   • Check your internet connection"
    echo "   • Try: docker system prune -a (frees up space)"
    exit 1
fi

echo ""
echo "🚀 Starting ICMS services..."
echo "   Creating and starting containers:"
echo "   • MongoDB (database)"
echo "   • Zookeeper (Kafka coordinator)"
echo "   • Kafka (message broker)"
echo "   • Producer (data generator)"
echo "   • Consumer (event processor)"
echo "   • Migrator (migration worker)"
echo "   • Optimizer (ML brain)"
echo "   • Dashboard (web UI)"
echo ""
run_compose up -d

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to start services! Check the errors above."
    echo "   Try: $COMPOSE_CMD logs"
    exit 1
fi

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo ""
echo "📊 Container Status:"
run_compose ps

# Count running containers
RUNNING_COUNT=$(run_compose ps --filter "status=running" --quiet | wc -l | tr -d ' ')
TOTAL_COUNT=8

echo ""
echo "   Containers running: $RUNNING_COUNT/$TOTAL_COUNT"

if [ "$RUNNING_COUNT" -lt "$TOTAL_COUNT" ]; then
    echo "   ⚠️  Not all containers started successfully"
    echo "   Check logs with: $COMPOSE_CMD logs"
fi

echo ""
echo "=========================================="
echo "✅ ICMS is starting up!"
echo "=========================================="
echo ""
echo "🌐 Dashboard: http://localhost:8080"
echo ""
echo "📝 Useful Commands:"
echo "  • View logs:        $COMPOSE_CMD logs -f"
echo "  • Stop services:    $COMPOSE_CMD down"
echo "  • Restart:          $COMPOSE_CMD restart"
echo "  • View status:      $COMPOSE_CMD ps"
echo ""
echo "⏰ Please wait 30-60 seconds for all services to initialize..."
echo "   The dashboard will be ready when you see data appearing"
echo ""

# Ask if user wants to follow logs
read -p "📋 Show live logs now? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Showing live logs (Press Ctrl+C to exit, services will keep running):"
    echo ""
    sleep 1
    run_compose logs -f
else
    echo "Services are running in the background."
    echo "To view logs later, run: $COMPOSE_CMD logs -f"
fi