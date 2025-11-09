#!/bin/bash

# ICMS Management Script
# Provides convenient commands for managing the system

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Functions
print_header() {
    echo ""
    echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}  🚀 ICMS Management Console${NC}"
    echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

cmd_start() {
    print_info "Starting all ICMS services..."
    docker-compose up -d
    print_status "All services started!"
    cmd_status
}

cmd_stop() {
    print_info "Stopping all ICMS services..."
    docker-compose stop
    print_status "All services stopped!"
}

cmd_restart() {
    print_info "Restarting all ICMS services..."
    docker-compose restart
    print_status "All services restarted!"
}

cmd_status() {
    print_info "Service Status:"
    docker-compose ps
}

cmd_logs() {
    if [ -z "$1" ]; then
        print_info "Showing logs for all services (Ctrl+C to exit)..."
        docker-compose logs -f
    else
        print_info "Showing logs for $1 (Ctrl+C to exit)..."
        docker-compose logs -f "$1"
    fi
}

cmd_shell() {
    if [ -z "$1" ]; then
        print_error "Please specify a service: producer, consumer, optimizer, migrator, dashboard"
        exit 1
    fi
    
    print_info "Opening shell in $1..."
    docker-compose exec "$1" /bin/bash
}

cmd_mongo() {
    print_info "Connecting to MongoDB..."
    docker-compose exec mongodb mongosh icms_db
}

cmd_analytics() {
    print_info "Generating analytics report..."
    docker-compose exec -T dashboard python3 /app/analytics.py
}

cmd_scale() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        print_error "Usage: ./manage.sh scale <service> <replicas>"
        print_info "Example: ./manage.sh scale migrator 3"
        exit 1
    fi
    
    print_info "Scaling $1 to $2 replicas..."
    docker-compose up -d --scale "$1=$2"
    print_status "Scaled $1 to $2 replicas!"
}

cmd_clean() {
    print_info "This will remove all containers, volumes, and data. Are you sure? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Cleaning up..."
        docker-compose down -v
        rm -rf logs/*
        print_status "Cleanup complete!"
    else
        print_info "Cleanup cancelled."
    fi
}

cmd_reset() {
    print_info "Resetting system (keeping data)..."
    docker-compose restart
    print_status "System reset complete!"
}

cmd_rebuild() {
    print_info "Rebuilding all Docker images..."
    docker-compose build --no-cache
    print_status "Rebuild complete!"
}

cmd_health() {
    print_info "Checking system health..."
    echo ""
    
    # Check MongoDB
    echo -n "MongoDB: "
    if docker-compose exec -T mongodb mongosh --eval "db.adminCommand('ping')" &> /dev/null; then
        print_status "Healthy"
    else
        print_error "Unhealthy"
    fi
    
    # Check Kafka
    echo -n "Kafka: "
    if docker-compose exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 &> /dev/null; then
        print_status "Healthy"
    else
        print_error "Unhealthy"
    fi
    
    # Check Dashboard
    echo -n "Dashboard: "
    if curl -s http://localhost:8080 &> /dev/null; then
        print_status "Healthy"
    else
        print_error "Unhealthy"
    fi
    
    echo ""
}

cmd_stats() {
    print_info "Quick Statistics"
    echo ""
    
    docker-compose exec -T mongodb mongosh --quiet icms_db --eval "
        print('Total Datasets:     ' + db.metadata.countDocuments());
        print('Total Migrations:   ' + db.migration_jobs.countDocuments());
        print('Pending Jobs:       ' + db.migration_jobs.countDocuments({status: 'PENDING'}));
        print('Running Jobs:       ' + db.migration_jobs.countDocuments({status: 'RUNNING'}));
        print('Completed Jobs:     ' + db.migration_jobs.countDocuments({status: 'COMPLETE'}));
    "
    
    echo ""
}

cmd_backup() {
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    print_info "Creating backup in $BACKUP_DIR..."
    
    # Backup MongoDB
    docker-compose exec -T mongodb mongodump --db icms_db --archive > "$BACKUP_DIR/mongodb.archive"
    
    # Backup models
    cp -r models "$BACKUP_DIR/"
    
    # Backup logs
    cp -r logs "$BACKUP_DIR/"
    
    print_status "Backup complete!"
}

cmd_restore() {
    if [ -z "$1" ]; then
        print_error "Usage: ./manage.sh restore <backup_directory>"
        exit 1
    fi
    
    if [ ! -d "$1" ]; then
        print_error "Backup directory not found: $1"
        exit 1
    fi
    
    print_info "Restoring from $1..."
    
    # Restore MongoDB
    if [ -f "$1/mongodb.archive" ]; then
        docker-compose exec -T mongodb mongorestore --archive < "$1/mongodb.archive"
        print_status "MongoDB restored"
    fi
    
    # Restore models
    if [ -d "$1/models" ]; then
        cp -r "$1/models/"* models/
        print_status "Models restored"
    fi
    
    print_status "Restore complete!"
}

cmd_watch() {
    print_info "Watching dashboard metrics (updates every 5s, Ctrl+C to exit)..."
    
    while true; do
        clear
        print_header
        cmd_stats
        cmd_health
        echo ""
        print_info "Next update in 5 seconds..."
        sleep 5
    done
}

cmd_help() {
    print_header
    
    echo -e "${CYAN}COMMANDS:${NC}"
    echo ""
    echo -e "  ${GREEN}start${NC}                Start all services"
    echo -e "  ${GREEN}stop${NC}                 Stop all services"
    echo -e "  ${GREEN}restart${NC}              Restart all services"
    echo -e "  ${GREEN}status${NC}               Show service status"
    echo -e "  ${GREEN}logs [service]${NC}       Show logs (all or specific service)"
    echo -e "  ${GREEN}shell <service>${NC}      Open shell in service container"
    echo -e "  ${GREEN}mongo${NC}                Connect to MongoDB shell"
    echo -e "  ${GREEN}analytics${NC}            Generate analytics report"
    echo -e "  ${GREEN}scale <svc> <n>${NC}      Scale service to n replicas"
    echo -e "  ${GREEN}health${NC}               Check system health"
    echo -e "  ${GREEN}stats${NC}                Show quick statistics"
    echo -e "  ${GREEN}watch${NC}                Watch metrics in real-time"
    echo -e "  ${GREEN}backup${NC}               Create system backup"
    echo -e "  ${GREEN}restore <dir>${NC}        Restore from backup"
    echo -e "  ${GREEN}rebuild${NC}              Rebuild Docker images"
    echo -e "  ${GREEN}reset${NC}                Reset system (keep data)"
    echo -e "  ${GREEN}clean${NC}                Remove everything (WARNING!)"
    echo -e "  ${GREEN}help${NC}                 Show this help"
    echo ""
    echo -e "${CYAN}SERVICES:${NC}"
    echo "  producer, consumer, optimizer, migrator, dashboard"
    echo ""
    echo -e "${CYAN}EXAMPLES:${NC}"
    echo "  ./manage.sh start"
    echo "  ./manage.sh logs producer"
    echo "  ./manage.sh scale migrator 5"
    echo "  ./manage.sh shell optimizer"
    echo ""
    echo -e "${CYAN}QUICK ACCESS:${NC}"
    echo "  Dashboard:  http://localhost:8080"
    echo "  MongoDB:    localhost:27017"
    echo "  Kafka:      localhost:9092"
    echo ""
}

# Main
case "$1" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs "$2"
        ;;
    shell)
        cmd_shell "$2"
        ;;
    mongo)
        cmd_mongo
        ;;
    analytics)
        cmd_analytics
        ;;
    scale)
        cmd_scale "$2" "$3"
        ;;
    health)
        cmd_health
        ;;
    stats)
        cmd_stats
        ;;
    watch)
        cmd_watch
        ;;
    backup)
        cmd_backup
        ;;
    restore)
        cmd_restore "$2"
        ;;
    rebuild)
        cmd_rebuild
        ;;
    reset)
        cmd_reset
        ;;
    clean)
        cmd_clean
        ;;
    help|--help|-h|"")
        cmd_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run './manage.sh help' for usage information."
        exit 1
        ;;
esac