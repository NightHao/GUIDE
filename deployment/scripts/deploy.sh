#!/bin/bash

# Research_CODE API Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/deployment/docker/docker-compose.prod.yml"
ENV_FILE="$PROJECT_DIR/.env"

echo -e "${GREEN}🚀 Research_CODE API Deployment Script${NC}"
echo "======================================"

# Check if .env file exists
if [[ ! -f "$ENV_FILE" ]]; then
    echo -e "${RED}❌ Error: .env file not found at $ENV_FILE${NC}"
    echo "Please create a .env file with required environment variables:"
    echo "  OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# Load environment variables
source "$ENV_FILE"

# Check required variables
required_vars=("OPENAI_API_KEY")
for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        echo -e "${RED}❌ Error: Required environment variable $var is not set${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ Environment variables validated${NC}"

# Function to stop and remove containers
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up existing containers...${NC}"
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans || true
}

# Function to build and start services
deploy() {
    echo -e "${YELLOW}🔨 Building and starting services...${NC}"

    # Build the application
    docker-compose -f "$COMPOSE_FILE" build --no-cache

    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d

    echo -e "${GREEN}✅ Services started successfully${NC}"
}

# Function to check service health
health_check() {
    echo -e "${YELLOW}🏥 Checking service health...${NC}"

    # Wait for API to be ready
    max_attempts=30
    attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API is healthy and responding${NC}"
            break
        else
            echo "Attempt $attempt/$max_attempts: Waiting for API to be ready..."
            sleep 10
            ((attempt++))
        fi
    done

    if [[ $attempt -gt $max_attempts ]]; then
        echo -e "${RED}❌ API health check failed after $max_attempts attempts${NC}"
        echo "Check logs with: docker-compose -f $COMPOSE_FILE logs research-code-api"
        exit 1
    fi
}

# Function to show service status
show_status() {
    echo -e "${YELLOW}📊 Service Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps

    echo -e "\n${GREEN}🌐 Access URLs:${NC}"
    echo "  API Documentation: http://localhost/docs"
    echo "  API Health Check:  http://localhost/health"
}

# Function to show logs
show_logs() {
    echo -e "${YELLOW}📝 Recent logs:${NC}"
    docker-compose -f "$COMPOSE_FILE" logs --tail=50 research-code-api
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "deploy")
            cleanup
            deploy
            health_check
            show_status
            ;;
        "stop")
            echo -e "${YELLOW}🛑 Stopping services...${NC}"
            docker-compose -f "$COMPOSE_FILE" down
            echo -e "${GREEN}✅ Services stopped${NC}"
            ;;
        "restart")
            echo -e "${YELLOW}🔄 Restarting services...${NC}"
            docker-compose -f "$COMPOSE_FILE" restart
            health_check
            show_status
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "update")
            echo -e "${YELLOW}🔄 Updating services...${NC}"
            docker-compose -f "$COMPOSE_FILE" pull
            cleanup
            deploy
            health_check
            show_status
            ;;
        "backup")
            echo -e "${YELLOW}💾 Creating backup...${NC}"
            backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$backup_dir"
            cp -r "$PROJECT_DIR/data" "$backup_dir/"
            tar -czf "${backup_dir}.tar.gz" "$backup_dir"
            rm -rf "$backup_dir"
            echo -e "${GREEN}✅ Backup created: ${backup_dir}.tar.gz${NC}"
            ;;
        "help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the application (default)"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart all services"
            echo "  status   - Show service status"
            echo "  logs     - Show recent logs"
            echo "  update   - Update and redeploy services"
            echo "  backup   - Create backup of data"
            echo "  help     - Show this help message"
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            echo "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Handle script interruption
trap 'echo -e "\n${YELLOW}⚠️  Deployment interrupted${NC}"; exit 1' INT

# Run main function
main "$@"
