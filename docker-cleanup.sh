#!/bin/bash
# Cleanup script for bookmark processor Docker resources

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}Bookmark Processor - Docker Cleanup${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "This script will help you clean up Docker resources used by bookmark-processor."
echo ""
echo "Options:"
echo "  1) Remove stopped containers only"
echo "  2) Remove checkpoint data (keeps model cache)"
echo "  3) Remove model cache (keeps checkpoint data)"
echo "  4) Remove ALL volumes (checkpoints + model cache) - CAUTION!"
echo "  5) Remove Docker image"
echo "  6) Full cleanup (containers + volumes + image) - CAUTION!"
echo "  7) Exit"
echo ""
read -p "Select option (1-7): " -n 1 -r
echo ""

case $REPLY in
    1)
        echo -e "${YELLOW}Removing stopped containers...${NC}"
        docker container prune -f
        echo -e "${GREEN}Done!${NC}"
        ;;
    2)
        echo -e "${YELLOW}Removing checkpoint data...${NC}"
        read -p "Are you sure? This will delete all checkpoint data. (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker volume rm bookmark-validator_checkpoint-data 2>/dev/null || true
            echo -e "${GREEN}Checkpoint data removed.${NC}"
        else
            echo "Cancelled."
        fi
        ;;
    3)
        echo -e "${YELLOW}Removing model cache...${NC}"
        echo -e "${RED}WARNING: This will force re-download of AI models (~2GB) on next run.${NC}"
        read -p "Are you sure? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker volume rm bookmark-validator_model-cache 2>/dev/null || true
            echo -e "${GREEN}Model cache removed.${NC}"
        else
            echo "Cancelled."
        fi
        ;;
    4)
        echo -e "${RED}Removing ALL volumes...${NC}"
        echo -e "${RED}WARNING: This will delete checkpoints AND force model re-download!${NC}"
        read -p "Are you sure? Type 'yes' to confirm: " confirm
        if [ "$confirm" = "yes" ]; then
            $DOCKER_COMPOSE down -v
            echo -e "${GREEN}All volumes removed.${NC}"
        else
            echo "Cancelled."
        fi
        ;;
    5)
        echo -e "${YELLOW}Removing Docker image...${NC}"
        read -p "Are you sure? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker rmi bookmark-processor:latest 2>/dev/null || true
            docker rmi bookmark-validator_bookmark-processor 2>/dev/null || true
            echo -e "${GREEN}Image removed.${NC}"
        else
            echo "Cancelled."
        fi
        ;;
    6)
        echo -e "${RED}FULL CLEANUP - This will remove everything!${NC}"
        echo "  - All containers"
        echo "  - All volumes (checkpoints + model cache)"
        echo "  - Docker image"
        echo ""
        read -p "Are you sure? Type 'DELETE' to confirm: " confirm
        if [ "$confirm" = "DELETE" ]; then
            echo -e "${YELLOW}Removing everything...${NC}"
            $DOCKER_COMPOSE down -v
            docker rmi bookmark-processor:latest 2>/dev/null || true
            docker rmi bookmark-validator_bookmark-processor 2>/dev/null || true
            docker container prune -f
            echo -e "${GREEN}Full cleanup complete!${NC}"
        else
            echo "Cancelled."
        fi
        ;;
    7)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}==================================================${NC}"
echo "Current Docker resource usage:"
echo -e "${GREEN}==================================================${NC}"
docker system df

echo ""
echo "To see detailed volume information:"
echo "  docker volume ls"
echo "  docker volume inspect bookmark-validator_model-cache"
echo "  docker volume inspect bookmark-validator_checkpoint-data"
