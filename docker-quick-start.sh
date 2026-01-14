#!/bin/bash
# Quick start script for running bookmark processor with Docker

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}Bookmark Processor - Docker Quick Start${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Warning: docker-compose command not found, trying 'docker compose'${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p data logs

# Check if CSV file exists
if [ ! -f data/raindrop_export.csv ]; then
    echo -e "${YELLOW}Warning: No raindrop_export.csv found in data/ directory${NC}"
    echo "Please copy your raindrop.io export CSV file to:"
    echo "  $(pwd)/data/raindrop_export.csv"
    echo ""
    read -p "Do you want to continue with the help command instead? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    COMMAND="--help"
else
    COMMAND="--input /app/data/raindrop_export.csv --output /app/data/enhanced_bookmarks.csv --resume --verbose"
fi

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
$DOCKER_COMPOSE build

# Run the processor
echo -e "${GREEN}Running bookmark processor...${NC}"
echo -e "${YELLOW}Command: ${COMMAND}${NC}"
echo ""

$DOCKER_COMPOSE run --rm bookmark-processor $COMMAND

echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""
echo "Output files (if any) are in: $(pwd)/data/"
echo "Logs are in: $(pwd)/logs/"
echo ""
echo "To run again with different options:"
echo "  $DOCKER_COMPOSE run --rm bookmark-processor --help"
