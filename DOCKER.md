# Docker Setup Guide for Bookmark Processor

This guide explains how to build and run the Bookmark Validation and Enhancement Tool using Docker.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Building the Docker Image](#building-the-docker-image)
- [Running with Docker Compose](#running-with-docker-compose)
- [Running with Docker CLI](#running-with-docker-cli)
- [Volume Management](#volume-management)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Quick Start

```bash
# 1. Clone the repository (if you haven't already)
git clone https://github.com/davistroy/bookmark-validator.git
cd bookmark-validator

# 2. Create data directory and add your CSV file
mkdir -p data logs
cp /path/to/your/raindrop_export.csv data/

# 3. Build and run with docker-compose
docker-compose build
docker-compose run --rm bookmark-processor \
  --input /app/data/raindrop_export.csv \
  --output /app/data/enhanced_bookmarks.csv
```

## Prerequisites

- Docker 20.10+ or Docker Desktop
- Docker Compose 2.0+ (included with Docker Desktop)
- At least 8GB RAM allocated to Docker
- At least 10GB free disk space (for AI models and data)

### System Requirements

- **CPU**: 2+ cores recommended (4+ for large datasets)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 10GB minimum (AI models ~2GB, data varies)
- **Network**: Internet connection for URL validation

## Building the Docker Image

### Standard Build

```bash
# Build the image
docker-compose build

# Or build with Docker CLI
docker build -t bookmark-processor:latest .
```

### Build with Custom Options

```bash
# Build without cache (clean build)
docker-compose build --no-cache

# Build with specific tag
docker build -t bookmark-processor:v1.0.0 .

# Build with build arguments
docker build --build-arg DEBIAN_FRONTEND=noninteractive -t bookmark-processor:latest .
```

## Running with Docker Compose

Docker Compose is the recommended way to run the bookmark processor as it handles volumes and configuration automatically.

### Basic Usage

1. **Edit docker-compose.yml** and uncomment the command you want to use
2. **Run the service**:

```bash
docker-compose up
```

### Common Commands

#### Process Bookmarks (One-time run)

```bash
# Process a CSV file
docker-compose run --rm bookmark-processor \
  --input /app/data/raindrop_export.csv \
  --output /app/data/enhanced_bookmarks.csv
```

#### Process with Resume Capability

```bash
# First run (may interrupt with Ctrl+C)
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --resume

# Resume after interruption
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --resume
```

#### Custom Processing Options

```bash
# Custom batch size for memory management
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --batch-size 50

# Verbose logging
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --verbose

# Clear checkpoints and start fresh
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --clear-checkpoints
```

#### Cloud AI Processing (Requires API Keys)

```bash
# Set API keys in .env file first (see Configuration section)

# Process with Claude
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --ai-engine claude

# Process with OpenAI
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --ai-engine openai
```

## Running with Docker CLI

If you prefer not to use Docker Compose, you can run the container directly with Docker CLI.

### Basic Run

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v bookmark-model-cache:/app/cache/models \
  -v bookmark-checkpoints:/app/checkpoints \
  bookmark-processor:latest \
  --input /app/data/raindrop_export.csv \
  --output /app/data/enhanced_bookmarks.csv
```

### With Custom Configuration

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v bookmark-model-cache:/app/cache/models \
  -v bookmark-checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  -e BATCH_SIZE=50 \
  -e MAX_RETRIES=5 \
  --memory=8g \
  --cpus=4 \
  bookmark-processor:latest \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --verbose
```

### Interactive Shell (for debugging)

```bash
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v bookmark-model-cache:/app/cache/models \
  -v bookmark-checkpoints:/app/checkpoints \
  --entrypoint /bin/bash \
  bookmark-processor:latest
```

## Volume Management

The Docker setup uses several volumes to persist data and improve performance.

### Volume Types

1. **data/** (bind mount)
   - Contains input CSV files and output results
   - Mounted from host directory for easy access

2. **model-cache** (named volume)
   - Stores downloaded AI models (~2GB)
   - Persists across container restarts
   - Prevents re-downloading models

3. **checkpoint-data** (named volume)
   - Stores processing checkpoints
   - Enables resume functionality
   - Automatically cleaned up after successful processing

4. **logs/** (bind mount)
   - Contains application logs
   - Mounted from host directory for easy monitoring

### Managing Volumes

```bash
# List all volumes
docker volume ls

# Inspect a volume
docker volume inspect bookmark-validator_model-cache

# Remove unused volumes (caution: deletes checkpoint data)
docker volume prune

# Remove specific volume
docker volume rm bookmark-validator_model-cache

# Backup a volume
docker run --rm \
  -v bookmark-validator_model-cache:/source \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/model-cache-backup.tar.gz -C /source .

# Restore a volume
docker run --rm \
  -v bookmark-validator_model-cache:/target \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/model-cache-backup.tar.gz -C /target
```

## Configuration

### Environment Variables

Create a `.env` file in the project root for configuration:

```bash
# .env file example
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-proj-your-key-here
PERPLEXITY_API_KEY=pplx-your-key-here

# Processing configuration
BATCH_SIZE=100
MAX_RETRIES=3
TIMEOUT=30
```

**Important**: Never commit the `.env` file to version control!

### API Keys for Cloud AI

If you want to use cloud AI providers (Claude, OpenAI, etc.):

1. **Create .env file**:
```bash
cp .env.example .env
```

2. **Edit .env** and add your API keys:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key
OPENAI_API_KEY=sk-proj-your-actual-key
```

3. **Run with cloud AI**:
```bash
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --ai-engine claude
```

### Custom Configuration Files

You can mount custom configuration files:

```bash
# Create custom config
cp bookmark_processor/config/user_config.toml.template my_config.toml

# Edit my_config.toml with your settings

# Run with custom config
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/my_config.toml:/app/config/user_config.toml:ro \
  -v bookmark-model-cache:/app/cache/models \
  bookmark-processor:latest \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv
```

## Troubleshooting

### Issue: "No such file or directory" when accessing CSV

**Cause**: The CSV file is not in the mounted data directory.

**Solution**:
```bash
# Ensure your CSV is in the data directory
cp /path/to/raindrop_export.csv data/

# Verify the mount
docker-compose run --rm bookmark-processor ls -la /app/data
```

### Issue: Out of Memory Error

**Cause**: Docker doesn't have enough memory allocated.

**Solution**:
1. **Increase Docker memory limit** (Docker Desktop → Settings → Resources)
2. **Reduce batch size**:
```bash
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --batch-size 25
```

### Issue: Model Downloads Very Slow

**Cause**: AI models are large (~2GB) and download on first run.

**Solution**:
- Be patient on first run (models are cached)
- Use local AI (default) instead of cloud AI
- Pre-download models (see Advanced Usage)

### Issue: Cannot Resume from Checkpoint

**Cause**: Checkpoint volume was removed or corrupted.

**Solution**:
```bash
# Clear checkpoints and start fresh
docker-compose run --rm bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv \
  --clear-checkpoints
```

### Issue: Permission Denied Errors

**Cause**: File ownership mismatch between host and container.

**Solution**:
```bash
# Fix ownership of data directory
sudo chown -R $(id -u):$(id -g) data/ logs/

# Or run with user override (Linux/Mac)
docker-compose run --rm --user $(id -u):$(id -g) bookmark-processor \
  --input /app/data/bookmarks.csv \
  --output /app/data/enhanced.csv
```

### Issue: Container Won't Start

**Cause**: Various issues with Docker or configuration.

**Solution**:
```bash
# Check Docker is running
docker ps

# View container logs
docker-compose logs

# Rebuild image
docker-compose build --no-cache

# Check health status
docker inspect bookmark-processor | grep -A 10 Health
```

## Advanced Usage

### Pre-downloading AI Models

To speed up first run, pre-download models:

```bash
# Run a dry-run to download models
docker-compose run --rm bookmark-processor \
  --input /app/data/sample.csv \
  --output /app/data/test.csv
```

### Running Multiple Processes

Process multiple CSV files in parallel:

```bash
# Terminal 1
docker-compose run --rm --name bp1 bookmark-processor \
  --input /app/data/bookmarks1.csv \
  --output /app/data/enhanced1.csv

# Terminal 2
docker-compose run --rm --name bp2 bookmark-processor \
  --input /app/data/bookmarks2.csv \
  --output /app/data/enhanced2.csv
```

### Monitoring Progress

```bash
# Follow logs in real-time
docker-compose logs -f bookmark-processor

# View resource usage
docker stats bookmark-processor

# Check disk usage
docker system df -v
```

### Building for Production

```bash
# Build optimized image
docker build \
  --build-arg DEBIAN_FRONTEND=noninteractive \
  --tag bookmark-processor:production \
  --no-cache \
  .

# Tag for registry
docker tag bookmark-processor:production your-registry/bookmark-processor:latest

# Push to registry
docker push your-registry/bookmark-processor:latest
```

### Using with Kubernetes

Example Kubernetes deployment (save as `k8s-deployment.yaml`):

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: bookmark-model-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: batch/v1
kind: Job
metadata:
  name: bookmark-processor-job
spec:
  template:
    spec:
      containers:
      - name: bookmark-processor
        image: bookmark-processor:latest
        command:
          - python
          - -m
          - bookmark_processor
          - --input
          - /app/data/bookmarks.csv
          - --output
          - /app/data/enhanced.csv
          - --resume
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: model-cache
          mountPath: /app/cache/models
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
      restartPolicy: OnFailure
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: bookmark-data
      - name: model-cache
        persistentVolumeClaim:
          claimName: bookmark-model-cache
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

## Performance Tips

1. **Allocate Sufficient Resources**
   - CPU: 4+ cores for large datasets
   - RAM: 8GB minimum, 16GB recommended
   - Disk: Fast SSD for better performance

2. **Optimize Batch Size**
   - Default: 100 bookmarks per batch
   - Reduce for limited memory: `--batch-size 25`
   - Increase for powerful systems: `--batch-size 200`

3. **Use Named Volumes**
   - Model cache prevents re-downloading
   - Checkpoints enable resume functionality
   - Logs help with monitoring

4. **Monitor Resource Usage**
   ```bash
   docker stats bookmark-processor
   ```

5. **Clean Up Regularly**
   ```bash
   # Remove old containers
   docker container prune

   # Remove unused volumes (caution!)
   docker volume prune

   # Remove unused images
   docker image prune
   ```

## Security Considerations

1. **Never Commit API Keys**
   - Use `.env` files (add to `.gitignore`)
   - Use Docker secrets in production
   - Rotate keys regularly

2. **Run as Non-Root User**
   - The Dockerfile uses `bookmarkuser` (UID 1000)
   - Reduces security risks

3. **Network Isolation**
   - Consider using custom Docker networks
   - Limit container network access if needed

4. **Volume Permissions**
   - Check file ownership in mounted volumes
   - Use appropriate file permissions

## Getting Help

- **Documentation**: Check [README.md](README.md) and [docs/](docs/)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/davistroy/bookmark-validator/issues)
- **Docker Logs**: `docker-compose logs -f`
- **Container Shell**: `docker-compose run --rm --entrypoint /bin/bash bookmark-processor`

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Project README](README.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

---

**Last Updated**: 2026-01-14
**Docker Image**: bookmark-processor:latest
**Base Image**: python:3.12-slim
