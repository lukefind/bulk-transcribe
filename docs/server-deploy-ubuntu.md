# Server Deployment Guide (Ubuntu x86_64)

Deploy Bulk Transcribe on an Ubuntu server using Docker Compose.

## Prerequisites

- Ubuntu 20.04+ (x86_64)
- Docker Engine 20.10+
- Docker Compose v2.0+
- At least 8GB RAM (16GB+ recommended for larger models)
- 10GB+ free disk space

## Quick Start (CPU)

```bash
# Clone the repository
git clone https://github.com/lukefind/bulk-transcribe.git
cd bulk-transcribe

# Create data directories
mkdir -p data/input data/output

# Build and start
docker compose up -d --build

# Verify it's running
curl http://localhost:8476/healthz
# Expected: {"ok":true}
```

The UI is now available at `http://<server-ip>:8476`

## Installation Steps

### 1. Install Docker

```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add your user to docker group (logout/login required)
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version
```

### 2. Clone and Configure

```bash
git clone https://github.com/lukefind/bulk-transcribe.git
cd bulk-transcribe

# Create data directories with correct permissions
mkdir -p data/input data/output
chmod 755 data data/input data/output
```

### 3. Deploy (CPU)

```bash
docker compose up -d --build
```

### 4. Verify Deployment

```bash
# Check container status
docker compose ps

# Check health
curl http://localhost:8476/healthz

# View logs
docker compose logs -f
```

## GPU Deployment (NVIDIA)

For NVIDIA GPU acceleration, you need the NVIDIA Container Toolkit.

### 1. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### 2. Deploy with GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### 3. Verify GPU Usage

```bash
# Check logs for CUDA detection
docker compose logs | grep -i cuda

# Monitor GPU usage during transcription
nvidia-smi -l 1
```

## Usage

### Uploading Files

Place audio files in `./data/input/` on the host:

```bash
cp /path/to/audio/*.mp3 ./data/input/
```

### Accessing the UI

Open `http://<server-ip>:8476` in your browser.

1. Input folder: `/data/input` (pre-configured)
2. Output folder: `/data/output` (pre-configured)
3. Select model and options
4. Start transcription

### Retrieving Output

Output files are saved to `./data/output/` on the host:

```bash
ls -la ./data/output/
```

## Management Commands

```bash
# Stop the service
docker compose down

# Restart the service
docker compose restart

# View logs
docker compose logs -f

# Update to latest version
git pull
docker compose up -d --build

# Clean up old images
docker image prune -f
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8476` | Server port |
| `INPUT_DIR` | `/data/input` | Default input directory |
| `OUTPUT_DIR` | `/data/output` | Default output directory |
| `DEVICE` | `cpu` | Processing device (`cpu` or `cuda`) |

### Changing the Port

Edit `docker-compose.yml`:

```yaml
ports:
  - "YOUR_PORT:8476"
```

## Reverse Proxy (Optional)

For TLS/HTTPS, put Bulk Transcribe behind a reverse proxy.

### Nginx Example

```nginx
server {
    listen 443 ssl http2;
    server_name transcribe.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8476;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Long timeout for transcription
        proxy_read_timeout 3600s;
    }
}
```

### Caddy Example

```
transcribe.example.com {
    reverse_proxy localhost:8476
}
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs

# Check if port is in use
sudo lsof -i :8476
```

### Health check failing

```bash
# Check if app is responding
docker compose exec app curl http://localhost:8476/healthz

# Check container health status
docker inspect bulk-transcribe | grep -A 10 Health
```

### Permission denied on data directory

```bash
# Fix permissions
sudo chown -R 1000:1000 ./data
chmod -R 755 ./data
```

### Out of disk space

```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a
```

### FFmpeg not found

The Docker image includes FFmpeg. If you see FFmpeg errors:

```bash
# Rebuild the image
docker compose build --no-cache
```

### GPU not detected (CUDA)

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify container toolkit
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Check Docker runtime config
cat /etc/docker/daemon.json
```

### Model download slow/failing

Whisper models are downloaded on first use. For large models:

```bash
# Increase timeout and retry
docker compose restart

# Or pre-download inside container
docker compose exec app python -c "import whisper; whisper.load_model('turbo')"
```

## Resource Requirements

| Model | RAM | Disk | Speed (CPU) | Speed (GPU) |
|-------|-----|------|-------------|-------------|
| tiny | ~1 GB | ~75 MB | ~10x realtime | ~30x realtime |
| base | ~1 GB | ~145 MB | ~7x realtime | ~25x realtime |
| small | ~2 GB | ~465 MB | ~4x realtime | ~15x realtime |
| medium | ~5 GB | ~1.5 GB | ~2x realtime | ~8x realtime |
| large | ~10 GB | ~3 GB | ~1x realtime | ~5x realtime |
| turbo | ~6 GB | ~800 MB | ~5x realtime | ~20x realtime |

*Speed estimates vary by hardware. GPU speeds assume NVIDIA RTX 3080 or similar.*
