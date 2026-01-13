# GPU Worker Deployment on Lambda Labs

Step-by-step guide to deploy the Bulk Transcribe GPU Worker on Lambda Labs.

## Prerequisites

- Lambda Labs account
- SSH key added to your Lambda account
- Docker image pushed to a registry

## Step 1: Push Your Worker Image

```bash
./scripts/build_worker.sh latest
./scripts/push_worker.sh latest your-dockerhub-username
```

## Step 2: Launch an Instance

1. Go to [Lambda Cloud](https://cloud.lambdalabs.com/instances)
2. Click "Launch Instance"
3. Select GPU type:

| GPU | VRAM | Cost | Notes |
|-----|------|------|-------|
| A10 | 24GB | ~$0.60/hr | Good balance |
| A100 40GB | 40GB | ~$1.10/hr | Fast |
| A100 80GB | 80GB | ~$1.50/hr | Long files |
| H100 | 80GB | ~$2.00/hr | Maximum speed |

4. Select your SSH key
5. Click "Launch"

## Step 3: SSH into Instance

```bash
ssh ubuntu@<instance-ip>
```

## Step 4: Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Step 5: Run the Worker

```bash
# Generate a token
export WORKER_TOKEN=$(openssl rand -hex 32)
echo "Your WORKER_TOKEN: $WORKER_TOKEN"

# Run the worker
docker run -d \
  --name bt-worker \
  --gpus all \
  -p 8477:8477 \
  -e WORKER_TOKEN=$WORKER_TOKEN \
  -e HF_TOKEN=your-huggingface-token \
  your-dockerhub-username/bulk-transcribe-worker:latest
```

## Step 6: Expose HTTPS (Optional but Recommended)

Option A: Use Caddy for automatic HTTPS:

```bash
# Install Caddy
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# Create Caddyfile
echo "your-domain.com {
    reverse_proxy localhost:8477
}" | sudo tee /etc/caddy/Caddyfile

sudo systemctl restart caddy
```

Option B: Use ngrok for quick testing:

```bash
# Install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Authenticate (get token from ngrok.com)
ngrok config add-authtoken YOUR_NGROK_TOKEN

# Start tunnel
ngrok http 8477
```

## Step 7: Test the Worker

```bash
# From your local machine
curl https://your-domain.com/health
curl https://your-domain.com/v1/ping
```

## Step 8: Configure Controller

```bash
REMOTE_WORKER_URL=https://your-domain.com
REMOTE_WORKER_TOKEN=your-generated-token
REMOTE_WORKER_MODE=optional
```

## Step 9: Run Smoke Test

```bash
export REMOTE_WORKER_URL=https://your-domain.com
export REMOTE_WORKER_TOKEN=your-generated-token
./scripts/smoke_remote_worker.sh
```

## Managing the Worker

```bash
# View logs
docker logs -f bt-worker

# Restart
docker restart bt-worker

# Stop
docker stop bt-worker

# Update to new version
docker pull your-dockerhub-username/bulk-transcribe-worker:latest
docker stop bt-worker
docker rm bt-worker
# Run again with same command from Step 5
```

## Cost Notes

- Lambda charges by the hour (rounded up)
- Instances continue charging when idle
- **Terminate instances when done** to stop charges
- No spot/preemptible option - all instances are on-demand

## Troubleshooting

### Docker permission denied

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### Port 8477 not accessible

- Lambda instances have all ports open by default
- Check `docker ps` to verify container is running
- Check firewall: `sudo ufw status`

### Worker crashes on start

```bash
# Check logs
docker logs bt-worker

# Common issues:
# - Missing WORKER_TOKEN
# - Invalid HF_TOKEN
# - Out of disk space
```
