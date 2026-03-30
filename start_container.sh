#!/usr/bin/env bash
# =============================================================
# start_container.sh — Quick-start the GPU dev container
# =============================================================
# Usage:
#   ./start_container.sh              # interactive bash shell
#   ./start_container.sh --build      # rebuild image first
#   ./start_container.sh --jupyter    # start with Jupyter Lab
#   ./start_container.sh --tensorboard # start TensorBoard
# =============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="deepfake-detect"
CONTAINER_NAME="deepfake-dev"

# Default data directory (override with DATA_DIR env var)
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"

# ---- Parse flags ----
BUILD=false
JUPYTER=false
TENSORBOARD=false

for arg in "$@"; do
    case $arg in
        -h|--help)
            echo "Usage: ./start_container.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -h, --help       Show this help message and exit"
            echo "  --build          Rebuild Docker image before starting"
            echo "  --jupyter        Start with Jupyter Lab"
            echo "  --tensorboard    Start TensorBoard"
            exit 0
            ;;
        --build)       BUILD=true ;;
        --jupyter)     JUPYTER=true ;;
        --tensorboard) TENSORBOARD=true ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: $0 [--build] [--jupyter] [--tensorboard] [-h|--help]"
            exit 1
            ;;
    esac
done

# ---- Build if requested or image doesn't exist ----
if $BUILD || ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "🔨 Building Docker image: $IMAGE_NAME"
    docker build \
        --build-arg UNAME="$(id -un)" \
        --build-arg UID="$(id -u)" \
        --build-arg GID="$(id -g)" \
        -t "$IMAGE_NAME" "$PROJECT_DIR"
fi

# ---- Stop existing container if running ----
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo "⏹  Stopping existing container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null
    docker rm "$CONTAINER_NAME" >/dev/null
fi

# ---- Ensure data directory exists ----
mkdir -p "$DATA_DIR"
mkdir -p "$PROJECT_DIR/outputs"

# ---- Determine command ----
if $JUPYTER; then
    CMD="jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    echo "🚀 Starting Jupyter Lab at http://localhost:8888"
elif $TENSORBOARD; then
    CMD="tensorboard --logdir=/workspace/outputs/runs --host=0.0.0.0 --port=6006"
    echo "📊 Starting TensorBoard at http://localhost:6006"
else
    CMD="/bin/bash"
    echo "🐚 Starting interactive shell"
fi

# ---- Run ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GPU:        RTX 3090 (24 GB)"
echo "  Project:    $PROJECT_DIR → /workspace"
echo "  Data:       $DATA_DIR → /workspace/data"
echo "  Shared mem: 8 GB"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

docker run \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --rm \
    -it \
    --user "$(id -u):$(id -g)" \
    --shm-size=8g \
    -v "$HOME":"$HOME" \
    -v "$PROJECT_DIR":/workspace \
    -v "$DATA_DIR":/workspace/data \
    -v "$PROJECT_DIR/outputs":/workspace/outputs \
    -p 6006:6006 \
    -p 8888:8888 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -w /workspace \
    "$IMAGE_NAME" \
    $CMD
