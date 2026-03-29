#!/usr/bin/env bash
# =============================================================
# build_image.sh — Build the DeepFakeDetection Docker image
# =============================================================
# Usage:
#   ./build_image.sh                  # standard build
#   ./build_image.sh --no-cache       # rebuild from scratch
#   ./build_image.sh --verbose        # show full build output
# =============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="deepfake-detect"
TAG="latest"

# ---- Parse flags ----
NO_CACHE=""
VERBOSE=false

for arg in "$@"; do
    case $arg in
        --no-cache) NO_CACHE="--no-cache" ;;
        --verbose)  VERBOSE=true ;;
        *)          echo "Unknown flag: $arg"; echo "Usage: $0 [--no-cache] [--verbose]"; exit 1 ;;
    esac
done

# ---- Pre-build info ----
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🔨 Building: ${IMAGE_NAME}:${TAG}"
echo "  📁 Context:  ${PROJECT_DIR}"
echo "  📝 Dockerfile: ${PROJECT_DIR}/Dockerfile"
if [ -n "$NO_CACHE" ]; then
    echo "  🚫 Cache:    disabled (--no-cache)"
else
    echo "  ✅ Cache:    enabled"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ---- Check Docker is available ----
if ! command -v docker &>/dev/null; then
    echo "❌ Docker is not installed or not in PATH."
    exit 1
fi

# ---- Check NVIDIA Container Toolkit ----
if dpkg -l nvidia-container-toolkit &>/dev/null; then
    echo "✅ NVIDIA Container Toolkit detected"
else
    echo "⚠️  NVIDIA Container Toolkit not found — GPU may not work in container"
fi

# ---- Check GPU ----
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "✅ GPU: ${GPU_NAME} (${GPU_MEM})"
else
    echo "⚠️  nvidia-smi not found — building without GPU verification"
fi

echo ""

# ---- Build ----
START_TIME=$(date +%s)

if $VERBOSE; then
    docker build \
        $NO_CACHE \
        -t "${IMAGE_NAME}:${TAG}" \
        -f "${PROJECT_DIR}/Dockerfile" \
        "${PROJECT_DIR}"
else
    docker build \
        $NO_CACHE \
        -t "${IMAGE_NAME}:${TAG}" \
        -f "${PROJECT_DIR}/Dockerfile" \
        "${PROJECT_DIR}" 2>&1 | while IFS= read -r line; do
        # Show step progress but skip verbose pip output
        if [[ "$line" == *"Step"* ]] || [[ "$line" == *"--->"* ]] || \
           [[ "$line" == *"Successfully"* ]] || [[ "$line" == *"ERROR"* ]] || \
           [[ "$line" == *"WARNING"* ]]; then
            echo "  $line"
        fi
    done
fi

BUILD_STATUS=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
if [ $BUILD_STATUS -eq 0 ]; then
    # ---- Post-build summary ----
    IMAGE_SIZE=$(docker image inspect "${IMAGE_NAME}:${TAG}" --format='{{.Size}}' 2>/dev/null)
    IMAGE_SIZE_GB=$(echo "scale=2; ${IMAGE_SIZE} / 1073741824" | bc 2>/dev/null || echo "?")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✅ Build successful!"
    echo "  🏷️  Image:  ${IMAGE_NAME}:${TAG}"
    echo "  💾 Size:   ${IMAGE_SIZE_GB} GB"
    echo "  ⏱️  Time:   ${ELAPSED}s"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Next steps:"
    echo "    ./start_container.sh              # interactive shell"
    echo "    ./start_container.sh --jupyter     # Jupyter Lab"
    echo "    ./start_container.sh --tensorboard # TensorBoard"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ❌ Build FAILED (exit code: ${BUILD_STATUS})"
    echo "  ⏱️  Time: ${ELAPSED}s"
    echo ""
    echo "  Troubleshooting:"
    echo "    1. Run with --verbose for full output"
    echo "    2. Run with --no-cache to rebuild from scratch"
    echo "    3. Check Docker daemon: docker info"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit $BUILD_STATUS
fi
