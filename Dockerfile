# =============================================================
# DeepFakeDetection — GPU Training Container
# =============================================================
# Base: NVIDIA PyTorch NGC container (includes CUDA, cuDNN,
#        PyTorch, torchvision, NCCL — all optimised for Ampere)
#
# Build:  docker build -t deepfake-detect .
# Run:    ./start_container.sh
# =============================================================

FROM nvcr.io/nvidia/pytorch:24.12-py3

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
        vim \
        tmux \
        htop \
        && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
WORKDIR /workspace

# Copy requirements first for Docker layer caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Environment tuning for RTX 3090 (Ampere, sm_86) ----
# Enable TF32 for faster matmuls on Ampere GPUs
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
# Set reasonable default for PyTorch memory allocator
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Faster cuDNN autotuning (helps with fixed input sizes)
ENV TORCH_CUDNN_V8_API_ENABLED=1

# ---- Default entrypoint ----
# Mount project as /workspace at runtime; drop into bash
CMD ["/bin/bash"]
