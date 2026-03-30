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
ENV PYTHONNOUSERSITE=1

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        git \
        vim \
        tmux \
        htop \
        && rm -rf /var/lib/apt/lists/*

# ---- Environment tuning for RTX 3090 (Ampere, sm_86) ----
# Enable TF32 for faster matmuls on Ampere GPUs
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
# Set reasonable default for PyTorch memory allocator
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Faster cuDNN autotuning (helps with fixed input sizes)
ENV TORCH_CUDNN_V8_API_ENABLED=1

# ---- Setup non-root user (Best Practice) ----
ARG UNAME=developer
ARG UID=1000
ARG GID=1000

# Create group and user if they don't already exist
RUN getent group ${GID} || groupadd -g ${GID} ${UNAME} && \
    getent passwd ${UID} || useradd -m -s /bin/bash -u ${UID} -g ${GID} ${UNAME}

# Create workspace directory and give ownership to the user
RUN mkdir -p /workspace && chown ${UID}:${GID} /workspace

# ---- Python dependencies ----
# Copy requirements first for Docker layer caching
COPY --chown=${UID}:${GID} requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt && \
    pip install --no-cache-dir --force-reinstall "numpy<2"

# Switch to non-root user for runtime
USER ${UID}:${GID}
WORKDIR /workspace

# ---- Default entrypoint ----
# Mount project as /workspace at runtime; drop into bash
CMD ["/bin/bash"]
