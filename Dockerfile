# =============================================================================
# mcp-models-cup-disc — RunPod serverless GPU worker
# =============================================================================
# Runs SegFormer optic cup/disc segmentation on GPU.
# This image is the inference-only half of the fundus-mcp-cup-disc stack; the
# Horizon MCP server handles image validation and MCP protocol, then calls
# this worker via the RunPod Serverless endpoint API.
#
# Base image: runpod/pytorch ships CUDA + cuDNN + a GPU-built torch/torchvision
# so we never pull a CPU-only torch from PyPI.
#
# Tested base: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Pin this in CI; "latest" drifts.
# =============================================================================

ARG PYTORCH_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${PYTORCH_IMAGE}

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY handler.py .

# ---------------------------------------------------------------------------
# Model weights — loaded from a RunPod network volume at runtime.
#
# Weights are NOT baked into the image. Attach a network volume in your
# RunPod serverless template and upload the contents of the weights/ directory
# (model.safetensors, config.json, preprocessor_config.json) to it.
# Set WEIGHTS_DIR in the template env vars to match the volume mount path
# (e.g. /runpod-volume).
#
# To upload weights to the volume:
#   1. Create a network volume in the RunPod console.
#   2. Spin up a temporary CPU pod with the volume attached.
#   3. scp or wget the weights/ directory contents onto the pod.
#   4. Terminate the pod — files persist on the volume.
#   5. Attach the same volume to your serverless template.
# ---------------------------------------------------------------------------

ENV WEIGHTS_DIR=/runpod-volume

# ---------------------------------------------------------------------------
# Non-root user
# ---------------------------------------------------------------------------
RUN useradd --no-create-home --shell /bin/false worker \
    && chown -R worker:worker /app
USER worker

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
CMD ["python", "-u", "handler.py"]

# =============================================================================
# Build instructions
# =============================================================================
#
# 1. Build:
#       docker build -t your-registry/mcp-models-cup-disc:latest .
#
# 2. Push to a registry RunPod can pull from (Docker Hub, GHCR, etc.):
#       docker push your-registry/mcp-models-cup-disc:latest
#
# 3. Create a RunPod Serverless Endpoint:
#       - Container image: your-registry/mcp-models-cup-disc:latest
#       - GPU type: any NVIDIA with ≥8 GB VRAM (RTX 3090, A5000, etc.)
#       - Attach the network volume containing weights/ directory contents
#       - Set WEIGHTS_DIR env var to match the volume mount path
#       - Min workers: 1 for production (eliminates cold-start latency);
#         0 for dev/low-traffic to save cost
#
# 4. Set in Horizon env vars:
#       RUNPOD_API_KEY      = <your RunPod API key>
#       RUNPOD_ENDPOINT_URL = https://api.runpod.ai/v2/<endpoint_id>
#
# 5. Local test (CPU, no GPU required):
#       docker run --rm \
#           -e WEIGHTS_DIR=/weights \
#           -v /path/to/local/weights:/weights \
#           your-registry/mcp-models-cup-disc:latest \
#           python handler.py --rp_serve_api
#       # then POST to http://localhost:8000/runsync
# =============================================================================
