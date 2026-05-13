# mcp-models-cup-disc

RunPod serverless GPU worker for SegFormer optic cup/disc segmentation.

This is the inference half of the `fundus-mcp-cup-disc` stack. The Horizon
MCP server validates images and handles the MCP protocol; this worker owns
the GPU forward pass.

## Repo layout

```
handler.py        RunPod serverless handler
Dockerfile        GPU worker image (runpod/pytorch base)
requirements.txt  Python deps (torch provided by base image)
```

## Weights

Weights are **not** committed here. Upload the contents of the `weights/`
directory from the Horizon repo to a RunPod network volume:

| File | Purpose |
|---|---|
| `model.safetensors` | SegFormer-B4 checkpoint |
| `config.json` | Model architecture config |
| `preprocessor_config.json` | Image processor config |

Steps:
1. Create a network volume in the RunPod console.
2. Spin up a temporary CPU pod with the volume attached.
3. Copy the three files above to the volume mount path (default `/runpod-volume`).
4. Terminate the pod — files persist on the volume.
5. Attach the same volume to your serverless endpoint template.

## Build & deploy

```bash
# Build
docker build -t your-registry/mcp-models-cup-disc:latest .

# Push
docker push your-registry/mcp-models-cup-disc:latest
```

RunPod serverless template settings:
- **Container image**: `your-registry/mcp-models-cup-disc:latest`
- **GPU**: any NVIDIA with ≥ 8 GB VRAM (RTX 3090, A5000, etc.)
- **Network volume**: attached, mounted at `/runpod-volume`
- **Env var**: `WEIGHTS_DIR=/runpod-volume` (adjust if mounted elsewhere)
- **Min workers**: `1` for production, `0` for dev

## Horizon env vars

Set these in your Horizon deployment:

```
RUNPOD_API_KEY      = <your RunPod API key>
RUNPOD_ENDPOINT_URL = https://api.runpod.ai/v2/<endpoint_id>
FASTMCP_DOCKET_URL  = rediss://<host>:<port>
```

## Local test (CPU)

```bash
docker run --rm \
    -e WEIGHTS_DIR=/weights \
    -v /path/to/local/weights:/weights \
    your-registry/mcp-models-cup-disc:latest \
    python handler.py --rp_serve_api
```

Then POST to `http://localhost:8000/runsync`:

```json
{
  "input": {
    "image_id": "test-001",
    "image_b64": "<base64-encoded JPEG or PNG>"
  }
}
```

## Input / output schema

**Input**
```json
{ "image_id": "str", "image_b64": "str (base64 JPEG/PNG)" }
```

**Output (success)**
```json
{
  "success": true,
  "image_id": "str",
  "shape": [H, W],
  "disc_pixel_count": 0,
  "cup_pixel_count": 0,
  "full_disc_pixel_count": 0,
  "cdr": 0.0,
  "masks_b64": "str (base64 NPZ: disc_annulus, cup, full_disc, cd_raw)",
  "model": "model.safetensors",
  "created_at": "ISO-8601"
}
```

**Output (error)**
```json
{ "success": false, "error": "str", "image_id": "str" }
```
