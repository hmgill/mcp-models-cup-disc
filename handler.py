"""
handler.py — cup-disc-runpod
==============================
RunPod serverless worker for SegFormer optic cup/disc segmentation.
This is the pure-inference half of the fundus-mcp-cup-disc stack; the
Horizon MCP server handles image validation, MCP protocol, and response
passthrough, then dispatches here for the GPU-bound forward pass.

Expected input schema (job["input"])
-------------------------------------
{
    "image_id": str,   # required — used in logs / response
    "image_b64": str,  # base64-encoded RGB fundus image (JPEG or PNG)
}

Output schema (returned inside RunPod's {"output": ...} envelope)
------------------------------------------------------------------
Success:
{
    "success":               true,
    "image_id":              str,
    "shape":                 [H, W],
    "disc_pixel_count":      int,   # annulus only (label == 1)
    "cup_pixel_count":       int,   # label == 2
    "full_disc_pixel_count": int,   # disc_annulus + cup (labels >= 1)
    "cdr":                   float, # cup_px / full_disc_px
    "masks_b64":             str,   # base64 NPZ: disc_annulus, cup, full_disc, cd_raw
    "model":                 str,
    "created_at":            str,   # UTC ISO-8601
}

Error:
{
    "success":  false,
    "error":    str,
    "image_id": str | null,
}

Design notes
------------
- The SegFormer model and processor are loaded ONCE at module level (_model_cache).
  RunPod reuses the same worker process across jobs, so cold-start cost (~5-10s)
  is paid only on the first job per container lifetime.

- /tmp cache: after the first load from safetensors, parsed weights are saved to
  /tmp/fundus-model-cache so that subsequent cold starts in the same container
  instance skip the slower safetensors parse step (~24s → ~1s).

- The processor size is overridden to 224x224 (matching image_size in config.json)
  regardless of the 512 default in preprocessor_config.json.

- Weights are loaded from the network volume at WEIGHTS_DIR (default /runpod-volume).
  The file must be named model.safetensors and sit alongside config.json and
  preprocessor_config.json (i.e. upload the entire weights/ directory).

- num_classes = 3  (0=background, 1=disc annulus, 2=optic cup)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import runpod

logging.basicConfig(
    format="%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("cup-disc-worker")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WEIGHTS_DIR   = Path(os.environ.get("WEIGHTS_DIR", "/runpod-volume"))
WEIGHTS_FILE  = WEIGHTS_DIR / "model.safetensors"
TMP_CACHE_DIR = Path("/tmp/fundus-model-cache")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _get_model():
    """
    Load SegFormer model and processor, caching in-process and on /tmp.

    Load order:
      1. /tmp cache (fastest — skips safetensors parse)
      2. WEIGHTS_DIR (network volume, slower first load)

    After loading from WEIGHTS_DIR, saves a parsed copy to /tmp for
    subsequent cold starts in the same container.
    """
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["processor"], _model_cache["device"]

    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS_FILE}\n"
            "Upload the weights/ directory contents to the attached RunPod "
            "network volume and ensure WEIGHTS_DIR points to it."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading SegFormer on {device} ...")

    if TMP_CACHE_DIR.exists():
        logger.info(f"Loading from /tmp cache (PID={os.getpid()}) ...")
        src = str(TMP_CACHE_DIR)
    else:
        logger.info(f"Loading from network volume (PID={os.getpid()}) ...")
        src = str(WEIGHTS_DIR)

    processor = AutoImageProcessor.from_pretrained(src, local_files_only=True)
    # Override to training size (224) — preprocessor_config.json says 512
    # but config.json image_size=224. Smaller input = faster inference.
    processor.size = {"height": 224, "width": 224}

    model = SegformerForSemanticSegmentation.from_pretrained(
        src, local_files_only=True,
    ).to(device)
    model.eval()

    # Persist parsed weights to /tmp for subsequent cold starts
    if not TMP_CACHE_DIR.exists():
        logger.info(f"Saving parsed weights to {TMP_CACHE_DIR} ...")
        model.save_pretrained(str(TMP_CACHE_DIR))
        processor.save_pretrained(str(TMP_CACHE_DIR))
        logger.info("Cache saved.")

    _model_cache["model"]     = model
    _model_cache["processor"] = processor
    _model_cache["device"]    = device
    logger.info(f"SegFormer ready on {device}.")

    return model, processor, device


# ---------------------------------------------------------------------------
# Warm up at module load — RunPod reuses the process; first job is free
# ---------------------------------------------------------------------------

logger.info("Pre-warming SegFormer model ...")
try:
    _get_model()
    logger.info("Model ready.")
except Exception as _e:
    logger.error(f"Model initialisation failed: {_e}", exc_info=True)
    raise


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_input(job_input: dict) -> tuple[str, str]:
    """
    Pull and validate required fields. Returns (image_id, image_b64).
    Raises ValueError with a descriptive message on missing / bad fields.
    """
    image_id = job_input.get("image_id")
    if not image_id:
        raise ValueError("'image_id' is required and must be a non-empty string.")

    image_b64 = job_input.get("image_b64")
    if not image_b64:
        raise ValueError("'image_b64' (base64-encoded RGB fundus image) is required.")

    return image_id, image_b64


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_segment(image_id: str, image_b64: str) -> dict:
    """
    Decode the image, run SegFormer inference, compute mask statistics,
    and return the full result payload.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image as _Image

    model, processor, device = _get_model()

    # Decode
    img_bytes = base64.b64decode(image_b64)
    image     = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h      = image.size
    logger.info(f"[{image_id}] Image size: {w}x{h}")

    # Preprocess
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Upsample logits to original image size
    upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    cd_raw    = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Derive binary masks
    disc_annulus = (cd_raw == 1).astype(np.uint8)
    cup          = (cd_raw == 2).astype(np.uint8)
    full_disc    = (cd_raw >= 1).astype(np.uint8)

    cup_px  = int(cup.sum())
    disc_px = int(full_disc.sum())
    cdr     = round(cup_px / disc_px, 4) if disc_px > 0 else 0.0

    # Encode masks as base64 NPZ
    npz_buf = io.BytesIO()
    np.savez_compressed(
        npz_buf,
        disc_annulus=disc_annulus,
        cup=cup,
        full_disc=full_disc,
        cd_raw=cd_raw,
    )
    masks_b64 = base64.b64encode(npz_buf.getvalue()).decode()

    logger.info(
        f"[{image_id}] Done — CDR={cdr}  "
        f"cup={cup_px}px  disc={disc_px}px  payload={len(masks_b64) / 1024:.1f}KB"
    )

    return {
        "success":               True,
        "image_id":              image_id,
        "shape":                 list(cd_raw.shape),
        "disc_pixel_count":      int(disc_annulus.sum()),
        "cup_pixel_count":       cup_px,
        "full_disc_pixel_count": disc_px,
        "cdr":                   cdr,
        "masks_b64":             masks_b64,
        "model":                 WEIGHTS_FILE.name,
        "created_at":            datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """
    RunPod synchronous handler.

    RunPod wraps the returned dict as {"output": <return value>} automatically.
    Structured errors are returned rather than raising so the Horizon caller
    always gets a parseable payload rather than an SDK-level FAILED status.
    """
    job_input = job.get("input", {})
    image_id  = job_input.get("image_id", "<unknown>")

    try:
        image_id, image_b64 = _validate_input(job_input)
    except ValueError as exc:
        logger.error(f"[{image_id}] Input validation error: {exc}")
        return {"success": False, "error": str(exc), "image_id": image_id}

    try:
        return _run_segment(image_id, image_b64)
    except Exception as exc:
        logger.error(f"[{image_id}] Inference error: {exc}", exc_info=True)
        return {"success": False, "error": str(exc), "image_id": image_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
