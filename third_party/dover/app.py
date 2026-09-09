"""
DOVER microservice - video quality assessment.
Port: 8006
Function: no-reference video quality assessment, emitting aesthetic and technical quality scores.
Uses the real DOVER model (ICCV 2023).
"""
import os
import sys
import tempfile
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import yaml

app = FastAPI(title="DOVER Video Quality Assessment Service", version="2.0")

# ============ Configuration ============
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(MODEL_ROOT, "dover"))
DOVER_SRC = os.path.join(MODEL_DIR, "DOVER-master")
WEIGHT_PATH = os.path.join(MODEL_DIR, "DOVER.pth")
CONFIG_PATH = os.path.join(DOVER_SRC, "dover.yml")

# Add the DOVER source tree to path
sys.path.insert(0, DOVER_SRC)

# Global model and sampler
evaluator = None
temporal_samplers = None
sample_types = None

# DOVER normalisation parameters
MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
STD = torch.FloatTensor([58.395, 57.12, 57.375])


def fuse_results(technical_score: float, aesthetic_score: float) -> float:
    """
    DOVER's official score-level fusion, emitting an overall score in [0,1].
    The parameters come from the DOVER paper (regressed on LSVQ-1080P).
    """
    x = (technical_score - 0.1107) / 0.07355 * 0.6104 + \
        (aesthetic_score + 0.08285) / 0.03774 * 0.3896
    return float(1.0 / (1.0 + np.exp(-x)))


def load_model():
    """Load the DOVER model and sampler."""
    global evaluator, temporal_samplers, sample_types

    if evaluator is not None:
        return

    from dover.models import DOVER as DOVERModel
    from dover.datasets import UnifiedFrameSampler

    # Read the config
    with open(CONFIG_PATH, "r") as f:
        opt = yaml.safe_load(f)

    # Instantiate the model
    model_args = opt["model"]["args"]
    evaluator = DOVERModel(**model_args).to(DEVICE)
    state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=False)
    evaluator.load_state_dict(state_dict)
    evaluator.eval()

    # Use the val-l1080p sampling config (suited to high-resolution video)
    dopt = opt["data"]["val-l1080p"]["args"]
    sample_types = dopt["sample_types"]

    temporal_samplers = {}
    for stype, sopt in sample_types.items():
        if "t_frag" not in sopt:
            # resized temporal sampling for TQE (technical quality)
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            # temporal sampling for AQE (aesthetic quality)
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )

    print(f"[DOVER] Model loaded on {DEVICE}, sample_types={list(sample_types.keys())}")


@app.on_event("startup")
async def startup():
    try:
        load_model()
    except Exception as e:
        import traceback
        print(f"[DOVER] Warning: Failed to load model: {e}")
        traceback.print_exc()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "DOVER (ICCV2023)",
        "device": DEVICE,
        "loaded": evaluator is not None
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    num_frames: int = Form(default=32)
):
    """
    Video quality assessment (DOVER).

    Input: a video file
    Output:
        - technical_quality: raw technical quality score (higher is better)
        - aesthetic_quality: raw aesthetic quality score (higher is better)
        - overall_score: fused normalised score [0,1]
        - details: the raw scores plus the fusion parameters
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        load_model()

        if evaluator is None:
            return JSONResponse(
                {"success": False, "error": "DOVER model not loaded"},
                status_code=500
            )

        from dover.datasets import spatial_temporal_view_decomposition

        # Video preprocessing: spatial-temporal view decomposition
        views, _ = spatial_temporal_view_decomposition(
            tmp_path, sample_types, temporal_samplers
        )

        # Normalise and reshape
        for k, v in views.items():
            num_clips = sample_types[k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - MEAN) / STD)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(DEVICE)
            )

        # Inference
        with torch.no_grad():
            results = [r.mean().item() for r in evaluator(views)]

        technical_score = results[0]
        aesthetic_score = results[1]
        overall_score = fuse_results(technical_score, aesthetic_score)

        return JSONResponse({
            "success": True,
            "technical_quality": round(technical_score, 6),
            "aesthetic_quality": round(aesthetic_score, 6),
            "overall_score": round(overall_score, 4),
            "details": {
                "raw_technical": round(technical_score, 6),
                "raw_aesthetic": round(aesthetic_score, 6),
                "fused_overall": round(overall_score, 4),
            },
            "model_used": "DOVER_ICCV2023"
        })

    except Exception as e:
        import traceback
        return JSONResponse(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()},
            status_code=500
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
