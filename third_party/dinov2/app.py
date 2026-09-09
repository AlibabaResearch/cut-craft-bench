"""
DINOv2 microservice - visual embeddings.
Port: 8003
Function: extract visual embedding vectors from video frames, for similarity computation.
"""
import os
import tempfile
import numpy as np
import torch
import cv2

_orig_torch_load = torch.load

def _torch_load_compat(*args, **kwargs):
    kwargs.pop("weights_only", None)
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_compat

if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    def _scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        scale_factor = scale if scale is not None else query.size(-1) ** -0.5
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        if attn_mask is not None:
            attn = attn + attn_mask
        if is_causal:
            mask = torch.ones(attn.size(-2), attn.size(-1), device=attn.device, dtype=torch.bool).tril()
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        if dropout_p and dropout_p > 0:
            attn = torch.nn.functional.dropout(attn, p=dropout_p, training=True)
        return torch.matmul(attn, value)
    torch.nn.functional.scaled_dot_product_attention = _scaled_dot_product_attention
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="DINOv2 Visual Embedding Service", version="1.0")

model = None
transform = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
LOCAL_WEIGHT_PATH = os.environ.get(
    "LOCAL_WEIGHT_PATH", os.path.join(MODEL_ROOT, "dinov2", "dinov2_vitb14_pretrain.pth")
)


def load_model():
    global model, transform
    if model is not None:
        return model
    
    from torchvision import transforms
    import sys
    
    # Use only the local model definition and local weights, to avoid a remote fallback at runtime
    local_weight = LOCAL_WEIGHT_PATH
    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    dinov2_repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    if not os.path.exists(local_weight):
        raise FileNotFoundError(f"DINOv2 local weights not found: {local_weight}")
    if not os.path.exists(dinov2_repo):
        raise FileNotFoundError(f"DINOv2 local hub repo not found: {dinov2_repo}")

    sys.path.insert(0, dinov2_repo)
    from dinov2.models.vision_transformer import vit_base
    model = vit_base(patch_size=14, img_size=518, init_values=1.0, block_chunks=0)
    state_dict = torch.load(local_weight, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    print(f"[DINOv2] Model loaded from LOCAL: {local_weight}")
    
    model.eval().to(DEVICE)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model


@app.on_event("startup")
async def startup():
    try:
        load_model()
        print(f"[DINOv2] Model loaded on {DEVICE}")
    except Exception as e:
        print(f"[DINOv2] Warning: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "DINOv2-ViT-B/14", "device": DEVICE, "loaded": model is not None, "weights_path": LOCAL_WEIGHT_PATH}


def extract_embedding(frame_rgb):
    """Extract the embedding of a single frame."""
    m = load_model()
    img_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = m(img_tensor)
    return embedding[0].cpu().numpy()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sample_fps: float = Form(default=2.0),
    max_frames: int = Form(default=30)
):
    """
    Extract visual embeddings of video frames.
    
    Input: a video file
    Parameters:
        - sample_fps: sampling frame rate (default 2fps)
        - max_frames: maximum number of frames to process (default 30)
    Output:
        - embeddings: [[float]] per-frame embedding vectors (768-d)
        - frame_times: [float] timestamp of each frame
        - pairwise_similarities: [[float]] cosine similarity matrix between frames
        - mean_similarity: mean inter-frame similarity
        - temporal_consistency: sequence of adjacent-frame similarities
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_interval = max(1, int(fps / sample_fps))
        frames = []
        frame_times = []
        frame_idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_times.append(round(frame_idx / fps, 3))
            frame_idx += 1
        cap.release()

        if len(frames) == 0:
            return JSONResponse({"success": False, "error": "No frames extracted"}, status_code=400)

        # Batch compute embeddings
        m = load_model()
        batch_tensors = torch.stack([transform(f) for f in frames]).to(DEVICE)
        with torch.no_grad():
            embeddings = m(batch_tensors).cpu().numpy()

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        sim_matrix = (normalized @ normalized.T).tolist()

        # Temporal consistency (adjacent frame similarity)
        temporal_sim = []
        for i in range(len(normalized) - 1):
            sim = float(np.dot(normalized[i], normalized[i + 1]))
            temporal_sim.append(round(sim, 4))

        mean_sim = float(np.mean([sim_matrix[i][j] 
                                   for i in range(len(sim_matrix)) 
                                   for j in range(i+1, len(sim_matrix))]))

        return JSONResponse({
            "success": True,
            "num_frames": len(frames),
            "embedding_dim": int(embeddings.shape[1]),
            "frame_times": frame_times,
            "embeddings": embeddings.tolist(),
            "pairwise_similarities": sim_matrix,
            "temporal_consistency": temporal_sim,
            "mean_similarity": round(mean_sim, 4),
            "style_consistency_score": round(float(np.mean(temporal_sim)) if temporal_sim else 1.0, 4)
        })

    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e), "traceback": traceback.format_exc()}, status_code=500)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
