"""
E2 image quality microservice (replacing DOVER).
Port: 8006
Models:
  - Aesthetic Quality: CLIP ViT-L/14 + the LAION aesthetic linear head (per frame, raw/10)
  - Imaging Quality:   MUSIQ (SPAQ pretrained weights) (per frame, raw/100)
Output: overall_score = (aesthetic_quality + imaging_quality) / 2, in [0,1]
Reference: VBench (aesthetic_quality.py / imaging_quality.py)
"""
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

_orig_torch_load = torch.load

def _torch_load_compat(*args, **kwargs):
    kwargs.pop("weights_only", None)
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_compat

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except Exception:
    from PIL import Image
    BICUBIC = Image.BICUBIC

app = FastAPI(title="E2 Quality (Aesthetic+Imaging) Service", version="1.0")

# ============ Configuration ============
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(MODEL_ROOT, "e2_quality"))
CLIP_PATH = os.path.join(MODEL_DIR, "ViT-L-14.pt")
AES_HEAD_PATH = os.path.join(MODEL_DIR, "sa_0_4_vit_l_14_linear.pth")
MUSIQ_PATH = os.path.join(MODEL_DIR, "musiq_spaq_ckpt-358bb6af.pth")

BATCH_SIZE = 32

# Global models
clip_model = None
aesthetic_head = None
musiq_model = None


def clip_transform(n_px):
    """CLIP image preprocessing (tensor version, matching VBench)."""
    return Compose([
        Resize(n_px, interpolation=BICUBIC, antialias=False),
        CenterCrop(n_px),
        T.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def musiq_transform(images):
    """MUSIQ preprocessing (longer mode: scale the long edge down to 512, then normalise to [0,1])."""
    _, _, h, w = images.size()
    if max(h, w) > 512:
        scale = 512. / max(h, w)
        images = T.Resize(size=(int(scale * h), int(scale * w)),
                          antialias=False)(images)
    return images / 255.


def get_aesthetic_head(path):
    """Load the LAION aesthetic linear head (768 -> 1)."""
    m = nn.Linear(768, 1)
    s = torch.load(path, map_location="cpu")
    m.load_state_dict(s)
    m.eval()
    return m


def load_models():
    """Load CLIP ViT-L/14 + the LAION aesthetic head + MUSIQ."""
    global clip_model, aesthetic_head, musiq_model
    if clip_model is not None:
        return

    import clip
    clip_model, _ = clip.load(CLIP_PATH, device=DEVICE)
    clip_model.eval()

    aesthetic_head = get_aesthetic_head(AES_HEAD_PATH).to(DEVICE)

    from pyiqa.archs.musiq_arch import MUSIQ
    musiq_model = MUSIQ(pretrained_model_path=MUSIQ_PATH)
    musiq_model.to(DEVICE)
    musiq_model.eval()

    print(f"[E2Quality] Models loaded on {DEVICE} "
          f"(CLIP ViT-L/14 + LAION aesthetic + MUSIQ-SPAQ)")


def load_frames(video_path, num_frames=32):
    """Uniformly sample video frames, returning an (N,C,H,W) uint8 RGB tensor."""
    frames = None
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            raise RuntimeError("empty video (decord)")
        if num_frames and num_frames < total:
            idxs = np.linspace(0, total - 1, num_frames).astype(int).tolist()
        else:
            idxs = list(range(total))
        frames = vr.get_batch(idxs).asnumpy()  # (N,H,W,C) RGB uint8
    except Exception:
        import cv2
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            all_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        total = len(all_frames)
        if total == 0:
            raise RuntimeError("empty video (cv2)")
        if num_frames and num_frames < total:
            idxs = np.linspace(0, total - 1, num_frames).astype(int).tolist()
        else:
            idxs = list(range(total))
        frames = np.stack([all_frames[i] for i in idxs])

    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    return frames  # uint8 (N,C,H,W)


@torch.no_grad()
def compute_aesthetic(frames):
    """Per-frame aesthetic scoring, returning the mean in [0,1]."""
    aes_tf = clip_transform(224)
    scores = []
    N = frames.shape[0]
    for i in range(0, N, BATCH_SIZE):
        batch = frames[i:i + BATCH_SIZE]
        batch = aes_tf(batch).to(DEVICE)
        feats = clip_model.encode_image(batch).to(torch.float32)
        feats = F.normalize(feats, dim=-1, p=2)
        s = aesthetic_head(feats).squeeze(-1)  # raw ~0-10
        scores.append(s.detach().cpu())
    scores = torch.cat(scores, dim=0) / 10.0
    return float(scores.mean()), scores.tolist()


@torch.no_grad()
def compute_imaging(frames):
    """Per-frame MUSIQ scoring, returning the mean in [0,1]."""
    scores = []
    N = frames.shape[0]
    for i in range(N):
        frame = frames[i:i + 1].float()          # (1,C,H,W) 0-255
        frame = musiq_transform(frame).to(DEVICE)  # -> [0,1]
        sc = musiq_model(frame)
        scores.append(float(sc))
    scores = [s / 100.0 for s in scores]           # MUSIQ raw ~0-100
    return float(np.mean(scores)), scores


@app.on_event("startup")
async def startup():
    try:
        load_models()
    except Exception as e:
        import traceback
        print(f"[E2Quality] Warning: Failed to load models: {e}")
        traceback.print_exc()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "CLIP-ViT-L14+LAION_aesthetic & MUSIQ-SPAQ",
        "device": DEVICE,
        "loaded": clip_model is not None and musiq_model is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    num_frames: int = Form(default=32),
):
    """
    E2 image quality: the mean of the aesthetic (CLIP+LAION) and imaging (MUSIQ) model scores.

    Output:
        - aesthetic_quality: aesthetic score [0,1]
        - imaging_quality:   imaging score [0,1]
        - overall_score:     the mean of the two [0,1]  <- the final E2 score
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        load_models()
        if clip_model is None or musiq_model is None:
            return JSONResponse(
                {"success": False, "error": "E2 quality models not loaded"},
                status_code=500,
            )

        frames = load_frames(tmp_path, num_frames=num_frames)
        aesthetic_score, aes_list = compute_aesthetic(frames)
        imaging_score, img_list = compute_imaging(frames)
        overall_score = (aesthetic_score + imaging_score) / 2.0

        return JSONResponse({
            "success": True,
            "aesthetic_quality": round(aesthetic_score, 6),
            "imaging_quality": round(imaging_score, 6),
            "overall_score": round(overall_score, 6),
            "details": {
                "num_frames": int(frames.shape[0]),
                "aesthetic_mean": round(aesthetic_score, 6),
                "imaging_mean": round(imaging_score, 6),
            },
            "model_used": "CLIP-ViT-L14+LAION_aesthetic & MUSIQ-SPAQ",
        })

    except Exception as e:
        import traceback
        return JSONResponse(
            {"success": False, "error": str(e),
             "traceback": traceback.format_exc()},
            status_code=500,
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
