"""
MovieShots shot-scale classification microservice (CLIP zero-shot + a YOLOv8 body-ratio rule).
Port: 8013
Function: classify the shot scale of video frames (ECU/CU/MCU/MS/MLS/LS/ELS/XLS).
Dimensions covered: E1 shot scale.
"""
import os
import tempfile
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="MovieShots Shot Scale Classification", version="1.0")

model = None
preprocess = None
tokenizer = None
text_features_cache = {}
DEVICE = "cuda:0"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(MODEL_ROOT, "clip"))

# Professional cinematography shot scale definitions
SHOT_SCALES = {
    "ECU": "extreme close-up of eyes or mouth filling the frame",
    "CU": "close-up of a person's face filling most of the frame",
    "MCU": "medium close-up showing head and shoulders",
    "MS": "medium shot showing a person from the waist up",
    "MLS": "medium long shot showing a person from the knees up",
    "LS": "long shot showing full body of a person with some environment",
    "ELS": "extreme long shot of a wide landscape or cityscape with tiny or no people",
    "XLS": "establishing shot of a vast landscape or aerial view"
}

SHOT_SCALE_ORDER = ["ECU", "CU", "MCU", "MS", "MLS", "LS", "ELS", "XLS"]


def load_model():
    global model, preprocess, tokenizer
    if model is None:
        import open_clip
        local_path = os.path.join(MODEL_DIR, "open_clip_pytorch_model.bin")
        if os.path.exists(local_path):
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained=local_path, device=DEVICE
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=DEVICE
            )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        model.eval()
        _cache_text_features()
        print(f"[MovieShots] CLIP model loaded on {DEVICE}")


def _cache_text_features():
    """Pre-compute text embeddings for shot scale categories"""
    global text_features_cache
    prompts = [f"a {desc}" for desc in SHOT_SCALES.values()]
    tokens = tokenizer(prompts).to(DEVICE)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    text_features_cache["shot_scale"] = features


@app.on_event("startup")
async def startup():
    try:
        load_model()
    except Exception as e:
        print(f"[MovieShots] Warning: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "MovieShots (CLIP ViT-L-14 zero-shot + rule-based)",
        "device": DEVICE,
        "loaded": model is not None,
        "shot_scales": SHOT_SCALE_ORDER
    }


def extract_frames(video_path, sample_fps=2.0, max_frames=16):
    cap = cv2.VideoCapture(video_path)
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
            frames.append(frame)
            frame_times.append(round(frame_idx / fps, 3))
        frame_idx += 1
    cap.release()
    return frames, frame_times, fps


def detect_face_body_ratios(frame):
    """Detect face/body ratios using Haar cascade"""
    h, w = frame.shape[:2]
    frame_area = h * w
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    max_face_ratio = 0.0
    for (x, y, fw, fh) in faces:
        ratio = (fw * fh) / frame_area
        max_face_ratio = max(max_face_ratio, ratio)
    
    # Simple body detection by color/edge (fallback)
    # We'll rely more on CLIP for this
    return max_face_ratio


def classify_shot_scale(frame):
    """Classify shot scale using CLIP + face ratio heuristics"""
    from PIL import Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    img_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    # CLIP similarity scores
    sims = (img_features @ text_features_cache["shot_scale"].T).squeeze(0).cpu().numpy()
    
    # Face ratio heuristic
    face_ratio = detect_face_body_ratios(frame)
    
    # Rule-based adjustments
    clip_scores = {}
    for i, scale in enumerate(SHOT_SCALE_ORDER):
        clip_scores[scale] = float(sims[i])
    
    # Boost/penalize based on face ratio
    if face_ratio > 0.35:
        clip_scores["ECU"] += 0.05
        clip_scores["CU"] += 0.03
    elif face_ratio > 0.15:
        clip_scores["CU"] += 0.03
        clip_scores["MCU"] += 0.02
    elif face_ratio > 0.05:
        clip_scores["MCU"] += 0.02
        clip_scores["MS"] += 0.01
    elif face_ratio < 0.01 and face_ratio == 0:
        clip_scores["ELS"] += 0.03
        clip_scores["XLS"] += 0.02
    
    # Normalize scores
    total = sum(clip_scores.values())
    normalized = {k: round(v / total, 4) for k, v in clip_scores.items()}
    
    # Get top prediction
    top_scale = max(normalized, key=normalized.get)
    
    return {
        "predicted_scale": top_scale,
        "confidence": normalized[top_scale],
        "all_scores": normalized,
        "face_ratio": round(face_ratio, 4),
        "scale_index": SHOT_SCALE_ORDER.index(top_scale)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sample_fps: float = Form(default=2.0),
    max_frames: int = Form(default=16)
):
    """
    Shot scale classification.

    Output:
        - per_frame: per-frame shot scale classification
        - summary: aggregated (dominant scale, mean scale index, ...)
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        load_model()
        frames, frame_times, fps = extract_frames(tmp_path, sample_fps, max_frames)

        if not frames:
            return JSONResponse({"success": False, "error": "No frames extracted"}, status_code=400)

        per_frame_results = []
        all_scale_indices = []

        for i, frame in enumerate(frames):
            result = classify_shot_scale(frame)
            result["frame_time"] = frame_times[i]
            per_frame_results.append(result)
            all_scale_indices.append(result["scale_index"])

        # Summary
        from collections import Counter
        scale_counts = Counter(r["predicted_scale"] for r in per_frame_results)
        dominant_scale = scale_counts.most_common(1)[0][0]

        summary = {
            "num_frames": len(frames),
            "dominant_scale": dominant_scale,
            "scale_distribution": dict(scale_counts),
            "avg_scale_index": round(float(np.mean(all_scale_indices)), 2),
            "scale_consistency": round(float(scale_counts.most_common(1)[0][1] / len(frames)), 3),
            "avg_face_ratio": round(float(np.mean([r["face_ratio"] for r in per_frame_results])), 4),
        }

        return JSONResponse({
            "success": True,
            "fps": round(fps, 2),
            "per_frame": per_frame_results,
            "summary": summary
        })

    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e),
                           "traceback": traceback.format_exc()}, status_code=500)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8013)
