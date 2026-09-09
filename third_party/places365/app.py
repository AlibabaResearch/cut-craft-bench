"""
Scene classification microservice (CLIP zero-shot based).
Port: 8012
Function: classify the scene of video frames (indoor/outdoor/landscape/cityscape, ...), replacing Places365.
Dimensions covered: D1 empty-shot transition (scene category decision).
"""
import os
import tempfile
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Scene Classification Service (CLIP-based)", version="1.0")

model = None
preprocess = None
tokenizer = None
text_features_cache = {}
DEVICE = "cuda:0"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(MODEL_ROOT, "clip"))

# Places365-style scene categories (grouped)
SCENE_CATEGORIES = {
    "landscape": ["landscape", "mountain", "field", "forest", "valley", "countryside"],
    "waterscape": ["ocean", "lake", "river", "waterfall", "beach", "coast"],
    "cityscape": ["city skyline", "street", "highway", "bridge", "building exterior"],
    "indoor_home": ["living room", "bedroom", "kitchen", "bathroom", "dining room"],
    "indoor_public": ["office", "restaurant", "hospital", "classroom", "library"],
    "object_closeup": ["close-up of an object", "food", "flower", "book", "artwork"],
    "sky": ["sky", "clouds", "sunset", "sunrise", "night sky with stars"],
    "empty_scene": ["empty room", "empty landscape", "abandoned place", "still life without people"],
}

# Flat list for classification
SCENE_LABELS = []
SCENE_GROUP_MAP = {}
for group, items in SCENE_CATEGORIES.items():
    for item in items:
        SCENE_LABELS.append(item)
        SCENE_GROUP_MAP[item] = group


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
        # Pre-compute text features for scene categories
        _cache_text_features()
        print(f"[Places365] CLIP model loaded on {DEVICE}")


def _cache_text_features():
    """Pre-compute text embeddings for all scene categories"""
    global text_features_cache
    prompts = [f"a photo of {label}" for label in SCENE_LABELS]
    tokens = tokenizer(prompts).to(DEVICE)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    text_features_cache["scene"] = features


@app.on_event("startup")
async def startup():
    try:
        load_model()
    except Exception as e:
        print(f"[Places365] Warning: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "CLIP ViT-L-14 (zero-shot scene classification)",
        "device": DEVICE,
        "loaded": model is not None,
        "num_categories": len(SCENE_LABELS)
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


def classify_frame(frame):
    """Classify a single frame into scene categories"""
    from PIL import Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    img_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    # Compute similarities
    sims = (img_features @ text_features_cache["scene"].T).squeeze(0).cpu().numpy()

    # Get top predictions
    top_indices = np.argsort(sims)[::-1][:5]
    predictions = []
    for idx in top_indices:
        label = SCENE_LABELS[idx]
        predictions.append({
            "label": label,
            "group": SCENE_GROUP_MAP[label],
            "score": round(float(sims[idx]), 4)
        })

    # Group-level aggregation
    group_scores = {}
    for group in SCENE_CATEGORIES.keys():
        group_indices = [i for i, l in enumerate(SCENE_LABELS) if SCENE_GROUP_MAP[l] == group]
        group_scores[group] = round(float(np.max(sims[group_indices])), 4)

    top_group = max(group_scores, key=group_scores.get)

    return {
        "top_predictions": predictions,
        "group_scores": group_scores,
        "top_group": top_group,
        "is_outdoor": top_group in ["landscape", "waterscape", "cityscape", "sky"],
        "is_empty_scene": group_scores.get("empty_scene", 0) > 0.2
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sample_fps: float = Form(default=2.0),
    max_frames: int = Form(default=16)
):
    """
    Scene classification.

    Output:
        - per_frame: per-frame scene classification
        - summary: aggregated (dominant scene type, indoor/outdoor ratio, ...)
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
        all_groups = []

        for i, frame in enumerate(frames):
            result = classify_frame(frame)
            result["frame_time"] = frame_times[i]
            per_frame_results.append(result)
            all_groups.append(result["top_group"])

        # Summary
        from collections import Counter
        group_counts = Counter(all_groups)
        dominant_group = group_counts.most_common(1)[0][0]
        outdoor_ratio = sum(1 for g in all_groups if g in ["landscape", "waterscape", "cityscape", "sky"]) / len(all_groups)

        summary = {
            "num_frames": len(frames),
            "dominant_scene": dominant_group,
            "scene_distribution": dict(group_counts),
            "outdoor_ratio": round(outdoor_ratio, 3),
            "indoor_ratio": round(1 - outdoor_ratio, 3),
            "is_predominantly_outdoor": outdoor_ratio > 0.5,
            "has_empty_scenes": any(r["is_empty_scene"] for r in per_frame_results),
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
    uvicorn.run(app, host="0.0.0.0", port=8012)
