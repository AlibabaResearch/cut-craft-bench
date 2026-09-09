"""
CLIP/OpenCLIP microservice - text-image semantic similarity.
Port: 8010
Function: semantic similarity between video frames and text descriptions, including cross-shot semantic distance.
Dimensions covered: C1 montage (semantic distance), D1 gilligan cut (semantic contrast), D1 logical cut (cross-shot similarity), D1 POV (frame consistency).
"""
import os
import tempfile
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="CLIP Semantic Similarity Service", version="1.0")

model = None
preprocess = None
tokenizer = None
DEVICE = "cuda:0"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(MODEL_ROOT, "clip"))
LOCAL_WEIGHT_PATH = os.path.join(MODEL_DIR, "ViT-L-14.pt")


def load_model():
    global model, preprocess, tokenizer
    if model is None:
        import open_clip
        from open_clip.openai import load_openai_model
        from open_clip.transform import image_transform
        if not os.path.exists(LOCAL_WEIGHT_PATH):
            raise FileNotFoundError(f"CLIP local weights not found: {LOCAL_WEIGHT_PATH}")
        model = load_openai_model(LOCAL_WEIGHT_PATH, precision="fp32", device=DEVICE)
        preprocess = image_transform(
            model.visual.image_size,
            is_train=False,
            mean=model.visual.image_mean,
            std=model.visual.image_std,
        )
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.eval()
        print(f"[CLIP] Model {MODEL_NAME} loaded from LOCAL: {LOCAL_WEIGHT_PATH} on {DEVICE}")


@app.on_event("startup")
async def startup():
    try:
        load_model()
    except Exception as e:
        print(f"[CLIP] Warning: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": f"OpenCLIP {MODEL_NAME}/{PRETRAINED}",
        "device": DEVICE,
        "loaded": model is not None,
        "weights_path": LOCAL_WEIGHT_PATH
    }


def extract_frames(video_path, sample_fps=2.0, max_frames=16):
    """Sample frames from the video."""
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


def frame_to_tensor(frame):
    """Convert a BGR frame into a CLIP input tensor."""
    from PIL import Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = preprocess(pil_img).unsqueeze(0)
    model_dtype = next(model.parameters()).dtype if model is not None else torch.float32
    return tensor.to(device=DEVICE, dtype=model_dtype)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    texts: str = Form(default=""),
    sample_fps: float = Form(default=2.0),
    max_frames: int = Form(default=16)
):
    """
    CLIP semantic similarity analysis.

    Input:
        - file: video file
        - texts: comma-separated text list (e.g. "a person falling,a happy person,landscape")
        - sample_fps: sampling frame rate
        - max_frames: maximum number of frames
    Output:
        - frame_embeddings_sim: cosine similarity matrix between frames
        - text_frame_sim: similarity of each text against each frame
        - cross_shot_similarity: semantic distance of adjacent frame pairs
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

        # Encode frames
        frame_tensors = torch.cat([frame_to_tensor(f) for f in frames], dim=0)
        with torch.no_grad():
            frame_features = model.encode_image(frame_tensors)
            frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)

        # Frame-to-frame similarity matrix
        sim_matrix = (frame_features @ frame_features.T).cpu().numpy()

        # Cross-shot similarity (adjacent frames)
        cross_shot_sims = []
        for i in range(len(frames) - 1):
            sim_val = float(sim_matrix[i, i + 1])
            cross_shot_sims.append({
                "pair": [frame_times[i], frame_times[i + 1]],
                "similarity": round(sim_val, 4)
            })

        result = {
            "success": True,
            "fps": round(fps, 2),
            "num_frames": len(frames),
            "frame_times": frame_times,
            "frame_sim_matrix": [[round(float(v), 4) for v in row] for row in sim_matrix],
            "cross_shot_similarities": cross_shot_sims,
            "avg_cross_shot_sim": round(float(np.mean([s["similarity"] for s in cross_shot_sims])), 4) if cross_shot_sims else 0.0,
        }

        # Text-frame similarity
        if texts.strip():
            text_list = [t.strip() for t in texts.split(",") if t.strip()]
            tokens = tokenizer(text_list).to(DEVICE)
            with torch.no_grad():
                text_features = model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            text_frame_sim = (text_features @ frame_features.T).cpu().numpy()
            result["text_frame_similarities"] = {}
            for i, text in enumerate(text_list):
                sims = [round(float(v), 4) for v in text_frame_sim[i]]
                result["text_frame_similarities"][text] = {
                    "per_frame": sims,
                    "mean": round(float(np.mean(sims)), 4),
                    "max": round(float(np.max(sims)), 4),
                    "max_frame_idx": int(np.argmax(sims))
                }

            # Semantic contrast (for the D1 gilligan cut: high contrast = strong irony)
            if len(text_list) >= 2:
                text_sims = (text_features @ text_features.T).cpu().numpy()
                result["text_text_similarities"] = {
                    f"{text_list[i]} vs {text_list[j]}": round(float(text_sims[i, j]), 4)
                    for i in range(len(text_list))
                    for j in range(i + 1, len(text_list))
                }

        return JSONResponse(result)

    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e),
                           "traceback": traceback.format_exc()}, status_code=500)
    finally:
        os.unlink(tmp_path)


@app.post("/encode_text")
async def encode_text(texts: str = Form(...)):
    """Encode text alone and return the embedding."""
    load_model()
    text_list = [t.strip() for t in texts.split(",") if t.strip()]
    tokens = tokenizer(text_list).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return JSONResponse({
        "success": True,
        "texts": text_list,
        "embedding_dim": text_features.shape[-1],
        "embeddings": text_features.cpu().numpy().tolist()
    })


@app.post("/style_match")
async def style_match(
    file: UploadFile = File(...),
    target_style: str = Form(...),
    candidate_styles: str = Form(...),
    template: str = Form(default="a {} video"),
    sample_fps: float = Form(default=2.0),
    max_frames: int = Form(default=16)
):
    """
    A4 style match (CLIP zero-shot style classification).

    Input:
        - file: video file
        - target_style: target style label (must appear in candidate_styles)
        - candidate_styles: comma-separated candidate style list (e.g. all 9 classes)
        - template: text template; {} is replaced with the style name
    Output:
        - a4_score: mean per-frame softmax probability of the target style, in [0,1]
        - predicted_style: the style with the highest mean probability
        - per_style_mean_prob: mean probability of every candidate style (for diagnosis)
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        load_model()
        candidates = [s.strip() for s in candidate_styles.split(",") if s.strip()]
        if not candidates:
            return JSONResponse({"success": False, "error": "no candidate_styles"}, status_code=400)
        target_style = target_style.strip()
        if target_style not in candidates:
            candidates = [target_style] + candidates

        frames, frame_times, fps = extract_frames(tmp_path, sample_fps, max_frames)
        if not frames:
            return JSONResponse({"success": False, "error": "No frames extracted"}, status_code=400)

        # Encode the frame images
        frame_tensors = torch.cat([frame_to_tensor(f) for f in frames], dim=0)
        with torch.no_grad():
            frame_features = model.encode_image(frame_tensors)
            frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)

            # Encode the candidate style texts
            texts = [template.format(s) for s in candidates]
            tokens = tokenizer(texts).to(DEVICE)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # logits = logit_scale * cos_sim, per-frame softmax over the candidates
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (frame_features @ text_features.T)  # (F, C)
            probs = logits.softmax(dim=-1).cpu().numpy()  # (F, C)

        target_idx = candidates.index(target_style)
        C = probs.shape[1]
        per_frame_target = [round(float(p), 4) for p in probs[:, target_idx]]
        target_prob = float(np.mean(probs[:, target_idx]))
        mean_probs = probs.mean(axis=0)
        per_style_mean_prob = {candidates[i]: round(float(mean_probs[i]), 4)
                               for i in range(len(candidates))}
        predicted_style = candidates[int(np.argmax(mean_probs))]

        # Per-frame rank score: the target style's rank inside a frame (1=highest) -> (C-rank)/(C-1)
        # The random baseline sits near 0.5 and a hit (rank 1) near 1.0, far more discriminative than the diluted probability
        order = (-probs).argsort(axis=1)
        frame_ranks = [int(list(order[f]).index(target_idx)) + 1
                       for f in range(probs.shape[0])]
        rank_score = float(np.mean([(C - r) / (C - 1) for r in frame_ranks]))
        # Overall rank (based on the mean probability)
        target_rank = int(list((-mean_probs).argsort()).index(target_idx)) + 1

        # Final A4 score: exponential decay on the overall rank (tau=1.5)
        # rank1=1.0, rank2~0.51, rank3~0.26 ... discriminative, yet the model average does not collapse
        tau = 1.5
        a4_score = float(np.exp(-(target_rank - 1) / tau))

        return JSONResponse({
            "success": True,
            "a4_score": round(a4_score, 6),
            "rank_score": round(rank_score, 6),
            "target_prob": round(target_prob, 6),
            "target_rank": target_rank,
            "num_candidates": C,
            "target_style": target_style,
            "predicted_style": predicted_style,
            "num_frames": len(frames),
            "per_frame_target_prob": per_frame_target,
            "per_style_mean_prob": per_style_mean_prob,
        })

    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e),
                           "traceback": traceback.format_exc()}, status_code=500)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
