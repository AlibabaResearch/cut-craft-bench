"""
Whisper microservice - speech recognition (ASR).
Port: 8004
Function: extract spoken text and timestamps from a video or audio file.
"""
import os
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Whisper ASR Service", version="1.0")

model = None
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
DEVICE = "cuda:0"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", os.path.join(MODEL_ROOT, "whisper"))


def load_model():
    global model
    if model is not None:
        return model
    import whisper
    # Load from the local model directory only, to avoid a remote download at runtime
    local_path = os.path.join(LOCAL_MODEL_DIR, f"{MODEL_SIZE}.pt")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Whisper local weights not found: {local_path}")
    model = whisper.load_model(MODEL_SIZE, device=DEVICE, download_root=LOCAL_MODEL_DIR)
    print(f"[Whisper] Model loaded from LOCAL: {local_path}")
    return model


@app.on_event("startup")
async def startup():
    try:
        load_model()
        print(f"[Whisper] Model '{MODEL_SIZE}' loaded on {DEVICE}")
    except Exception as e:
        print(f"[Whisper] Warning: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": f"Whisper-{MODEL_SIZE}", "device": DEVICE, "loaded": model is not None, "weights_path": os.path.join(LOCAL_MODEL_DIR, f"{MODEL_SIZE}.pt")}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    task: str = Form(default="transcribe"),
    word_timestamps: bool = Form(default=False),
    condition_on_previous_text: bool = Form(default=True),
    word_gap_threshold: float = Form(default=0.1)
):
    """
    Speech recognition.
    
    Input: a video or audio file
    Parameters:
        - language: language code (auto-detected by default, e.g. 'zh', 'en', 'ja')
        - task: 'transcribe' or 'translate' (translate into English)
        - word_timestamps: whether to emit word-level timestamps (off by default; the evaluation uses Whisper's raw segments)
        - condition_on_previous_text: whether the current window may reference the preceding text
        - word_gap_threshold: split into two speech segments when the inter-word pause exceeds this threshold (only effective with word_timestamps on)
    Output:
        - text: the full transcript
        - segments: [{id, start, end, text, confidence, words}] segment results
        - words: [{word, start, end, probability, segment_id}] word-level timestamps
        - pause_split_segments: speech segments after the secondary split on inter-word pauses
        - language: the detected language
        - duration: total audio duration
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        m = load_model()
        
        options = {
            "task": task,
            "word_timestamps": word_timestamps,
            "condition_on_previous_text": condition_on_previous_text,
        }
        if language:
            options["language"] = language
        
        result = m.transcribe(tmp_path, **options)
        
        segments = []
        words = []
        for seg in result.get("segments", []):
            seg_words = []
            for word in seg.get("words", []) or []:
                word_item = {
                    "word": str(word.get("word", "")).strip(),
                    "start": round(float(word.get("start", 0.0)), 3),
                    "end": round(float(word.get("end", 0.0)), 3),
                    "probability": round(float(word.get("probability", 0.0)), 4),
                    "segment_id": seg["id"],
                }
                if word_item["word"]:
                    seg_words.append(word_item)
                    words.append(word_item)
            segments.append({
                "id": seg["id"],
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"].strip(),
                "confidence": round(1.0 - seg.get("no_speech_prob", 0), 4),
                "words": seg_words,
            })

        pause_split_segments = []
        current_words = []
        for word in words:
            if current_words:
                gap = float(word["start"]) - float(current_words[-1]["end"])
                if gap >= word_gap_threshold:
                    pause_split_segments.append({
                        "id": len(pause_split_segments),
                        "start": current_words[0]["start"],
                        "end": current_words[-1]["end"],
                        "text": " ".join(w["word"] for w in current_words).strip(),
                        "words": current_words,
                        "split_reason": f"word_gap>={word_gap_threshold}",
                    })
                    current_words = []
            current_words.append(word)
        if current_words:
            pause_split_segments.append({
                "id": len(pause_split_segments),
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": " ".join(w["word"] for w in current_words).strip(),
                "words": current_words,
                "split_reason": "final",
            })
        
        # Compute total speech duration
        speech_duration = sum(s["end"] - s["start"] for s in segments)
        
        return JSONResponse({
            "success": True,
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": segments,
            "words": words,
            "pause_split_segments": pause_split_segments,
            "num_segments": len(segments),
            "num_words": len(words),
            "num_pause_split_segments": len(pause_split_segments),
            "speech_duration": round(speech_duration, 3),
            "word_count": len(words) if words else len(result["text"].split()),
            "word_gap_threshold": word_gap_threshold
        })

    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e), "traceback": traceback.format_exc()}, status_code=500)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    # SERVICE_PORT allows starting a second instance without changing the default behaviour (e.g. an offset port for the edit agent's self-evaluation)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("SERVICE_PORT", 8004)))
