"""
Demucs microservice - audio source separation.
Port: 8005
Function: split audio into four stems (vocals / drums / bass / other).
"""
import os
import tempfile
import shutil
import subprocess
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

app = FastAPI(title="Demucs Audio Source Separation Service", version="1.0")

model = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.environ.get(
    "DEMUCS_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "demucs_output")
)
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(MODEL_ROOT, "demucs"))
WEIGHTS_PATH = os.path.join(MODEL_DIR, "955717e8-8726e21a.th")


def load_model():
    global model
    if model is not None:
        return model
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from pathlib import Path
    # Load from local files only, to avoid a remote download at runtime
    local_repo = Path(MODEL_DIR)
    local_weight = Path(WEIGHTS_PATH)
    if not local_weight.exists():
        raise FileNotFoundError(f"Demucs local weights not found: {local_weight}")
    model = get_model("htdemucs", repo=local_repo)
    print(f"[Demucs] Model loaded from LOCAL: {local_repo}")
    model.to(DEVICE)
    model.eval()
    return model


@app.on_event("startup")
async def startup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        load_model()
        print(f"[Demucs] Model loaded on {DEVICE}")
    except Exception as e:
        print(f"[Demucs] Warning: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "htdemucs", "device": DEVICE, "loaded": model is not None, "weights_path": WEIGHTS_PATH}


def _extract_audio_to_wav(input_path: str, sample_rate: int = 44100, channels: int = 2) -> str:
    """Decode video/container audio to a WAV file that torchaudio can reliably read."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", str(channels),
        wav_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr.strip()[:500]}")
    return wav_path


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    return_stats: bool = Form(default=True)
):
    """
    Audio source separation.
    
    Input: a video or audio file
    Parameters:
        - return_stats: whether to return per-stem statistics (default True)
    Output:
        - sources: statistics for each of {vocals, drums, bass, other}
        - source_files: paths of the separated audio files (downloadable)
        - duration: audio duration
        - sample_rate: sample rate
    """
    from demucs.apply import apply_model
    
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    audio_path = None
    try:
        m = load_model()
        
        # Decode container audio first; torchaudio 0.12 is unreliable on MP4 directly.
        audio_path = _extract_audio_to_wav(tmp_path, sample_rate=m.samplerate, channels=2)
        wav, sr = torchaudio.load(audio_path)
        
        # Resample to model's sample rate if needed
        if sr != m.samplerate:
            wav = torchaudio.functional.resample(wav, sr, m.samplerate)
            sr = m.samplerate
        
        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        
        duration = wav.shape[1] / sr
        
        # Detect silent audio: when the track is virtually silent, return an all-zero separation directly
        ref = wav.mean(0)
        if ref.std().item() < 1e-6:
            source_names = m.sources  # ['drums', 'bass', 'other', 'vocals']
            source_info = {}
            source_files = {}
            session_dir = tempfile.mkdtemp(dir=OUTPUT_DIR)
            for name in source_names:
                source_info[name] = {"rms": 0.0, "peak": 0.0, "energy_ratio": 0.0}
                out_path = os.path.join(session_dir, f"{name}.wav")
                torchaudio.save(out_path, torch.zeros_like(wav), sr)
                source_files[name] = out_path
            return JSONResponse({
                "success": True,
                "duration": round(duration, 3),
                "sample_rate": sr,
                "sources": source_info,
                "source_files": source_files,
                "source_names": source_names,
                "note": "audio is silent, returning zero sources"
            })
        
        # htdemucs needs audio longer than its segment (7.8s), otherwise the internal STFT padding trips an assertion
        # Short audio is loop-padded to the training length and truncated back after inference
        TRAINING_LENGTH = 343980  # segment=7.8s * sr=44100
        original_length = wav.shape[1]
        padded = False
        if original_length < TRAINING_LENGTH:
            # Loop the audio to reach the model's training length
            repeats = (TRAINING_LENGTH // original_length) + 1
            wav = wav.repeat(1, repeats)[:, :TRAINING_LENGTH]
            padded = True
        
        # Apply model (shift-trick and split are disabled for short audio to avoid tiny chunks)
        wav = (wav - ref.mean()) / ref.std()
        
        shifts = 0 if padded else 1
        split = not padded  # do not split short audio, process it in one pass
        with torch.no_grad():
            sources = apply_model(m, wav[None].to(DEVICE), device=DEVICE,
                                  progress=False, shifts=shifts, split=split)[0]
        
        sources = sources * ref.std() + ref.mean()
        sources = sources.cpu()
        
        # Truncate back to the original length when padding was applied
        if padded:
            sources = sources[:, :, :original_length]
        
        # Source names
        source_names = m.sources  # ['drums', 'bass', 'other', 'vocals']
        
        # Save separated sources and compute stats
        session_dir = tempfile.mkdtemp(dir=OUTPUT_DIR)
        source_info = {}
        source_files = {}
        
        for i, name in enumerate(source_names):
            src = sources[i]
            
            # Stats
            rms = float(torch.sqrt(torch.mean(src ** 2)))
            peak = float(torch.max(torch.abs(src)))
            energy_ratio = float(torch.sum(src ** 2) / (torch.sum(sources ** 2) + 1e-8))
            
            source_info[name] = {
                "rms": round(rms, 6),
                "peak": round(peak, 6),
                "energy_ratio": round(energy_ratio, 4),
            }
            
            # Save file
            out_path = os.path.join(session_dir, f"{name}.wav")
            torchaudio.save(out_path, src, sr)
            source_files[name] = out_path
        
        return JSONResponse({
            "success": True,
            "duration": round(duration, 3),
            "sample_rate": sr,
            "sources": source_info,
            "source_files": source_files,
            "source_names": source_names
        })

    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e), "traceback": traceback.format_exc()}, status_code=500)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        os.unlink(tmp_path)


if __name__ == "__main__":
    # SERVICE_PORT allows starting a second instance without changing the default behaviour (e.g. an offset port for the edit agent's self-evaluation)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("SERVICE_PORT", 8005)))
