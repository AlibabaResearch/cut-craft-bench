"""
Audio signal quality analysis microservice (originally DNSMOS, heavily reworked).
Port: 8008
Function: multi-dimensional audio quality feature extraction via signal processing, plus an overall objective quality score.
      Works on general audio (music / sound effects / ambience), not speech only.
"""
import os
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Audio Signal Quality Analysis Service", version="2.0")

SAMPLE_RATE = 22050  # librosa default sample rate, fine for general audio analysis


def compute_audio_quality_features(audio_path: str) -> dict:
    """
    Multi-dimensional audio quality feature extraction based on librosa.
    
    Output features:
      - snr_db: estimated signal-to-noise ratio (dB)
      - dynamic_range_db: dynamic range (dB)
      - spectral_richness: spectral richness (0-1, from spectral entropy)
      - spectral_centroid_hz: spectral centroid (Hz)
      - clip_ratio: clipping ratio
      - silence_ratio: silence ratio
      - rms_mean_db: mean loudness (dB)
      - onset_density: onsets per second
      - bandwidth_hz: effective bandwidth (Hz)
      - objective_quality_score: overall objective quality score [0,1]
    """
    import librosa

    # Load the audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr

    if duration < 0.3:
        return None

    # === 1. SNR estimation ===
    # Use a low RMS quantile as the noise floor estimate
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    noise_floor = float(np.percentile(rms_db, 10))
    signal_level = float(np.percentile(rms_db, 90))
    snr_db = max(0.0, signal_level - noise_floor)

    # === 2. Dynamic range ===
    dynamic_range_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    # === 3. Spectral richness (spectral entropy) ===
    # Normalised entropy of the mel spectrum; higher means a richer spectrum
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    mel_power = mel + 1e-10
    # Spectral entropy per frame
    mel_norm = mel_power / mel_power.sum(axis=0, keepdims=True)
    spectral_entropy = -np.sum(mel_norm * np.log2(mel_norm + 1e-10), axis=0)
    max_entropy = np.log2(128)  # maximum possible entropy
    spectral_richness = float(np.mean(spectral_entropy) / max_entropy)

    # === 4. Spectral centroid ===
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_hz = float(np.mean(centroid))

    # === 5. Clipping detection ===
    clip_ratio = float(np.sum(np.abs(y) > 0.99)) / len(y)

    # === 6. Silence ratio ===
    # Below -50dB counts as silence
    frame_rms = rms
    silence_threshold_linear = 10 ** (-50 / 20)
    silence_ratio = float(np.sum(frame_rms < silence_threshold_linear)) / len(frame_rms)

    # === 7. Mean loudness ===
    rms_mean = float(np.mean(rms))
    rms_mean_db = float(20 * np.log10(rms_mean + 1e-10))

    # === 8. Onset density (audio activity) ===
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_density = float(len(onsets) / duration) if duration > 0 else 0.0

    # === 9. Effective bandwidth ===
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bandwidth_hz = float(np.mean(bandwidth))

    # === 10. Overall objective quality score ===
    # Each sub-component is normalised to [0,1] and then weighted
    snr_score = min(snr_db / 35.0, 1.0)                    # 35dB scores full marks
    dr_score = min(dynamic_range_db / 30.0, 1.0)           # 30dB scores full marks
    richness_score = spectral_richness                      # already in 0-1
    clip_penalty = min(clip_ratio * 20.0, 0.5)             # penalty capped at 0.5
    silence_penalty = max(0.0, (silence_ratio - 0.5) * 1.0)  # penalise once silence exceeds 50%
    # Onset density: moderate is best (2-8/s); too low or too high is bad
    if onset_density < 0.5:
        onset_score = onset_density / 0.5 * 0.5  # very few onsets -> low score
    elif onset_density <= 10.0:
        onset_score = 0.5 + 0.5 * min((onset_density - 0.5) / 5.0, 1.0)
    else:
        onset_score = max(0.5, 1.0 - (onset_density - 10.0) / 20.0)  # penalise an over-dense track

    objective_quality_score = (
        0.25 * snr_score +
        0.15 * dr_score +
        0.25 * richness_score +
        0.15 * onset_score +
        0.20 * min(bandwidth_hz / 8000.0, 1.0)  # bandwidth component
        - clip_penalty
        - silence_penalty
    )
    objective_quality_score = max(0.0, min(1.0, objective_quality_score))

    return {
        "snr_db": round(snr_db, 2),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "spectral_richness": round(spectral_richness, 4),
        "spectral_centroid_hz": round(spectral_centroid_hz, 1),
        "clip_ratio": round(clip_ratio, 6),
        "silence_ratio": round(silence_ratio, 4),
        "rms_mean_db": round(rms_mean_db, 2),
        "onset_density": round(onset_density, 2),
        "bandwidth_hz": round(bandwidth_hz, 1),
        "duration": round(duration, 3),
        "objective_quality_score": round(objective_quality_score, 4),
    }


@app.on_event("startup")
async def startup():
    # Pre-import librosa to speed up the first request
    try:
        import librosa
        print("[AudioQuality] librosa loaded, service ready")
    except Exception as e:
        print(f"[AudioQuality] Warning: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "signal_analysis", "version": "2.0"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Audio signal quality analysis.

    Input: a video or audio file
    Output:
        - snr_db, dynamic_range_db, spectral_richness, ...
        - objective_quality_score: overall objective quality score [0,1]
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = compute_audio_quality_features(tmp_path)
        if result is None:
            return JSONResponse(
                {"success": False, "error": "Audio too short (< 0.3s)"},
                status_code=400
            )

        result["success"] = True
        result["model_used"] = "signal_analysis"
        return JSONResponse(result)

    except Exception as e:
        import traceback
        return JSONResponse(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()},
            status_code=500
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
