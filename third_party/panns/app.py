"""
PANNs microservice - audio embedding / classification.
Port: 8007
Function: extract audio embedding vectors, for audio event classification and similarity computation.
"""
import os
import tempfile
import subprocess
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="PANNs Audio Embedding Service", version="1.0")

model = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(MODEL_ROOT, "panns"))
MODEL_PATH = os.path.join(MODEL_DIR, "Cnn14_mAP=0.431.pth")
SAMPLE_RATE = 32000
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels_indices.csv")

# Load label mapping
LABEL_MAP = {}  # index -> display_name
def load_labels():
    global LABEL_MAP
    import csv
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 3:
                    LABEL_MAP[int(row[0])] = row[2].strip('"').strip()
        print(f"[PANNs] Loaded {len(LABEL_MAP)} labels")
    else:
        raise FileNotFoundError(f"PANNs labels file not found: {LABELS_PATH}")

load_labels()


class Cnn14(torch.nn.Module):
    """PANNs CNN14 model for audio tagging"""
    def __init__(self):
        super().__init__()
        from torchlibrosa.stft import Spectrogram, LogmelFilterBank
        from torchlibrosa.augmentation import SpecAugmentation
        
        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, 
                                                   win_length=1024, window='hann',
                                                   center=True, pad_mode='reflect',
                                                   freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=SAMPLE_RATE, n_fft=1024, 
                                                  n_mels=64, fmin=50, fmax=14000,
                                                  ref=1.0, amin=1e-10, top_db=None,
                                                  freeze_parameters=True)
        
        self.bn0 = torch.nn.BatchNorm2d(64)
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.fc1 = torch.nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = torch.nn.Linear(2048, 527, bias=True)
        
    def forward(self, input):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        embedding = torch.nn.functional.relu(self.fc1(x))
        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))
        return clipwise_output, embedding


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, (3,3), padding=(1,1), bias=False)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, (3,3), padding=(1,1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
    def forward(self, x, pool_size=(2,2), pool_type='avg'):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = torch.nn.functional.max_pool2d(x, pool_size)
        elif pool_type == 'avg':
            x = torch.nn.functional.avg_pool2d(x, pool_size)
        return x


def load_model():
    global model
    if model is not None:
        return model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"PANNs local weights not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_instance = Cnn14()
    model_instance.load_state_dict(checkpoint['model'])
    model_instance.eval().to(DEVICE)
    model = model_instance
    print(f"[PANNs] Model loaded from LOCAL: {MODEL_PATH}")
    return model


def _extract_audio_to_wav(input_path: str, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> str:
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


@app.on_event("startup")
async def startup():
    try:
        load_model()
        print(f"[PANNs] Model loaded on {DEVICE}")
    except Exception as e:
        print(f"[PANNs] Warning: {e}")


@app.get("/health")
async def health():
    loaded = model is not None
    return {"status": "ok", "model": "PANNs-CNN14", "device": DEVICE, "loaded": loaded, "mode": "model" if loaded else "missing", "weights_path": MODEL_PATH, "labels_path": LABELS_PATH}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    segment_duration: float = Form(default=0.0)
):
    """
    Audio embedding extraction and event classification.
    
    Input: a video or audio file
    Parameters:
        - segment_duration: segment length in seconds; 0 means no segmentation (default 0)
    Output:
        - embedding: [float] 2048-d audio embedding vector
        - audio_tags: [{label, confidence}] top-10 audio event tags
        - duration: audio duration
        - segment_embeddings: per-segment embeddings when segmentation is on
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    audio_path = None
    try:
        m = load_model()
        
        # Decode container audio first; torchaudio 0.12 is unreliable on MP4 directly.
        audio_path = _extract_audio_to_wav(tmp_path, sample_rate=SAMPLE_RATE, channels=1)
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        
        wav = wav.squeeze(0)
        duration = len(wav) / SAMPLE_RATE
        
        # Process full audio
        with torch.no_grad():
            input_tensor = wav.unsqueeze(0).to(DEVICE)
            clipwise_output, embedding = m(input_tensor)
        
        embedding_np = embedding[0].cpu().numpy()
        tags_probs = clipwise_output[0].cpu().numpy()
        
        # Get top tags
        top_indices = np.argsort(tags_probs)[::-1][:10]
        audio_tags = [{"index": int(idx), "label": LABEL_MAP.get(int(idx), f"class_{idx}"), "confidence": round(float(tags_probs[idx]), 4)} for idx in top_indices]
        
        # Extract mood/emotion tags (AudioSet indices 276-282)
        MOOD_INDICES = {
            276: "happy", 277: "funny", 278: "sad",
            279: "tender", 280: "exciting", 281: "angry", 282: "scary"
        }
        mood_tags = {name: round(float(tags_probs[idx]), 4) for idx, name in MOOD_INDICES.items()}
        
        # Also extract music presence indicator (index 137=Music, 267=Background music)
        music_presence = {
            "music": round(float(tags_probs[137]), 4),
            "background_music": round(float(tags_probs[267]), 4),
        }
        
        result = {
            "success": True,
            "model_used": "PANNs-CNN14",
            "embedding": embedding_np.tolist(),
            "embedding_dim": len(embedding_np),
            "audio_tags": audio_tags,
            "mood_tags": mood_tags,
            "music_presence": music_presence,
            "duration": round(duration, 3)
        }
        
        # Segment processing
        if segment_duration > 0:
            segment_samples = int(segment_duration * SAMPLE_RATE)
            min_segment_samples = max(1, int(min(segment_duration, 0.25) * SAMPLE_RATE))
            segment_embeddings = []
            segment_tags = []
            segment_times = []
            for start in range(0, len(wav), segment_samples):
                raw_seg = wav[start:start + segment_samples]
                if len(raw_seg) < min_segment_samples:
                    continue
                seg_start_sec = start / SAMPLE_RATE
                seg_end_sec = min((start + len(raw_seg)) / SAMPLE_RATE, duration)
                seg = raw_seg
                # Pad if needed
                if len(seg) < segment_samples:
                    seg = torch.nn.functional.pad(seg, (0, segment_samples - len(seg)))
                with torch.no_grad():
                    seg_output, seg_emb = m(seg.unsqueeze(0).to(DEVICE))
                segment_embeddings.append(seg_emb[0].cpu().numpy().tolist())
                # Get top-5 tags for this segment
                seg_probs = seg_output[0].cpu().numpy()
                seg_top = np.argsort(seg_probs)[::-1][:5]
                seg_tag_list = [{"index": int(idx), "label": LABEL_MAP.get(int(idx), f"class_{idx}"), "confidence": round(float(seg_probs[idx]), 4)} for idx in seg_top]
                segment_tags.append(seg_tag_list)
                segment_times.append({"start": round(float(seg_start_sec), 3), "end": round(float(seg_end_sec), 3)})
            result["segment_embeddings"] = segment_embeddings
            result["segment_tags"] = segment_tags
            result["segment_times"] = segment_times
            result["segment_duration"] = segment_duration
            result["num_segments"] = len(segment_embeddings)
        
        return JSONResponse(result)

    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e), "traceback": traceback.format_exc()}, status_code=500)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)
        os.unlink(tmp_path)


if __name__ == "__main__":
    # SERVICE_PORT allows starting a second instance without changing the default behaviour (e.g. an offset port for the edit agent's self-evaluation)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("SERVICE_PORT", 8007)))
