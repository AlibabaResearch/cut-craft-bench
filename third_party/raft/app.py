"""
RAFT microservice - optical flow estimation (using torchvision's built-in RAFT).
Port: 8002
Function: compute optical flow between video frames, emitting the motion field (magnitude/direction).
"""
import os
import tempfile
import numpy as np
import torch
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.transforms.functional import to_tensor, resize

app = FastAPI(title="RAFT Optical Flow Service", version="1.0")

model = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
WEIGHTS_PATH = os.environ.get(
    "WEIGHTS_PATH", os.path.join(MODEL_ROOT, "raft", "raft_small_C_T_V2-01064c6d.pth")
)


def load_model():
    global model
    if model is not None:
        return model
    
    # Prefer loading the weights from the local directory
    local_weight = WEIGHTS_PATH
    if not os.path.exists(local_weight):
        raise FileNotFoundError(f"RAFT local weights not found: {local_weight}")
    model_instance = raft_small(weights=None, progress=False).to(DEVICE)
    state_dict = torch.load(local_weight, map_location=DEVICE)
    model_instance.load_state_dict(state_dict)
    model = model_instance
    print(f"[RAFT] Model loaded from LOCAL: {local_weight}")
    model.eval()
    return model


@app.on_event("startup")
async def startup():
    try:
        load_model()
        print(f"[RAFT] Model loaded on {DEVICE}")
    except Exception as e:
        print(f"[RAFT] Warning: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "RAFT-Small (torchvision)", "device": DEVICE, "loaded": model is not None, "weights_path": WEIGHTS_PATH}


def preprocess_frame(frame_rgb, target_size=(520, 960)):
    """Preprocess a frame into the RAFT input format."""
    h, w = frame_rgb.shape[:2]
    # Resize maintaining aspect ratio, pad to divisible by 8
    scale = min(target_size[0] / h, target_size[1] / w, 1.0)
    new_h, new_w = int(h * scale), int(w * scale)
    # Make dimensions divisible by 8
    new_h = (new_h // 8) * 8
    new_w = (new_w // 8) * 8
    if new_h == 0:
        new_h = 8
    if new_w == 0:
        new_w = 8
    
    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
    tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()  # [C, H, W]
    return tensor


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sample_fps: float = Form(default=5.0),
    max_frames: int = Form(default=60)
):
    """
    Compute the optical flow information of a video.
    
    Input: a video file
    Parameters:
        - sample_fps: sampling frame rate (default 5fps)
        - max_frames: maximum number of frames to process (default 60)
    Output:
        - flow_magnitudes: mean flow magnitude between consecutive frames (pixels/frame)
        - flow_directions: dominant flow direction between consecutive frames (degrees)
        - mean_magnitude: mean motion strength
        - motion_energy: motion energy over time
        - camera_motion_estimate: estimated camera motion class (static/slow/moderate/fast)
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
        frame_idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_idx += 1
        cap.release()

        if len(frames) < 2:
            return JSONResponse({"success": False, "error": "Not enough frames"}, status_code=400)

        m = load_model()
        
        magnitudes = []
        directions = []
        motion_energies = []

        for i in range(len(frames) - 1):
            img1 = preprocess_frame(frames[i]).unsqueeze(0).to(DEVICE)
            img2 = preprocess_frame(frames[i + 1]).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                flow_predictions = m(img1, img2)
                flow = flow_predictions[-1][0]  # Last iteration, first batch
            
            flow_np = flow.cpu().numpy()  # [2, H, W]
            flow_x, flow_y = flow_np[0], flow_np[1]
            
            mag = np.sqrt(flow_x**2 + flow_y**2)
            angle = np.arctan2(flow_y, flow_x) * 180 / np.pi
            
            mean_mag = float(np.mean(mag))
            mean_dir = float(np.mean(angle))
            energy = float(np.sum(mag**2)) / mag.size
            
            magnitudes.append(round(mean_mag, 4))
            directions.append(round(mean_dir, 2))
            motion_energies.append(round(energy, 4))

        avg_mag = float(np.mean(magnitudes))
        mag_std = float(np.std(magnitudes))
        
        if avg_mag < 0.5:
            cam_motion = "static"
        elif avg_mag < 2.0:
            cam_motion = "slow"
        elif avg_mag > 8.0:
            cam_motion = "fast"
        else:
            cam_motion = "moderate"

        return JSONResponse({
            "success": True,
            "num_frame_pairs": len(magnitudes),
            "fps": round(fps, 2),
            "sample_fps": sample_fps,
            "flow_magnitudes": magnitudes,
            "flow_directions": directions,
            "motion_energy": motion_energies,
            "mean_magnitude": round(avg_mag, 4),
            "std_magnitude": round(mag_std, 4),
            "max_magnitude": round(float(max(magnitudes)), 4),
            "camera_motion_estimate": cam_motion
        })

    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e), "traceback": traceback.format_exc()}, status_code=500)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
