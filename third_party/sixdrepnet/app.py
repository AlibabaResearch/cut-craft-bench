"""
6DRepNet head pose estimation microservice.
Port: 8014
Function: estimate the head pose (yaw/pitch/roll) of faces in video frames.
Dimensions covered: D1 POV shot (head yaw/pitch), E1 camera angle (camera pitch estimate).
"""
import os
import tempfile
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="6DRepNet Head Pose Estimation Service", version="1.0")

model = None
load_error = None
DEVICE = "cuda:0"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
MODEL_WEIGHT_PATH = os.environ.get(
    "MODEL_WEIGHT_PATH", os.path.join(MODEL_ROOT, "6DRepNet_300W_LP_AFLW2000.pth")
)


def load_model():
    global model, load_error
    if model is None:
        load_error = None
        if not os.path.exists(MODEL_WEIGHT_PATH):
            raise FileNotFoundError(f"6DRepNet local weights not found: {MODEL_WEIGHT_PATH}")
        from sixdrepnet import SixDRepNet
        model = SixDRepNet(gpu_id=0, dict_path=MODEL_WEIGHT_PATH)
        print(f"[6DRepNet] Model loaded on GPU 0 from {MODEL_WEIGHT_PATH}")


@app.on_event("startup")
async def startup():
    global load_error
    try:
        load_model()
    except Exception as e:
        load_error = str(e)
        print(f"[6DRepNet] Warning: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "error",
        "model": "6DRepNet (300W_LP + AFLW2000)",
        "device": DEVICE,
        "loaded": model is not None,
        "weights_path": MODEL_WEIGHT_PATH,
        "error": load_error,
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


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sample_fps: float = Form(default=2.0),
    max_frames: int = Form(default=16)
):
    """
    Head pose estimation.

    Output:
        - per_frame: head poses detected in each frame (yaw/pitch/roll)
        - summary: aggregated statistics (mean angles, angle ranges, ...)
    Notes:
        - yaw: horizontal rotation (left negative, right positive, -90~+90)
        - pitch: pitch angle (looking down positive, looking up negative, -90~+90)
        - roll: roll angle (counter-clockwise positive, -90~+90)
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
        all_yaws = []
        all_pitches = []
        all_rolls = []

        for i, frame in enumerate(frames):
            # 6DRepNet predicts head pose for each detected face
            # It internally detects faces and returns poses
            pitch, yaw, roll = model.predict(frame)
            
            faces_in_frame = []
            if len(pitch) > 0:
                for j in range(len(pitch)):
                    p = float(pitch[j])
                    y = float(yaw[j])
                    r = float(roll[j])
                    faces_in_frame.append({
                        "yaw": round(y, 2),
                        "pitch": round(p, 2),
                        "roll": round(r, 2)
                    })
                    all_yaws.append(y)
                    all_pitches.append(p)
                    all_rolls.append(r)

            per_frame_results.append({
                "frame_time": frame_times[i],
                "num_faces": len(faces_in_frame),
                "faces": faces_in_frame,
                # Gaze direction estimation from head pose
                "gaze_direction": _estimate_gaze(faces_in_frame) if faces_in_frame else None
            })

        # Summary
        summary = {
            "num_frames": len(frames),
            "total_faces_detected": len(all_yaws),
            "frames_with_faces": sum(1 for r in per_frame_results if r["num_faces"] > 0),
        }

        if all_yaws:
            summary.update({
                "avg_yaw": round(float(np.mean(all_yaws)), 2),
                "avg_pitch": round(float(np.mean(all_pitches)), 2),
                "avg_roll": round(float(np.mean(all_rolls)), 2),
                "yaw_range": [round(float(min(all_yaws)), 2), round(float(max(all_yaws)), 2)],
                "pitch_range": [round(float(min(all_pitches)), 2), round(float(max(all_pitches)), 2)],
                # Camera angle estimation based on pitch
                "estimated_camera_angle": _estimate_camera_angle(np.mean(all_pitches)),
                # Gaze alignment (for POV detection)
                "gaze_consistency": round(float(1.0 - np.std(all_yaws) / 90.0), 4),
            })
        else:
            summary.update({
                "avg_yaw": None, "avg_pitch": None, "avg_roll": None,
                "estimated_camera_angle": "unknown",
                "gaze_consistency": 0.0,
            })

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


def _estimate_gaze(faces):
    """Estimate gaze direction from head poses"""
    if not faces:
        return None
    # Use the primary face (first detected)
    face = faces[0]
    yaw = face["yaw"]
    pitch = face["pitch"]
    
    # Determine horizontal direction
    if yaw < -30:
        h_dir = "looking_left"
    elif yaw > 30:
        h_dir = "looking_right"
    else:
        h_dir = "looking_forward"
    
    # Determine vertical direction
    if pitch > 20:
        v_dir = "looking_down"
    elif pitch < -20:
        v_dir = "looking_up"
    else:
        v_dir = "looking_level"
    
    return {
        "horizontal": h_dir,
        "vertical": v_dir,
        "yaw_deg": yaw,
        "pitch_deg": pitch
    }


def _estimate_camera_angle(avg_pitch):
    """Estimate camera angle from average face pitch"""
    if avg_pitch > 15:
        return "high_angle"  # Camera looking down at subject
    elif avg_pitch < -15:
        return "low_angle"   # Camera looking up at subject
    else:
        return "eye_level"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8014)
