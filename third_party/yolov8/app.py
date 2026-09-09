"""
YOLOv8 microservice - general detection (people / objects / faces).
Port: 8009
Function: detect people, generic objects and faces in video frames, emitting bboxes, confidences and classes.
Dimensions covered: D1 POV shot, D1 exit-and-entry, D1 empty shot, E1 shot scale, E1 focus.
"""
import os
import tempfile
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="YOLOv8 Multi-Detection Service", version="1.0")

model_detect = None
face_cascade = None
DEVICE = "cuda:0"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights sit next to this service by default, override with MODEL_DIR
MODEL_DIR = os.environ.get("MODEL_DIR", SERVICE_DIR)


def load_models():
    global model_detect, face_cascade
    from ultralytics import YOLO

    if model_detect is None:
        model_path = os.path.join(MODEL_DIR, "yolov8m.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, "yolov8n.pt")
        model_detect = YOLO(model_path)
        model_detect.to(DEVICE)

    if face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)


@app.on_event("startup")
async def startup():
    try:
        load_models()
        print(f"[YOLOv8] Models loaded on {DEVICE}")
    except Exception as e:
        print(f"[YOLOv8] Warning: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "YOLOv8m (detect) + Haar (face)",
        "device": DEVICE,
        "loaded": model_detect is not None
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


def detect_faces(frame_gray, frame_area):
    """Detect faces with a Haar cascade."""
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))
    results = []
    max_area = 0
    for (x, y, w, h) in faces:
        area = w * h
        max_area = max(max_area, area)
        results.append({
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "area_ratio": round(float(area / frame_area), 4)
        })
    return results, max_area


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sample_fps: float = Form(default=2.0),
    max_frames: int = Form(default=16),
    conf_threshold: float = Form(default=0.3)
):
    """
    General object detection.

    Input: a video file
    Parameters:
        - sample_fps: sampling frame rate (default 2fps)
        - max_frames: maximum number of frames to process (default 16)
        - conf_threshold: confidence threshold (default 0.3)
    Output:
        - per_frame: per-frame detections (objects, faces, person_count, ...)
        - summary: aggregated statistics (avg_person_count, face_ratio, body_ratio, ...)
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        load_models()
        frames, frame_times, fps = extract_frames(tmp_path, sample_fps, max_frames)

        if not frames:
            return JSONResponse({"success": False, "error": "No frames extracted"}, status_code=400)

        per_frame_results = []
        all_person_counts = []
        all_face_ratios = []
        all_body_ratios = []

        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            frame_area = h * w

            # 1. YOLO object detection
            det_results = model_detect(frame, conf=conf_threshold, verbose=False)[0]
            objects = []
            person_count = 0
            max_person_area = 0

            for box in det_results.boxes:
                cls_id = int(box.cls[0])
                cls_name = model_detect.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox_area = (x2 - x1) * (y2 - y1)
                obj_entry = {
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(float(x1), 1), round(float(y1), 1),
                             round(float(x2), 1), round(float(y2), 1)],
                    "area_ratio": round(float(bbox_area / frame_area), 4)
                }
                objects.append(obj_entry)
                if cls_name == "person":
                    person_count += 1
                    max_person_area = max(max_person_area, bbox_area)

            # 2. Face detection (Haar cascade)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces, max_face_area = detect_faces(gray, frame_area)

            face_ratio = max_face_area / frame_area if frame_area > 0 else 0
            body_ratio = max_person_area / frame_area if frame_area > 0 else 0

            # Bbox edge distances (for exit/entry detection)
            edge_distances = []
            for obj in objects:
                if obj["class"] == "person":
                    bx1, by1, bx2, by2 = obj["bbox"]
                    center_x = (bx1 + bx2) / 2
                    center_y = (by1 + by2) / 2
                    dist_left = bx1 / w
                    dist_right = (w - bx2) / w
                    dist_top = by1 / h
                    dist_bottom = (h - by2) / h
                    min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)
                    edge_distances.append({
                        "center": [round(center_x / w, 3), round(center_y / h, 3)],
                        "min_edge_dist": round(min_edge_dist, 4),
                        "edge_dists": {
                            "left": round(dist_left, 3),
                            "right": round(dist_right, 3),
                            "top": round(dist_top, 3),
                            "bottom": round(dist_bottom, 3)
                        }
                    })

            per_frame_results.append({
                "frame_time": frame_times[i],
                "objects": objects[:15],
                "faces": faces,
                "person_count": person_count,
                "face_count": len(faces),
                "face_ratio": round(float(face_ratio), 4),
                "body_ratio": round(float(body_ratio), 4),
                "person_edge_distances": edge_distances,
                "frame_size": [w, h]
            })

            all_person_counts.append(person_count)
            all_face_ratios.append(face_ratio)
            all_body_ratios.append(body_ratio)

        # Summary statistics
        summary = {
            "num_frames": len(frames),
            "avg_person_count": round(float(np.mean(all_person_counts)), 2),
            "max_person_count": int(max(all_person_counts)),
            "min_person_count": int(min(all_person_counts)),
            "avg_face_ratio": round(float(np.mean(all_face_ratios)), 4),
            "max_face_ratio": round(float(max(all_face_ratios)), 4),
            "avg_body_ratio": round(float(np.mean(all_body_ratios)), 4),
            "max_body_ratio": round(float(max(all_body_ratios)), 4),
            "has_person": any(c > 0 for c in all_person_counts),
            "is_empty_shot": all(c == 0 for c in all_person_counts),
        }

        # Shot scale estimation based on face/body ratio
        # ECU: face_ratio > 0.4, CU: > 0.15, MS: body > 0.3, LS: body < 0.15
        avg_fr = summary["avg_face_ratio"]
        avg_br = summary["avg_body_ratio"]
        if avg_fr > 0.4:
            estimated_scale = "ECU"
        elif avg_fr > 0.15:
            estimated_scale = "CU"
        elif avg_br > 0.5:
            estimated_scale = "MS"
        elif avg_br > 0.25:
            estimated_scale = "MLS"
        elif avg_br > 0.1:
            estimated_scale = "LS"
        else:
            estimated_scale = "ELS"
        summary["estimated_shot_scale"] = estimated_scale

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
    uvicorn.run(app, host="0.0.0.0", port=8009)
