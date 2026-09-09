"""
MonST3R microservice - 6DoF camera pose + focal length estimation.
Port: 8015
Function: estimate the per-frame camera pose (6DoF: tx,ty,tz,rx,ry,rz) and focal length from a video frame sequence.
      Used by the E1 camera-motion judgement (pan/tilt/dolly/zoom/crane/static, ...).
"""
import os
import sys
import faulthandler
import tempfile
import numpy as np
import torch
import cv2

# Enable faulthandler to debug crashes
faulthandler.enable()

# Make CUDA errors synchronous so they can be caught in Python
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
from scipy.spatial.transform import Rotation

# MonST3R repo path setup
MONST3R_REPO = os.path.join(os.path.dirname(__file__), "monst3r_repo")
sys.path.insert(0, MONST3R_REPO)
sys.path.insert(0, os.path.join(MONST3R_REPO, "croco"))
sys.path.insert(0, os.path.join(MONST3R_REPO, "third_party", "sam2"))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

app = FastAPI(title="MonST3R Camera Pose Service", version="1.0")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # actual GPU controlled by CUDA_VISIBLE_DEVICES
MODEL_NAME = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", os.path.join(MODEL_ROOT, "monst3r"))
model = None
load_error = None


def load_model():
    global model, load_error
    if model is not None:
        return model
    load_error = None
    from dust3r.model import AsymmetricCroCo3DStereo
    local_safetensors = os.path.join(LOCAL_MODEL_DIR, "model.safetensors")
    if not os.path.exists(local_safetensors):
        raise FileNotFoundError(f"MonST3R local weights not found: {local_safetensors}")
    model = AsymmetricCroCo3DStereo.from_pretrained(LOCAL_MODEL_DIR).to(DEVICE)
    print(f"[MonST3R] Model loaded from LOCAL: {LOCAL_MODEL_DIR}")
    model.eval()
    return model


@app.on_event("startup")
async def startup():
    global load_error
    try:
        load_model()
        print(f"[MonST3R] Model loaded on {DEVICE}")
    except Exception as e:
        load_error = str(e)
        print(f"[MonST3R] Warning: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "error",
        "model": "DUSt3R/MonST3R ViT-Large",
        "device": DEVICE,
        "loaded": model is not None,
        "weights_path": LOCAL_MODEL_DIR,
        "error": load_error,
    }


def extract_frames(video_path: str, max_frames: int = 16, target_fps: float = 4.0):
    """Extract frames from the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30.0
    
    # Compute the sampling interval
    frame_interval = max(1, int(fps / target_fps))
    
    frames = []
    frame_indices = []
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_indices.append(idx)
        idx += 1
    cap.release()
    
    return frames, frame_indices, fps, total_frames


def prepare_images_for_dust3r(frames, image_size=512):
    """Convert a frame list into the DUSt3R input format."""
    from dust3r.utils.image import _resize_pil_image
    from PIL import Image
    
    imgs = []
    for i, frame in enumerate(frames):
        pil_img = Image.fromarray(frame)
        # Resize while maintaining aspect ratio
        W, H = pil_img.size
        # Target: short edge = image_size, keep aspect
        if W > H:
            new_H = image_size
            new_W = int(W * image_size / H)
        else:
            new_W = image_size
            new_H = int(H * image_size / W)
        # Make divisible by 16
        new_W = (new_W // 16) * 16
        new_H = (new_H // 16) * 16
        
        pil_img = pil_img.resize((new_W, new_H), Image.LANCZOS)
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        imgs.append({
            'img': img_tensor,
            'true_shape': torch.tensor([[new_H, new_W]], dtype=torch.int32),
            'idx': i,
            'instance': str(i),
        })
    return imgs


def run_pairwise_inference(frames, image_size=512):
    """Run DUSt3R pairwise inference plus global optimisation to obtain the camera poses.
    
    Includes robust error handling so a CUDA error cannot crash the process.
    """
    import gc
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    
    imgs = prepare_images_for_dust3r(frames, image_size)
    
    if len(imgs) < 2:
        return None, None
    
    try:
        # Free GPU memory to make sure there is enough room
        gc.collect()
        torch.cuda.empty_cache()
        
        # Make pairs with sliding window
        pairs = make_pairs(imgs, scene_graph='swin-3-noncyclic', prefilter=None, symmetrize=True)
        
        # Run inference
        output = inference(pairs, model, DEVICE, batch_size=8, verbose=False)
        torch.cuda.synchronize()  # force a sync so CUDA errors surface in Python
        
        # Global alignment
        if len(imgs) > 2:
            mode = GlobalAlignerMode.PointCloudOptimizer
            scene = global_aligner(
                output, device=DEVICE, mode=mode, verbose=False,
                shared_focal=True, temporal_smoothing_weight=0.01,
                translation_weight=1.0, flow_loss_weight=0.0,
                num_total_iter=150, batchify=False
            )
            scene.compute_global_alignment(init='mst', niter=150, schedule='cosine', lr=0.01)
        else:
            mode = GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=DEVICE, mode=mode, verbose=False)
        
        torch.cuda.synchronize()  # sync again to check the optimisation succeeded
        
        # Extract poses and focals
        cams2world = scene.get_im_poses().detach().cpu().numpy()  # (N, 4, 4)
        focals = scene.get_focals().detach().cpu().numpy()  # (N,) or (N,1)
        
        # Check for NaN/Inf
        if np.isnan(cams2world).any() or np.isinf(cams2world).any():
            print("[MonST3R] Warning: NaN/Inf in poses, returning None")
            return None, None
        if np.isnan(focals).any() or np.isinf(focals).any():
            print("[MonST3R] Warning: NaN/Inf in focals, returning None")
            return None, None
        
        return cams2world, focals
    
    except RuntimeError as e:
        # Catch CUDA errors (OOM, CUBLAS, ...)
        err_msg = str(e).lower()
        print(f"[MonST3R] RuntimeError during inference: {e}")
        if "cuda" in err_msg or "cublas" in err_msg or "out of memory" in err_msg:
            torch.cuda.empty_cache()
        return None, None
    except Exception as e:
        print(f"[MonST3R] Unexpected error during inference: {e}")
        return None, None
    finally:
        # Clean up after every inference
        gc.collect()
        torch.cuda.empty_cache()


def analyze_camera_motion(poses, focals, fps, frame_indices):
    """Classify the camera motion from the camera pose sequence."""
    n_frames = len(poses)
    if n_frames < 2:
        return {"motion_type": "static", "confidence": 1.0}
    
    # Extract translations and rotations
    translations = []
    rotations_euler = []
    
    for i in range(n_frames):
        T = poses[i]  # 4x4 camera-to-world matrix
        trans = T[:3, 3]
        rot_mat = T[:3, :3]
        translations.append(trans)
        # Convert to euler angles (in degrees)
        r = Rotation.from_matrix(rot_mat)
        euler = r.as_euler('xyz', degrees=True)
        rotations_euler.append(euler)
    
    translations = np.array(translations)
    rotations_euler = np.array(rotations_euler)
    
    # Compute frame-to-frame changes
    dt = np.diff(translations, axis=0)  # (N-1, 3)
    dr = np.diff(rotations_euler, axis=0)  # (N-1, 3)
    
    # Focal length changes (zoom indicator)
    focal_arr = np.array(focals, dtype=np.float64).flatten()
    if len(focal_arr) > 1:
        focal_change = (focal_arr[-1] - focal_arr[0]) / focal_arr[0]  # relative change
        focal_std = np.std(focal_arr) / np.mean(focal_arr)
    else:
        focal_change = 0.0
        focal_std = 0.0
    
    # Compute motion statistics
    # Translation magnitudes
    trans_total = np.linalg.norm(translations[-1] - translations[0])
    trans_per_frame = np.mean(np.linalg.norm(dt, axis=1))
    
    # Rotation magnitudes per axis
    rot_x_total = rotations_euler[-1, 0] - rotations_euler[0, 0]  # tilt
    rot_y_total = rotations_euler[-1, 1] - rotations_euler[0, 1]  # pan
    rot_z_total = rotations_euler[-1, 2] - rotations_euler[0, 2]  # roll
    
    # Translation per axis (camera local frame approximation)
    trans_x_total = translations[-1, 0] - translations[0, 0]  # track (lateral)
    trans_y_total = translations[-1, 1] - translations[0, 1]  # crane (vertical)
    trans_z_total = translations[-1, 2] - translations[0, 2]  # dolly (forward/back)
    
    # Classify motion type
    motion_types = []
    
    # Thresholds
    ROT_THRESH = 5.0     # degrees
    TRANS_THRESH = 0.1    # relative to scene scale
    FOCAL_THRESH = 0.05   # 5% focal change = zoom
    
    # Normalize translations by scene scale
    scene_scale = max(np.std(translations[:, 0]), np.std(translations[:, 1]), 
                      np.std(translations[:, 2]), 0.001)
    
    norm_trans_x = abs(trans_x_total) / scene_scale
    norm_trans_y = abs(trans_y_total) / scene_scale
    norm_trans_z = abs(trans_z_total) / scene_scale
    
    # Detect zoom (focal length change)
    if abs(focal_change) > FOCAL_THRESH or focal_std > FOCAL_THRESH:
        motion_types.append(("zoom", abs(focal_change) + focal_std))
    
    # Detect pan (rotation around Y)
    if abs(rot_y_total) > ROT_THRESH:
        motion_types.append(("pan", abs(rot_y_total)))
    
    # Detect tilt (rotation around X)
    if abs(rot_x_total) > ROT_THRESH:
        motion_types.append(("tilt", abs(rot_x_total)))
    
    # Detect roll (rotation around Z)
    if abs(rot_z_total) > ROT_THRESH:
        motion_types.append(("roll", abs(rot_z_total)))
    
    # Detect dolly (translation along Z / forward)
    if norm_trans_z > 1.0:
        direction = "dolly_in" if trans_z_total > 0 else "dolly_out"
        motion_types.append((direction, norm_trans_z))
    
    # Detect tracking (lateral translation)
    if norm_trans_x > 1.0:
        motion_types.append(("tracking", norm_trans_x))
    
    # Detect crane (vertical translation)
    if norm_trans_y > 1.0:
        direction = "crane_up" if trans_y_total > 0 else "crane_down"
        motion_types.append((direction, norm_trans_y))
    
    # Sort by magnitude
    motion_types.sort(key=lambda x: x[1], reverse=True)
    
    # Determine primary motion
    if not motion_types:
        primary_motion = "static"
        confidence = 1.0
    else:
        primary_motion = motion_types[0][0]
        # Confidence based on dominance
        total_mag = sum(m[1] for m in motion_types)
        confidence = motion_types[0][1] / total_mag if total_mag > 0 else 1.0
    
    # Build per-frame trajectory
    time_stamps = [idx / fps for idx in frame_indices]
    
    trajectory = []
    for i in range(n_frames):
        trajectory.append({
            "frame_idx": int(frame_indices[i]),
            "time_sec": round(time_stamps[i], 3),
            "position": [round(float(x), 4) for x in translations[i]],
            "rotation_euler_xyz_deg": [round(float(x), 2) for x in rotations_euler[i]],
            "focal_length": round(float(focal_arr[i]) if i < len(focal_arr) else focal_arr[-1], 2),
        })
    
    return {
        "primary_motion": primary_motion,
        "confidence": round(confidence, 3),
        "all_motions": [(m[0], round(m[1], 3)) for m in motion_types],
        "summary": {
            "translation_total": round(float(trans_total), 4),
            "pan_degrees": round(float(rot_y_total), 2),
            "tilt_degrees": round(float(rot_x_total), 2),
            "roll_degrees": round(float(rot_z_total), 2),
            "dolly_z": round(float(trans_z_total), 4),
            "track_x": round(float(trans_x_total), 4),
            "crane_y": round(float(trans_y_total), 4),
            "focal_change_ratio": round(float(focal_change), 4),
            "focal_std_ratio": round(float(focal_std), 4),
        },
        "trajectory": trajectory,
    }


@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    max_frames: int = Form(default=16),
    target_fps: float = Form(default=4.0),
    image_size: int = Form(default=512),
):
    """
    Analyse the camera motion of a video.
    Returns the 6DoF trajectory, the focal length changes and the camera-motion class.
    
    Inference runs in a thread pool so the event loop is never blocked (keeping /health responsive).
    """
    import asyncio
    import concurrent.futures
    
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    
    # Save uploaded video
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    def _do_inference(tmp_path, max_frames, target_fps, image_size):
        """CPU/GPU heavy inference, executed in a thread pool."""
        frames, frame_indices, fps, total_frames = extract_frames(
            tmp_path, max_frames=max_frames, target_fps=target_fps
        )
        
        if len(frames) < 2:
            return {"error": "Not enough frames extracted", "n_frames": len(frames)}
        
        poses, focals = run_pairwise_inference(frames, image_size=image_size)
        
        if poses is None:
            return {"error": "Inference failed"}
        
        result = analyze_camera_motion(poses, focals, fps, frame_indices)
        result["video_info"] = {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "analyzed_frames": len(frames),
            "duration_sec": round(total_frames / fps, 2) if fps > 0 else 0,
        }
        
        import json
        result = json.loads(json.dumps(result, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))
        return result
    
    try:
        # Run the inference in a thread pool without blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # use the default thread pool
            _do_inference, tmp_path, max_frames, target_fps, image_size
        )
        
        if "error" in result and len(result) <= 2:
            return JSONResponse(content=result)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8015)
