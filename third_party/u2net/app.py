"""
Saliency detection + sharpness analysis microservice.
Port: 8011
Function: compute saliency maps, subject regions, sharpness distribution and composition of video frames.
Dimensions covered: D1 occlusion (mask area ratio), D1 empty shot (salient area), E1 composition (subject centroid), E1 focus (in/out-of-subject sharpness).
"""
import os
import tempfile
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Saliency & Sharpness Analysis Service", version="1.0")

DEVICE = "cpu"  # this service uses OpenCV, no GPU needed


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "OpenCV Saliency + Laplacian Sharpness",
        "device": DEVICE,
        "loaded": True
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


def compute_saliency(frame):
    """Compute the saliency map - FFT spectral residual plus colour saliency."""
    # Resize for speed
    small = cv2.resize(frame, (256, 256))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Spectral Residual method (pure numpy FFT implementation)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1e-8)
    phase = np.angle(fshift)
    
    # Smooth magnitude spectrum
    avg_magnitude = cv2.blur(magnitude, (3, 3))
    # Spectral residual = difference
    spectral_residual = magnitude - avg_magnitude
    
    # Reconstruct saliency map
    sr_complex = np.exp(spectral_residual + 1j * phase)
    sr_ifft = np.fft.ifftshift(sr_complex)
    sal_sr = np.abs(np.fft.ifft2(sr_ifft)) ** 2
    sal_sr = cv2.GaussianBlur(sal_sr.astype(np.float32), (9, 9), 2.5)
    if sal_sr.max() > 0:
        sal_sr = sal_sr / sal_sr.max()
    
    # Color-based saliency (mean color distance)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_lab = lab.mean(axis=(0, 1))
    color_dist = np.sqrt(np.sum((lab - mean_lab) ** 2, axis=2))
    if color_dist.max() > 0:
        color_dist = color_dist / color_dist.max()
    
    # Combine spectral residual + color distance
    sal_map = 0.5 * sal_sr + 0.5 * color_dist
    
    # Resize back to original frame size
    h, w = frame.shape[:2]
    sal_map = cv2.resize(sal_map, (w, h))
    
    return sal_map


def compute_sharpness_map(frame, block_size=32):
    """Compute the sharpness map (Laplacian variance per block)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows = h // block_size
    cols = w // block_size
    sharpness_map = np.zeros((rows, cols), dtype=np.float32)
    
    for r in range(rows):
        for c in range(cols):
            block = gray[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size]
            lap = cv2.Laplacian(block, cv2.CV_64F)
            sharpness_map[r, c] = lap.var()
    
    return sharpness_map


def analyze_composition(sal_map, frame_shape):
    """Analyse the composition - subject centroid, rule-of-thirds distance, symmetry."""
    h, w = sal_map.shape[:2]
    
    # Find salient region centroid
    threshold = 0.5 * sal_map.max()
    binary = (sal_map > threshold).astype(np.uint8)
    
    # Centroid
    moments = cv2.moments(binary)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx, cy = w / 2, h / 2
    
    # Normalized coordinates
    norm_cx = cx / w
    norm_cy = cy / h
    
    # Rule of thirds distance (distance to nearest intersection point)
    thirds_points = [(1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)]
    min_thirds_dist = min(
        np.sqrt((norm_cx - px)**2 + (norm_cy - py)**2)
        for px, py in thirds_points
    )
    
    # Symmetry (left-right flip SSIM approximation)
    left_half = sal_map[:, :w//2]
    right_half = cv2.flip(sal_map[:, w//2:2*(w//2)], 1)
    if left_half.shape == right_half.shape:
        diff = np.abs(left_half - right_half)
        symmetry_score = 1.0 - float(np.mean(diff))
    else:
        symmetry_score = 0.5
    
    # Salient area ratio
    salient_ratio = float(np.sum(binary)) / (h * w)
    
    return {
        "subject_center": [round(norm_cx, 4), round(norm_cy, 4)],
        "rule_of_thirds_dist": round(float(min_thirds_dist), 4),
        "symmetry_score": round(symmetry_score, 4),
        "salient_area_ratio": round(salient_ratio, 4)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sample_fps: float = Form(default=2.0),
    max_frames: int = Form(default=16)
):
    """
    Saliency + sharpness analysis.

    Output:
        - per_frame: per-frame saliency / sharpness / composition info
        - summary: aggregated statistics
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        frames, frame_times, fps = extract_frames(tmp_path, sample_fps, max_frames)

        if not frames:
            return JSONResponse({"success": False, "error": "No frames extracted"}, status_code=400)

        per_frame_results = []
        all_salient_ratios = []
        all_sharpness_inside = []
        all_sharpness_outside = []

        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            
            # 1. Saliency map
            sal_map = compute_saliency(frame)
            
            # 2. Composition analysis
            composition = analyze_composition(sal_map, frame.shape)
            
            # 3. Sharpness analysis
            sharpness_map = compute_sharpness_map(frame)
            
            # Inside vs outside salient region sharpness
            sal_resized = cv2.resize(sal_map, (sharpness_map.shape[1], sharpness_map.shape[0]))
            sal_binary = (sal_resized > 0.5 * sal_resized.max()).astype(bool)
            
            if sal_binary.any():
                sharpness_inside = float(sharpness_map[sal_binary].mean())
            else:
                sharpness_inside = float(sharpness_map.mean())
            
            if (~sal_binary).any():
                sharpness_outside = float(sharpness_map[~sal_binary].mean())
            else:
                sharpness_outside = sharpness_inside
            
            # DOF score: high when subject is sharp and background is blurry
            if sharpness_outside > 0:
                dof_score = sharpness_inside / (sharpness_inside + sharpness_outside)
            else:
                dof_score = 0.5
            
            # Mask ratio for occlusion detection
            # High salient_ratio + centered = normal; high + edge = possible occlusion
            mask_ratio = composition["salient_area_ratio"]
            
            frame_result = {
                "frame_time": frame_times[i],
                "composition": composition,
                "sharpness_inside": round(sharpness_inside, 2),
                "sharpness_outside": round(sharpness_outside, 2),
                "dof_score": round(dof_score, 4),
                "mask_ratio": round(mask_ratio, 4),
                "overall_sharpness": round(float(sharpness_map.mean()), 2)
            }
            per_frame_results.append(frame_result)
            
            all_salient_ratios.append(mask_ratio)
            all_sharpness_inside.append(sharpness_inside)
            all_sharpness_outside.append(sharpness_outside)

        # Summary
        summary = {
            "num_frames": len(frames),
            "avg_salient_ratio": round(float(np.mean(all_salient_ratios)), 4),
            "max_salient_ratio": round(float(max(all_salient_ratios)), 4),
            "avg_sharpness_inside": round(float(np.mean(all_sharpness_inside)), 2),
            "avg_sharpness_outside": round(float(np.mean(all_sharpness_outside)), 2),
            "avg_dof_score": round(float(np.mean([r["dof_score"] for r in per_frame_results])), 4),
            "avg_subject_center": [
                round(float(np.mean([r["composition"]["subject_center"][0] for r in per_frame_results])), 4),
                round(float(np.mean([r["composition"]["subject_center"][1] for r in per_frame_results])), 4)
            ],
            "avg_rule_of_thirds_dist": round(float(np.mean([r["composition"]["rule_of_thirds_dist"] for r in per_frame_results])), 4),
            "avg_symmetry_score": round(float(np.mean([r["composition"]["symmetry_score"] for r in per_frame_results])), 4),
            # Occlusion indicator: mask_ratio > 0.7 in any frame
            "has_high_occlusion": any(r > 0.7 for r in all_salient_ratios),
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
    uvicorn.run(app, host="0.0.0.0", port=8011)
