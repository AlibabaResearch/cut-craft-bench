"""
TransNetV2 microservice - shot boundary detection (PyTorch version).
Port: 8001
Function: detect the shot cuts of a video, returning the cut timestamps and the transition type.
Five transition classes: hard_cut / dissolve / wipe / flash_white / flash_black
"""
import os
import sys
import tempfile
import numpy as np
import torch

def _disable_torch_determinism():
    """Ensure TransNetV2 inference can use CUDA ops that lack deterministic kernels."""
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    try:
        torch.set_deterministic_debug_mode("default")
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

_disable_torch_determinism()
import cv2
import open_clip
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="TransNetV2 Shot Boundary Detection Service", version="2.0")

# Global models
model = None
clip_model = None
clip_preprocess = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights root: defaults to third_party/models, override with MODEL_ROOT
MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.join(os.path.dirname(SERVICE_DIR), "models"))
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(MODEL_ROOT, "transnetv2", "TransNetV2", "inference-pytorch")
)
WEIGHTS_PATH = os.path.join(MODEL_DIR, "transnetv2-pytorch-weights.pth")
CLIP_WEIGHTS_PATH = os.environ.get(
    "CLIP_WEIGHTS_PATH", os.path.join(MODEL_ROOT, "clip", "ViT-B-32.pt")
)

sys.path.insert(0, MODEL_DIR)


def load_clip_model():
    """Load OpenCLIP ViT-B/32 for semantic scene change detection."""
    global clip_model, clip_preprocess
    local_clip_path = CLIP_WEIGHTS_PATH
    if not os.path.exists(local_clip_path):
        raise FileNotFoundError(f"OpenCLIP ViT-B-32 local weights not found: {local_clip_path}")
    from open_clip.openai import load_openai_model
    from open_clip.transform import image_transform
    clip_model = load_openai_model(local_clip_path, precision="fp32", device=DEVICE)
    clip_preprocess = image_transform(
        clip_model.visual.image_size,
        is_train=False,
        mean=clip_model.visual.image_mean,
        std=clip_model.visual.image_std,
    )
    clip_model.eval().to(DEVICE)
    print(f"[OpenCLIP] ViT-B-32 loaded from LOCAL: {local_clip_path} on {DEVICE}")


def load_model():
    global model
    if model is not None:
        return model
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"TransNetV2 local weights not found: {WEIGHTS_PATH}")
    from transnetv2_pytorch import TransNetV2
    model = TransNetV2()
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval().to(DEVICE)
    print(f"[TransNetV2] Model loaded from LOCAL: {WEIGHTS_PATH}")
    return model


def extract_frames(video_path, frame_size=(48, 27)):
    """Extract video frames and resize them to the TransNetV2 input size (27x48)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.uint8), fps


def extract_frames_range(video_path, start_frame, end_frame, analysis_size=(320, 180)):
    """Extract the frames of a given range (for transition type analysis, at a higher resolution).
    
    Args:
        video_path: path to the video file
        start_frame: first frame (inclusive)
        end_frame: last frame (inclusive)
        analysis_size: analysis resolution (width, height)
    
    Returns:
        frames: list of numpy arrays (BGR)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, analysis_size)
        frames.append(frame)
    cap.release()
    return frames


def classify_transition_type(video_path, start_frame, end_frame, fps):
    """
    Classify a transition into one of five types:
      - hard_cut:    an instant cut with no optical effect
      - dissolve:    uniform alpha blending across the whole frame
      - wipe:        a spatially moving boundary
      - flash_white: a transition through high-brightness frames
      - flash_black: a transition through low-brightness frames
    
    Algorithm:
      1. widen the analysis window (half a second on each side) and detect flash white/black
      2. measure how concentrated the frame differences are -> hard cut vs gradual
      3. for a gradual transition, analyse the spatial uniformity -> dissolve vs wipe
    """
    span = end_frame - start_frame
    
    # Analysis window: half_window frames on each side (about 0.5s)
    half_window = max(int(fps * 0.5), 8)
    center_frame = (start_frame + end_frame) // 2
    ctx_start = max(0, center_frame - half_window)
    ctx_end = center_frame + half_window
    
    frames = extract_frames_range(video_path, ctx_start, ctx_end)
    if len(frames) < 5:
        return "hard_cut" if span <= 2 else "dissolve"
    
    # ============================================================
    # Step 1: detect flash white/black (brightness extremes)
    # First confirm a real scene change around the flash, rather than a lighting change inside one scene
    # ============================================================
    brightness_values = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))
    
    max_brightness = max(brightness_values)
    min_brightness = min(brightness_values)
    
    # Flash white/black detection: an absolute brightness threshold plus a relative spike condition
    # The window edge frames give the baseline, to avoid false positives in bright scenes
    edge_brightness = np.mean(brightness_values[:3] + brightness_values[-3:])
    brightness_spike = max_brightness - edge_brightness
    brightness_dip = edge_brightness - min_brightness
    
    # Flash white/black candidate detection
    flash_candidate = None
    if (max_brightness > 200 and brightness_spike > 50) or \
       (max_brightness > 140 and brightness_spike > 80):
        flash_candidate = "flash_white"
    elif (min_brightness < 40 and brightness_dip > 50) or \
         (min_brightness < 60 and brightness_dip > 80):
        flash_candidate = "flash_black"
    
    # When a flash candidate is found, verify there really is a scene change around it
    # rather than flickering lights or an explosion inside the same scene
    if flash_candidate is not None:
        # Take the stable frames before and after the flash for the scene comparison
        # The first 3 and the last 3 frames represent the two scenes
        n_edge = min(3, len(frames) // 4)
        before_frames = frames[:n_edge]
        after_frames = frames[-n_edge:]
        
        # Mean difference between the two scenes
        g_before = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in before_frames], axis=0)
        g_after = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in after_frames], axis=0)
        scene_diff = float(np.mean(np.abs(g_after - g_before)))
        
        # Only a large enough scene difference confirms a flash transition
        # Otherwise it is just a lighting change inside one scene and is not a transition
        if scene_diff > 20:
            return flash_candidate
        # The scene did not change: fall through to the hard cut / gradual check
    
    # ============================================================
    # Step 2: frame-difference concentration -> hard cut vs gradual
    # ============================================================
    # Per-frame difference
    frame_diffs = []
    for i in range(1, len(frames)):
        g_prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        frame_diffs.append(float(np.mean(np.abs(g_curr - g_prev))))
    
    total_diff = sum(frame_diffs)
    if total_diff < 1.0:
        return "hard_cut"
    
    # Concentration: the largest single-frame difference over the total difference
    # Hard cut: most of the change sits in 1-2 frames -> high concentration
    # Dissolve: the change spreads over many frames -> low concentration
    sorted_diffs = sorted(frame_diffs, reverse=True)
    top2_sum = sum(sorted_diffs[:2])
    concentration = top2_sum / total_diff
    
    # Peak ratio: the largest frame difference over the median one
    # Hard cut: a very high spike (ratio >> 1)
    # Dissolve: similar differences across frames (a lower ratio)
    median_diff = float(np.median(frame_diffs))
    peak_ratio = sorted_diffs[0] / median_diff if median_diff > 0.5 else 0
    
    # Hard cut decision:
    #   condition 1: concentration > 0.28 (the change sits in a few frames)
    #   condition 2: peak_ratio > 8 (a single-frame jump far above the normal inter-frame change)
    # Either one is enough for a hard cut
    if concentration > 0.28 or peak_ratio > 8.0:
        return "hard_cut"
    
    # ============================================================
    # Step 3: dissolve vs wipe (spatial uniformity analysis)
    # ============================================================
    frame_before = frames[0]
    frame_after = frames[-1]
    mid_idx = len(frames) // 2
    frame_mid = frames[mid_idx]
    
    GRID_ROWS, GRID_COLS = 4, 4
    h, w = frame_before.shape[:2]
    cell_h, cell_w = h // GRID_ROWS, w // GRID_COLS
    
    gray_before = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_after = cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_mid = cv2.cvtColor(frame_mid, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    cell_progress = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            cell_before = gray_before[y1:y2, x1:x2]
            cell_after = gray_after[y1:y2, x1:x2]
            cell_mid = gray_mid[y1:y2, x1:x2]
            
            diff_total = np.mean(np.abs(cell_after - cell_before))
            diff_progress = np.mean(np.abs(cell_mid - cell_before))
            
            if diff_total > 5.0:
                progress = diff_progress / (diff_total + 1e-6)
                cell_progress.append(np.clip(progress, 0.0, 1.0))
    
    if len(cell_progress) < 4:
        return "dissolve"
    
    progress_std = float(np.std(cell_progress))
    
    if progress_std > 0.20:
        return "wipe"
    else:
        return "dissolve"


def detect_flash_transitions(video_path, fps, existing_transitions, total_frames):
    """
    Supplementary flash white/black transition detection (independent of TransNetV2).
    
    TransNetV2 is weak at flash white/black transitions, so this function scans the video
    brightness curve directly to catch the flash transitions it misses.
    
    Detection conditions:
      1. the brightness is far above/below the baseline (a spike)
      2. the content before and after the spike differs clearly (confirming a scene change)
      3. it does not overlap an already detected transition
    
    Returns:
        list of dict: the supplementary transitions found
    """
    # Read the brightness of every frame (at a lower resolution for speed)
    cap = cv2.VideoCapture(video_path)
    brightness_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        brightness_list.append(float(np.mean(gray)))
    cap.release()
    
    if len(brightness_list) < 30:
        return []
    
    brightness = np.array(brightness_list)
    mean_b = float(np.mean(brightness))
    std_b = float(np.std(brightness))
    
    # Flash white spike threshold: brightness > max(mean + 2.5*std, 160) and > mean * 1.6
    flash_white_thresh = max(mean_b + 2.5 * std_b, 160, mean_b * 1.6)
    # Flash black spike threshold: brightness < min(mean - 2.5*std, 40)
    flash_black_thresh = min(mean_b - 2.5 * std_b, 40)
    
    # Frame ranges of the known transitions (to avoid duplicates)
    existing_ranges = set()
    for t in existing_transitions:
        for f in range(max(0, t["start_frame"] - 15), min(total_frames, t["end_frame"] + 15)):
            existing_ranges.add(f)
    
    supplementary = []
    
    # Detect flash white
    flash_white_frames = np.where(brightness > flash_white_thresh)[0]
    if len(flash_white_frames) > 0:
        groups = np.split(flash_white_frames, np.where(np.diff(flash_white_frames) > 5)[0] + 1)
        for group in groups:
            if len(group) < 2:
                continue
            peak_frame = int(group[np.argmax(brightness[group])])
            # Skip the already detected transitions
            if peak_frame in existing_ranges:
                continue
            # Skip the frames near the very start / end
            if peak_frame < 10 or peak_frame > total_frames - 10:
                continue
            # Verify the scene change: compare the content before and after the flash
            before_frame = max(0, int(group[0]) - 5)
            after_frame = min(total_frames - 1, int(group[-1]) + 5)
            frames_check = extract_frames_range(video_path, before_frame, after_frame, (160, 90))
            if len(frames_check) >= 3:
                g_before = cv2.cvtColor(frames_check[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
                g_after = cv2.cvtColor(frames_check[-1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                scene_diff = float(np.mean(np.abs(g_after - g_before)))
                # Only a large enough scene difference confirms a transition
                if scene_diff > 15:
                    supplementary.append({
                        "start_frame": int(group[0]),
                        "end_frame": int(group[-1]),
                        "start_time": round(float(group[0]) / fps, 3),
                        "end_time": round(float(group[-1]) / fps, 3),
                        "type": "flash_white",
                        "confidence": round(float(brightness[peak_frame]) / 255.0, 4)
                    })
    
    # Detect flash black
    if flash_black_thresh > 5:
        flash_black_frames = np.where(brightness < flash_black_thresh)[0]
        if len(flash_black_frames) > 0:
            groups = np.split(flash_black_frames, np.where(np.diff(flash_black_frames) > 5)[0] + 1)
            for group in groups:
                if len(group) < 2:
                    continue
                peak_frame = int(group[np.argmin(brightness[group])])
                if peak_frame in existing_ranges:
                    continue
                if peak_frame < 10 or peak_frame > total_frames - 10:
                    continue
                before_frame = max(0, int(group[0]) - 5)
                after_frame = min(total_frames - 1, int(group[-1]) + 5)
                frames_check = extract_frames_range(video_path, before_frame, after_frame, (160, 90))
                if len(frames_check) >= 3:
                    g_before = cv2.cvtColor(frames_check[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    g_after = cv2.cvtColor(frames_check[-1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    scene_diff = float(np.mean(np.abs(g_after - g_before)))
                    if scene_diff > 15:
                        supplementary.append({
                            "start_frame": int(group[0]),
                            "end_frame": int(group[-1]),
                            "start_time": round(float(group[0]) / fps, 3),
                            "end_time": round(float(group[-1]) / fps, 3),
                            "type": "flash_black",
                            "confidence": round(1.0 - float(brightness[peak_frame]) / 255.0, 4)
                        })
    
    return supplementary


def merge_adjacent_flash_transitions(transitions, fps):
    """
    Merge adjacent flash transitions of the same type.
    
    During a flash white/black transition TransNetV2 may drop below the threshold on the dark/bright
    middle frames and split one complete flash into two. This function merges flash_white/flash_black
    transitions of the same type that are less than 1 second apart.
    """
    if len(transitions) < 2:
        return transitions
    
    flash_types = {"flash_white", "flash_black"}
    max_gap_frames = int(fps * 1.0)  # a gap within 1 second counts as the same transition
    
    merged = [transitions[0]]
    for t in transitions[1:]:
        prev = merged[-1]
        # Same flash type and closer than the threshold -> merge
        if (prev["type"] in flash_types and 
            t["type"] == prev["type"] and
            t["start_frame"] - prev["end_frame"] <= max_gap_frames):
            # Extend the range of the previous transition
            merged[-1] = {
                "start_frame": prev["start_frame"],
                "end_frame": t["end_frame"],
                "start_time": prev["start_time"],
                "end_time": t["end_time"],
                "type": prev["type"],
                "confidence": round(max(prev["confidence"], t["confidence"]), 4)
            }
        else:
            merged.append(t)
    
    return merged


def merge_dense_cuts(transitions, fps, video_path):
    """
    Merge dense fragment cuts: several transition points within a short span that produce an extremely
    short micro shot (<0.5s) usually come from brightness swings inside one scene (flickering lights, explosions).
    
    Trigger conditions (all must hold):
    1. adjacent transitions less than 1.5s apart, 3+ in a row
    2. at least one micro shot < 0.5s inside the cluster (so normal fast cutting is excluded)
    
    Handling:
    - when the scene changes across the cluster, keep only the highest-confidence transition
    - when the scene does not change, drop them all
    """
    if len(transitions) < 3:
        return transitions
    
    cluster_gap_sec = 1.5   # a gap below this counts as the same cluster
    min_cluster_size = 3    # at least 3 transition points to count as a dense cluster
    micro_shot_thresh = 0.5 # micro shot threshold (seconds)
    
    # Step 1: group the transition points into clusters
    clusters = []
    current_cluster = [0]
    for i in range(1, len(transitions)):
        per_gap = transitions[i]["start_time"] - transitions[i-1]["end_time"]
        if per_gap < cluster_gap_sec:
            current_cluster.append(i)
        else:
            clusters.append(current_cluster)
            current_cluster = [i]
    clusters.append(current_cluster)
    
    # Step 2: handle the dense clusters
    keep_indices = set(range(len(transitions)))
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        
        # Check for a micro shot inside the cluster (a gap between adjacent transitions < 0.5s)
        has_micro_shot = False
        for k in range(len(cluster) - 1):
            idx_a = cluster[k]
            idx_b = cluster[k + 1]
            gap_between = transitions[idx_b]["start_time"] - transitions[idx_a]["end_time"]
            if gap_between < micro_shot_thresh:
                has_micro_shot = True
                break
        
        if not has_micro_shot:
            continue  # no micro shot, so this is normal fast cutting: keep it
        
        # Check whether the scene really changed across the cluster
        first_t = transitions[cluster[0]]
        last_t = transitions[cluster[-1]]
        
        before_frame = max(0, first_t["start_frame"] - int(fps * 0.3))
        after_frame = min(int(fps * 15), last_t["end_frame"] + int(fps * 0.3))
        
        frames_check = extract_frames_range(video_path, before_frame, after_frame, (160, 90))
        if len(frames_check) >= 3:
            g_before = cv2.cvtColor(frames_check[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
            g_after = cv2.cvtColor(frames_check[-1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            scene_diff = float(np.mean(np.abs(g_after - g_before)))
        else:
            scene_diff = 0
        
        if scene_diff > 20:
            # The scene did change: keep the highest-confidence transition
            best_idx = max(cluster, key=lambda i: transitions[i]["confidence"])
            for idx in cluster:
                if idx != best_idx:
                    keep_indices.discard(idx)
        else:
            # The scene did not change: drop them all
            for idx in cluster:
                keep_indices.discard(idx)
    
    return [transitions[i] for i in sorted(keep_indices)]


def predictions_to_scenes(predictions, threshold=0.5):
    """Convert the predictions into a scene list."""
    predictions = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if len(scenes) == 0:
        scenes.append([0, len(predictions) - 1])
    return np.array(scenes)


def rebuild_shots_from_transitions(transitions, fps, total_frames):
    """
    Rebuild the shots from the current transition list.
    Used after merging dense fragment cuts, when fewer transitions require a fresh shot list.
    """
    if not transitions:
        return [{
            "shot_index": 0,
            "start_frame": 0,
            "end_frame": total_frames - 1,
            "start_time": 0.0,
            "end_time": round(float(total_frames - 1) / fps, 3),
            "duration": round(float(total_frames - 1) / fps, 3)
        }]
    
    shots = []
    # First shot: from the start of the video to the first transition
    first_end = transitions[0]["start_frame"] - 1
    if first_end >= 0:
        shots.append({
            "shot_index": 0,
            "start_frame": 0,
            "end_frame": first_end,
            "start_time": 0.0,
            "end_time": round(float(first_end) / fps, 3),
            "duration": round(float(first_end) / fps, 3)
        })
    
    # Middle shots: between adjacent transitions
    for i in range(len(transitions) - 1):
        s_start = transitions[i]["end_frame"] + 1
        s_end = transitions[i + 1]["start_frame"] - 1
        if s_end >= s_start:
            shots.append({
                "shot_index": len(shots),
                "start_frame": s_start,
                "end_frame": s_end,
                "start_time": round(float(s_start) / fps, 3),
                "end_time": round(float(s_end) / fps, 3),
                "duration": round(float(s_end - s_start) / fps, 3)
            })
    
    # Last shot: from the last transition to the end of the video
    last_start = transitions[-1]["end_frame"] + 1
    if last_start < total_frames:
        shots.append({
            "shot_index": len(shots),
            "start_frame": last_start,
            "end_frame": total_frames - 1,
            "start_time": round(float(last_start) / fps, 3),
            "end_time": round(float(total_frames - 1) / fps, 3),
            "duration": round(float(total_frames - 1 - last_start) / fps, 3)
        })
    
    return shots


def rebuild_shots_with_supplements(shots, supplements, fps, total_frames):
    """
    Insert the supplementary transitions into the shot list and re-split the shots.
    
    When detect_flash_transitions finds a new transition, the long shot that contains it
    has to be split into two.
    
    Args:
        shots: the original shot list
        supplements: the supplementary transitions found
        fps: frame rate
        total_frames: total number of frames
    
    Returns:
        the updated shot list
    """
    if not supplements:
        return shots
    
    # Collect every new split point (a transition's end_frame+1 starts the new shot)
    new_boundaries = []
    for t in supplements:
        split_frame = t["end_frame"] + 1
        if split_frame < total_frames:
            new_boundaries.append(split_frame)
    
    if not new_boundaries:
        return shots
    
    new_boundaries.sort()
    
    # Rebuild the shots: for each existing shot, check whether a new boundary falls inside it
    new_shots = []
    for shot in shots:
        s_start = shot["start_frame"]
        s_end = shot["end_frame"]
        
        # Find every new boundary inside this shot's range
        splits_in_shot = [b for b in new_boundaries if s_start < b <= s_end]
        
        if not splits_in_shot:
            # No new split point, keep it as is
            new_shots.append(shot)
        else:
            # A new split point: cut this shot into several segments
            boundaries = [s_start] + splits_in_shot + [s_end + 1]
            for i in range(len(boundaries) - 1):
                seg_start = boundaries[i]
                seg_end = boundaries[i + 1] - 1
                if seg_end >= seg_start:
                    new_shots.append({
                        "start_frame": seg_start,
                        "end_frame": seg_end,
                        "start_time": round(float(seg_start) / fps, 3),
                        "end_time": round(float(seg_end) / fps, 3),
                        "duration": round(float(seg_end - seg_start) / fps, 3)
                    })
    
    # Renumber shot_index
    for i, shot in enumerate(new_shots):
        shot["shot_index"] = i
    
    return new_shots


def detect_long_shot_splits(shots, transitions, predictions, fps, video_path,
                           total_frames, length_ratio=2.5, low_threshold=0.12):
    """
    Second-pass detection on long shots (plan A).
    
    When a shot lasts longer than length_ratio * the median shot duration, its predictions are
    re-examined with a lower threshold, to find a slow transition that was missed.
    
    Args:
        shots: the current shot list
        transitions: the current transition list
        predictions: the model's raw prediction scores (after sigmoid)
        fps: frame rate
        video_path: path to the video file (for transition type classification)
        total_frames: total number of frames
        length_ratio: abnormal-length multiplier threshold (times the median)
        low_threshold: the low threshold used by the second pass
    
    Returns:
        (updated_shots, updated_transitions)
    """
    if len(shots) == 0:
        return shots, transitions
    
    # === Pick the second-pass trigger strategy from the shot count ===
    durations = [s["duration"] for s in shots]
    n_shots = len(shots)
    shots_to_split = []
    
    if n_shots == 1:
        # 1 shot: lower the threshold over the whole video (a slow transition may have been missed entirely)
        shots_to_split = list(shots)
    elif n_shots == 2:
        # 2 shots: when one shot is at least twice as long as the other, re-examine the longer one
        long_dur, short_dur = max(durations), min(durations)
        if short_dur > 0 and long_dur >= 2.0 * short_dur:
            long_idx = durations.index(long_dur)
            shots_to_split = [shots[long_idx]]
        else:
            return shots, transitions
    else:
        # 3+ shots: keep the median-based logic
        median_duration = float(np.median(durations))
        threshold_duration = median_duration * length_ratio
        # A minimum trigger duration, to avoid firing on videos where every shot is short
        min_trigger_duration = 3.0  # at least 3 seconds before a second pass fires
        threshold_duration = max(threshold_duration, min_trigger_duration)
        shots_to_split = [s for s in shots if s["duration"] > threshold_duration]
    
    if not shots_to_split:
        return shots, transitions
    
    new_splits = []  # frame numbers of the newly found cuts
    
    for shot in shots_to_split:
        # Re-detect inside the selected shot with the low threshold
        s_start = shot["start_frame"]
        s_end = shot["end_frame"]
        
        # Take this segment's prediction scores
        seg_preds = predictions[s_start:s_end + 1]
        
        # Look for frames above low_threshold inside the segment
        above_low = np.where(seg_preds > low_threshold)[0]
        if len(above_low) == 0:
            continue
        
        # Group the adjacent frames
        groups = np.split(above_low, np.where(np.diff(above_low) != 1)[0] + 1)
        
        for group in groups:
            if len(group) == 0:
                continue
            # Absolute frame number
            abs_start = s_start + int(group[0])
            abs_end = s_start + int(group[-1])
            peak_score = float(np.max(seg_preds[group]))
            
            # Skip anything too close to the shot edges (0.5s of margin on each side)
            buffer_frames = int(fps * 0.5)
            if abs_start - s_start < buffer_frames or s_end - abs_end < buffer_frames:
                continue
            
            # A new cut point was found
            # Use the peak frame as the split point
            peak_local_idx = int(group[np.argmax(seg_preds[group])])
            split_frame = s_start + peak_local_idx
            
            new_splits.append({
                "split_frame": split_frame,
                "start_frame": abs_start,
                "end_frame": abs_end,
                "confidence": round(peak_score, 4),
            })
    
    if not new_splits:
        return shots, transitions
    
    # Classify each new cut point and append it to transitions
    new_transitions = []
    for sp in new_splits:
        t_type = classify_transition_type(
            video_path, sp["start_frame"], sp["end_frame"], fps
        )
        new_transitions.append({
            "start_frame": sp["start_frame"],
            "end_frame": sp["end_frame"],
            "start_time": round(float(sp["start_frame"]) / fps, 3),
            "end_time": round(float(sp["end_frame"]) / fps, 3),
            "type": t_type,
            "confidence": sp["confidence"],
        })
    
    # Merge the transition lists
    transitions = transitions + new_transitions
    transitions.sort(key=lambda x: x["start_frame"])
    
    # Rebuild the shots: insert the new cut points into the shot split
    split_frames = [sp["split_frame"] for sp in new_splits]
    shots = rebuild_shots_with_supplements(
        shots, 
        [{"end_frame": sf} for sf in split_frames],  # reuse the rebuild helper
        fps, total_frames
    )
    
    return shots, transitions


def detect_semantic_scene_changes(video_path, fps, total_frames, existing_transitions,
                                  n_missing):
    """
    Semantic scene change detection based on OpenCLIP ViT-B/32.
    
    When TransNetV2 cannot detect a high-level transition (occlusion transitions, natural wipes, ...),
    the cosine similarity between the semantic features of neighbouring windows locates the scene change.
    
    Args:
        video_path: path to the video file
        fps: frame rate
        total_frames: total number of frames
        existing_transitions: the transitions already detected
        n_missing: how many transitions still need to be found
    
    Returns:
        list of supplementary transitions, or an empty list
    """
    global clip_model, clip_preprocess
    if clip_model is None or clip_preprocess is None:
        return []
    
    # Sampling parameters
    sample_step = max(int(fps * 0.25), 4)  # sample one frame every 0.25s
    window_size = 4  # 4 sample points = a 1 second semantic window
    min_distance_frames = int(fps * 1.5)  # candidates must be at least 1.5s apart
    
    # Step 1: sample frames and extract the CLIP features
    cap = cv2.VideoCapture(video_path)
    sample_frames = []
    sample_indices = []
    
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to a PIL Image for the CLIP preprocessing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        sample_frames.append(pil_img)
        sample_indices.append(frame_idx)
        frame_idx += sample_step
    cap.release()
    
    if len(sample_frames) < window_size * 2 + 1:
        return []
    
    # Extract the features in batches
    features = []
    batch_size = 32
    for i in range(0, len(sample_frames), batch_size):
        batch_imgs = sample_frames[i:i+batch_size]
        batch_tensors = torch.stack([clip_preprocess(img) for img in batch_imgs]).to(DEVICE)
        with torch.no_grad():
            batch_feats = clip_model.encode_image(batch_tensors)
            batch_feats = batch_feats / batch_feats.norm(dim=-1, keepdim=True)  # L2 normalisation
        features.append(batch_feats.cpu().numpy())
    
    features = np.concatenate(features, axis=0)  # shape: (N, 512)
    
    # Step 2: sliding window, semantic similarity between the two windows
    similarities = []
    sim_indices = []  # the matching frame numbers
    
    for i in range(window_size, len(features) - window_size):
        # Mean feature of the preceding window
        feat_before = np.mean(features[i-window_size:i], axis=0)
        feat_before = feat_before / (np.linalg.norm(feat_before) + 1e-8)
        # Mean feature of the following window
        feat_after = np.mean(features[i:i+window_size], axis=0)
        feat_after = feat_after / (np.linalg.norm(feat_after) + 1e-8)
        # Cosine similarity
        sim = float(np.dot(feat_before, feat_after))
        similarities.append(sim)
        sim_indices.append(sample_indices[i])
    
    if not similarities:
        return []
    
    similarities = np.array(similarities)
    
    # Step 3: find the local minima (the points of largest semantic change)
    # A simple local minimum detection
    candidates = []
    for i in range(1, len(similarities) - 1):
        if similarities[i] < similarities[i-1] and similarities[i] < similarities[i+1]:
            # This is a local minimum
            candidates.append({
                "frame": sim_indices[i],
                "similarity": float(similarities[i]),
                "dissimilarity": 1.0 - float(similarities[i]),
            })
    
    # Also accept non-strict minima whose similarity is very low (for a flat curve)
    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    low_threshold = mean_sim - 1.5 * std_sim  # below 1.5 standard deviations
    
    for i in range(1, len(similarities) - 1):
        if similarities[i] < low_threshold:
            # Check whether it is already among the candidates
            frame = sim_indices[i]
            already_added = any(abs(c["frame"] - frame) < min_distance_frames for c in candidates)
            if not already_added:
                candidates.append({
                    "frame": frame,
                    "similarity": float(similarities[i]),
                    "dissimilarity": 1.0 - float(similarities[i]),
                })
    
    if not candidates:
        return []
    
    # Step 4: drop the candidates less than 1s away from an existing transition
    existing_frames = set()
    for t in existing_transitions:
        center = (t["start_frame"] + t["end_frame"]) // 2
        existing_frames.add(center)
    
    filtered = []
    for c in candidates:
        too_close = False
        for ef in existing_frames:
            if abs(c["frame"] - ef) < int(fps * 1.0):
                too_close = True
                break
        if not too_close:
            filtered.append(c)
    
    if not filtered:
        return []
    
    # Step 5: sort by ascending similarity (largest semantic change first) and take the top n_missing
    filtered.sort(key=lambda x: x["similarity"])
    
    # Apply the minimum distance constraint, to avoid clustering
    selected = []
    for c in filtered:
        if len(selected) >= n_missing:
            break
        # Check the distance to the already selected points
        too_close = any(abs(c["frame"] - s["frame"]) < min_distance_frames for s in selected)
        if not too_close:
            selected.append(c)
    
    if not selected:
        return []
    
    # Step 6: classify the transition type of every selected point
    supplements = []
    for s in selected:
        frame = s["frame"]
        # Use a small window as the transition interval
        half_w = int(fps * 0.3)  # 0.3s on each side
        sf = max(0, frame - half_w)
        ef = min(total_frames - 1, frame + half_w)
        
        t_type = classify_transition_type(video_path, sf, ef, fps)
        supplements.append({
            "start_frame": sf,
            "end_frame": ef,
            "start_time": round(float(sf) / fps, 3),
            "end_time": round(float(ef) / fps, 3),
            "type": t_type,
            "confidence": round(s["dissimilarity"], 4),
            "method": "semantic_clip",
        })
    
    return supplements


@app.on_event("startup")
async def startup():
    try:
        load_model()
        print(f"[TransNetV2] Model loaded on {DEVICE}")
    except Exception as e:
        print(f"[TransNetV2] Warning: {e}")
    try:
        load_clip_model()
    except Exception as e:
        print(f"[OpenCLIP] Warning: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "TransNetV2",
        "device": DEVICE,
        "loaded": model is not None,
        "clip_loaded": clip_model is not None,
        "weights_path": WEIGHTS_PATH,
        "clip_weights_path": CLIP_WEIGHTS_PATH,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = Form(default=0.5),
                  expected_shots: int = Form(default=0)):
    """
    Detect the shot boundaries of a video.
    
    Input: a video file (mp4/avi/mov, ...)
    Parameters:
        threshold - cut threshold (default 0.5)
        expected_shots - GT expected shot count (default 0; >0 triggers the semantic supplementary detection)
    Output:
        - shots: [{shot_index, start_frame, end_frame, start_time, end_time, duration}]
        - transitions: [{start_frame, end_frame, start_time, end_time, type, confidence}]
        - num_shots: total number of shots
        - fps: video frame rate
    """
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extract frames
        frames, fps = extract_frames(tmp_path)
        if fps <= 0:
            fps = 30.0
        total_frames = len(frames)

        # Run model in batches of 100 frames
        _disable_torch_determinism()
        m = load_model()
        WINDOW = 100
        all_preds = []

        for start_idx in range(0, total_frames, WINDOW):
            end_idx = min(start_idx + WINDOW, total_frames)
            batch = frames[start_idx:end_idx]
            
            # Pad if needed
            if len(batch) < WINDOW:
                pad_size = WINDOW - len(batch)
                batch = np.concatenate([batch, np.zeros((pad_size, 27, 48, 3), dtype=np.uint8)])
            
            input_tensor = torch.from_numpy(batch).unsqueeze(0).to(DEVICE)
            _disable_torch_determinism()
            with torch.no_grad():
                single_pred, _ = m(input_tensor)
                single_pred = torch.sigmoid(single_pred).cpu().numpy()[0]
            
            actual_len = end_idx - start_idx
            all_preds.append(single_pred[:actual_len])

        predictions = np.concatenate(all_preds).flatten()
        
        # Get scenes
        scenes = predictions_to_scenes(predictions, threshold=threshold)
        
        shots = []
        for i, (start, end) in enumerate(scenes):
            shots.append({
                "shot_index": i,
                "start_frame": int(start),
                "end_frame": int(end),
                "start_time": round(float(start) / fps, 3),
                "end_time": round(float(end) / fps, 3),
                "duration": round(float(end - start) / fps, 3)
            })

        # Detect transitions with 5-class type classification
        # Detect the transitions and run the five-way type classification
        transitions = []
        above_thresh = np.where(predictions > threshold)[0]
        if len(above_thresh) > 0:
            groups = np.split(above_thresh, np.where(np.diff(above_thresh) != 1)[0] + 1)
            for group in groups:
                if len(group) > 0:
                    sf, ef = int(group[0]), int(group[-1])
                    # Use the five-way algorithm instead of the original binary one
                    t_type = classify_transition_type(tmp_path, sf, ef, fps)
                    transitions.append({
                        "start_frame": sf,
                        "end_frame": ef,
                        "start_time": round(float(sf) / fps, 3),
                        "end_time": round(float(ef) / fps, 3),
                        "type": t_type,
                        "confidence": round(float(np.mean(predictions[sf:ef+1])), 4)
                    })

        # Merge adjacent flash transitions of the same type
        # (TransNetV2 may leave a short low-confidence gap in the middle of one flash and split it)
        transitions = merge_adjacent_flash_transitions(transitions, fps)

        # Supplementary flash white/black detection (for the flashes TransNetV2 misses)
        flash_supplements = detect_flash_transitions(tmp_path, fps, transitions, total_frames)
        if flash_supplements:
            transitions.extend(flash_supplements)
            transitions.sort(key=lambda x: x["start_frame"])
            shots = rebuild_shots_with_supplements(shots, flash_supplements, fps, total_frames)

        # === Merge dense fragment cuts ===
        # Run this after all the basic detection, so the supplementary passes cannot add fragments back
        original_trans_count = len(transitions)
        transitions = merge_dense_cuts(transitions, fps, tmp_path)
        if len(transitions) < original_trans_count:
            shots = rebuild_shots_from_transitions(transitions, fps, total_frames)

        # === Second-pass detection on long shots (plan A) ===
        # When a shot lasts far longer than the median, re-examine it with a lower threshold
        shots, transitions = detect_long_shot_splits(
            shots, transitions, predictions, fps, tmp_path, total_frames,
            length_ratio=2.0, low_threshold=0.12
        )

        # === Semantic scene change supplementary detection (OpenCLIP) ===
        # Compute the effective shot count: exclude the pseudo shots from short transition intervals (< 1.0s)
        # Those short segments are usually flash transition regions wrongly split into their own shots
        min_shot_duration = 1.0
        effective_shots = [s for s in shots if s["duration"] >= min_shot_duration]
        
        if expected_shots > 0 and len(effective_shots) < expected_shots and clip_model is not None:
            # Ask for 1 extra candidate as a buffer, since the long-shot second pass may have produced a false cut
            n_missing = expected_shots - len(effective_shots) + 1
            semantic_supplements = detect_semantic_scene_changes(
                tmp_path, fps, total_frames, transitions, n_missing
            )
            # Filter out low-confidence semantic detections (dissimilarity < 0.10 counts as noise)
            if semantic_supplements:
                semantic_supplements = [
                    s for s in semantic_supplements if s["confidence"] >= 0.10
                ]
            if semantic_supplements:
                transitions.extend(semantic_supplements)
                transitions.sort(key=lambda x: x["start_frame"])
                shots = rebuild_shots_with_supplements(
                    shots, semantic_supplements, fps, total_frames
                )

        return JSONResponse({
            "success": True,
            "num_shots": len(shots),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration": round(total_frames / fps, 3),
            "shots": shots,
            "transitions": transitions
        })

    except Exception as e:
        import traceback
        return JSONResponse({"success": False, "error": str(e), "traceback": traceback.format_exc()}, status_code=500)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    # SERVICE_PORT allows starting a second instance without changing the default behaviour (e.g. an offset port for the edit agent's self-evaluation)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("SERVICE_PORT", 8001)))
