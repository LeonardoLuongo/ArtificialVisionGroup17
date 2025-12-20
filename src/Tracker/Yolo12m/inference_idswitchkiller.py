import os
import sys
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device

DEVICE = get_best_device()

try:
    from torchreid.utils import FeatureExtractor
    reid_model = FeatureExtractor(
        model_name='osnet_ibn_x1_0', 
        device=DEVICE
    )
except ImportError:
    reid_model = None
    print("ATTENZIONE: torchreid non trovato.")

# --- CAMERA MOTION DETECTOR ---
class CameraMotionDetector:
    """Rileva movimento significativo della camera usando optical flow."""
    def __init__(self, motion_threshold=5.0):
        self.prev_gray = None
        self.motion_threshold = motion_threshold
        
    def detect_camera_motion(self, frame):
        """
        Ritorna True se la camera si è mossa significativamente.
        Usa sparse optical flow su feature del campo.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0
        
        # Trova feature sul campo (linee bianche, punti fissi)
        corners = cv2.goodFeaturesToTrack(
            self.prev_gray, 
            maxCorners=100,
            qualityLevel=0.1,
            minDistance=20,
            blockSize=7
        )
        
        if corners is None or len(corners) < 10:
            self.prev_gray = gray
            return False, 0.0
        
        # Calcola optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, corners, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Filtra punti validi
        if next_pts is None:
            self.prev_gray = gray
            return False, 0.0
            
        good_old = corners[status == 1]
        good_new = next_pts[status == 1]
        
        if len(good_old) < 10:
            self.prev_gray = gray
            return False, 0.0
        
        # Calcola movimento medio (vettore di camera motion)
        motion_vectors = good_new - good_old
        median_motion = np.median(np.abs(motion_vectors), axis=0)
        motion_magnitude = np.linalg.norm(median_motion)
        
        self.prev_gray = gray
        
        is_moving = motion_magnitude > self.motion_threshold
        return is_moving, float(motion_magnitude)

# --- KALMAN CON CAMERA AWARENESS ---
class CameraAwareKalmanFilter:
    """Kalman che NON predice quando la camera si muove."""
    def __init__(self, initial_box):
        self.state = np.zeros(8, dtype=np.float32)
        self.state[:4] = initial_box
        
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0
        
        self.H = np.eye(4, 8, dtype=np.float32)
        
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0
        
        self.Q = np.eye(8, dtype=np.float32) * 0.01
        self.R = np.eye(4, dtype=np.float32) * 0.5
        
        self.F_T = self.F.T
        self.H_T = self.H.T
        
        self.last_position = initial_box.copy()

    def update_only(self, measurement):
        """Update senza predict - per quando la camera si muove."""
        S = self.H @ self.P @ self.H_T + self.R
        K = (np.linalg.solve(S.T, (self.P @ self.H_T).T)).T
        
        y = measurement - self.H @ self.state
        self.state += K @ y
        
        I_KH = np.eye(8, dtype=np.float32) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Reset velocità (la camera ha cambiato il reference frame)
        self.state[4:] = 0
        
        self.last_position = self.state[:4].copy()
        return self.state[:4].copy()

    def smooth(self, measurement, camera_moving=False):
        """Smooth con awareness del movimento camera."""
        if camera_moving:
            # Camera in movimento: NO predict, solo update
            return self.update_only(measurement)
        else:
            # Camera statica: predict + update normale
            self.state = self.F @ self.state
            self.P = self.F @ self.P @ self.F_T + self.Q
            
            S = self.H @ self.P @ self.H_T + self.R
            K = (np.linalg.solve(S.T, (self.P @ self.H_T).T)).T
            
            y = measurement - self.H @ self.state
            self.state += K @ y
            
            I_KH = np.eye(8, dtype=np.float32) - K @ self.H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
            
            self.last_position = self.state[:4].copy()
            return self.state[:4].copy()

class CameraAwareSmootherManager:
    """Manager Kalman con rilevamento movimento camera."""
    def __init__(self):
        self.filters = {}
        self.camera_detector = CameraMotionDetector(motion_threshold=3.0)

    def smooth_detections(self, detections, frame_bgr):
        """
        Smooth detection considerando il movimento camera.
        detections: {id: (coords, conf)}
        frame_bgr: immagine BGR per analisi motion
        """
        # Rileva movimento camera
        camera_moving, motion_mag = self.camera_detector.detect_camera_motion(frame_bgr)
        
        smoothed = {}
        
        for tid, (coords, conf) in detections.items():
            if tid not in self.filters:
                self.filters[tid] = CameraAwareKalmanFilter(coords)
                smoothed_coords = coords
            else:
                smoothed_coords = self.filters[tid].smooth(coords, camera_moving)
            
            smoothed[tid] = (smoothed_coords, conf)
        
        # Cleanup
        active_ids = set(detections.keys())
        old_ids = set(self.filters.keys()) - active_ids
        for old_id in old_ids:
            self.filters.pop(old_id, None)
        
        return smoothed, camera_moving, motion_mag

def run_optimized_tracking():
    # --- CONFIGURAZIONE ---
    CONF_THRES = 0.35
    MAX_ENTITIES = 28 
    REID_THRESHOLD = 0.85 
    IMG_SIZE = 1280
    GALLERY_MAX_AGE = 750
    USE_CAMERA_AWARE_KALMAN = True  # Toggle per test
    # ----------------------
    
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / "best.pt"

    output_folder = TRACKER_OUTPUT_DIR / f"hota_camera_aware"
    os.makedirs(output_folder, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted(os.listdir(base_path))

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            model = YOLO(weights_path)
            model.to(DEVICE)

            mode = "Camera-Aware" if USE_CAMERA_AWARE_KALMAN else "Standard"
            print(f"--> {mode}: {seq_name} (Res: {IMG_SIZE})")
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            # Strutture dati
            id_gallery = {}
            id_mapping = {}
            id_last_seen = {}
            id_similarity_scores = {}
            tracker_to_mapped_history = defaultdict(list)
            
            if USE_CAMERA_AWARE_KALMAN:
                smoother = CameraAwareSmootherManager()
            
            camera_motion_frames = 0  # Stats
            
            results = model.track(
                source=img_source, conf=CONF_THRES, iou=0.5, imgsz=IMG_SIZE,
                tracker=str(TRACKER_CONFIG / "botsort.yaml"),
                persist=True, verbose=False, stream=True, half=True
            )
            
            with open(res_file_path, 'w') as f:
                for result in results:
                    frame_id = int(os.path.basename(result.path).split('.')[0])
                    img_bgr = result.orig_img
                    img_h, img_w = img_bgr.shape[:2]
                    
                    crops_to_process = []
                    tracker_ids_to_process = []
                    boxes_data = []

                    if result.boxes is not None and result.boxes.id is not None:
                        boxes = result.boxes.cpu().numpy()
                        
                        # 1. Raccolta detection + Re-ID prep
                        for box in boxes:
                            if box.id is None: continue
                            tracker_id = int(box.id[0])
                            x1, y1, x2, y2 = box.xyxy[0].astype(int)
                            
                            boxes_data.append({
                                'tracker_id': tracker_id,
                                'coords': (x1, y1, x2, y2),
                                'conf': float(box.conf[0])
                            })

                            if reid_model and tracker_id not in id_mapping:
                                crop = img_bgr[y1:y2, x1:x2]
                                if crop.size > 0:
                                    crops_to_process.append(crop)
                                    tracker_ids_to_process.append(tracker_id)

                        # 2. Batch Re-ID
                        if crops_to_process:
                            batch_features = reid_model(crops_to_process).cpu().numpy()
                            
                            if id_gallery:
                                gallery_ids = list(id_gallery.keys())
                                gallery_features = np.vstack([id_gallery[gid] for gid in gallery_ids])
                                similarities = cosine_similarity(batch_features, gallery_features)
                                
                                for i, t_id in enumerate(tracker_ids_to_process):
                                    max_idx = similarities[i].argmax()
                                    max_sim = float(similarities[i][max_idx])
                                    
                                    if max_sim > REID_THRESHOLD:
                                        best_match_id = gallery_ids[max_idx]
                                        id_mapping[t_id] = best_match_id
                                        id_similarity_scores[t_id] = max_sim
                                        id_last_seen[best_match_id] = frame_id
                                        tracker_to_mapped_history[t_id].append(best_match_id)
                                    elif len(id_gallery) < MAX_ENTITIES:
                                        id_gallery[t_id] = batch_features[i:i+1]
                                        id_mapping[t_id] = t_id
                                        id_similarity_scores[t_id] = 1.0
                                        id_last_seen[t_id] = frame_id
                                        tracker_to_mapped_history[t_id].append(t_id)
                            else:
                                for i, t_id in enumerate(tracker_ids_to_process):
                                    if len(id_gallery) < MAX_ENTITIES:
                                        id_gallery[t_id] = batch_features[i:i+1]
                                        id_mapping[t_id] = t_id
                                        id_similarity_scores[t_id] = 1.0
                                        id_last_seen[t_id] = frame_id
                                        tracker_to_mapped_history[t_id].append(t_id)

                    # 3. Risoluzione conflitti ID
                    id_to_trackers = defaultdict(list)
                    for b in boxes_data:
                        t_id = b['tracker_id']
                        mapped_id = id_mapping.get(t_id, t_id)
                        id_to_trackers[mapped_id].append({
                            'tracker_id': t_id,
                            'data': b,
                            'similarity': id_similarity_scores.get(t_id, 0.0),
                            'conf': b['conf']
                        })
                    
                    final_detections = {}
                    used_final_ids = set()
                    
                    for mapped_id in sorted(id_to_trackers.keys()):
                        candidates = id_to_trackers[mapped_id]
                        
                        if len(candidates) > 1:
                            candidates.sort(key=lambda x: (
                                x['similarity'], x['conf'],
                                len(tracker_to_mapped_history[x['tracker_id']])
                            ), reverse=True)
                        
                        winner = candidates[0]
                        x1, y1, x2, y2 = winner['data']['coords']
                        
                        norm_coords = np.array([
                            x1 / img_w, y1 / img_h,
                            (x2 - x1) / img_w, (y2 - y1) / img_h
                        ], dtype=np.float32)
                        
                        final_detections[mapped_id] = (norm_coords, winner['conf'])
                        used_final_ids.add(mapped_id)

                    # 4. CAMERA-AWARE KALMAN SMOOTHING
                    camera_moving = False
                    motion_mag = 0.0
                    
                    if USE_CAMERA_AWARE_KALMAN:
                        final_detections, camera_moving, motion_mag = smoother.smooth_detections(
                            final_detections, img_bgr
                        )
                        if camera_moving:
                            camera_motion_frames += 1

                    # 5. Scrittura output
                    frame_output = []
                    for tid, (norm_coords, conf) in final_detections.items():
                        x, y, w, h = norm_coords
                        x1, y1 = x * img_w, y * img_h
                        w, h = w * img_w, h * img_h
                        
                        x1 = max(0, min(x1, img_w))
                        y1 = max(0, min(y1, img_h))
                        w = max(1, min(w, img_w - x1))
                        h = max(1, min(h, img_h - y1))
                        
                        line = f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                        frame_output.append((conf, line))

                    frame_output.sort(key=lambda x: x[0], reverse=True)
                    
                    for conf, line in frame_output[:MAX_ENTITIES]:
                        f.write(line)
                    # f.flush()

                    # 6. Cleanup periodico
                    if frame_id % GALLERY_MAX_AGE == 0:
                        old_ids = [gid for gid, lf in id_last_seen.items() 
                                   if frame_id - lf > GALLERY_MAX_AGE]
                        for old_id in old_ids:
                            id_gallery.pop(old_id, None)
                            id_last_seen.pop(old_id, None)

            cam_motion_pct = (camera_motion_frames / frame_id * 100) if frame_id > 0 else 0
            print(f"✓ {seq_name} - Camera motion: {cam_motion_pct:.1f}% frames")

if __name__ == "__main__":
    run_optimized_tracking()