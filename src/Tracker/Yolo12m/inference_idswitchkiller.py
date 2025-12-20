import os
import sys
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# Setup path (mantenendo la tua struttura)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device

DEVICE = get_best_device()

# Caricamento Re-ID
try:
    from torchreid.utils import FeatureExtractor
    reid_model = FeatureExtractor(
        model_name='osnet_x0_25', 
        device=DEVICE
    )
except ImportError:
    reid_model = None
    print("ATTENZIONE: torchreid non trovato.")

# --- 1. CAMERA MOTION DETECTOR (GMC) ---
class CameraMotionDetector:
    def __init__(self, motion_threshold=3.0):
        self.prev_gray = None
        self.motion_threshold = motion_threshold
        
    def detect_and_compensate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0, np.eye(2, 3, dtype=np.float32)
        
        corners = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=15)
        if corners is None or len(corners) < 20:
            self.prev_gray = gray
            return False, 0.0, np.eye(2, 3, dtype=np.float32)
        
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, corners, None)
        if next_pts is None:
            self.prev_gray = gray
            return False, 0.0, np.eye(2, 3, dtype=np.float32)
            
        good_old, good_new = corners[status == 1], next_pts[status == 1]
        if len(good_old) < 20:
            self.prev_gray = gray
            return False, 0.0, np.eye(2, 3, dtype=np.float32)
        
        motion_magnitude = np.mean(np.linalg.norm(good_new - good_old, axis=1))
        affine_matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
        
        if affine_matrix is None: affine_matrix = np.eye(2, 3, dtype=np.float32)
        
        self.prev_gray = gray
        return motion_magnitude > self.motion_threshold, float(motion_magnitude), affine_matrix

# --- 2. SLOT-BASED KALMAN FILTER (GMC AWARE) ---
class SlotKalmanFilter:
    def __init__(self, bbox):
        # Stato: [x1, y1, w, h, vx, vy, vw, vh]
        self.state = np.zeros(8, dtype=np.float32)
        self.state[:4] = bbox
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.Q = np.eye(8, dtype=np.float32) * 0.1
        self.R = np.eye(4, dtype=np.float32) * 0.5
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4): self.F[i, i+4] = 1.0

    def predict_and_compensate(self, affine_matrix):
        # 1. Applica GMC allo stato corrente
        if affine_matrix is not None:
            x, y, w, h = self.state[:4]
            center = np.array([[x + w/2, y + h/2]], dtype=np.float32)
            new_center = cv2.transform(center.reshape(1,1,2), affine_matrix).flatten()
            scale = np.sqrt(affine_matrix[0,0]**2 + affine_matrix[1,0]**2)
            self.state[0], self.state[1] = new_center[0] - (w*scale)/2, new_center[1] - (h*scale)/2
            self.state[2], self.state[3] = w * scale, h * scale
            # Trasforma velocità
            vel = self.state[4:6].reshape(1,1,2)
            self.state[4:6] = (vel @ affine_matrix[:, :2].T).flatten()

        # 2. Predizione Kalman standard
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        H = np.eye(4, 8, dtype=np.float32)
        y = measurement - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

# --- 3. GLOBAL SLOT MANAGER (CONSTRAINED TO 28) ---
class GlobalSlotManager:
    def __init__(self, max_slots=28, reid_weight=0.65):
        self.max_slots = max_slots
        self.reid_weight = reid_weight
        self.slots = {} # {id: {'kf': kf, 'feat': feat, 'last_seen': frame, 'conf': conf}}
        self.frame_count = 0
        self.camera_detector = CameraMotionDetector()

    def update(self, frame, detections):
        self.frame_count += 1
        img_h, img_w = frame.shape[:2]
        is_moving, _, affine_matrix = self.camera_detector.detect_and_compensate(frame)
        
        slot_ids = list(self.slots.keys())
        for sid in slot_ids:
            self.slots[sid]['kf'].predict_and_compensate(affine_matrix)

        if not detections:
            return self._get_results()

        matched_dets = set()
        if slot_ids:
            # Matrice di costo: Re-ID + Spazio
            cost_matrix = np.zeros((len(slot_ids), len(detections)))
            for i, sid in enumerate(slot_ids):
                slot_box = self.slots[sid]['kf'].state[:4]
                for j, det in enumerate(detections):
                    dist_reid = 1.0 - cosine_similarity(self.slots[sid]['feat'].reshape(1,-1), det['feat'].reshape(1,-1))[0,0]
                    dist_spatial = np.linalg.norm(slot_box[:2] - det['box'][:2]) / np.sqrt(img_h**2 + img_w**2)
                    cost_matrix[i, j] = (dist_reid * self.reid_weight) + (dist_spatial * (1-self.reid_weight))
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.8:
                    sid = slot_ids[r]
                    self.slots[sid]['kf'].update(detections[c]['box'])
                    self.slots[sid]['feat'] = 0.9 * self.slots[sid]['feat'] + 0.1 * detections[c]['feat']
                    self.slots[sid]['last_seen'] = self.frame_count
                    self.slots[sid]['conf'] = detections[c]['conf']
                    matched_dets.add(c)

        # Gestione nuovi slot (fino a 28)
        if len(self.slots) < self.max_slots:
            for j, det in enumerate(detections):
                if j not in matched_dets and len(self.slots) < self.max_slots:
                    # Trova il primo ID libero tra 0 e 27
                    for potential_id in range(self.max_slots):
                        if potential_id not in self.slots:
                            self.slots[potential_id] = {
                                'kf': SlotKalmanFilter(det['box']),
                                'feat': det['feat'],
                                'last_seen': self.frame_count,
                                'conf': det['conf']
                            }
                            matched_dets.add(j)
                            break

        # Cleanup: rimuovi chi è sparito da troppo tempo (es. 5 secondi / 125 frame)
        to_remove = [sid for sid, s in self.slots.items() if self.frame_count - s['last_seen'] > 125]
        for sid in to_remove: del self.slots[sid]

        return self._get_results()

    def _get_results(self):
        return {sid: (s['kf'].state[:4], s['conf']) for sid, s in self.slots.items()}

# --- 4. MAIN RUNNER ---
def run_realtime_tracking():
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / "best.pt"
    output_folder = TRACKER_OUTPUT_DIR / "final_slot_tracker"
    os.makedirs(output_folder, exist_ok=True)
    
    model = YOLO(weights_path)
    base_path = DATASET_TR23_TEST
    
    for seq_name in sorted(os.listdir(base_path)):
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')
        if not os.path.isdir(seq_path): continue
        
        print(f"Processing: {seq_name}")
        slot_manager = GlobalSlotManager(max_slots=28)
        res_file = os.path.join(output_folder, f"{seq_name}.txt")
        
        # model.predict è più veloce di .track perché non usiamo BoTSORT interno
        results = model.predict(source=img_source, conf=0.3, imgsz=1280, stream=True, verbose=False)
        
        with open(res_file, 'w') as f:
            for result in results:
                frame_id = int(os.path.basename(result.path).split('.')[0])
                img_bgr = result.orig_img
                
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    crops = []
                    valid_indices = []
                    for i, (box, conf) in enumerate(zip(boxes, confs)):
                        x1, y1, x2, y2 = box.astype(int)
                        crop = img_bgr[y1:y2, x1:x2]
                        if crop.size > 0:
                            crops.append(crop)
                            valid_indices.append(i)
                    
                    if crops and reid_model:
                        # Batch Re-ID (molto più veloce)
                        feats = reid_model(crops).cpu().numpy()
                        for i, idx in enumerate(valid_indices):
                            x1, y1, x2, y2 = boxes[idx]
                            detections.append({
                                'box': np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32),
                                'feat': feats[i],
                                'conf': confs[idx]
                            })

                # Logica competitiva 28 Slot
                active_slots = slot_manager.update(img_bgr, detections)
                
                # Scrittura output MOT
                for sid, (box, conf) in active_slots.items():
                    # Solo se lo slot è stato aggiornato recentemente lo consideriamo visibile
                    if frame_id - slot_manager.slots[sid]['last_seen'] < 2:
                        x, y, w, h = box
                        f.write(f"{frame_id},{sid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")

if __name__ == "__main__":
    run_realtime_tracking()