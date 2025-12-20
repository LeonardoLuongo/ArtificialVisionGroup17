import os
import sys
import torch
import numpy as np
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
        model_name='osnet_x0_25', 
        device=DEVICE
    )
except ImportError:
    reid_model = None
    print("ATTENZIONE: torchreid non trovato.")

# --- KALMAN OTTIMIZZATO ---
class SoccerKalmanFilter:
    """Filtro di Kalman ottimizzato per tracking giocatori."""
    def __init__(self, initial_box):
        self.state = np.zeros(8, dtype=np.float32)  # float32 per velocità
        self.state[:4] = initial_box
        
        # Pre-calcola matrici costanti
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0
        
        self.H = np.eye(4, 8, dtype=np.float32)
        
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0
        
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:4, :4] *= 0.1
        self.Q[4:, 4:] *= 0.01
        
        self.R = np.eye(4, dtype=np.float32) * 1.0
        
        # Pre-calcola F^T per riutilizzo
        self.F_T = self.F.T
        self.H_T = self.H.T

    def predict(self):
        """Predizione ottimizzata."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F_T + self.Q
        return self.state[:4].copy()

    def update(self, measurement):
        """Update ottimizzato usando solve() invece di inv()."""
        # S = H * P * H^T + R
        S = self.H @ self.P @ self.H_T + self.R
        
        # K = P * H^T * inv(S) -> risolviamo K^T * S^T = (P * H^T)^T
        K = (np.linalg.solve(S.T, (self.P @ self.H_T).T)).T
        
        # Update stato
        y = measurement - self.H @ self.state
        self.state += K @ y
        
        # Update covarianza (formula Joseph per stabilità numerica)
        I_KH = np.eye(8, dtype=np.float32) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_uncertainty(self):
        """Restituisce incertezza posizione."""
        return float(np.trace(self.P[:2, :2]))

# --- KALMAN MANAGER OTTIMIZZATO ---
class KalmanTrackerManager:
    def __init__(self, prediction_limit=25):
        self.filters = {}
        self.last_seen = {}
        self.prediction_limit = prediction_limit

    def update_or_predict(self, active_detections, frame_id):
        """
        active_detections: {id: (coords, conf)}
        Ritorna: {id: (coords, conf, is_predicted)}
        """
        results = {}
        
        # Update tracce attive
        for tid, (coords, conf) in active_detections.items():
            if tid not in self.filters:
                self.filters[tid] = SoccerKalmanFilter(coords)
            else:
                self.filters[tid].predict()
                self.filters[tid].update(coords)
            
            results[tid] = (coords, conf, False)
            self.last_seen[tid] = frame_id

        # Predici tracce occultate
        missing_ids = set(self.filters.keys()) - set(active_detections.keys())
        for tid in missing_ids:
            frames_missing = frame_id - self.last_seen.get(tid, frame_id)
            
            if frames_missing <= self.prediction_limit:
                pred_coords = self.filters[tid].predict()
                uncertainty = self.filters[tid].get_uncertainty()
                
                # Confidenza decrescente con incertezza
                pred_conf = max(0.1, 0.3 - 0.01 * frames_missing - 0.05 * uncertainty)
                results[tid] = (pred_coords, pred_conf, True)
            else:
                # Rimuovi tracce troppo vecchie
                self.filters.pop(tid, None)
                self.last_seen.pop(tid, None)

        return results

    def cleanup_old_tracks(self, active_ids, frame_id, max_age=100):
        """Rimuove tracce non viste da troppo tempo."""
        all_ids = list(self.filters.keys())
        for tid in all_ids:
            if tid not in active_ids and frame_id - self.last_seen.get(tid, 0) > max_age:
                self.filters.pop(tid, None)
                self.last_seen.pop(tid, None)

def run_optimized_tracking():
    # --- CONFIGURAZIONE ---
    CONF_THRES = 0.35
    MAX_ENTITIES = 28 
    REID_THRESHOLD = 0.85 
    IMG_SIZE = 1280
    GALLERY_MAX_AGE = 100
    PREDICTION_LIMIT = 25
    # ----------------------
    
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / "best.pt"
    model = YOLO(weights_path)
    model.to(DEVICE)

    output_folder = TRACKER_OUTPUT_DIR / f"hota_kalman_optimized"
    os.makedirs(output_folder, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted(os.listdir(base_path))

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Kalman-Enhanced HOTA: {seq_name} (Res: {IMG_SIZE})")
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            # Strutture dati
            id_gallery = {}
            id_mapping = {}
            id_last_seen = {}
            id_similarity_scores = {}
            tracker_to_mapped_history = defaultdict(list)
            kf_manager = KalmanTrackerManager(prediction_limit=PREDICTION_LIMIT)
            
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
                        
                        # 1. Raccolta detection + preparazione Re-ID
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

                        # 2. Batch Re-ID ottimizzato
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
                    
                    # Risolvi conflitti e prepara detections per Kalman
                    active_detections = {}
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
                        
                        # Normalizza per Kalman (0-1)
                        norm_coords = np.array([
                            x1 / img_w, y1 / img_h,
                            (x2 - x1) / img_w, (y2 - y1) / img_h
                        ], dtype=np.float32)
                        
                        active_detections[mapped_id] = (norm_coords, winner['conf'])
                        used_final_ids.add(mapped_id)

                    # 4. KALMAN: Update attivi + Predici occultati
                    all_tracks = kf_manager.update_or_predict(active_detections, frame_id)
                    
                    # 5. Scrittura output (detections + predictions)
                    frame_output = []
                    for tid, (norm_coords, conf, is_predicted) in all_tracks.items():
                        # Denormalizza
                        x, y, w, h = norm_coords
                        x1, y1 = x * img_w, y * img_h
                        w, h = w * img_w, h * img_h
                        
                        # Clamp coordinate
                        x1 = max(0, min(x1, img_w))
                        y1 = max(0, min(y1, img_h))
                        w = max(1, min(w, img_w - x1))
                        h = max(1, min(h, img_h - y1))
                        
                        line = f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                        frame_output.append((conf, line, is_predicted))

                    # Ordina: reali prima, poi predette, tutte per confidenza
                    frame_output.sort(key=lambda x: (x[2], -x[0]))  # False < True, conf desc
                    
                    for conf, line, _ in frame_output[:MAX_ENTITIES]:
                        f.write(line)
                    f.flush()

                    # 6. Cleanup periodico
                    if frame_id % GALLERY_MAX_AGE == 0:
                        # Gallery Re-ID
                        old_ids = [gid for gid, lf in id_last_seen.items() 
                                   if frame_id - lf > GALLERY_MAX_AGE]
                        for old_id in old_ids:
                            id_gallery.pop(old_id, None)
                            id_last_seen.pop(old_id, None)
                        
                        # Kalman filters
                        kf_manager.cleanup_old_tracks(
                            set(active_detections.keys()), frame_id, GALLERY_MAX_AGE
                        )

            print(f"✓ Completato: {seq_name}")

if __name__ == "__main__":
    run_optimized_tracking()