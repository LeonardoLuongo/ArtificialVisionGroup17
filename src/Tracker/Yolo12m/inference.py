import os
import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import json

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device
import torchreid

# ============ CONFIGURAZIONE ============
CONF_THRES = 0.1
MAX_ENTITIES = 28

# Soglie di matching
REID_THRESHOLD = 0.65        # Soglia base per ReID matching
MOTION_WEIGHT = 0.3          # Peso del motion nella metrica combinata
SIZE_WEIGHT = 0.15           # Peso della size similarity

# Track management
MAX_AGE = 30                 # Frame prima di eliminare track dormiente
MIN_HITS = 3                 # Conferme prima di considerare track "confirmed"
IOU_THRESHOLD = 0.3          # IOU per match spaziale diretto

# Gallery
GALLERY_SIZE = 30
MIN_GALLERY_SIMILARITY = 0.97  # Aggiungi solo embeddings diversi


def iou_batch(bb_test, bb_gt):
    """Calcola IOU tra detection e tracks"""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


class KalmanBoxTracker:
    """Kalman Filter per tracking del bounding box"""
    count = 0
    
    def __init__(self, bbox):
        """bbox: [x1, y1, x2, y2]"""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State: [x, y, s, r, vx, vy, vs] dove s=area, r=aspect ratio
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        # Measurement: [x, y, s, r]
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        # Noise
        self.kf.R[2:, 2:] *= 10.  # Measurement uncertainty per s e r
        self.kf.P[4:, 4:] *= 1000.  # Incertezza iniziale velocità
        self.kf.Q[-1, -1] *= 0.01  # Process noise per vs
        self.kf.Q[4:, 4:] *= 0.01  # Process noise per velocità
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """Aggiorna il Kalman con una nuova detection"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        
    def predict(self):
        """Predici la prossima posizione"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """Ritorna il bbox corrente"""
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox):
        """Converte [x1,y1,x2,y2] in [x,y,s,r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x):
        """Converte [x,y,s,r] in [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / (w + 1e-6)
        return np.array([
            x[0] - w/2.,
            x[1] - h/2.,
            x[0] + w/2.,
            x[1] + h/2.
        ]).reshape((1, 4))[0]


class Track:
    """Rappresenta un singolo track con ReID gallery e Kalman"""
    
    def __init__(self, track_id, bbox, embedding, conf, frame_id):
        self.track_id = track_id
        self.kf = KalmanBoxTracker(bbox)
        self.gallery = [embedding] if conf > 0.6 else []
        self.last_detection_conf = conf
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.last_bbox = bbox
        self.state = 'tentative'  # tentative -> confirmed -> lost
        
    def predict(self):
        """Predici la prossima posizione con Kalman"""
        self.last_bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.last_bbox
    
    def update(self, bbox, embedding, conf, frame_id):
        """Aggiorna il track con una nuova detection"""
        self.kf.update(bbox)
        self.last_bbox = bbox
        self.last_detection_conf = conf
        self.hits += 1
        self.time_since_update = 0
        
        # Aggiorna gallery con logica intelligente
        if conf > 0.6 and len(self.gallery) > 0:
            sims = self._compute_similarities(embedding)
            if np.max(sims) < MIN_GALLERY_SIMILARITY:  # Solo se diverso
                self.gallery.append(embedding)
                if len(self.gallery) > GALLERY_SIZE:
                    self.gallery.pop(0)
        elif conf > 0.6 and len(self.gallery) == 0:
            self.gallery.append(embedding)
        
        # Aggiorna stato
        if self.state == 'tentative' and self.hits >= MIN_HITS:
            self.state = 'confirmed'
    
    def compute_reid_similarity(self, embedding):
        """Calcola similarità ReID con la gallery"""
        if len(self.gallery) == 0:
            return 0.0
        gallery_arr = np.array(self.gallery)
        dists = cdist(embedding.reshape(1, -1), gallery_arr, metric='cosine')
        return 1.0 - np.min(dists)  # Min distance = max similarity
    
    def _compute_similarities(self, embedding):
        """Helper per calcolare similarità con tutta la gallery"""
        gallery_arr = np.array(self.gallery)
        dists = cdist(embedding.reshape(1, -1), gallery_arr, metric='cosine')
        return 1.0 - dists[0]
    
    def mark_missed(self):
        """Marca il track come missed (no detection match)"""
        self.time_since_update += 1
        if self.state == 'tentative':
            self.state = 'lost'
        elif self.time_since_update > MAX_AGE:
            self.state = 'lost'


class ReIDExtractor:
    """Estrattore di feature ReID"""
    
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        weights_str = str(weights_path).lower()
        self.resize_wh = (128, 256)
        
        if 'vit' in weights_str:
            self.resize_wh = (224, 224)
            model_name = 'vit_b_16'
        elif 'ain' in weights_str:
            model_name = 'osnet_ain_x1_0'
        else:
            model_name = 'osnet_x1_0'
            
        print(f"--> ReID Model: {model_name} | Input: {self.resize_wh}")
        
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=False
        )
        
        if weights_path and os.path.exists(weights_path):
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
            
        self.model.to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    @torch.no_grad()
    def get_embeddings(self, frame, bboxes_xyxy):
        """Estrai embeddings per tutti i bbox"""
        if len(bboxes_xyxy) == 0:
            return np.array([])
        
        crops = []
        h_img, w_img = frame.shape[:2]
        
        for box in bboxes_xyxy:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                crop = np.zeros((self.resize_wh[1], self.resize_wh[0], 3), dtype=np.uint8)
            
            crop = cv2.resize(crop, self.resize_wh)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(np.transpose(crop, (2, 0, 1)))
        
        input_t = torch.from_numpy(np.array(crops)).float().to(self.device) / 255.0
        input_t = (input_t - self.mean) / self.std
        return self.model(input_t).cpu().numpy()


class AdvancedReIDTracker:
    """Tracker principale basato su ReID con Kalman Filter"""
    
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections, embeddings, confs, frame):
        """
        Aggiorna il tracker con le detection del frame corrente
        
        Args:
            detections: array Nx4 [x1, y1, x2, y2]
            embeddings: array NxD features ReID
            confs: array N confidence scores
            frame: frame corrente (per debug/vis se necessario)
        
        Returns:
            track_ids: array N con gli ID assegnati
        """
        self.frame_count += 1
        
        # 1. Predici tutte le posizioni dei track esistenti
        for track in self.tracks:
            track.predict()
        
        # 2. Matching: detection → tracks
        matched, unmatched_dets, unmatched_trks = self._match(
            detections, embeddings, confs
        )
        
        # 3. Crea output PRIMA di modificare la lista tracks
        track_ids = np.full(len(detections), -1, dtype=int)
        for det_idx, trk_idx in matched:
            track = self.tracks[trk_idx]
            # Restituisci solo track confermati
            if track.state == 'confirmed' or track.hits >= MIN_HITS:
                track_ids[det_idx] = track.track_id
        
        # 4. Aggiorna i track matchati
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(
                detections[det_idx],
                embeddings[det_idx],
                confs[det_idx],
                self.frame_count
            )
        
        # 5. Marca i track non matchati come missed
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # 6. Crea nuovi track per detection non matchate
        for det_idx in unmatched_dets:
            new_track = Track(
                self.next_id,
                detections[det_idx],
                embeddings[det_idx],
                confs[det_idx],
                self.frame_count
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # 7. Rimuovi track morti
        self.tracks = [t for t in self.tracks if t.state != 'lost']
        
        return track_ids
    
    def _match(self, detections, embeddings, confs):
        """
        Matching tra detection e track usando ReID + Motion + Size
        
        Returns:
            matched: lista di (det_idx, trk_idx)
            unmatched_dets: lista di det_idx
            unmatched_trks: lista di trk_idx
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Costruisci matrice di costo combinata
        cost_matrix = self._build_cost_matrix(detections, embeddings, confs)
        
        # Hungarian matching
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        
        # Filtra match con costo troppo alto
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))
        
        for det_idx, trk_idx in zip(det_indices, trk_indices):
            if cost_matrix[det_idx, trk_idx] < 0.5:  # Soglia di accettazione
                matched.append((det_idx, trk_idx))
                unmatched_dets.remove(det_idx)
                if trk_idx in unmatched_trks:
                    unmatched_trks.remove(trk_idx)
        
        return matched, unmatched_dets, unmatched_trks
    
    def _build_cost_matrix(self, detections, embeddings, confs):
        """
        Costruisce matrice di costo combinando:
        - ReID similarity (peso principale)
        - Motion/spatial distance
        - Size similarity
        """
        num_dets = len(detections)
        num_trks = len(self.tracks)
        
        cost_matrix = np.zeros((num_dets, num_trks))
        
        # Ottieni predicted bboxes dei track
        trk_boxes = np.array([t.last_bbox for t in self.tracks])
        
        # Calcola IOU per motion matching
        iou_matrix = iou_batch(detections, trk_boxes)
        
        for d in range(num_dets):
            det_box = detections[d]
            det_emb = embeddings[d]
            det_size = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
            
            for t in range(num_trks):
                track = self.tracks[t]
                trk_box = track.last_bbox
                trk_size = (trk_box[2] - trk_box[0]) * (trk_box[3] - trk_box[1])
                
                # 1. ReID similarity
                reid_sim = track.compute_reid_similarity(det_emb)
                
                # 2. Motion (IOU)
                motion_sim = iou_matrix[d, t]
                
                # 3. Size similarity
                size_ratio = min(det_size, trk_size) / (max(det_size, trk_size) + 1e-6)
                size_sim = size_ratio
                
                # Combina le metriche (similarity → cost, quindi inverti)
                combined_sim = (
                    reid_sim * (1 - MOTION_WEIGHT - SIZE_WEIGHT) +
                    motion_sim * MOTION_WEIGHT +
                    size_sim * SIZE_WEIGHT
                )
                
                # Converti similarity in cost
                cost_matrix[d, t] = 1.0 - combined_sim
                
                # Penalizza fortemente se il track è troppo vecchio
                if track.time_since_update > 5:
                    cost_matrix[d, t] += 0.3
        
        return cost_matrix


def count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h):
    """Conta giocatori nelle ROI"""
    counts = {1: 0, 2: 0}
    if not roi_data:
        return counts
    
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx = x1 + (w / 2.0)
        cy = y2  # Bottom center
        
        for roi_id in [1, 2]:
            roi = roi_data[f"roi{roi_id}"]
            rx = roi["x"] * img_w
            ry = roi["y"] * img_h
            rw = roi["width"] * img_w
            rh = roi["height"] * img_h
            
            if rx <= cx <= (rx + rw) and ry <= cy <= (ry + rh):
                counts[roi_id] += 1
    
    return counts


def run_advanced_tracking():
    """Funzione principale di tracking"""
    TEAM_ID = "17"
    DEVICE = get_best_device()
    REID_WEIGHTS = MODEL_PATH / "re-ids" / "osnet_x1_0.pt"
    
    reid_extractor = ReIDExtractor(str(REID_WEIGHTS), DEVICE)
    
    output_folder = TRACKER_OUTPUT_DIR / "advanced_reid_tracker"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    base_path = DATASET_TR23_TEST
    sequences = sorted([
        s for s in os.listdir(base_path) 
        if os.path.isdir(os.path.join(base_path, s))
    ])
    
    for seq_name in sequences:
        print(f"Processing {seq_name}...")
        
        # Carica ROI
        roi_path = Path(base_path) / seq_name / "roi.json"
        roi_data = json.load(open(roi_path)) if roi_path.exists() else None
        
        # Crea tracker per questa sequenza
        tracker = AdvancedReIDTracker()
        
        # YOLO detection senza tracking
        yolo = YOLO(MODEL_PATH / "bestyolo8.pt")
        results_gen = yolo.predict(
            source=os.path.join(base_path, seq_name, 'img1'),
            conf=CONF_THRES,
            iou=0.6,
            imgsz=1280,
            device=0,
            classes=[0],
            stream=True,
            verbose=False
        )
        
        track_file = output_folder / f"tracking_{seq_name}_{TEAM_ID}.txt"
        behav_file = output_folder / f"behavior_{seq_name}_{TEAM_ID}.txt"
        
        with open(track_file, 'w') as f_t, open(behav_file, 'w') as f_b:
            for result in results_gen:
                if result.boxes is None:
                    continue
                
                frame_id = int(os.path.basename(result.path).split('.')[0])
                frame_img = result.orig_img
                img_h, img_w = frame_img.shape[:2]
                
                boxes = result.boxes
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                # Limita numero di detection
                if len(boxes_xyxy) > MAX_ENTITIES:
                    top_indices = np.argsort(confs)[::-1][:MAX_ENTITIES]
                    boxes_xyxy = boxes_xyxy[top_indices]
                    confs = confs[top_indices]
                
                # Estrai embeddings
                embeddings = reid_extractor.get_embeddings(frame_img, boxes_xyxy)
                
                # Tracking
                if len(embeddings) > 0:
                    track_ids = tracker.update(boxes_xyxy, embeddings, confs, frame_img)
                else:
                    track_ids = []
                
                # Scrivi tracking
                for i, tid in enumerate(track_ids):
                    if tid != -1:  # Solo track confermati
                        x1, y1, x2, y2 = boxes_xyxy[i]
                        f_t.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f}\n")
                
                # Scrivi behavior
                roi_counts = count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h)
                f_b.write(f"{frame_id},1,{roi_counts[1]}\n")
                f_b.write(f"{frame_id},2,{roi_counts[2]}\n")
        
        print(f"  ✓ {seq_name} completed")


if __name__ == "__main__":
    run_advanced_tracking()