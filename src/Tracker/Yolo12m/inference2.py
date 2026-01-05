import os
import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import deque
import json
import timm
import torch.nn as nn
# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device
import torchreid

# --- CONFIGURAZIONE ---
CONF_THRES = 0.1           
MAX_ENTITIES = 28           
GALLERY_SIZE = 20           

# Funzione per caricare le ROI e verificare la posizione dei giocatori
def count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h):
    counts = {1: 0, 2: 0} # ROI 1 e ROI 2
    if not roi_data:
        return counts

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Requisito: "center of the basis of the bounding box"
        # Ovvero: (x_centro, y_bottom)
        cx = x1 + (w / 2.0)
        cy = y2 

        for roi_id in [1, 2]:
            roi = roi_data[f"roi{roi_id}"]
            # Convertiamo coordinate relative in assolute
            rx = roi["x"] * img_w
            ry = roi["y"] * img_h
            rw = roi["width"] * img_w
            rh = roi["height"] * img_h
            
            if rx <= cx <= (rx + rw) and ry <= cy <= (ry + rh):
                counts[roi_id] += 1
    return counts

class ReIDExtractor:
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        weights_str = str(weights_path).lower()
        self.resize_wh = (128, 256) # Default OSNet
        
        if 'vit' in weights_str:
            self.resize_wh = (224, 224) 
            if 'l_16' in weights_str: model_name = 'vit_l_16'
            elif 'b_16' in weights_str: model_name = 'vit_b_16'
            else: model_name = 'vit_b_16'
        elif 'ain' in weights_str:
            model_name = 'osnet_ain_x1_0'
        else:
            model_name = 'osnet_x1_0'
            
        print(f"--> ReID Model: {model_name} | Input: {self.resize_wh}")

        try:
            self.model = torchreid.models.build_model(name=model_name, num_classes=1000, pretrained=False)
        except KeyError:
            print(f"ERRORE: Modello {model_name} non supportato.")
            sys.exit(1)

        if weights_path and os.path.exists(weights_path):
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
            
        self.model.to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    @torch.no_grad() 
    def get_embeddings(self, frame, bboxes_xyxy):
        if len(bboxes_xyxy) == 0: return np.array([])
        crops = []
        h_img, w_img = frame.shape[:2]
        
        for box in bboxes_xyxy:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: crop = np.zeros((self.resize_wh[1], self.resize_wh[0], 3), dtype=np.uint8)
            crop = cv2.resize(crop, self.resize_wh)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(np.transpose(crop, (2, 0, 1)))

        input_t = torch.from_numpy(np.array(crops)).float().to(self.device) / 255.0
        input_t = (input_t - self.mean) / self.std
        features = self.model(input_t)
        # --- AGGIUNTA FONDAMENTALE: NORMALIZZAZIONE L2 ---
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy()



class CameraCompensator:
    def __init__(self):
        self.prev_gray = None
        # Parametri per rilevare buoni punti da tracciare (angoli, linee di campo)
        self.feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
        # Parametri per Lucas-Kanade Optical Flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def update_and_get_transform(self, curr_frame):
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return np.eye(2, 3, dtype=np.float32) # Nessun movimento al primo frame

        # 1. Trova punti interessanti nel vecchio frame (angoli, linee)
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        
        if p0 is None:
            self.prev_gray = curr_gray
            return np.eye(2, 3, dtype=np.float32)

        # 2. Calcola dove sono finiti quei punti nel nuovo frame (Optical Flow)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None, **self.lk_params)

        # Filtra solo i punti validi
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 3. Stima la trasformazione Affine (traslazione + zoom + rotazione)
        # Usiamo RANSAC per ignorare outlier (es. giocatori che si muovono diversamente dallo sfondo)
        m = np.eye(2, 3, dtype=np.float32)
        if len(good_new) > 8: # Servono abbastanza punti
            m, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
            if m is None: m = np.eye(2, 3, dtype=np.float32)

        self.prev_gray = curr_gray
        return m

    def apply_transform(self, bbox, transform_matrix):
        # Applica la matrice (rotazione/zoom/traslazione) al centro del bbox
        # bbox è [x1, y1, x2, y2]
        # Trasformiamo i due punti (top-left e bottom-right) per approssimare
        pts = np.array([
            [bbox[0], bbox[1]], # TL
            [bbox[2], bbox[3]]  # BR
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        dst = cv2.transform(pts, transform_matrix)
        dst = dst.reshape(2, 2)
        return np.array([dst[0,0], dst[0,1], dst[1,0], dst[1,1]], dtype=np.float32)

class GlobalTrack:
    def __init__(self, global_id, embedding, bbox, frame_id, conf, state='probation'):
        self.global_id = global_id
        self.gallery = deque(maxlen=30) # Usiamo deque per efficienza
        
        self.state = state 
        self.probation_age = 0
        
        # Gallery Update
        if conf > 0.60:
            self.gallery.append(embedding)
            
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32) # Per stima posizione futura

    def update(self, embedding, bbox, frame_id, conf):
        # Calcolo velocità semplice (bbox center delta)
        curr_center = (bbox[:2] + bbox[2:]) / 2.0
        prev_center = (self.last_bbox[:2] + self.last_bbox[2:]) / 2.0
        
        dt = frame_id - self.last_seen_frame
        if dt > 0 and dt < 10:
            current_vel = (curr_center - prev_center) / dt
            # Media mobile esponenziale per la velocità
            self.velocity = 0.7 * self.velocity + 0.3 * current_vel
            
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
        
        if self.state == 'probation':
            self.probation_age += 1
            if self.probation_age > 10: self.state = 'locked'
        
        if conf > 0.60:
            self.gallery.append(embedding)

    def predict_location(self, target_frame_id):
        # Predizione lineare semplice della posizione
        dt = target_frame_id - self.last_seen_frame
        if dt <= 0: return self.last_bbox
        
        # Prediciamo solo il centro, manteniamo w,h
        center = (self.last_bbox[:2] + self.last_bbox[2:]) / 2.0
        pred_center = center + (self.velocity * dt)
        
        w = self.last_bbox[2] - self.last_bbox[0]
        h = self.last_bbox[3] - self.last_bbox[1]
        
        return np.array([
            pred_center[0] - w/2,
            pred_center[1] - h/2,
            pred_center[0] + w/2,
            pred_center[1] + h/2
        ], dtype=np.float32)

    def compute_distance(self, candidate_emb):
        # Distanza minima coseno rispetto alla gallery (Best Match Strategy)
        if not self.gallery: return 1.0
        gallery_arr = np.array(self.gallery)
        dists = cdist(candidate_emb.reshape(1, -1), gallery_arr, metric='cosine')
        return np.min(dists) # 0 = identico, 1 = diverso

# --- NUOVO ROBUST MAPPER CON MATRICE UNGHERESE ---
class RobustMapper:
    def __init__(self):
        self.botsort_to_global = {} 
        self.tracks = {} 
        self.next_global_id = 1
        self.gmc = CameraCompensator()
        
        # --- TUNING PARAMETRI ---
        self.SPATIAL_GATE_PIXELS = 150  
        self.MAX_TIME_LOST = 600        
        self.MATCH_THRESH_REID = 0.25   
        
    def process_frame(self, botsort_ids, embeddings, boxes, confs, frame_id, current_frame_img):
        # 1. Update Camera Motion
        transform_m = self.gmc.update_and_get_transform(current_frame_img)
        
        # Applica trasformazione ai track esistenti
        for gid, track in self.tracks.items():
            track.last_bbox = self.gmc.apply_transform(track.last_bbox, transform_m)

        # Liste di indici per gestire il matching
        det_indices = list(range(len(botsort_ids)))
        unmatched_dets = []
        
        # --- FASE 1: HARD LINK (BotSORT Continuity) ---
        active_track_ids = set()
        
        # NOTA: Iteriamo basandoci sull'ordine in cui arrivano (spesso ordinati per confidenza nel main)
        for i in det_indices:
            bid = botsort_ids[i]
            
            # Verifichiamo se c'è un link storico
            if bid in self.botsort_to_global:
                gid = self.botsort_to_global[bid]
                
                # CHECK CRUCIALE: L'ID Globale è ancora vivo e NON è già stato assegnato in questo frame?
                if gid in self.tracks and gid not in active_track_ids:
                    # Assegnazione valida
                    self.tracks[gid].update(embeddings[i], boxes[i], frame_id, confs[i])
                    active_track_ids.add(gid)
                
                elif gid in active_track_ids:
                    # DUPLICATO RILEVATO: 
                    # Questo Global ID è già stato preso da un altro BotSort ID in questo frame.
                    # Questa detection (bid) è probabilmente un 'ghost' o un errore di split di BotSORT.
                    # La trattiamo come unmatched: verrà associata a un altro ID o ne creerà uno nuovo.
                    unmatched_dets.append(i)
                    
                else:
                    # Il track era stato cancellato dal garbage collector ma è rimasto nel mapping
                    del self.botsort_to_global[bid]
                    unmatched_dets.append(i)
            else:
                unmatched_dets.append(i)

        # --- FASE 2: RECUPERO (HUNGARIAN ALGORITHM) ---
        candidate_tracks = []
        candidate_gids = [] # Lista parallela per tenere traccia dei GID
        
        for gid, track in self.tracks.items():
            # Consideriamo solo track NON aggiornati nella fase 1
            if gid not in active_track_ids:
                if (frame_id - track.last_seen_frame) < self.MAX_TIME_LOST:
                    candidate_tracks.append(track)
                    candidate_gids.append(gid)
        
        if len(unmatched_dets) > 0 and len(candidate_tracks) > 0:
            cost_matrix = np.ones((len(unmatched_dets), len(candidate_tracks)), dtype=np.float32)
            
            for r, det_idx in enumerate(unmatched_dets):
                det_emb = embeddings[det_idx]
                det_box = boxes[det_idx]
                det_center = (det_box[:2] + det_box[2:]) / 2.0
                
                for c, track in enumerate(candidate_tracks):
                    # 1. Gate Spaziale
                    pred_box = track.predict_location(frame_id)
                    pred_center = (pred_box[:2] + pred_box[2:]) / 2.0
                    
                    dist_px = np.linalg.norm(det_center - pred_center)
                    
                    dt = frame_id - track.last_seen_frame
                    gate_radius = self.SPATIAL_GATE_PIXELS + (10 * dt)
                    
                    if dist_px < gate_radius:
                        # 2. Distanza ReID
                        reid_dist = track.compute_distance(det_emb)
                        cost_matrix[r, c] = reid_dist + (dist_px / 2000.0)
                    else:
                        cost_matrix[r, c] = 1000.0 

            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_inds, col_inds):
                cost = cost_matrix[r, c]
                if cost < self.MATCH_THRESH_REID:
                    det_idx = unmatched_dets[r]
                    track_gid = candidate_gids[c] # Recuperiamo il GID corretto
                    bid = botsort_ids[det_idx]
                    
                    # Link trovato
                    self.botsort_to_global[bid] = track_gid
                    self.tracks[track_gid].update(embeddings[det_idx], boxes[det_idx], frame_id, confs[det_idx])
                    active_track_ids.add(track_gid)
                    
                    unmatched_dets[r] = -1 

        # --- FASE 3: NUOVI TRACK ---
        for i in unmatched_dets:
            if i == -1: continue 
            
            if confs[i] > 0.40:
                bid = botsort_ids[i]
                new_gid = self.next_global_id
                self.next_global_id += 1
                
                self.tracks[new_gid] = GlobalTrack(new_gid, embeddings[i], boxes[i], frame_id, confs[i])
                self.botsort_to_global[bid] = new_gid
                active_track_ids.add(new_gid)

        self._garbage_collection(frame_id)
        
        final_ids = []
        for bid in botsort_ids:
            # Recupera ID, se non esiste (appena scartato o conf bassa) ritorna -1
            gid = self.botsort_to_global.get(bid, -1)
            # Controllo finale di sicurezza: ritorniamo l'ID solo se è attivo (è stato confermato in questo frame)
            if gid in active_track_ids:
                final_ids.append(gid)
            else:
                final_ids.append(-1)
                
        return final_ids

    def _garbage_collection(self, frame_id):
        to_del = []
        for gid, t in self.tracks.items():
            dt = frame_id - t.last_seen_frame
            limit = 1500 if t.state == 'locked' else 60
            if dt > limit:
                to_del.append(gid)
        
        for gid in to_del:
            del self.tracks[gid]
            keys = [k for k, v in self.botsort_to_global.items() if v == gid]
            for k in keys: del self.botsort_to_global[k]


def run_hybrid_tracking():
    TEAM_ID = "17"

    DEVICE = get_best_device()
    REID_WEIGHTS = MODEL_PATH / "re-ids" / "osnet_x1_0.pt" # O osnet_ain_x1_0.pt
    
    reid_extractor = ReIDExtractor(str(REID_WEIGHTS), DEVICE)
    
    output_folder = TRACKER_OUTPUT_DIR / "final_provisional_tracker2"
    output_folder.mkdir(parents=True, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted([s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s))])

    for seq_name in sequences:
        
        print(f"Processing {seq_name}...")

        # Caricamento ROI per il behavior
        roi_path = Path(base_path) / seq_name / "roi.json"
        roi_data = json.load(open(roi_path)) if roi_path.exists() else None
        
        mapper = RobustMapper()
        
        yolo = YOLO(MODEL_PATH /  "bestyolo8.pt")
        results_gen = yolo.track(
            source=os.path.join(base_path, seq_name, 'img1'),
            conf=CONF_THRES,
            iou=0.6,
            imgsz=1280,
            device=0,
            classes=[0],
            tracker=str(TRACKER_CONFIG / "bot_sort.yaml"),
            persist=True,
            stream=True,
            verbose=False
        )

        track_file = output_folder / f"tracking_{seq_name}_{TEAM_ID}.txt"
        behav_file = output_folder / f"behavior_{seq_name}_{TEAM_ID}.txt"
        
        with open(track_file, 'w') as f_t, open(behav_file, 'w') as f_b:
            for result in results_gen:
                if result.boxes is None or result.boxes.id is None:
                    continue
                
                frame_id = int(os.path.basename(result.path).split('.')[0])
                frame_img = result.orig_img
                img_h, img_w = frame_img.shape[:2]

                boxes = result.boxes
                botsort_ids = boxes.id.cpu().numpy().astype(int)
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                if len(botsort_ids) > MAX_ENTITIES:
                    top_indices = np.argsort(confs)[::-1][:MAX_ENTITIES]
                    botsort_ids = botsort_ids[top_indices]
                    boxes_xyxy = boxes_xyxy[top_indices]
                    confs = confs[top_indices]
                
                embeddings = reid_extractor.get_embeddings(frame_img, boxes_xyxy)
                
                if len(embeddings) > 0:
                    global_ids = mapper.process_frame(botsort_ids, embeddings, boxes_xyxy, confs, frame_id, frame_img)
                else:
                    # IMPORTANTE: Anche se non ci sono detection, dobbiamo aggiornare la camera!
                    # Altrimenti la compensazione perde il filo del movimento
                    mapper.gmc.update_and_get_transform(frame_img)
                    global_ids = []

                # --- WRITING TRACKING (6 colonne come richiesto) ---
                for i, gid in enumerate(global_ids):
                    if gid == -1: continue 
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    f_t.write(f"{frame_id},{gid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f}\n")

                # --- PROCESSING & WRITING BEHAVIOR ---
                roi_counts = count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h)
                f_b.write(f"{frame_id},1,{roi_counts[1]}\n")
                f_b.write(f"{frame_id},2,{roi_counts[2]}\n")

        print(f"Sequenza {seq_name} completata.")

if __name__ == "__main__":
    run_hybrid_tracking()