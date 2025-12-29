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
VERIFY_THRESHOLD = 0.45     
MATCH_THRESHOLD = 0.65      
SPATIAL_LIMIT = 100         
GALLERY_SIZE = 20           
# --- NUOVI PARAMETRI PROVISIONAL ---
PROVISIONAL_FRAMES = 1      # K: Quanti frame aspettare prima di decidere
PROVISIONAL_THRESH = 0.70   # Soglia severa per il merge ritardato
# ----------------------

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
        return self.model(input_t).cpu().numpy()


class GlobalTrack:
    def __init__(self, global_id, embedding, bbox, frame_id, conf, state='probation'):
        self.global_id = global_id
        self.gallery = [] 
        self.max_gallery_size = 50 
        
        # Stato: 'probation' (accumulo info) o 'locked' (decisione presa)
        self.state = state 
        self.probation_age = 0  # Contatore frame in probation
        
        # Inizializziamo gallery se decente
        if conf > 0.60:
            self.gallery.append(embedding)
            
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
    
    def update(self, embedding, bbox, frame_id, conf):
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
        
        if self.state == 'probation':
            self.probation_age += 1
        
        # AGGIORNAMENTO GALLERY
        # In probation: Accumuliamo tutto ciò che è nitido per decidere meglio
        # In locked: Accumuliamo solo pose nuove per robustezza futura
        if conf > 0.60:
            if not self.gallery:
                self.gallery.append(embedding)
            else:
                sims = self.compute_sim_vector(embedding)
                max_sim = np.max(sims)
                
                # Se Locked: solo se diverso (0.95)
                # Se Probation: siamo più avidi, vogliamo capire chi è
                thresh = 0.95 if self.state == 'locked' else 0.98
                
                if max_sim < thresh: 
                    if len(self.gallery) >= self.max_gallery_size:
                        self.gallery.pop(0) 
                    self.gallery.append(embedding)

    def compute_sim(self, candidate_emb):
        if not self.gallery: return 0.0 
        gallery_arr = np.array(self.gallery)
        candidate_emb = candidate_emb.reshape(1, -1)
        dists = cdist(candidate_emb, gallery_arr, metric='cosine')
        return 1.0 - np.min(dists)

    def compute_sim_vector(self, candidate_emb):
        if not self.gallery: return np.array([0.0])
        gallery_arr = np.array(self.gallery)
        dists = cdist(candidate_emb.reshape(1, -1), gallery_arr, metric='cosine')
        return 1.0 - dists[0]


class RobustMapper:
    def __init__(self):
        self.botsort_to_global = {} 
        self.tracks = {} 
        self.next_global_id = 1
        
        # CONFIGURAZIONE
        self.PROBATION_FRAMES = 8   
        self.MERGE_THRESH = 0.70    
        self.EARLY_MERGE_THRESH = 0.85 
        
        # SOGLIA RECUPERO IMMEDIATO
        # Deve essere alta. Se recuperiamo l'ID sbagliato qui, facciamo un disastro.
        self.RECOVERY_THRESH = 0.75 

    def process_frame(self, botsort_ids, embeddings, boxes, confs, frame_id):
        current_centroids = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        num_dets = len(botsort_ids)
        
        active_global_ids = set() 
        
        mapped_indices = []
        unmapped_indices = []

        # --- FASE 1: GESTIONE LINK ESISTENTI (BotSORT Priority) ---
        for i in range(num_dets):
            bid = botsort_ids[i]
            
            if bid in self.botsort_to_global:
                gid = self.botsort_to_global[bid]
                
                # Check Duplicati (One-to-One)
                if gid in active_global_ids:
                    del self.botsort_to_global[bid]
                    unmapped_indices.append(i)
                    continue 

                track = self.tracks.get(gid)
                if track:
                    active_global_ids.add(gid)
                    track.update(embeddings[i], current_centroids[i], frame_id, confs[i])
                else:
                    del self.botsort_to_global[bid]
                    unmapped_indices.append(i)
            else:
                unmapped_indices.append(i)

        # --- FASE 2: VALUTAZIONE PROBATION (Il cuore della logica) ---
        # Controlliamo se qualche track in probation è pronto per essere promosso o mergiato
        
        # Copiamo le chiavi per poter modificare il dizionario durante il loop
        current_bids = list(self.botsort_to_global.keys())
        
        for bid in current_bids:
            gid = self.botsort_to_global[bid]
            track = self.tracks.get(gid)
            if not track: continue # Garbage collected

            if track.state == 'probation':
                # CERCHIAMO MATCH CON I DORMANT (Vecchi ID non attivi ora)
                best_match = None
                max_sim = -1.0
                if track.gallery:
                    curr_emb = track.gallery[-1]
                    for c_gid, c_track in self.tracks.items():
                        if c_gid == gid or c_gid in active_global_ids or c_track.state != 'locked': continue
                        dt = frame_id - c_track.last_seen_frame
                        if dt > 600: continue
                        if np.linalg.norm(track.last_bbox - c_track.last_bbox) < (100 + 40 * dt):
                            sim = c_track.compute_sim(curr_emb)
                            if sim > max_sim: max_sim = sim; best_match = c_track
                
                target = -1
                if max_sim > self.EARLY_MERGE_THRESH: target = best_match.global_id
                elif track.probation_age >= self.PROBATION_FRAMES:
                    if max_sim > self.MERGE_THRESH: target = best_match.global_id
                    else: track.state = 'locked'
                
                if target != -1:
                    self.botsort_to_global[bid] = target
                    tgt_t = self.tracks[target]
                    tgt_t.last_seen_frame = track.last_seen_frame
                    tgt_t.last_bbox = track.last_bbox
                    tgt_t.gallery.extend(track.gallery)
                    if len(tgt_t.gallery) > tgt_t.max_gallery_size: tgt_t.gallery = tgt_t.gallery[-tgt_t.max_gallery_size:]
                    del self.tracks[gid]
                    if gid in active_global_ids: active_global_ids.remove(gid)
                    active_global_ids.add(target)

        # --- FASE 2.5: RECUPERO IMMEDIATO (La tua richiesta) ---
        # Prima di creare nuovi ID, cerchiamo tra i 'Dormant' (quelli non attivi ora)
        
        final_new_indices = [] # Quelli che proprio non troviamo
        
        # Ordiniamo gli unmapped per confidence (i migliori scelgono per primi)
        # Così se abbiamo un detection buono e uno sfocato, quello buono si prende l'ID giusto.
        unmapped_indices.sort(key=lambda idx: confs[idx], reverse=True)

        for i in unmapped_indices:
            emb = embeddings[i]
            conf = confs[i]
            centroid = current_centroids[i]
            bid = botsort_ids[i]
            
            # Recuperiamo solo se l'immagine è decente
            if conf > 0.55:
                best_gid = -1
                best_sim = -1.0
                
                # Cerca tra TUTTI i track disponibili
                for gid, track in self.tracks.items():
                    # 1. Non deve essere già in campo (One-to-One globale)
                    if gid in active_global_ids: continue
                    
                    # 2. Filtro Temporale (non recuperare fantasmi di un'ora fa)
                    dt = frame_id - track.last_seen_frame
                    if dt > 900: continue # 30 secondi
                    
                    # 3. Filtro Spaziale Generoso (Recupero)
                    # Velocità max stimata: 50 pixel/frame (sprint + camera pan)
                    dist = np.linalg.norm(centroid - track.last_bbox)
                    if dist < (100 + 50 * dt):
                        sim = track.compute_sim(emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_gid = gid
                
                # CHECK SOGLIA
                if best_sim > self.RECOVERY_THRESH:
                    # TROVATO! È LUI!
                    self.botsort_to_global[bid] = best_gid
                    
                    # Aggiorniamo subito il track
                    recovered_track = self.tracks[best_gid]
                    recovered_track.update(emb, centroid, frame_id, conf)
                    
                    # Se era probation, lo teniamo probation. Se era locked, resta locked.
                    # Ma lo segniamo come ATTIVO così nessuno altro può prenderlo in questo frame
                    active_global_ids.add(best_gid)
                    continue # Passa al prossimo unmapped

            # Se siamo qui, non abbiamo trovato match validi
            final_new_indices.append(i)

        # --- FASE 3: NUOVI ID (Solo per chi non ha trovato casa) ---
        for i in final_new_indices:
            bid = botsort_ids[i]
            emb = embeddings[i]
            conf = confs[i]
            centroid = current_centroids[i]
            
            new_gid = self.next_global_id
            self.next_global_id += 1
            
            # Nasce sempre in Probation
            self.tracks[new_gid] = GlobalTrack(new_gid, emb, centroid, frame_id, conf, state='probation')
            self.botsort_to_global[bid] = new_gid
            
            # Aggiungiamo ai globali attivi per correttezza (anche se loop finito)
            active_global_ids.add(new_gid)

        # --- OUTPUT & CLEANUP ---
        final_ids = []
        for bid in botsort_ids:
            final_ids.append(self.botsort_to_global.get(bid, -1))
        
        self._garbage_collection(frame_id)
        return final_ids

    def _garbage_collection(self, frame_id):
        to_del = []
        for gid, t in self.tracks.items():
            # Teniamo i Locked molto a lungo per permettere il recupero (Fase 2.5)
            timeout = 1500 if t.state == 'locked' else 60 
            if (frame_id - t.last_seen_frame) > timeout:
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
    
    output_folder = TRACKER_OUTPUT_DIR / "final_provisional_tracker"
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
                    global_ids = mapper.process_frame(botsort_ids, embeddings, boxes_xyxy, confs, frame_id)
                else:
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