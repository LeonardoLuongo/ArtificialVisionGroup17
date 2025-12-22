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
import time
import json

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device
import torchreid

# --- CONFIGURAZIONE AVANZATA ---
CONF_THRES = 0.20           # Soglia detection
MAX_ENTITIES = 28           # Closed Set (max giocatori in campo + arbitri)
VERIFY_THRESHOLD = 0.35     # (Basso) Se scendi sotto questo, BotSORT ha sbagliato -> Spezzo il link
MATCH_THRESHOLD = 0.65      # (Alto) Per unire due tracce spezzate serve molta sicurezza
SPATIAL_LIMIT = 300         # Pixel massimi di spostamento per frame (velocity gating)
GALLERY_SIZE = 20           # Quanti embeddings ricordare per ogni identità (NON usiamo la media)
# -------------------------------

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
        
        # --- 1. Rilevamento Architettura e Risoluzione ---
        self.resize_wh = (128, 256) # Default (Width, Height) per OSNet
        
        if 'vit' in weights_str:
            # I ViT solitamente richiedono input quadrati per i Positional Embeddings
            self.resize_wh = (224, 224) 
            if 'l_16' in weights_str:
                model_name = 'vit_l_16'  # Large
            elif 'b_16' in weights_str:
                model_name = 'vit_b_16'  # Base
            else:
                model_name = 'vit_b_16'  # Fallback
        elif 'ain' in weights_str:
            model_name = 'osnet_ain_x1_0'
        else:
            model_name = 'osnet_x1_0'
            
        print(f"--> File pesi: {os.path.basename(weights_str)}")
        print(f"--> Architettura rilevata: {model_name}")
        print(f"--> Input Resolution (WxH): {self.resize_wh}")

        # --- 2. Costruzione Modello ---
        try:
            self.model = torchreid.models.build_model(
                name=model_name, 
                num_classes=1000, 
                pretrained=False 
            )
        except KeyError:
            print(f"ERRORE CRITICO: Il modello '{model_name}' non è nel registry di torchreid.")
            sys.exit(1)

        # --- 3. Caricamento Pesi ---
        if weights_path and os.path.exists(weights_path):
            try:
                torchreid.utils.load_pretrained_weights(self.model, weights_path)
                print("--> Pesi caricati CORRETTAMENTE (Nessun layer scartato).")
            except Exception as e:
                print(f"ERRORE nel caricamento pesi: {e}")
        else:
            print(f"ATTENZIONE: Pesi non trovati in {weights_path}")
            
        self.model.to(device).eval()
        
        # Pre-allocazione tensori normalizzazione per velocità
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
            
            # Gestione crop vuoti/invalidi
            if crop.size == 0: 
                # Crea immagine nera delle dimensioni target
                crop = np.zeros((self.resize_wh[1], self.resize_wh[0], 3), dtype=np.uint8)
            
            # RESIZE DINAMICO (224x224 per ViT, 128x256 per OSNet)
            crop = cv2.resize(crop, self.resize_wh)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(np.transpose(crop, (2, 0, 1)))

        if not crops: return np.array([])

        input_t = torch.from_numpy(np.array(crops)).float().to(self.device) / 255.0
        input_t = (input_t - self.mean) / self.std
        
        return self.model(input_t).cpu().numpy()



class GlobalTrack:
    """Rappresenta un'identità unica (Giocatore X)"""
    def __init__(self, global_id, embedding, bbox, frame_id):
        self.global_id = global_id
        # Problema 2: Non usiamo la media, ma una deque di campioni
        self.gallery = deque(maxlen=GALLERY_SIZE)
        self.gallery.append(embedding)
        
        self.last_seen_frame = frame_id
        self.last_bbox = bbox # (cx, cy)
    
    def update(self, embedding, bbox, frame_id, conf):
        # Problema 4: Aggiornamento selettivo
        if conf > 0.6: 
            self.gallery.append(embedding)
        self.last_seen_frame = frame_id
        self.last_bbox = bbox

    def compute_sim(self, candidate_emb):
        """Confronta il candidato con TUTTA la gallery e prende il miglior match (Best Link)"""
        gallery_arr = np.array(self.gallery)
        # Cosine distance: 0=uguale, 1=diverso. Sim = 1 - dist
        dists = cdist([candidate_emb], gallery_arr, metric='cosine')[0]
        min_dist = np.min(dists)
        return 1.0 - min_dist

class RobustMapper:
    def __init__(self):
        self.botsort_to_global = {} # Mappa ID Volatile -> ID Stabile
        self.tracks = {}            # ID Stabile -> Oggetto GlobalTrack
        self.next_global_id = 1

    def _get_centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def process_frame(self, botsort_ids, embeddings, boxes, confs, frame_id):
        current_centroids = [self._get_centroid(b) for b in boxes]
        
        # Liste per gestire i due gruppi
        verified_indices = []  # Indices in input arrays
        unassigned_indices = [] # Indices in input arrays
        
        active_global_ids = set() # Per evitare duplicati nello stesso frame

        # --- FASE 1: VERIFICA LINK ESISTENTI (Problema 1: Fiducia condizionata) ---
        for i, bid in enumerate(botsort_ids):
            is_verified = False
            
            if bid in self.botsort_to_global:
                gid = self.botsort_to_global[bid]
                
                # Se l'abbiamo già usato in questo frame (errore BotSORT duplicazione ID), abortiamo
                if gid not in active_global_ids:
                    track = self.tracks.get(gid)
                    if track:
                        # Check Visivo
                        sim = track.compute_sim(embeddings[i])
                        if sim > VERIFY_THRESHOLD:
                            is_verified = True
                            track.update(embeddings[i], current_centroids[i], frame_id, confs[i])
                            active_global_ids.add(gid)
                            verified_indices.append(i)
                        else:
                            # LINK SPEZZATO: BotSORT ha mantenuto ID, ma sembra un altro.
                            # Rimuoviamo il mapping e lo trattiamo come nuovo
                            del self.botsort_to_global[bid]
            
            if not is_verified:
                unassigned_indices.append(i)

        # --- FASE 2: RECUPERO IDENTITÀ (Problema 3 e 5) ---
        # Cerchiamo di matchare gli unassigned con i track "dormienti" (non attivi in questo frame)
        
        if len(unassigned_indices) > 0:
            candidate_indices = unassigned_indices
            # Tracks disponibili: quelli non usati in questo frame e non troppo vecchi
            dormant_tracks = [t for gid, t in self.tracks.items() 
                              if gid not in active_global_ids 
                              and (frame_id - t.last_seen_frame) < 150]
            
            if len(dormant_tracks) > 0:
                cost_matrix = np.ones((len(candidate_indices), len(dormant_tracks))) * 100.0
                
                for r, idx in enumerate(candidate_indices):
                    cand_emb = embeddings[idx]
                    cand_cen = current_centroids[idx]
                    
                    for c, track in enumerate(dormant_tracks):
                        # A. Spatial Gating (Problema 3)
                        # Calcoliamo distanza pixel per frame
                        delta_frames = frame_id - track.last_seen_frame
                        max_allowed_dist = SPATIAL_LIMIT * delta_frames
                        
                        dist_spatial = np.linalg.norm(np.array(cand_cen) - np.array(track.last_bbox))
                        
                        if dist_spatial < max_allowed_dist:
                            # B. Visual Matching
                            sim = track.compute_sim(cand_emb)
                            if sim > MATCH_THRESHOLD:
                                cost_matrix[r, c] = 1.0 - sim # Costo basso = alta sim

                # Hungarian Algo
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                assigned_in_this_step = set()
                
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < (1.0 - MATCH_THRESHOLD):
                        # MATCH TROVATO
                        real_idx = candidate_indices[r]
                        track = dormant_tracks[c]
                        bid = botsort_ids[real_idx]
                        
                        # Creiamo/Ripristiniamo link
                        self.botsort_to_global[bid] = track.global_id
                        track.update(embeddings[real_idx], current_centroids[real_idx], frame_id, confs[real_idx])
                        
                        active_global_ids.add(track.global_id)
                        assigned_in_this_step.add(real_idx)
                
                # Aggiorniamo gli unassigned togliendo quelli matchati
                unassigned_indices = [idx for idx in unassigned_indices if idx not in assigned_in_this_step]

        # --- FASE 3: CREAZIONE NUOVI ID ---
        # Quelli rimasti sono veramente nuovi
        for idx in unassigned_indices:
            bid = botsort_ids[idx]
            new_gid = self.next_global_id
            self.next_global_id += 1
            
            self.botsort_to_global[bid] = new_gid
            self.tracks[new_gid] = GlobalTrack(new_gid, embeddings[idx], current_centroids[idx], frame_id)

        # --- FASE 4: OUTPUT ---
        # Costruiamo la lista di ID finali rispettando l'ordine originale
        final_ids = []
        for i in range(len(botsort_ids)):
            bid = botsort_ids[i]
            if bid in self.botsort_to_global:
                final_ids.append(self.botsort_to_global[bid])
            else:
                # Fallback estremo (non dovrebbe accadere con la logica sopra)
                final_ids.append(-1)
                
        # --- FASE 5: CLEANUP (Garbage Collection) ---
        to_del = [gid for gid, t in self.tracks.items() if (frame_id - t.last_seen_frame) > 300]
        for gid in to_del:
            del self.tracks[gid]
            # Pulisci anche le chiavi botsort inverse
            keys_to_remove = [k for k, v in self.botsort_to_global.items() if v == gid]
            for k in keys_to_remove:
                del self.botsort_to_global[k]

        return final_ids

def run_hybrid_tracking():
    DEVICE = get_best_device()
    REID_WEIGHTS = MODEL_PATH / "re-ids" / "osnet_x1_0.pt"
    TEAM_ID = "17"
    
    # Inizializza Modelli
    reid_extractor = ReIDExtractor(str(REID_WEIGHTS), DEVICE)
    yolo = YOLO(MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / "best.pt")
    
    output_folder = TRACKER_OUTPUT_DIR / "final_robust_tracker"
    output_folder.mkdir(parents=True, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted([s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s))])

    for seq_name in sequences:
        start = time.time()
        print(f"Processing {seq_name}...")

        # Caricamento ROI per il behavior
        roi_path = Path(base_path) / seq_name / "roi.json"
        roi_data = json.load(open(roi_path)) if roi_path.exists() else None
        
        # Reset Mapper per ogni sequenza
        mapper = RobustMapper()
        
        # Generatore BotSORT
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
                
                # Estrai Dati Grezzi
                boxes = result.boxes
                botsort_ids = boxes.id.cpu().numpy().astype(int)
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                # --- PRE-FILTERING (Closed Set) ---
                if len(botsort_ids) > MAX_ENTITIES:
                    top_indices = np.argsort(confs)[::-1][:MAX_ENTITIES]
                    botsort_ids = botsort_ids[top_indices]
                    boxes_xyxy = boxes_xyxy[top_indices]
                    confs = confs[top_indices]
                
                # --- PROCESSING ---
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

        end = time.time()
        elapsed = end - start
        print(f"Elapsed time: {elapsed:.3f} seconds")
        print(f"Sequenza {seq_name} completata.")

if __name__ == "__main__":
    run_hybrid_tracking()