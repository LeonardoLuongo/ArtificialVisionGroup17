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
CONF_THRES = 0.35           
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

# --- NUOVA CLASSE: Stima Movimento Camera ---
class CameraMotionEstimator:
    def __init__(self):
        self.prev_gray = None
        # Parametri per trovare punti salienti (angoli)
        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        # Parametri per Optical Flow Lucas-Kanade
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def estimate(self, frame):
        """Calcola dx, dy globali del frame rispetto al precedente"""
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy = 0.0, 0.0
        
        if self.prev_gray is not None:
            # Trova punti buoni nel frame precedente
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if p0 is not None:
                # Calcola dove sono finiti nel frame attuale
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None, **self.lk_params)
                
                # Filtra i punti validi
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    
                    if len(good_new) > 0:
                        # Calcola spostamenti
                        shifts = good_new - good_old
                        # La MEDIANA è robusta: ignora i giocatori che corrono e prende lo sfondo
                        dx = np.median(shifts[:, 0])
                        dy = np.median(shifts[:, 1])
        
        self.prev_gray = curr_gray
        return dx, dy

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
        
        # NUOVO: Velocity Tracker (vx, vy)
        self.velocity = np.zeros(2, dtype=np.float32) 
        self.last_centroid = (bbox[:2] + bbox[2:]) / 2.0
    
    # NUOVO: Metodo per compensare il movimento camera
    def compensate_camera_motion(self, dx, dy):
        # Se la camera si sposta a destra (+dx), i pixel si spostano a destra.
        # Aggiorniamo le coordinate memorizzate
        self.last_bbox[0] += dx
        self.last_bbox[2] += dx
        self.last_bbox[1] += dy
        self.last_bbox[3] += dy
        self.last_centroid[0] += dx
        self.last_centroid[1] += dy

    def update(self, embedding, bbox, frame_id, conf, is_collision=False):
        current_centroid = (bbox[:2] + bbox[2:]) / 2.0
        
        # Calcolo Velocità (con semplice smoothing)
        dt = frame_id - self.last_seen_frame
        if dt > 0 and dt < 5: # Se il salto è piccolo, aggiorna velocità
            new_vel = (current_centroid - self.last_centroid) / dt
            # Smoothing factor 0.7 per ridurre il jitter
            self.velocity = 0.7 * self.velocity + 0.3 * new_vel
        
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
        self.last_centroid = current_centroid
        
        if self.state == 'probation':
            self.probation_age += 1
        # --- COLLISION GUARD ---
        # Se c'è collisione, NON aggiornare MAI la gallery.
        # Rischiamo di memorizzare pezzi dell'altro giocatore.
        if is_collision:
            return
        
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

    def predict_location(self, target_frame_id):
        """Predice dove sarà il centroide al frame target"""
        dt = target_frame_id - self.last_seen_frame
        if dt <= 0: return self.last_centroid
        # Predizione lineare: pos = pos + vel * dt
        return self.last_centroid + (self.velocity * dt)

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
        
        # Soglia di ambiguità (gap minimo tra 1° e 2° miglior match)
        self.AMBIGUITY_THRESH = 0.05 
        
        # Inizializziamo il GMC
        self.gmc = CameraMotionEstimator()

    def _detect_collisions(self, boxes):
        """
        Ritorna una mask booleana: True se la box i si sovrappone o è vicinissima a un'altra.
        """
        n = len(boxes)
        collision_mask = np.zeros(n, dtype=bool)
        if n < 2: return collision_mask

        # Calcolo IoU vettorizzato o distanza centri
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        # Distanze a coppie
        dists = cdist(centers, centers)
        
        # Riempie diagonale con infinito per non auto-matchare
        np.fill_diagonal(dists, np.inf)
        
        # Se due centri sono più vicini di 50 pixel, è una collisione/mischia
        # 50px è circa la larghezza media di un giocatore in FHD
        min_dists = np.min(dists, axis=1)
        collision_mask = min_dists < 60 # Aumentato slightly per sicurezza
        
        return collision_mask

    def process_frame(self, botsort_ids, embeddings, boxes, confs, frame_img, frame_id):
        # 1. Stima movimento camera
        cam_dx, cam_dy = self.gmc.estimate(frame_img)
        
        # 2. Aggiorna TUTTI i track (anche quelli persi) con il movimento camera
        for t in self.tracks.values():
            t.compensate_camera_motion(cam_dx, cam_dy)

        current_centroids = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        num_dets = len(botsort_ids)
        
        collision_mask = self._detect_collisions(boxes)
        active_global_ids = set() 
        
        mapped_indices = []
        unmapped_indices = []

        # --- FASE 1: GESTIONE LINK ESISTENTI (BotSORT Priority) ---
        for i in range(num_dets):
            bid = botsort_ids[i]
            is_colliding = collision_mask[i]
            
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
                    track.update(embeddings[i], boxes[i], frame_id, confs[i], is_collision=is_colliding)
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
            if bid not in self.botsort_to_global: continue
            gid = self.botsort_to_global[bid]
            track = self.tracks.get(gid)
            if not track: continue # Garbage collected

            if track.state == 'probation':
                # --- GUARDIA: Se è in mischia, NON prendiamo decisioni ora ---
                # Aspettiamo che esca dalla mischia per capire chi è.
                # Per sapere se è in mischia ora, dovremmo mappare gid -> index -> collision_mask
                # Ma possiamo usare la euristica: se non ha gallery (perché collision), non mergiamo.
                if not track.gallery: 
                    continue

                # AMBIGUITY CHECK LOGIC
                matches = [] # List of (score, track)
                
                if track.gallery:
                    curr_emb = track.gallery[-1]
                    for c_gid, c_track in self.tracks.items():
                        if c_gid == gid or c_gid in active_global_ids or c_track.state != 'locked': continue
                        dt = frame_id - c_track.last_seen_frame
                        if dt > 600: continue
                        
                        # USA PREDIZIONE QUI (Se il track target si sta muovendo, cerchiamo dove sarà ORA)
                        pred_loc = c_track.predict_location(frame_id)
                        
                        if np.linalg.norm(track.last_centroid - pred_loc) < (100 + 40 * dt):
                            sim = c_track.compute_sim(curr_emb)
                            if sim > 0.5: # Considera solo candidati minimamente validi
                                matches.append((sim, c_track))
                
                # Ordina match
                matches.sort(key=lambda x: x[0], reverse=True)
                
                target = -1
                is_ambiguous = False
                
                # Verifica Ambiguità
                if len(matches) >= 2:
                    if (matches[0][0] - matches[1][0]) < self.AMBIGUITY_THRESH:
                        is_ambiguous = True
                
                if not is_ambiguous and matches:
                    best_score, best_track = matches[0]
                    
                    if best_score > self.EARLY_MERGE_THRESH: 
                        target = best_track.global_id
                    elif track.probation_age >= self.PROBATION_FRAMES:
                        if best_score > self.MERGE_THRESH: 
                            target = best_track.global_id
                        else: 
                            track.state = 'locked'
                elif track.probation_age >= self.PROBATION_FRAMES:
                    # Tempo scaduto e situazione ambigua o nessun match -> Promote a nuovo
                    track.state = 'locked'
                
                if target != -1:
                    self.botsort_to_global[bid] = target
                    tgt_t = self.tracks[target]
                    tgt_t.last_seen_frame = track.last_seen_frame
                    tgt_t.last_bbox = track.last_bbox
                    tgt_t.velocity = track.velocity # Ereditiamo la velocità corrente!
                    tgt_t.last_centroid = track.last_centroid
                    
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
            in_collision = collision_mask[i]
            
            # --- SPATIAL INERTIA (Mischia) ---
            # Se è in collisione, ignoriamo Re-ID e cerchiamo match spaziale strettissimo
            if in_collision:
                best_spatial_gid = -1
                min_spatial_dist = float('inf')
                
                # Cerca tra TUTTI i track disponibili
                for gid, track in self.tracks.items():
                    # 1. Non deve essere già in campo (One-to-One globale)
                    if gid in active_global_ids: continue
                    
                    # 2. Filtro Temporale (non recuperare fantasmi di un'ora fa)
                    dt = frame_id - track.last_seen_frame
                    if dt > 5: continue 
                    
                    # Confrontiamo dove siamo ORA con dove il track DOVREBBE essere
                    pred_loc = track.predict_location(frame_id)
                    
                    dist = np.linalg.norm(centroid - pred_loc)
                    if dist < 40 and dist < min_spatial_dist:
                        min_spatial_dist = dist
                        best_spatial_gid = gid
                
                if best_spatial_gid != -1:
                    # RECUPERO SPAZIALE FORZATO (Glue)
                    self.botsort_to_global[bid] = best_spatial_gid
                    recovered_track = self.tracks[best_spatial_gid]
                    recovered_track.update(emb, boxes[i], frame_id, conf, is_collision=True)
                    active_global_ids.add(best_spatial_gid)
                    continue
                else:
                    # In mischia e nessun match spaziale -> Nuovo ID (evita danni)
                    final_new_indices.append(i)
                    continue

            # --- STANDARD VISUAL RECOVERY (Nitido) ---
            if conf > 0.55:
                matches = []
                for gid, track in self.tracks.items():
                    if gid in active_global_ids: continue
                    dt = frame_id - track.last_seen_frame
                    if dt > 900: continue
                    
                    # Filtro spaziale sulla PREDIZIONE
                    pred_loc = track.predict_location(frame_id)
                    dist = np.linalg.norm(centroid - pred_loc)
                    
                    if dist < (100 + 50 * dt):
                        sim = track.compute_sim(emb)
                        if sim > 0.6:
                            matches.append((sim, gid))
                
                matches.sort(key=lambda x: x[0], reverse=True)
                
                # AMBIGUITY CHECK
                is_ambiguous = False
                if len(matches) >= 2:
                    if (matches[0][0] - matches[1][0]) < self.AMBIGUITY_THRESH:
                        is_ambiguous = True

                if not is_ambiguous and matches:
                    best_sim, best_gid = matches[0]
                    if best_sim > self.RECOVERY_THRESH:
                        self.botsort_to_global[bid] = best_gid
                        recovered_track = self.tracks[best_gid]
                        recovered_track.update(emb, boxes[i], frame_id, conf)
                        active_global_ids.add(best_gid)
                        continue 

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
            self.tracks[new_gid] = GlobalTrack(new_gid, emb, boxes[i], frame_id, conf, state='probation')
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
        
        yolo = YOLO(MODEL_PATH / "bestyolo8.pt")
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
                    global_ids = mapper.process_frame(botsort_ids, embeddings, boxes_xyxy, confs, frame_img, frame_id)
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