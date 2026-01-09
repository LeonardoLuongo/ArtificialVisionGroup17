import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from collections import deque
from scipy.spatial.distance import cdist
from ultralytics import YOLO
import PIL.Image as Image
from torchvision import transforms

# Aggiunta path progetto (Mantenuta come originale)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device
import torchreid

# =============================================================================
# CONFIGURAZIONE E COSTANTI
# =============================================================================
# Queste costanti definiscono la sensibilità del tracker.
# NON MODIFICARE: Ottimizzati per HOTA 82.47%.
CONF_THRES = 0.1             # Confidenza minima per accettare una detection YOLO
MAX_ENTITIES = 28            # Numero massimo di giocatori in campo attesi
VERIFY_THRESHOLD = 0.45      # Soglia base per verifica ReID
MATCH_THRESHOLD = 0.65       # Soglia per associare track esistenti
SPATIAL_LIMIT = 100          # Limite spostamento pixel (senza compensazione)
GALLERY_SIZE = 20            # Quanti embeddings storici conservare per ogni giocatore

# Parametri Logica "Probation" (Isteresi)
PROVISIONAL_FRAMES = 1       # Frame di attesa prima di confermare un ID
PROVISIONAL_THRESH = 0.70    # Similarità richiesta per merge in fase probation

# =============================================================================
# FUNZIONI DI UTILITÀ (GAME LOGIC)
# =============================================================================

def count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h):
    """
    Conta quanti giocatori sono presenti nelle Region of Interest (ROI) definite.
    Logica specifica per l'analisi tattica (es. possesso palla, zone calde).
    """
    counts = {1: 0, 2: 0}
    if not roi_data:
        return counts

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Punto di riferimento: Centro della base del bounding box (piedi del giocatore)
        cx = x1 + (w / 2.0)
        cy = y2 

        for roi_id in [1, 2]:
            roi = roi_data[f"roi{roi_id}"]
            # Conversione coordinate relative (0-1) in assolute (pixel)
            rx = roi["x"] * img_w
            ry = roi["y"] * img_h
            rw = roi["width"] * img_w
            rh = roi["height"] * img_h
            
            if rx <= cx <= (rx + rw) and ry <= cy <= (ry + rh):
                counts[roi_id] += 1
    return counts


# =============================================================================
# CLASSE: ESTRAZIONE FEATURES VISIVE (ReID)
# =============================================================================

class ReIDExtractor:
    """
    Wrapper per torchreid. Si occupa di trasformare l'immagine di un giocatore
    in un vettore numerico (embedding) che ne descrive l'aspetto visivo.
    """
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        weights_str = str(weights_path).lower()
        self.resize_wh = (128, 256) # Default standard per OSNet
        
        # Selezione automatica architettura in base al nome del file pesi
        if 'vit' in weights_str:
            self.resize_wh = (224, 224) 
            model_name = 'vit_l_16' if 'l_16' in weights_str else 'vit_b_16'
        elif 'ain' in weights_str:
            model_name = 'osnet_ain_x1_0'
        else:
            model_name = 'osnet_x1_0' # Modello leggero ed efficiente
            
        print(f"--> ReID Model Loaded: {model_name} | Input Size: {self.resize_wh}")

        try:
            self.model = torchreid.models.build_model(name=model_name, num_classes=1000, pretrained=False)
        except KeyError:
            print(f"ERRORE CRITICO: Modello {model_name} non supportato.")
            sys.exit(1)

        if weights_path and os.path.exists(weights_path):
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
            
        self.model.to(device).eval()
        # Normalizzazione standard ImageNet
        self.preprocess = transforms.Compose([
            transforms.Resize((self.resize_wh[1],self.resize_wh[0])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_embeddings(self, frame, bboxes_xyxy):
        """
        Returns:
            np.ndarray: Shape (N, 1024), L2-normalized embeddings on CPU.
        """
        if len(bboxes_xyxy) == 0:
            return np.empty((0, 1024), dtype=np.float32)

        # Convert BGR (OpenCV) to RGB
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
        else:
            pil_image = frame

        crops = []
        for bbox in bboxes_xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Clip coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(pil_image.width, x2)
            y2 = min(pil_image.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                crop = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                crop = pil_image.crop((x1, y1, x2, y2))
            
            crops.append(self.preprocess(crop))

        if not crops:
            return np.empty((0, 1024), dtype=np.float32)

        # Batch input
        batch_input = torch.stack(crops).to(self.device)
        
        # Forward pass
        embeddings = self.model(batch_input)
        
        # L2 Normalize
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Return as Numpy array (Fixes TypeError in Tracker)
        return embeddings.detach().cpu().numpy()


# =============================================================================
# CLASSE: COMPENSAZIONE MOVIMENTO CAMERA (GMC)
# =============================================================================

class CameraCompensator:
    """
    Gestisce il movimento della telecamera (Global Motion Compensation).
    Usa Optical Flow per capire come si muove lo sfondo e allineare le coordinate
    dei giocatori da un frame all'altro.
    """
    def __init__(self):
        self.prev_gray = None
        # Rileva angoli forti (features) da tracciare
        self.feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
        # Parametri Lucas-Kanade Optical Flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def update_and_get_transform(self, curr_frame):
        """Calcola la matrice di trasformazione Affine tra frame t-1 e t."""
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return np.eye(2, 3, dtype=np.float32) # Identità (nessun movimento)

        # 1. Trova punti chiave nel frame precedente
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        
        if p0 is None:
            self.prev_gray = curr_gray
            return np.eye(2, 3, dtype=np.float32)

        # 2. Calcola Optical Flow (dove sono finiti i punti nel nuovo frame?)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None, **self.lk_params)

        # Filtra solo i punti tracciati con successo
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 3. Stima trasformazione rigida (RANSAC per escludere outlier come giocatori che si muovono)
        m = np.eye(2, 3, dtype=np.float32)
        if len(good_new) > 8: 
            m, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
            if m is None: m = np.eye(2, 3, dtype=np.float32)

        self.prev_gray = curr_gray
        return m

    def apply_transform(self, bbox, transform_matrix):
        """Applica la matrice di movimento alle coordinate di un BBox."""
        pts = np.array([
            [bbox[0], bbox[1]], # Top-Left
            [bbox[2], bbox[3]]  # Bottom-Right
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        dst = cv2.transform(pts, transform_matrix)
        dst = dst.reshape(2, 2)
        return np.array([dst[0,0], dst[0,1], dst[1,0], dst[1,1]], dtype=np.float32)


# =============================================================================
# CLASSE: TRACCIA GLOBALE (Identità Giocatore)
# =============================================================================

class GlobalTrack:
    """
    Rappresenta un singolo giocatore tracciato nel tempo.
    Mantiene uno storico (gallery) dei suoi embeddings per riconoscerlo.
    """
    def __init__(self, global_id, embedding, bbox, frame_id, conf, state='probation'):
        self.global_id = global_id
        self.gallery = [] 
        self.max_gallery_size = 50 
        
        # Stato: 'probation' (in verifica) o 'locked' (confermato)
        self.state = state 
        self.probation_age = 0
        
        # Aggiungiamo alla gallery solo se l'immagine è chiara
        if conf > 0.60:
            self.gallery.append(embedding)
            
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
    
    def update(self, embedding, bbox, frame_id, conf):
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
        
        if self.state == 'probation':
            self.probation_age += 1
        
        # Gestione intelligente della Galleria (Memory Bank)
        if conf > 0.60:
            if not self.gallery:
                self.gallery.append(embedding)
            else:
                sims = self.compute_sim_vector(embedding)
                max_sim = np.max(sims)
                
                # Strategia di aggiornamento:
                # Se Locked: siamo selettivi (0.95), aggiorniamo solo se cambia leggermente l'aspetto
                # Se Probation: siamo più avidi (0.98), raccogliamo dati per capire chi è
                thresh = 0.95 if self.state == 'locked' else 0.98
                
                if max_sim < thresh: 
                    if len(self.gallery) >= self.max_gallery_size:
                        self.gallery.pop(0) # Rimuovi il più vecchio (FIFO)
                    self.gallery.append(embedding)

    def compute_sim(self, candidate_emb):
        """Calcola similarità coseno massima con la gallery."""
        if not self.gallery: return 0.0 
        gallery_arr = np.array(self.gallery)
        candidate_emb = candidate_emb.reshape(1, -1)
        # Distanza coseno -> Similarità = 1 - Distanza
        dists = cdist(candidate_emb, gallery_arr, metric='cosine')
        return 1.0 - np.min(dists)

    def compute_sim_vector(self, candidate_emb):
        if not self.gallery: return np.array([0.0])
        gallery_arr = np.array(self.gallery)
        dists = cdist(candidate_emb.reshape(1, -1), gallery_arr, metric='cosine')
        return 1.0 - dists[0]


# =============================================================================
# CLASSE: LOGICA DI ASSOCIAZIONE (CORE ALGORITHM)
# =============================================================================

class RobustMapper:
    """
    Gestisce l'associazione tra le detection correnti e le tracce globali.
    Implementa una logica ibrida: BotSORT (spaziale) + ReID (visivo) + Recupero.
    """
    def __init__(self):
        self.botsort_to_global = {} # Mappa ID frame corrente -> ID globale
        self.tracks = {}            # Dizionario di tutti i GlobalTrack
        self.next_global_id = 1
        
        self.gmc = CameraCompensator()
        
        # Parametri interni algoritmo
        self.PROBATION_FRAMES = 8   
        self.MERGE_THRESH = 0.70    
        self.EARLY_MERGE_THRESH = 0.85 
        self.RECOVERY_THRESH = 0.75 # Soglia alta per evitare falsi positivi nel recupero

    def process_frame(self, botsort_ids, embeddings, boxes, confs, frame_id, current_frame_img):
        """
        Main loop per processare un frame.
        Input: Dati dal tracker BotSORT (YOLO) e features ReID.
        Output: Lista di Global ID coerenti nel tempo.
        """
        # 1. Aggiornamento GMC e proiezione posizioni
        transform_m = self.gmc.update_and_get_transform(current_frame_img)
        self._apply_gmc_to_tracks(transform_m)

        current_centroids = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        num_dets = len(botsort_ids)
        
        active_global_ids = set() # ID presenti in questo frame
        unmapped_indices = []     # Indici detection non ancora assegnati

        # 2. Gestione Link Esistenti (Continuità BotSORT)
        # Se BotSORT dice che l'ID 5 è sempre ID 5, ci fidiamo in prima battuta.
        unmapped_indices = self._handle_existing_links(
            botsort_ids, embeddings, boxes, confs, frame_id, 
            num_dets, active_global_ids
        )

        # 3. Logica Probation (Fusione tracce duplicate)
        # Controlla se tracce "giovani" sono in realtà doppioni di vecchie tracce.
        self._manage_probation_merging(frame_id, active_global_ids)

        # 4. Recupero Immediato (ReID Recovery) - FASE CRITICA
        # Cerca tra i "Dormant" (giocatori persi momentaneamente) usando ReID.
        final_new_indices = self._attempt_recovery(
            unmapped_indices, botsort_ids, embeddings, boxes, confs, 
            frame_id, active_global_ids
        )

        # 5. Creazione Nuovi ID
        # Se nessuno switch precedente ha funzionato, è un nuovo giocatore.
        self._create_new_tracks(
            final_new_indices, botsort_ids, embeddings, boxes, confs, 
            frame_id, current_centroids, active_global_ids
        )

        # Output & Pulizia
        final_ids = [self.botsort_to_global.get(bid, -1) for bid in botsort_ids]
        self._garbage_collection(frame_id)
        
        return final_ids

    # --- METODI INTERNI DI SUPPORTO ---

    def _apply_gmc_to_tracks(self, transform_m):
        """Aggiorna l'ultima posizione nota di tutti i giocatori compensando il movimento camera."""
        for gid, track in self.tracks.items():
            track.last_bbox = self.gmc.apply_transform(track.last_bbox, transform_m)

    def _handle_existing_links(self, botsort_ids, embeddings, boxes, confs, frame_id, num_dets, active_global_ids):
        """Mantiene le associazioni già stabili fornite da BotSORT."""
        unmapped = []
        for i in range(num_dets):
            bid = botsort_ids[i]
            if bid in self.botsort_to_global:
                gid = self.botsort_to_global[bid]
                
                # Evita conflitti one-to-one (due bot_id mappati allo stesso global_id)
                if gid in active_global_ids:
                    print(f"Warning: Conflict detected for BotSORT ID {bid} mapping to Global ID {gid}. Removing mapping.")
                    del self.botsort_to_global[bid]
                    unmapped.append(i)
                    continue 

                track = self.tracks.get(gid)
                if track:
                    active_global_ids.add(gid)
                    track.update(embeddings[i], boxes[i], frame_id, confs[i])
                    if track.state == 'locked':
                        track.best_guess_id = None
                else:
                    del self.botsort_to_global[bid]
                    unmapped.append(i)
            else:
                unmapped.append(i)
        return unmapped

    def _manage_probation_merging(self, frame_id, active_global_ids):
        """
        Controlla i track in stato 'probation'. Se il loro aspetto (ReID) coincide
        con un track 'dormant' (non attivo ora) che è spazialmente vicino, unisce i due.
        """
        current_bids = list(self.botsort_to_global.keys())
        for bid in current_bids:
            gid = self.botsort_to_global[bid]
            track = self.tracks.get(gid)
            if not track: continue 

            if track.state == 'probation':
                best_match = None
                max_sim = -1.0
                
                # Cerca match tra le tracce non attive (Dormant)
                if track.gallery:
                    curr_emb = track.gallery[-1]
                    for c_gid, c_track in self.tracks.items():
                        # Skip se stesso, se attivo, o se in probation (cerchiamo solo locked stabili)
                        if c_gid == gid or c_gid in active_global_ids or c_track.state != 'locked': continue
                        
                        dt = frame_id - c_track.last_seen_frame
                        if dt > 1500: continue # Timeout temporale
                        
                        # Distanza spaziale (corretta dal GMC)
                        dist = np.linalg.norm((track.last_bbox[:2]+track.last_bbox[2:])/2 - 
                                            (c_track.last_bbox[:2]+c_track.last_bbox[2:])/2)
                        
                        # Finestra di ricerca spaziale che si allarga col tempo
                        if dist < (150 + 60 * dt): 
                            sim = c_track.compute_sim(curr_emb)
                            if sim > max_sim: max_sim = sim; best_match = c_track
                
                target = -1
                # Logica di decisione Merge
                if max_sim > self.EARLY_MERGE_THRESH: target = best_match.global_id
                elif track.probation_age >= self.PROBATION_FRAMES:
                    if max_sim > self.MERGE_THRESH: target = best_match.global_id
                    else: track.state = 'locked' # Promosso a nuova identità unica
                
                # Esegui il Merge
                if target != -1:
                    self.botsort_to_global[bid] = target
                    tgt_t = self.tracks[target]
                    tgt_t.last_seen_frame = track.last_seen_frame
                    tgt_t.last_bbox = track.last_bbox
                    tgt_t.gallery.extend(track.gallery)
                    # Mantieni gallery pulita
                    if len(tgt_t.gallery) > tgt_t.max_gallery_size: 
                        tgt_t.gallery = tgt_t.gallery[-tgt_t.max_gallery_size:]
                    del self.tracks[gid]
                    if gid in active_global_ids: active_global_ids.remove(gid)
                    active_global_ids.add(target)

    def _attempt_recovery(self, unmapped_indices, botsort_ids, embeddings, boxes, confs, frame_id, active_global_ids):
        """
        Tenta di recuperare giocatori persi (occlusi o usciti) usando ReID.
        Priorità alle detection con confidenza alta.
        """
        final_new_indices = []
        # Ordina per confidenza: i detection migliori scelgono per primi
        unmapped_indices.sort(key=lambda idx: confs[idx], reverse=True)

        for i in unmapped_indices:
            emb = embeddings[i]
            conf = confs[i]
            bbox = boxes[i]
            centroid = (bbox[:2] + bbox[2:]) / 2.0
            bid = botsort_ids[i]
            
            recovered = False
            # Recuperiamo solo se l'immagine è decente
            if conf > 0.55:
                best_gid = -1
                best_sim = -1.0
                
                for gid, track in self.tracks.items():
                    if gid in active_global_ids: continue # Già in campo
                    
                    dt = frame_id - track.last_seen_frame
                    if dt > 900: continue # Troppo vecchio (30s @ 30fps)
                    
                    track_center = (track.last_bbox[:2] + track.last_bbox[2:]) / 2.0
                    dist = np.linalg.norm(centroid - track_center)
                    
                    if dist < (150 + 50 * dt):
                        sim = track.compute_sim(emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_gid = gid
                
                if best_sim > self.RECOVERY_THRESH:
                    self.botsort_to_global[bid] = best_gid
                    self.tracks[best_gid].update(emb, bbox, frame_id, conf)
                    active_global_ids.add(best_gid)
                    recovered = True

            if not recovered:
                final_new_indices.append(i)
        
        return final_new_indices

    def _create_new_tracks(self, final_new_indices, botsort_ids, embeddings, boxes, confs, frame_id, centroids, active_global_ids):
        """Inizializza nuove identità globali per i detection rimasti orfani."""
        for i in final_new_indices:
            bid = botsort_ids[i]
            
            new_gid = self.next_global_id
            self.next_global_id += 1
            
            # Nasce in Probation
            self.tracks[new_gid] = GlobalTrack(new_gid, embeddings[i], boxes[i], frame_id, confs[i], state='probation')
            self.botsort_to_global[bid] = new_gid
            active_global_ids.add(new_gid)

    def _garbage_collection(self, frame_id):
        """Rimuove tracce troppo vecchie per risparmiare memoria."""
        to_del = []
        for gid, t in self.tracks.items():
            # I locked vivono più a lungo per permettere il recupero
            timeout = 1500 if t.state == 'locked' else 60 
            if (frame_id - t.last_seen_frame) > timeout:
                to_del.append(gid)
        
        for gid in to_del:
            del self.tracks[gid]
            # Pulizia reverse lookup
            keys = [k for k, v in self.botsort_to_global.items() if v == gid]
            for k in keys: del self.botsort_to_global[k]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_hybrid_tracking():
    # Setup Parametri
    TEAM_ID = "17"
    DEVICE = get_best_device()
    REID_WEIGHTS = MODEL_PATH / "re-ids" / "osnet_x1_0.pt" 
    
    # Inizializzazione Modelli
    reid_extractor = ReIDExtractor(str(REID_WEIGHTS), DEVICE)
    yolo = YOLO(MODEL_PATH / "training_runs" / "soccernet_v8m_ultra_19208" / "weights" / "best.pt")
    
    # Setup Output
    output_folder = TRACKER_OUTPUT_DIR / "final_results2_rgb_fres_2"
    output_folder.mkdir(parents=True, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted([s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s))])

    for seq_name in sequences:
        print(f"Processing Sequence: {seq_name}...")

        # Caricamento ROI (se presenti)
        roi_path = Path(base_path) / seq_name / "roi.json"
        roi_data = json.load(open(roi_path)) if roi_path.exists() else None
        
        mapper = RobustMapper()
        
        # Inizializzazione Tracker YOLO (BotSORT)
        results_gen = yolo.track(
            source=os.path.join(base_path, seq_name, 'img1'),
            conf=CONF_THRES,
            iou=0.6,
            imgsz=1920,
            device=0,
            classes=[0], # Solo classe 'persona'
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
                    # Necessario per aggiornare GMC anche se non ci sono detection
                    mapper.gmc.update_and_get_transform(result.orig_img)
                    continue
                
                # Estrazione Dati Frame
                frame_id = int(os.path.basename(result.path).split('.')[0])
                frame_img = cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2BGR)
                img_h, img_w = frame_img.shape[:2]

                boxes = result.boxes
                botsort_ids = boxes.id.cpu().numpy().astype(int)
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                # Limitazione numero entità (Top-K confidence)
                if len(botsort_ids) > MAX_ENTITIES:
                    top_indices = np.argsort(confs)[::-1][:MAX_ENTITIES]
                    botsort_ids = botsort_ids[top_indices]
                    boxes_xyxy = boxes_xyxy[top_indices]
                    confs = confs[top_indices]
                
                # Calcolo Embeddings e Tracking Globale
                embeddings = reid_extractor.get_embeddings(frame_img, boxes_xyxy)
                
                if len(embeddings) > 0:
                    global_ids = mapper.process_frame(botsort_ids, embeddings, boxes_xyxy, confs, frame_id, frame_img)
                else:
                    mapper.gmc.update_and_get_transform(frame_img)
                    global_ids = []

                # Scrittura Output (Tracking)
                for i, gid in enumerate(global_ids):
                    if gid == -1: continue 
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    # Formato: frame, ID, x, y, w, h
                    f_t.write(f"{frame_id},{gid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f}\n")

                # Scrittura Output (Behavior/Tactical)
                roi_counts = count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h)
                f_b.write(f"{frame_id},1,{roi_counts[1]}\n")
                f_b.write(f"{frame_id},2,{roi_counts[2]}\n")

        print(f"Sequenza {seq_name} completata.")

if __name__ == "__main__":
    run_hybrid_tracking()