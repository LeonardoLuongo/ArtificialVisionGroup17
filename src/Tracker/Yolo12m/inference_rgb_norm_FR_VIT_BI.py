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
import timm
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import numpy as np


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
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        
        # 1. Load and inspect the weights *first* to determine the architecture
        print(f"Loading weights from {weights_path}...")
        clean_state_dict = self._load_and_clean_weights(weights_path)
        
        # 2. Check pos_embed shape to decide if we need a Class Token
        # Standard ViT-L/16: 197 tokens (196 patches + 1 CLS)
        # GAP ViT-L/16:      196 tokens (196 patches)
        use_cls_token = True
        global_pool = 'token' # Default: use CLS token output
        
        if 'pos_embed' in clean_state_dict:
            num_tokens = clean_state_dict['pos_embed'].shape[1]
            print(f"Detected 'pos_embed' with {num_tokens} tokens.")
            
            if num_tokens == 196:
                print("-> Configuring model WITHOUT Class Token (using Global Average Pooling).")
                use_cls_token = False
                global_pool = 'avg'
            elif num_tokens == 197:
                print("-> Configuring model WITH Class Token.")
                use_cls_token = True
                global_pool = 'token'

        # 3. Create the model with the correct configuration
        print("Creating ViT-L/16 model...")
        self.model = timm.create_model(
            'vit_large_patch16_224', 
            pretrained=False, 
            num_classes=0,          # No classification head
            class_token=use_cls_token,
            global_pool=global_pool 
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # 4. Load the cleaned weights
        # strict=False allows ignoring missing head weights or 'fc_norm'
        msg = self.model.load_state_dict(clean_state_dict, strict=False)
        print(f"Model loaded. Missing keys (usually head/fc): {msg.missing_keys[:5]}...")

        # Standard Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_and_clean_weights(self, weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError(f"Weights file not found at: {weights_path}")
        
        # Unwrap state dictionary
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()

        # --- Smart Prefix Detection ---
        # Find the prefix used in this specific file (e.g., 'model.' or 'module.')
        target_key = 'patch_embed.proj.weight'
        prefix = ""
        
        for k in state_dict.keys():
            if k.endswith(target_key):
                prefix = k[:-len(target_key)]
                print(f"Detected weight prefix: '{prefix}'")
                break
        
        # Create a new dict with the prefix removed
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                clean_state_dict[new_k] = v
                
        return clean_state_dict

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
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
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
# CLASSE: LOGICA DI ASSOCIAZIONE (ROBUST MAPPER)
# =============================================================================

class RobustMapper:
    """
    Gestisce l'associazione tra detection e identità globali.
    Utilizza un approccio ibrido:
    1. Bi-directional Softmax (ReID globale)
    2. Continuità Spaziale (BotSORT Boost)
    3. Global Linear Assignment (Algoritmo Ungherese)
    """
    def __init__(self):
        self.botsort_to_global = {} # Mappa ID frame corrente -> ID globale
        self.tracks = {}            # Dizionario di tutti i GlobalTrack attivi/dormienti
        self.next_global_id = 1
        
        self.gmc = CameraCompensator()
        
        # Parametri
        self.TEMPERATURE = 0.05      # Temperatura per la Softmax (più basso = più piccato)
        self.MATCH_THRESH = 0.40     # Soglia minima di similarità Cosine pura
        self.CONTINUITY_BOOST = 0.5  # Bonus dato se BotSORT conferma l'identità
        
    def process_frame(self, botsort_ids, embeddings, boxes, confs, frame_id, current_frame_img):
        """
        Main loop per processare un frame.
        """
        # 1. Aggiornamento GMC (Global Motion Compensation)
        transform_m = self.gmc.update_and_get_transform(current_frame_img)
        self._apply_gmc_to_tracks(transform_m)

        # 2. Preparazione dati (Embeddings dei track esistenti)
        # track_ids_list mantiene l'ordine delle righe della matrice
        track_ids_list, track_embs = self._get_active_track_embeddings()

        # 3. Matching Globale (Tracks vs Detections)
        matches, unmatched_dets = self._global_matching(
            track_ids_list, track_embs, 
            botsort_ids, embeddings
        )

        # 4. Aggiornamento Tracks Matchati
        active_global_ids = self._update_matched_tracks(
            matches, botsort_ids, embeddings, boxes, confs, frame_id
        )

        # 5. Creazione Nuovi Tracks (Detection non associate)
        self._create_new_tracks(
            unmatched_dets, botsort_ids, embeddings, boxes, confs, frame_id, active_global_ids
        )

        # 6. Pulizia memoria (Garbage Collection)
        self._garbage_collection(frame_id)
        
        # Output: Lista ID allineata con l'input
        return [self.botsort_to_global.get(bid, -1) for bid in botsort_ids]

    # =========================================================================
    # METODI PRIVATI (LOGICA CORE)
    # =========================================================================

    def _get_active_track_embeddings(self):
        """Estrae l'ultimo embedding valido da tutti i track conosciuti."""
        track_ids = list(self.tracks.keys())
        embs = []
        for gid in track_ids:
            track = self.tracks[gid]
            # Usa l'ultimo embedding della gallery o un vettore nullo (fallback)
            if track.gallery:
                embs.append(track.gallery[-1])
            else:
                embs.append(np.zeros(1024, dtype=np.float32))
        return track_ids, np.array(embs)

    def _global_matching(self, track_ids, track_embs, botsort_ids, det_embs):
        """
        Esegue l'associazione globale usando Bi-directional Softmax + Ungherese.
        Restituisce:
        - matches: Lista di tuple (global_id, detection_index)
        - unmatched_dets: Set di indici detection non assegnati
        """
        num_tracks = len(track_ids)
        num_dets = len(det_embs)
        
        # Caso base: Nessun track o nessuna detection
        if num_tracks == 0 or num_dets == 0:
            return [], set(range(num_dets))

        # A. Calcolo Matrice Affinità Base (Bi-directional Softmax)
        affinity_matrix, raw_sim_matrix = self._compute_bi_directional_affinity(track_embs, det_embs)

        # B. Hybrid Boosting (Aiuta il ReID con la continuità spaziale di BotSORT)
        affinity_matrix = self._apply_continuity_boost(
            affinity_matrix, track_ids, botsort_ids
        )

        # C. Risoluzione Conflitti (Algoritmo Ungherese)
        # Minimizziamo il costo negativo (-affinity) per massimizzare l'affinità
        row_indices, col_indices = linear_sum_assignment(-affinity_matrix)

        matches = []
        unmatched_dets = set(range(num_dets))

        for t_idx, d_idx in zip(row_indices, col_indices):
            # Recuperiamo i valori
            gid = track_ids[t_idx]
            raw_sim = raw_sim_matrix[t_idx, d_idx]
            
            # D. Filtro di Validità
            # Accettiamo il match solo se visivamente ha senso (raw_sim > soglia)
            if raw_sim > self.MATCH_THRESH:
                matches.append((gid, d_idx))
                if d_idx in unmatched_dets:
                    unmatched_dets.remove(d_idx)
            
        return matches, unmatched_dets

    def _compute_bi_directional_affinity(self, track_embs, det_embs):
        """
        Calcola matrice affinità dove A[i,j] è alto solo se track[i] e det[j]
        si 'scelgono' a vicenda.
        """
        # 1. Matrice Similarità Coseno (Tracks x Detections)
        sim_matrix = np.dot(track_embs, det_embs.T) # Shape: (N_tracks, N_dets)
        sim_tensor = torch.tensor(sim_matrix)

        # 2. Softmax Row-wise (Preferenze del Track)
        row_soft = F.softmax(sim_tensor / self.TEMPERATURE, dim=1)

        # 3. Softmax Col-wise (Preferenze della Detection)
        col_soft = F.softmax(sim_tensor / self.TEMPERATURE, dim=0)

        # 4. Consenso (Moltiplicazione Element-wise)
        affinity = (row_soft * col_soft).numpy()
        
        return affinity, sim_matrix

    def _apply_continuity_boost(self, affinity_matrix, track_ids, botsort_ids):
        """
        Aumenta lo score di affinità se l'ID di BotSORT corrisponde 
        allo storico (mapping frame precedente).
        """
        for t_idx, gid in enumerate(track_ids):
            for d_idx, bid in enumerate(botsort_ids):
                # Se questo bot_id era mappato a questo global_id
                prev_mapping = self.botsort_to_global.get(bid)
                if prev_mapping == gid:
                    affinity_matrix[t_idx, d_idx] += self.CONTINUITY_BOOST
        return affinity_matrix

    def _update_matched_tracks(self, matches, botsort_ids, embeddings, boxes, confs, frame_id):
        """Aggiorna i track esistenti con le nuove detection assegnate."""
        active_ids = set()
        for gid, d_idx in matches:
            track = self.tracks[gid]
            bid = botsort_ids[d_idx]
            
            # Aggiorna lo stato del track
            track.update(embeddings[d_idx], boxes[d_idx], frame_id, confs[d_idx])
            
            # Aggiorna mapping inverso (importante per il frame successivo)
            self.botsort_to_global[bid] = gid
            active_ids.add(gid)
        return active_ids

    def _create_new_tracks(self, unmatched_dets, botsort_ids, embeddings, boxes, confs, frame_id, active_ids):
        """Inizializza nuovi track per le detection rimaste orfane."""
        for d_idx in unmatched_dets:
            bid = botsort_ids[d_idx]
            
            new_gid = self.next_global_id
            self.next_global_id += 1
            
            # Crea nuovo track in stato 'probation'
            self.tracks[new_gid] = GlobalTrack(
                new_gid, embeddings[d_idx], boxes[d_idx], frame_id, confs[d_idx], state='probation'
            )
            
            self.botsort_to_global[bid] = new_gid
            active_ids.add(new_gid)

    def _apply_gmc_to_tracks(self, transform_m):
        """Compensa il movimento della camera per tutti i track."""
        for gid, track in self.tracks.items():
            track.last_bbox = self.gmc.apply_transform(track.last_bbox, transform_m)

    def _garbage_collection(self, frame_id):
        """Rimuove tracce vecchie."""
        to_del = []
        for gid, t in self.tracks.items():
            # Timeout più lungo per i locked (confermati), breve per i probation
            timeout = 1500 if t.state == 'locked' else 30
            if (frame_id - t.last_seen_frame) > timeout:
                to_del.append(gid)
        
        for gid in to_del:
            del self.tracks[gid]
            # Pulisci lookup table inversa
            keys_to_remove = [k for k, v in self.botsort_to_global.items() if v == gid]
            for k in keys_to_remove:
                del self.botsort_to_global[k]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_hybrid_tracking():
    # Setup Parametri
    TEAM_ID = "17"
    DEVICE = get_best_device()
    REID_WEIGHTS = MODEL_PATH / "re-ids" / "ViT-L_16.pt" 
    
    # Inizializzazione Modelli
    reid_extractor = ReIDExtractor(str(REID_WEIGHTS), DEVICE)
    yolo = YOLO(MODEL_PATH / "bestyolo8.pt")
    
    # Setup Output
    output_folder = TRACKER_OUTPUT_DIR / "final_results2_rgb_norm_FR_VIT_BI"
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