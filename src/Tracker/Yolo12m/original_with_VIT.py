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
from PIL import Image
from torchvision import transforms

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
        # --- COMPENSATORE CAMERA ---
        self.gmc = CameraCompensator()
        # CONFIGURAZIONE
        self.PROBATION_FRAMES = 8   
        self.MERGE_THRESH = 0.70    
        self.EARLY_MERGE_THRESH = 0.85 
        
        # SOGLIA RECUPERO IMMEDIATO
        # Deve essere alta. Se recuperiamo l'ID sbagliato qui, facciamo un disastro.
        self.RECOVERY_THRESH = 0.75 

    def process_frame(self, botsort_ids, embeddings, boxes, confs, frame_id, current_frame_img):
        # 1. Update GMC e Trasforma i BBox vecchi
        transform_m = self.gmc.update_and_get_transform(current_frame_img)
        
        # Aggiorniamo TUTTE le posizioni note applicando lo spostamento della camera
        for gid, track in self.tracks.items():
            track.last_bbox = self.gmc.apply_transform(track.last_bbox, transform_m)

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
                    track.update(embeddings[i], boxes[i], frame_id, confs[i])
                    
                    # Se il track è locked o probation avanzato, resettiamo il guess
                    # perché ormai "è lui"
                    if track.state == 'locked':
                        track.best_guess_id = None
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
                        if dt > 1500: continue # Timeout temporale ampio
                        
                        # Distanza spaziale (ora corretta dal GMC!)
                        dist = np.linalg.norm((track.last_bbox[:2]+track.last_bbox[2:])/2 - 
                                            (c_track.last_bbox[:2]+c_track.last_bbox[2:])/2)
                        
                        # Soglia spaziale dinamica
                        if dist < (150 + 60 * dt): 
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
            bbox = boxes[i]
            centroid = (bbox[:2] + bbox[2:]) / 2.0
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
                    
                    # Distanza con GMC applicato
                    track_center = (track.last_bbox[:2] + track.last_bbox[2:]) / 2.0
                    dist = np.linalg.norm(centroid - track_center)
                    
                    if dist < (150 + 50 * dt):
                        sim = track.compute_sim(emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_gid = gid
                
                # CHECK SOGLIA
                if best_sim > self.RECOVERY_THRESH:
                    # TROVATO! È LUI!
                    self.botsort_to_global[bid] = best_gid
                    self.tracks[best_gid].update(emb, bbox, frame_id, conf)
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
            
            # Nasce in Probation e cercherà subito un Best Guess al prossimo loop
            self.tracks[new_gid] = GlobalTrack(new_gid, embeddings[i], boxes[i], frame_id, confs[i], state='probation')
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
    REID_WEIGHTS = MODEL_PATH / "re-ids" / "ViT-L_16.pt" # O osnet_ain_x1_0.pt
    
    reid_extractor = ReIDExtractor(str(REID_WEIGHTS), DEVICE)
    
    output_folder = TRACKER_OUTPUT_DIR / "final_provisional_tracker_vit"
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