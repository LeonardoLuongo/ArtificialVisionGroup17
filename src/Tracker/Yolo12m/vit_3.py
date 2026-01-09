# RGB -> final_results


import os
import sys
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from ultralytics import YOLO
import timm
from PIL import Image
from torchvision import transforms

# Aggiunge la root del progetto al path per importare moduli custom
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

# Import moduli custom (assicurati che esistano nel tuo progetto)
from config import *
from src.utils.get_best_device import get_best_device
import torchreid

# =================================================================================================
# 1. PARAMETRI DI CONFIGURAZIONE
# =================================================================================================
# Questi parametri controllano "quanto severo" è il sistema nel giudicare le somiglianze.

# Soglia minima di confidenza per accettare una detection dallo YOLO
CONF_THRES = 0.1           

# Numero massimo di giocatori/arbitri attesi in campo (per filtrare rumore)
MAX_ENTITIES = 50           

# Soglie per il Re-Identification (ReID)
MERGE_SIM_THRESH = 0.97      # Se la similarità è sotto questo valore, non è la stessa persona
RECOVERY_THRESH = 0.975      # Soglia alta per recuperare un giocatore perso (evita errori)
MERGE_THRESH = 0.97         # Soglia per unire due tracce frammentate

# Parametri Temporali
PROBATION_FRAMES = 8        # Quanti frame un ID deve esistere prima di essere considerato "stabile"
MAX_TIME_LOST = 1500        # Frame massimi di memoria per un giocatore scomparso (es. uscito dall'inquadratura)

# =================================================================================================
# 2. CLASSI DI UTILITÀ
# =================================================================================================

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


class GlobalTrack:
    """
    Rappresenta un singolo "Giocatore" nel tempo.
    Mantiene una "gallery" (storia) dei suoi vettori visivi per riconoscerlo anche se cambia posa.
    """
    def __init__(self, global_id, embedding, bbox, frame_id, conf, state='probation'):
        self.global_id = global_id
        self.gallery = []           # Lista di vettori (embeddings) passati
        self.max_gallery_size = 100  # Non teniamo infinite pose, solo le ultime 50
        
        # Stati possibili:
        # - 'probation': Appena apparso, stiamo verificando se è un vero giocatore o un errore.
        # - 'locked': Confermato, è un giocatore stabile.
        self.state = state 
        self.probation_age = 0      # Contatore: da quanti frame è in osservazione?
        
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
        
        # Aggiungiamo il primo embedding alla memoria
        if conf > 0.1:
            self.gallery.append(embedding)
    
    def update(self, embedding, bbox, frame_id, conf):
        """Aggiorna la posizione e, se l'immagine è chiara, aggiorna la memoria visiva."""
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
        
        if self.state == 'probation':
            self.probation_age += 1
        
        # Aggiornamento intelligente della Gallery:
        # Aggiungiamo una nuova posa solo se è abbastanza diversa da quelle che abbiamo già
        # (per avere varietà) ma non troppo diversa (per evitare errori).
        if conf > 0.1:
            if not self.gallery:
                self.gallery.append(embedding)
            else:
                sims = self.compute_sim_vector(embedding)
                max_sim = np.max(sims)
                
                # Se è 'locked' (stabile), siamo selettivi (0.95). 
                # Se è 'probation', siamo più avidi per imparare in fretta (0.98).
                thresh = 0.95 if self.state == 'locked' else 0.98
                
                if max_sim < thresh: 
                    if len(self.gallery) >= self.max_gallery_size:
                        self.gallery.pop(0) # Rimuovi il più vecchio
                    self.gallery.append(embedding)

    def compute_sim(self, candidate_emb):
        """Calcola la somiglianza massima tra il candidato e la storia di questo giocatore."""
        if not self.gallery: return 0.0 
        gallery_arr = np.array(self.gallery)
        candidate_emb = candidate_emb.reshape(1, -1)
        # Cosine distance: 0=uguali, 1=diversi. Noi vogliamo Similarità: 1=uguali.
        dists = cdist(candidate_emb, gallery_arr, metric='cosine')
        return 1.0 - np.min(dists)

    def compute_sim_vector(self, candidate_emb):
        """Restituisce vettore di similarità per usi interni."""
        if not self.gallery: return np.array([0.0])
        gallery_arr = np.array(self.gallery)
        dists = cdist(candidate_emb.reshape(1, -1), gallery_arr, metric='cosine')
        return 1.0 - dists[0]


class RobustMapper:
    """
    CUORE DEL SISTEMA.
    Questa classe mappa gli ID temporanei del tracker (BotSORT) agli ID Globali persistenti.
    Gestisce occlusioni, uscite dal campo e rientri.
    """
    def __init__(self):
        self.botsort_to_global = {} # Dizionario: ID BotSORT -> ID Globale
        self.tracks = {}            # Dizionario: ID Globale -> Oggetto GlobalTrack
        self.next_global_id = 1     # Contatore per nuovi ID

    def process_frame(self, botsort_ids, embeddings, boxes, confs, frame_id):
        """
        Elabora un singolo frame video.
        Input: Dati dallo YOLO/BotSORT (ID temporanei, box, embeddings).
        Output: Lista di ID Globali coerenti.
        """
        # Calcolo centroidi (x, y) dei box
        current_centroids = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        num_dets = len(botsort_ids)
        
        # Insieme degli ID globali che abbiamo già assegnato in QUESTO frame
        # (Serve per evitare che due persone abbiano lo stesso ID nello stesso momento)
        active_global_ids = set() 
        
        # Liste per gestire chi è stato trovato e chi no
        mapped_indices = []
        unmapped_indices = []

        # --- FASE 1: CONTINUITÀ TEMPORALE ---
        # Se un ID BotSORT era già associato a un ID Globale, manteniamo il legame.
        for i in range(num_dets):
            bid = botsort_ids[i]
            
            if bid in self.botsort_to_global:
                gid = self.botsort_to_global[bid]
                
                # Controllo coerenza: un ID globale non può apparire due volte
                if gid in active_global_ids:
                    # Anomalia: rimuoviamo il legame e lo trattiamo come nuovo
                    del self.botsort_to_global[bid]
                    unmapped_indices.append(i)
                else:
                    # Tutto ok, aggiorniamo il track
                    if gid in self.tracks:
                        active_global_ids.add(gid)
                        self.tracks[gid].update(embeddings[i], current_centroids[i], frame_id, confs[i])
                    else:
                        # Il track globale è stato cancellato (timeout), ripartiamo
                        del self.botsort_to_global[bid]
                        unmapped_indices.append(i)
            else:
                # BotSORT ID mai visto prima
                unmapped_indices.append(i)

        # --- FASE 2: GESTIONE FRAMMENTAZIONI (MERGE) ---
        # A volte BotSORT cambia ID allo stesso giocatore.
        # Qui controlliamo se un track "giovane" (probation) è in realtà un track vecchio.
        self._merge_probationary_tracks(frame_id, active_global_ids)

        # --- FASE 3: RECUPERO ID PERSI (RE-ID) ---
        # Per le detection che non hanno ancora un ID Globale (unmapped),
        # cerchiamo nella memoria dei giocatori scomparsi ("Dormant").
        final_new_indices = self._recover_lost_tracks(
            unmapped_indices, embeddings, confs, current_centroids, 
            botsort_ids, active_global_ids, frame_id
        )

        # --- FASE 4: CREAZIONE NUOVI GIOCATORI ---
        # Chi non è stato recuperato nemmeno ora, è davvero un nuovo giocatore.
        for i in final_new_indices:
            self._create_new_track(
                botsort_ids[i], embeddings[i], current_centroids[i], 
                confs[i], frame_id, active_global_ids
            )

        # --- FASE 5: PULIZIA ---
        # Rimuoviamo tracce troppo vecchie per risparmiare memoria
        self._garbage_collection(frame_id)

        # Costruzione output finale
        final_ids = [self.botsort_to_global.get(bid, -1) for bid in botsort_ids]
        return final_ids

    def _merge_probationary_tracks(self, frame_id, active_global_ids):
        """Controlla se tracce in 'probation' possono essere unite a tracce esistenti."""
        current_bids = list(self.botsort_to_global.keys())
        
        for bid in current_bids:
            gid = self.botsort_to_global[bid]
            track = self.tracks.get(gid)
            if not track or track.state != 'probation': continue

            # Cerchiamo il miglior match tra i track inattivi (non in active_global_ids)
            best_match = None
            max_sim = -1.0
            
            if track.gallery:
                curr_emb = track.gallery[-1]
                for c_gid, c_track in self.tracks.items():
                    # Ignora se stesso, se è attivo ora, o se anche l'altro è in probation
                    if c_gid == gid or c_gid in active_global_ids or c_track.state != 'locked': 
                        continue
                    
                    # Filtro temporale e spaziale veloce
                    dt = frame_id - c_track.last_seen_frame
                    if dt > 600: continue # Troppo vecchio
                    
                    # Se sono vicini nello spazio, calcoliamo ReID
                    if np.linalg.norm(track.last_bbox - c_track.last_bbox) < (100 + 40 * dt):
                        sim = c_track.compute_sim(curr_emb)
                        if sim > max_sim: 
                            max_sim = sim
                            best_match = c_track
            
            # Decisione: Uniamo o Promuoviamo?
            target = -1
            if max_sim > MERGE_SIM_THRESH: # Similarità altissima -> Merge immediato
                target = best_match.global_id
            elif track.probation_age >= PROBATION_FRAMES:
                if max_sim > MERGE_THRESH: # Tempo scaduto, similarità buona -> Merge
                    target = best_match.global_id
                else: 
                    track.state = 'locked' # Nessun match trovato -> Promosso a nuovo giocatore stabile
            print(f"Track {gid} (age {track.probation_age}) merge check: max_sim={max_sim:.3f} -> target={target}")

            # Esecuzione del Merge
            if target != -1:
                self.botsort_to_global[bid] = target
                # Uniamo le memorie visive
                tgt_t = self.tracks[target]
                tgt_t.last_seen_frame = track.last_seen_frame
                tgt_t.last_bbox = track.last_bbox
                tgt_t.gallery.extend(track.gallery)
                # Mantieni dimensione gallery
                if len(tgt_t.gallery) > tgt_t.max_gallery_size: 
                    tgt_t.gallery = tgt_t.gallery[-tgt_t.max_gallery_size:]
                
                # Elimina il track temporaneo
                del self.tracks[gid]
                if gid in active_global_ids: active_global_ids.remove(gid)
                active_global_ids.add(target)

    def _recover_lost_tracks(self, unmapped_indices, embeddings, confs, centroids, botsort_ids, active_global_ids, frame_id):
        """Cerca di assegnare detection orfane a vecchi ID globali (es. dopo occlusione)."""
        final_new_indices = []
        
        # Ordiniamo per confidenza: i box migliori scelgono per primi
        unmapped_indices.sort(key=lambda idx: confs[idx], reverse=True)

        for i in unmapped_indices:
            conf = confs[i]
            
            # Tentiamo il recupero solo se l'immagine è decente
            if conf > 0.1:
                best_gid = -1
                best_sim = -1.0
                emb = embeddings[i]
                centroid = centroids[i]

                for gid, track in self.tracks.items():
                    # 1. Non deve essere già in campo ORA
                    if gid in active_global_ids: continue
                    
                    # 2. Non deve essere scomparso da troppo tempo
                    dt = frame_id - track.last_seen_frame
                    if dt > 900: continue 
                    
                    # 3. Filtro Spaziale Generoso (Recupero)
                    # Ammettiamo uno spostamento massimo basato sul tempo trascorso
                    #dist = np.linalg.norm(centroid - track.last_bbox)
                    dist = 0
                    if dist < (100 + 50 * dt):
                        sim = track.compute_sim(emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_gid = gid
                
                # Se abbiamo trovato un match forte
                if best_sim > RECOVERY_THRESH:
                    bid = botsort_ids[i]
                    self.botsort_to_global[bid] = best_gid
                    
                    # Aggiorniamo il vecchio track con i nuovi dati
                    self.tracks[best_gid].update(emb, centroid, frame_id, conf)
                    active_global_ids.add(best_gid)
                    continue # Assegnato, passa al prossimo

            # Se arriviamo qui, non abbiamo trovato nessun match
            final_new_indices.append(i)
            
        return final_new_indices

    def _create_new_track(self, bid, emb, centroid, conf, frame_id, active_global_ids):
        """Crea una nuova identità globale."""
        new_gid = self.next_global_id
        self.next_global_id += 1
        
        # Nasce sempre in 'probation'
        self.tracks[new_gid] = GlobalTrack(new_gid, emb, centroid, frame_id, conf, state='probation')
        self.botsort_to_global[bid] = new_gid
        active_global_ids.add(new_gid)

    def _garbage_collection(self, frame_id):
        """Rimuove ID che non si vedono da troppo tempo."""
        to_del = []
        for gid, t in self.tracks.items():
            # I track 'locked' li teniamo molto tempo per permettere il recupero
            # I track 'probation' li scartiamo subito se scompaiono
            timeout = MAX_TIME_LOST if t.state == 'locked' else 60 
            if (frame_id - t.last_seen_frame) > timeout:
                to_del.append(gid)
        
        for gid in to_del:
            del self.tracks[gid]
            # Rimuovi anche dal dizionario di mappatura
            keys = [k for k, v in self.botsort_to_global.items() if v == gid]
            for k in keys: del self.botsort_to_global[k]


# Funzione ausiliaria per la logica di gioco (Behavior)
def count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h):
    counts = {1: 0, 2: 0} 
    if not roi_data: return counts

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        # Punto di contatto a terra (piedi del giocatore)
        cx = x1 + ((x2 - x1) / 2.0)
        cy = y2 

        for roi_id in [1, 2]:
            roi = roi_data[f"roi{roi_id}"]
            rx, ry = roi["x"] * img_w, roi["y"] * img_h
            rw, rh = roi["width"] * img_w, roi["height"] * img_h
            
            if rx <= cx <= (rx + rw) and ry <= cy <= (ry + rh):
                counts[roi_id] += 1
    return counts


# =================================================================================================
# 3. MAIN LOOP
# =================================================================================================

def run_hybrid_tracking():
    # Setup Iniziale
    TEAM_ID = "17"
    DEVICE = get_best_device()
    
    # Percorso pesi ReID
    reid_weights_path = MODEL_PATH / "re-ids" / "ViT-L_16.pt"
    reid_extractor = ReIDExtractor(str(reid_weights_path), DEVICE)
    
    # Preparazione cartella output
    output_folder = TRACKER_OUTPUT_DIR / "final_results_VIT3"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Lettura dataset
    base_path = DATASET_TR23_TEST
    sequences = sorted([s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s))])

    # Loop su ogni sequenza video (partita)
    for seq_name in sequences:
        print(f"\nProcessing Sequence: {seq_name}...")

        # Caricamento ROI (Zone del campo)
        roi_path = Path(base_path) / seq_name / "roi.json"
        roi_data = json.load(open(roi_path)) if roi_path.exists() else None
        
        # Inizializza il Mapper per questa sequenza
        mapper = RobustMapper()
        
        # Inizializza YOLO con BotSORT
        yolo = YOLO(MODEL_PATH / "bestyolo8.pt")
        yolo.eval()
        # Generatore di frame (Stream)
        results_gen = yolo.track(
            source=os.path.join(base_path, seq_name, 'img1'),
            conf=CONF_THRES,
            iou=0.6,
            imgsz=1280,
            device=0, # Usa GPU 0
            classes=[0], # Solo classe 'persona'
            tracker=str(TRACKER_CONFIG / "bot_sort.yaml"),
            persist=True,
            stream=True,
            verbose=False
        )

        track_file = output_folder / f"tracking_{seq_name}_{TEAM_ID}.txt"
        behav_file = output_folder / f"behavior_{seq_name}_{TEAM_ID}.txt"
        
        # Apertura file di output
        with open(track_file, 'w') as f_t, open(behav_file, 'w') as f_b:
            
            for result in results_gen:
                # Se non ci sono detection, passa oltre
                if result.boxes is None or result.boxes.id is None:
                    continue
                
                # Estrazione dati base
                frame_id = int(os.path.basename(result.path).split('.')[0])
                frame_img = result.orig_img
                img_h, img_w = frame_img.shape[:2]

                # Dati da YOLO/BotSORT
                boxes = result.boxes
                botsort_ids = boxes.id.cpu().numpy().astype(int)
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                # Filtro: teniamo solo le top N detection per evitare folla/rumore
                if len(botsort_ids) > MAX_ENTITIES:
                    top_indices = np.argsort(confs)[::-1][:MAX_ENTITIES]
                    botsort_ids = botsort_ids[top_indices]
                    boxes_xyxy = boxes_xyxy[top_indices]
                    confs = confs[top_indices]
                
                # Estrazione Embeddings (Vettori visivi) per ReID
                #frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
                embeddings = reid_extractor.get_embeddings(frame_img, boxes_xyxy)
                
                # --- PROCESSING CORE ---
                if len(embeddings) > 0:
                    global_ids = mapper.process_frame(botsort_ids, embeddings, boxes_xyxy, confs, frame_id)
                else:
                    global_ids = []

                # --- SALVATAGGIO TRACKING ---
                # Formato: frame, id, x, y, w, h
                for i, gid in enumerate(global_ids):
                    if gid == -1: continue 
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    f_t.write(f"{frame_id},{gid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f}\n")

                # --- SALVATAGGIO BEHAVIOR ---
                roi_counts = count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h)
                f_b.write(f"{frame_id},1,{roi_counts[1]}\n")
                f_b.write(f"{frame_id},2,{roi_counts[2]}\n")

        print(f"Sequenza {seq_name} completata.")

if __name__ == "__main__":
    run_hybrid_tracking()