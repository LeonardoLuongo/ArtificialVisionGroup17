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
MAX_ENTITIES = 28           

# Soglie per il Re-Identification (ReID)
MERGE_SIM_THRESH = 0.85      # Se la similarità è sotto questo valore, non è la stessa persona
RECOVERY_THRESH = 0.75      # Soglia alta per recuperare un giocatore perso (evita errori)
MERGE_THRESH = 0.70         # Soglia per unire due tracce frammentate

# Parametri Temporali
PROBATION_FRAMES = 8        # Quanti frame un ID deve esistere prima di essere considerato "stabile"
MAX_TIME_LOST = 1500        # Frame massimi di memoria per un giocatore scomparso (es. uscito dall'inquadratura)

# =================================================================================================
# 2. CLASSI DI UTILITÀ
# =================================================================================================

class ReIDExtractor:
    """
    Questa classe gestisce la Rete Neurale per il Re-Identification.
    Il suo scopo è trasformare l'immagine di un giocatore (crop) in un vettore numerico (embedding).
    Se due vettori sono vicini matematicamente, le immagini ritraggono la stessa persona.
    """
    def __init__(self, weights_path, device='cuda'):
        self.device = device
        self.resize_wh = (128, 256) # Dimensione standard per modelli OSNet
        
        # Selezione automatica della dimensione in base al nome del modello
        weights_str = str(weights_path).lower()
        if 'vit' in weights_str:
            self.resize_wh = (224, 224)
            model_name = 'vit_b_16'
        elif 'ain' in weights_str:
            model_name = 'osnet_ain_x1_0'
        else:
            model_name = 'osnet_x1_0'
            
        print(f"--> Caricamento Modello ReID: {model_name} su {device}")

        # Costruzione del modello usando la libreria torchreid
        try:
            self.model = torchreid.models.build_model(name=model_name, num_classes=1000, pretrained=False)
            if weights_path and os.path.exists(weights_path):
                torchreid.utils.load_pretrained_weights(self.model, weights_path)
        except Exception as e:
            print(f"ERRORE nel caricamento del modello: {e}")
            sys.exit(1)
            
        self.model.to(device).eval()
        # Normalizzazione standard per immagini (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    @torch.no_grad()
    def get_embeddings(self, frame, bboxes_xyxy):
        """
        Prende il frame intero e le bounding box, ritaglia i giocatori 
        e restituisce i vettori (embeddings) corrispondenti.
        """
        if len(bboxes_xyxy) == 0: 
            return np.array([])
            
        crops = []
        h_img, w_img = frame.shape[:2]
        
        for box in bboxes_xyxy:
            x1, y1, x2, y2 = map(int, box)
            # Clip delle coordinate per non uscire dall'immagine
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            crop = frame[y1:y2, x1:x2]
            
            # Gestione caso crop vuoto (raro ma possibile)
            if crop.size == 0: 
                crop = np.zeros((self.resize_wh[1], self.resize_wh[0], 3), dtype=np.uint8)
            
            # Ridimensionamento e conversione colore
            crop = cv2.resize(crop, self.resize_wh)
            crops.append(np.transpose(crop, (2, 0, 1))) # Da HWC a CHW per PyTorch

        # Preparazione tensore batch
        input_t = torch.from_numpy(np.array(crops)).float().to(self.device) / 255.0
        input_t = (input_t - self.mean) / self.std
        
        # Inferenza
        features = self.model(input_t)
        
        # L2 Normalize
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.detach().cpu().numpy()


class GlobalTrack:
    """
    Rappresenta un singolo "Giocatore" nel tempo.
    Mantiene una "gallery" (storia) dei suoi vettori visivi per riconoscerlo anche se cambia posa.
    """
    def __init__(self, global_id, embedding, bbox, frame_id, conf, state='probation'):
        self.global_id = global_id
        self.gallery = []           # Lista di vettori (embeddings) passati
        self.max_gallery_size = 50  # Non teniamo infinite pose, solo le ultime 50
        
        # Stati possibili:
        # - 'probation': Appena apparso, stiamo verificando se è un vero giocatore o un errore.
        # - 'locked': Confermato, è un giocatore stabile.
        self.state = state 
        self.probation_age = 0      # Contatore: da quanti frame è in osservazione?
        
        self.last_seen_frame = frame_id
        self.last_bbox = np.array(bbox, dtype=np.float32)
        
        # Aggiungiamo il primo embedding alla memoria
        if conf > 0.60:
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
        if conf > 0.60:
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
            if conf > 0.55:
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
                    dist = np.linalg.norm(centroid - track.last_bbox)
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
    reid_weights_path = MODEL_PATH / "re-ids" / "osnet_x1_0.pt"
    reid_extractor = ReIDExtractor(str(reid_weights_path), DEVICE)
    
    # Preparazione cartella output
    output_folder = TRACKER_OUTPUT_DIR / "final_results2_norm2"
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
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
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