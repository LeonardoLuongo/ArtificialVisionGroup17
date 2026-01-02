import os
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from pathlib import Path
import json

# --- IMPORT LIBRARY ESTERNA PER IL TEST PURO ---
from boxmot import BotSort

# Setup path (Manteniamo la tua struttura)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device
import torchreid

# --- CONFIGURAZIONE ---
CONF_THRES = 0.1
MAX_ENTITIES = 28
# Parametri BoT-SORT simili a quelli che usavi, ma gestiti internamente da lui
TRACK_BUFFER = 90      # Quanti frame ricorda un giocatore perso
MATCH_THRESH = 0.70    # Soglia di similarità (essendo ViT molto preciso, possiamo stare alti)
    

# Manteniamo la tua funzione helper ESATTA
def count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h):
    counts = {1: 0, 2: 0} 
    if not roi_data:
        return counts

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx = x1 + (w / 2.0)
        cy = y2 

        for roi_id in [1, 2]:
            roi = roi_data[f"roi{roi_id}"]
            rx = roi["x"] * img_w
            ry = roi["y"] * img_h
            rw = roi["width"] * img_w
            rh = roi["height"] * img_h
            
            if rx <= cx <= (rx + rw) and ry <= cy <= (ry + rh):
                counts[roi_id] += 1
    return counts

# La TUA classe ReIDExtractor (quella basata su torchreid che mi hai passato)
class ReIDExtractor:
    def __init__(self, weights_path=None, device='cuda'):
        self.device = device
        if not weights_path:
            use_custom_weights = False
            model_name = 'osnet_x1_0'
            self.resize_wh = (128, 256)
            print(f"--> Modalità BASELINE: Uso {model_name} pre-addestrato su ImageNet.")
        else:
            use_custom_weights = True
            weights_str = str(weights_path).lower()
            if 'vit' in weights_str:
                self.resize_wh = (224, 224)
                model_name = 'vit_l_16' if 'l_16' in weights_str else 'vit_b_16'
                print(f"--> Rilevato Custom ViT. Model: {model_name} | Input: {self.resize_wh}")
            else:
                self.resize_wh = (128, 256)
                model_name = 'osnet_x1_0'
                print(f"--> Rilevato Custom OSNet. Model: {model_name} | Input: {self.resize_wh}")

        try:
            self.model = torchreid.models.build_model(
                name=model_name, 
                num_classes=1000, 
                pretrained=(not use_custom_weights) 
            )
        except KeyError:
            print(f"ERRORE: Modello {model_name} non supportato.")
            sys.exit(1)

        if use_custom_weights:
            if os.path.exists(weights_path):
                try:
                    checkpoint = torch.load(weights_path, map_location='cpu')
                    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('module.'): k = k[7:]
                        if k.startswith('model.'): k = k[6:]
                        if 'classifier' in k or 'fc' in k or 'head' in k: continue
                        new_state_dict[k] = v
                    self.model.load_state_dict(new_state_dict, strict=False)
                    print(f"✅ Pesi Custom caricati (Backbone only).")
                except Exception as e:
                    print(f"ERRORE GRAVE nel caricamento pesi custom: {e}")
                    sys.exit(1)
            else:
                print(f"WARNING: File pesi {weights_path} non trovato! Uso inizializzazione random.")
        else:
            print("ℹ️  Disattivazione classifier standard per estrazione feature...")
            if hasattr(self.model, 'classifier'): self.model.classifier = nn.Sequential()
            elif hasattr(self.model, 'head'): self.model.head = nn.Sequential()
            elif hasattr(self.model, 'fc'): self.model.fc = nn.Sequential()
            
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
            else: crop = cv2.resize(crop, self.resize_wh)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(np.transpose(crop, (2, 0, 1)))

        if not crops: return np.array([])
        input_t = torch.from_numpy(np.array(crops)).float().to(self.device) / 255.0
        input_t = (input_t - self.mean) / self.std
        
        features = self.model(input_t)
        # IMPORTANTE: Normalizziamo per BoTSORT (cosine distance)
        return F.normalize(features, p=2, dim=1).cpu().numpy()

def run_test_1_botsort():
    TEAM_ID = "17"
    DEVICE = get_best_device()
    
    # Path del tuo modello ReID ViT o OSNet
    # REID_WEIGHTS = MODEL_PATH / "re-ids" / "ViT-L_16.pt" 
    # Oppure usa None per testare imagenet baseline, come nel tuo esempio ibrido
    REID_WEIGHTS = MODEL_PATH / "re-ids" / "osnet_x1_0.pt" 

    reid_extractor = ReIDExtractor(str(REID_WEIGHTS), DEVICE)
    yolo = YOLO(MODEL_PATH / "bestyolo8.pt")

    # --- INIZIALIZZA BoT-SORT (Libreria BoxMOT) ---
    tracker = BotSort(
        reid_weights=Path('osnet_x0_25_msmt17.pt'), # BoxMOT richiede un path, ma lo ignoreremo passando feats
        device=0,
        half=True,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        match_thresh=MATCH_THRESH,
        track_buffer=TRACK_BUFFER,
        cmc_method='orb',
        with_reid=True, # Abilitiamo la logica ReID, ma useremo i NOSTRI embedding
        proximity_thresh=0.5, 
        appearance_thresh=0.25
    )
    
    # Cartella Output
    output_folder = TRACKER_OUTPUT_DIR / "test1_pure_botsort"
    output_folder.mkdir(parents=True, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted([s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s))])

    for seq_name in sequences:
        print(f"Processing {seq_name} [Test 1: BoT-SORT Pure]...")
        tracker.reset() # Reset Kalman Filters per nuova sequenza

        roi_path = Path(base_path) / seq_name / "roi.json"
        roi_data = json.load(open(roi_path)) if roi_path.exists() else None

        # PREDICT (Non Track) - Facciamo detection pura
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
                frame_id = int(os.path.basename(result.path).split('.')[0])
                frame_img = result.orig_img
                img_h, img_w = frame_img.shape[:2]

                # 1. Estrai Detections
                # [x1, y1, x2, y2, conf, cls]
                dets = result.boxes.data.cpu().numpy()

                # Filtro Top-K Entities
                if len(dets) > MAX_ENTITIES:
                    dets = dets[np.argsort(dets[:, 4])[::-1][:MAX_ENTITIES]]

                # 2. Gestione frame vuoti o pieni
                if len(dets) == 0:
                    empty_dets = np.empty((0, 6))
                    empty_embs = np.empty((0, 1024)) # O 512 per OSNet
                    tracks = tracker.update(empty_dets, frame_img, empty_embs)
                else:
                    # 3. Estrazione Embedding Custom
                    bboxes_xyxy = dets[:, :4]
                    embeddings = reid_extractor.get_embeddings(frame_img, bboxes_xyxy)
                    
                    # 4. Update Tracker
                    tracks = tracker.update(dets, frame_img, embeddings)

                # tracks output: [x1, y1, x2, y2, id, conf, cls, index]

                # --- SCRITTURA TRACKING FILE ---
                # Usiamo i box TRACCIATI (tracks) per l'output, non i detection grezzi
                tracked_xyxy = []
                for t in tracks:
                    x1, y1, x2, y2 = t[:4]
                    gid = int(t[4]) # ID assegnato da BoTSORT
                    
                    # Salva per behavior
                    tracked_xyxy.append([x1, y1, x2, y2])
                    
                    # Scrittura: frame, id, x, y, w, h
                    f_t.write(f"{frame_id},{gid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f}\n")

                # --- PROCESSING & WRITING BEHAVIOR ---
                # Usiamo i box stabilizzati dal tracker
                roi_counts = count_players_in_rois(np.array(tracked_xyxy), roi_data, img_w, img_h)
                f_b.write(f"{frame_id},1,{roi_counts[1]}\n")
                f_b.write(f"{frame_id},2,{roi_counts[2]}\n")

        print(f"Sequenza {seq_name} completata.")

if __name__ == "__main__":
    run_test_1_botsort()