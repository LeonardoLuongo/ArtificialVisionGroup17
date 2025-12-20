import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from boxmot import BoostTrack  # Utilizzo di BoostTrack per massimizzare AssA

# Setup path (mantenendo la tua struttura)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device

def run_boosttrack_soccer():
    # --- CONFIGURAZIONE OTTIMIZZATA PER ASSA ---
    CONF_THRES = 0.40      # Soglia leggermente più alta per pulire la DetA [cite: 80]
    MAX_ENTITIES = 28      # Limite giocatori/arbitri [cite: 18]
    REID_MODEL = 'osnet_x1_0_market1501.pt' # Re-ID necessario per re-identificare 
    
    DEVICE = get_best_device()
    print(f"Utilizzo dispositivo: {DEVICE}")
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / "best.pt"
    
    model = YOLO(weights_path)
    model.to(DEVICE)

    output_folder = TRACKER_OUTPUT_DIR / "boosttrack_results"
    os.makedirs(output_folder, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted(os.listdir(base_path))

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Tracking con BoostTrack: {seq_name}")
            
            # Formato file richiesto: tracking_K_XX.txt [cite: 59]
            # Assumendo XX come team '01'
            res_file_path = os.path.join(output_folder, f"tracking_{seq_name}_01.txt")
            
            # Inizializza BoostTrack
            # Fondamentale per AssA: calibrazione GMC e Re-ID [cite: 71, 81]
            tracker = BoostTrack(
                model_weights=Path(REID_MODEL),
                device=DEVICE,
                fp16=True,
                det_thresh=CONF_THRES,
                max_age=150,        # Gestisce sparizioni temporanee dovute a pan della camera 
                min_hits=3,
                iou_threshold=0.3,  # Soglia per associazione spaziale
                match_thresh=0.8,   # Soglia per Re-ID (apparenza)
                gmc_method='ecc',   # Indispensabile per Moving Camera 
                lambda_m=0.7,       # Peso del movimento nell'associazione
                lambda_a=0.3,       # Peso dell'apparenza (Re-ID)
                use_dfl=True        # Deep Feature Learner per migliorare l'associazione
            )

            frame_files = sorted([f for f in os.listdir(img_source) if f.endswith(('.jpg', '.png'))])

            with open(res_file_path, 'w') as f:
                for img_file in frame_files:
                    frame_path = os.path.join(img_source, img_file)
                    # frame_id deve partire da 1 [cite: 62]
                    frame_id = int(''.join(filter(str.isdigit, img_file)))
                    
                    img = cv2.imread(frame_path)
                    # Risoluzione 1280 per mantenere dettagli dei giocatori a 1080p [cite: 22]
                    results = model.predict(img, conf=CONF_THRES, iou=0.5, imgsz=1280, device=DEVICE, classes=[0], verbose=False)
                    dets = results[0].boxes.data.cpu().numpy()
                    
                    # Update BoostTrack
                    tracks = tracker.update(dets, img)
                    
                    valid_candidates = []
                    for track in tracks:
                        x1, y1, x2, y2, track_id, conf, cls, ind = track
                        w, h = x2 - x1, y2 - y1
                        
                        # Filtro giocatori: evita cartelloni orizzontali scambiati per persone
                        if (w / h) > 0.9: continue
                        
                        valid_candidates.append({
                            # Formato richiesto: frame_id,object_id,top_left_x,top_left_y,width,height 
                            'line': f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,1,-1\n",
                            'conf': conf
                        })
                    
                    # Seleziona i migliori candidati fino al limite MAX_ENTITIES
                    valid_candidates.sort(key=lambda x: x['conf'], reverse=True)
                    for cand in valid_candidates[:MAX_ENTITIES]:
                        f.write(cand['line'])

            print(f"Salvati: {res_file_path}")
        
        seqmap_file = SEQMAP_FILE
        with open(seqmap_file, 'w') as f:
            for seq_name in sequences:
                f.write(f"{seq_name}\n")

if __name__ == "__main__":
    run_boosttrack_soccer()