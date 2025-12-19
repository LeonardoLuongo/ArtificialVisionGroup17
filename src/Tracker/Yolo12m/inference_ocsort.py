import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from boxmot import OcSort  # <--- Libreria esterna per OC-SORT

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device

def run_optimized_tracking():
    # --- CONFIGURAZIONE ---
    CONF_THRES = 0.35
    MAX_ENTITIES = 28  
    # ----------------------

    DEVICE = get_best_device()
    model_name = 'best.pt'
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / model_name
    
    print(f"Caricamento modello: {weights_path}")
    
    # Inizializziamo il modello YOLO (solo per detection)
    model = YOLO(weights_path)
    model.to(DEVICE)

    output_folder = TRACKER_OUTPUT_DIR / "ocsort_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_path = DATASET_TR23_TEST
    sequences = sorted(os.listdir(base_path))

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Tracking sequenza (OC-SORT): {seq_name}")
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            # Inizializza OC-SORT per ogni sequenza (per resettare gli ID)
            tracker = OcSort(
                model_weights=None, # Non usiamo ReID basato su modelli esterni qui
                device=DEVICE,
                per_class=True,
                det_thresh=CONF_THRES,
                max_age=60,       # track_buffer
                min_hits=3,       # frame minimi per confermare
                iou_threshold=0.3 # match_thresh
            )

            # Prendiamo la lista dei frame
            frame_files = sorted([f for f in os.listdir(img_source) if f.endswith(('.jpg', '.png'))])

            with open(res_file_path, 'w') as f:
                for img_file in frame_files:
                    frame_path = os.path.join(img_source, img_file)
                    frame_id = int(img_file.split('.')[0])
                    
                    # 1. Detection pura (senza .track)
                    img = cv2.imread(frame_path)
                    results = model.predict(img, conf=CONF_THRES, iou=0.6, imgsz=1280, device=DEVICE, classes=[0], verbose=False)
                    
                    # Prepara le detezioni per OC-SORT: [x1, y1, x2, y2, conf, class]
                    dets = results[0].boxes.data.cpu().numpy()
                    
                    # 2. Update manuale del Tracker OC-SORT
                    # Il tracker ritorna: [x1, y1, x2, y2, id, conf, class, ind]
                    tracks = tracker.update(dets, img)
                    
                    valid_candidates = []
                    
                    for track in tracks:
                        x1, y1, x2, y2, track_id, conf, cls, ind = track
                        
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Filtro geometrico (cartelloni bassi e larghi) [cite: 18, 20]
                        if (w / h) > 1.3: continue
                        
                        valid_candidates.append({
                            'line': f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,1,-1\n",
                            'conf': conf
                        })
                    
                    # Ordina per confidenza e tieni i Top-K richiesti [cite: 18, 22]
                    valid_candidates.sort(key=lambda x: x['conf'], reverse=True)
                    final_candidates = valid_candidates[:MAX_ENTITIES]
                    
                    for cand in final_candidates:
                        f.write(cand['line'])

            print(f"Salvati: {res_file_path}")

if __name__ == "__main__":
    run_optimized_tracking()