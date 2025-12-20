import os
import sys
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device

# Carichiamo un estrattore di feature per il Re-ID (SOTA per lo sport)
try:
    from torchreid.utils import FeatureExtractor
    reid_model = FeatureExtractor(
        model_name='osnet_x0_25', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
except ImportError:
    reid_model = None
    print("ATTENZIONE: torchreid non trovato. La Galleria ID non funzionerà al meglio.")

def run_optimized_tracking():
    # --- CONFIGURAZIONE ---
    CONF_THRES = 0.35
    MAX_ENTITIES = 28 # Numero chiuso di giocatori + arbitri
    REID_THRESHOLD = 0.85 # Somiglianza minima per dire "è lo stesso giocatore"
    # ----------------------

    DEVICE = get_best_device()
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / "best.pt"
    
    model = YOLO(weights_path)
    model.to(DEVICE)

    output_folder = TRACKER_OUTPUT_DIR / f"botsort_gallery_results"
    os.makedirs(output_folder, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted(os.listdir(base_path))

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Tracking con ID Gallery (Batch Mode): {seq_name}")
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            # Reset memoria per ogni video
            id_gallery = {} # { original_id: embedding_vector }
            id_mapping = {} # { tracker_new_id: original_id }

            results = model.track(
                source=img_source, conf=CONF_THRES, iou=0.6, imgsz=1280,
                tracker=str(TRACKER_CONFIG / "botsort.yaml"),
                persist=True, verbose=False, stream=True
            )
            
            with open(res_file_path, 'w') as f:
                for result in results:
                    frame_id = int(os.path.basename(result.path).split('.')[0])
                    img_bgr = result.orig_img # Frame originale per il Re-ID
                    
                    if result.boxes is None or result.boxes.id is None:
                        continue

                    boxes = result.boxes.cpu().numpy()
                    frame_entities = []

                    # --- LOGICA BATCH RE-ID ---
                    crops_to_process = []
                    tracker_ids_to_process = []
                    boxes_data = []

                    # 1. Identifichiamo quali box necessitano di un controllo Re-ID
                    for box in boxes:
                        if box.id is None: continue
                        
                        tracker_id = int(box.id[0])
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        conf = box.conf[0]
                        
                        boxes_data.append({
                            'tracker_id': tracker_id,
                            'coords': (x1, y1, x2, y2),
                            'conf': conf
                        })

                        # Se l'ID è nuovo e abbiamo il modello, prepariamo il crop per il batch
                        if reid_model and tracker_id not in id_mapping:
                            crop = img_bgr[y1:y2, x1:x2]
                            if crop.size > 0:
                                crops_to_process.append(crop)
                                tracker_ids_to_process.append(tracker_id)

                    # 2. Eseguiamo l'inferenza Re-ID in un unico batch per questo frame
                    if crops_to_process:
                        # Estraiamo tutte le feature in una sola chiamata alla GPU
                        batch_features = reid_model(crops_to_process).cpu().numpy()
                        
                        for i, t_id in enumerate(tracker_ids_to_process):
                            feat = batch_features[i:i+1] # Singolo embedding
                            
                            best_match_id = None
                            max_sim = -1
                            
                            # Confronto con la galleria esistente
                            for gid, g_feat in id_gallery.items():
                                sim = cosine_similarity(feat, g_feat)[0][0]
                                if sim > max_sim:
                                    max_sim = sim
                                    best_match_id = gid
                            
                            # Logica di assegnazione ID
                            if best_match_id and max_sim > REID_THRESHOLD:
                                id_mapping[t_id] = best_match_id
                            else:
                                if len(id_gallery) < MAX_ENTITIES:
                                    id_gallery[t_id] = feat
                                    id_mapping[t_id] = t_id

                    # 3. Costruiamo le righe di output usando il mapping aggiornato
                    for b in boxes_data:
                        t_id = b['tracker_id']
                        x1, y1, x2, y2 = b['coords']
                        conf = b['conf']
                        
                        final_id = id_mapping.get(t_id, t_id)
                        
                        frame_entities.append({
                            'line': f"{frame_id},{final_id},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{conf:.4f},-1,1,-1\n",
                            'conf': conf
                        })

                    # Filtro finale Top-K (Assicuriamoci di non scrivere più di 28 persone)
                    frame_entities.sort(key=lambda x: x['conf'], reverse=True)
                    for ent in frame_entities[:MAX_ENTITIES]:
                        f.write(ent['line'])

            print(f"Salvati: {res_file_path}")

if __name__ == "__main__":
    run_optimized_tracking()