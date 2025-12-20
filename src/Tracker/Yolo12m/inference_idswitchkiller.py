import os
import sys
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device

DEVICE = get_best_device()

try:
    from torchreid.utils import FeatureExtractor
    reid_model = FeatureExtractor(
        model_name='osnet_x0_25', 
        device=DEVICE
    )
except ImportError:
    reid_model = None
    print("ATTENZIONE: torchreid non trovato.")

def run_optimized_tracking():
    # --- CONFIGURAZIONE ---
    CONF_THRES = 0.35
    MAX_ENTITIES = 28 
    REID_THRESHOLD = 0.85 
    IMG_SIZE = 1280
    GALLERY_MAX_AGE = 100
    MIN_SIMILARITY_DELTA = 0.05  # Margine minimo per "rubare" un ID
    # ----------------------
    
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
            print(f"--> Modalità HOTA-Optimized: {seq_name} (Res: {IMG_SIZE})")
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            # Strutture dati per tracking
            id_gallery = {}           # {id: feature_vector}
            id_mapping = {}           # {tracker_id: mapped_id}
            id_last_seen = {}         # {id: frame_id}
            id_similarity_scores = {} # {tracker_id: similarity_score}
            tracker_to_mapped_history = defaultdict(list)  # Storia dei mapping
            
            results = model.track(
                source=img_source, conf=CONF_THRES, iou=0.5, imgsz=IMG_SIZE,
                tracker=str(TRACKER_CONFIG / "botsort.yaml"),
                persist=True, verbose=False, stream=True, half=True
            )
            
            with open(res_file_path, 'w') as f:
                for result in results:
                    frame_id = int(os.path.basename(result.path).split('.')[0])
                    img_bgr = result.orig_img 
                    
                    if result.boxes is None or result.boxes.id is None:
                        continue

                    boxes = result.boxes.cpu().numpy()
                    crops_to_process = []
                    tracker_ids_to_process = []
                    boxes_data = []

                    # 1. Raccolta dati e preparazione batch Re-ID
                    for box in boxes:
                        if box.id is None: continue
                        tracker_id = int(box.id[0])
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        
                        boxes_data.append({
                            'tracker_id': tracker_id, 
                            'coords': (x1, y1, x2, y2), 
                            'conf': float(box.conf[0])
                        })

                        # Re-ID solo se non mappato O se vogliamo aggiornare
                        if reid_model and tracker_id not in id_mapping:
                            crop = img_bgr[y1:y2, x1:x2]
                            if crop.size > 0:
                                crops_to_process.append(crop)
                                tracker_ids_to_process.append(tracker_id)

                    # 2. Batch Re-ID con similarity tracking
                    if crops_to_process:
                        batch_features = reid_model(crops_to_process).cpu().numpy()
                        
                        if id_gallery:
                            gallery_ids = list(id_gallery.keys())
                            gallery_features = np.vstack([id_gallery[gid] for gid in gallery_ids])
                            similarities = cosine_similarity(batch_features, gallery_features)
                            
                            for i, t_id in enumerate(tracker_ids_to_process):
                                max_idx = similarities[i].argmax()
                                max_sim = float(similarities[i][max_idx])
                                
                                if max_sim > REID_THRESHOLD:
                                    best_match_id = gallery_ids[max_idx]
                                    id_mapping[t_id] = best_match_id
                                    id_similarity_scores[t_id] = max_sim
                                    id_last_seen[best_match_id] = frame_id
                                    tracker_to_mapped_history[t_id].append(best_match_id)
                                elif len(id_gallery) < MAX_ENTITIES:
                                    id_gallery[t_id] = batch_features[i:i+1]
                                    id_mapping[t_id] = t_id
                                    id_similarity_scores[t_id] = 1.0
                                    id_last_seen[t_id] = frame_id
                                    tracker_to_mapped_history[t_id].append(t_id)
                        else:
                            for i, t_id in enumerate(tracker_ids_to_process):
                                if len(id_gallery) < MAX_ENTITIES:
                                    id_gallery[t_id] = batch_features[i:i+1]
                                    id_mapping[t_id] = t_id
                                    id_similarity_scores[t_id] = 1.0
                                    id_last_seen[t_id] = frame_id
                                    tracker_to_mapped_history[t_id].append(t_id)

                    # 3. Pruning Gallery
                    if frame_id % GALLERY_MAX_AGE == 0:
                        old_ids = [gid for gid, last_frame in id_last_seen.items() 
                                   if frame_id - last_frame > GALLERY_MAX_AGE]
                        for old_id in old_ids:
                            id_gallery.pop(old_id, None)
                            id_last_seen.pop(old_id, None)

                    # 4. RISOLUZIONE CONFLITTI ID (HOTA-Optimized)
                    # Raggruppa box per ID mappato
                    id_to_trackers = defaultdict(list)
                    for b in boxes_data:
                        t_id = b['tracker_id']
                        mapped_id = id_mapping.get(t_id, t_id)
                        id_to_trackers[mapped_id].append({
                            'tracker_id': t_id,
                            'data': b,
                            'similarity': id_similarity_scores.get(t_id, 0.0),
                            'conf': b['conf']
                        })
                    
                    # Risolvi conflitti: se più tracker_id mappano allo stesso ID
                    frame_output_lines = []
                    used_final_ids = set()
                    
                    # Ordina per ID mappato per consistenza
                    for mapped_id in sorted(id_to_trackers.keys()):
                        candidates = id_to_trackers[mapped_id]
                        
                        if len(candidates) == 1:
                            # Nessun conflitto
                            winner = candidates[0]
                            final_id = mapped_id
                        else:
                            # CONFLITTO: Scegli il vincitore basandosi su:
                            # 1. Similarity Re-ID (priorità massima)
                            # 2. Confidence detection
                            # 3. Storia del mapping (consistenza temporale)
                            
                            # Ordina: similarity > conf > storia
                            candidates.sort(key=lambda x: (
                                x['similarity'],
                                x['conf'],
                                len(tracker_to_mapped_history[x['tracker_id']])
                            ), reverse=True)
                            
                            winner = candidates[0]
                            final_id = mapped_id
                            
                            # I "perdenti" ottengono nuovi ID temporanei
                            for loser in candidates[1:]:
                                t_id = loser['tracker_id']
                                # Cerca un nuovo ID libero basato sul tracker_id originale
                                temp_id = t_id
                                offset = 0
                                while temp_id in used_final_ids:
                                    offset += 1
                                    temp_id = t_id + offset * 1000
                                
                                # Scrivi il perdente con ID temporaneo
                                x1, y1, x2, y2 = loser['data']['coords']
                                line = f"{frame_id},{temp_id},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{loser['conf']:.4f},-1,1,-1\n"
                                frame_output_lines.append((loser['conf'], line))
                                used_final_ids.add(temp_id)
                        
                        # Scrivi il vincitore con l'ID mappato
                        if final_id not in used_final_ids:
                            x1, y1, x2, y2 = winner['data']['coords']
                            line = f"{frame_id},{final_id},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{winner['conf']:.4f},-1,1,-1\n"
                            frame_output_lines.append((winner['conf'], line))
                            used_final_ids.add(final_id)

                    # Ordina per confidenza e limita a MAX_ENTITIES
                    frame_output_lines.sort(key=lambda x: x[0], reverse=True)
                    for _, line in frame_output_lines[:MAX_ENTITIES]:
                        f.write(line)
                    # f.flush()

            print(f"✓ Completato: {seq_name}")

if __name__ == "__main__":
    run_optimized_tracking()