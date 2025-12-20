import os
import sys
from ultralytics import YOLO
from pathlib import Path
import numpy as np

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
    MAX_PREDICTION_FRAMES = 15 # Frame per cui continuare a predire il movimento
    # ----------------------

    DEVICE = get_best_device()
    model_name = 'best.pt'
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / model_name
    
    print(f"Caricamento modello: {weights_path}")
    model = YOLO(weights_path)
    model.to(DEVICE)

    output_folder = TRACKER_OUTPUT_DIR / f"botsort_results"
    os.makedirs(output_folder, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted(os.listdir(base_path))

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Tracking con Predizione Lineare: {seq_name}")
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            model = YOLO(weights_path)
            model.to(DEVICE)
            
            # Storia avanzata: { id: {'box': [x,y,w,h], 'velocity': [dx,dy,dw,dh], 'lost': 0} }
            track_history = {}

            results = model.track(
                source=img_source, conf=CONF_THRES, iou=0.6, imgsz=1280,
                device=0, classes=[0], tracker=str(TRACKER_CONFIG / "botsort.yaml"),
                persist=True, verbose=False, stream=True
            )
            
            with open(res_file_path, 'w') as f:
                for result in results:
                    frame_id = int(os.path.basename(result.path).split('.')[0])
                    active_this_frame = []
                    current_frame_ids = set()

                    if result.boxes is not None and result.boxes.id is not None:
                        boxes = result.boxes.cpu().numpy()
                        for box in boxes:
                            if box.id is None: continue
                            
                            tid = int(box.id[0])
                            x1, y1, x2, y2 = box.xyxy[0]
                            w, h = x2 - x1, y2 - y1
                            curr_box = [x1, y1, w, h]

                            if (w / h) > 1.3: continue # Filtro cartelloni

                            current_frame_ids.add(tid)
                            
                            # --- CALCOLO VELOCITÀ ---
                            velocity = [0, 0, 0, 0]
                            if tid in track_history:
                                prev_box = track_history[tid]['box']
                                # Velocità = Posizione_attuale - Posizione_precedente
                                velocity = [curr_box[i] - prev_box[i] for i in range(4)]
                            
                            # Aggiorna storia con dati REALI
                            track_history[tid] = {
                                'box': curr_box,
                                'velocity': velocity,
                                'lost': 0
                            }

                            active_this_frame.append({'id': tid, 'box': curr_box, 'conf': box.conf[0]})

                    # --- LOGICA DI PREDIZIONE (Simil-Kalman) ---
                    predicted_candidates = []
                    for tid in list(track_history.keys()):
                        if tid not in current_frame_ids:
                            track_history[tid]['lost'] += 1
                            lost_count = track_history[tid]['lost']

                            if lost_count <= MAX_PREDICTION_FRAMES:
                                last_box = track_history[tid]['box']
                                vel = track_history[tid]['velocity']
                                
                                # Applichiamo l'equazione del moto rettilineo uniforme: P_nuova = P_vecchia + V
                                # Moltiplichiamo vel per un fattore di smorzamento (0.9) per evitare che il box "scappi" via
                                damping = 0.9 
                                pred_box = [last_box[i] + (vel[i] * damping) for i in range(4)]
                                
                                # Aggiorniamo la storia con la predizione (per il prossimo frame)
                                track_history[tid]['box'] = pred_box
                                
                                predicted_candidates.append({
                                    'id': tid,
                                    'box': pred_box,
                                    'conf': 0.10 # Confidenza bassa per i predetti
                                })
                            else:
                                del track_history[tid]

                    # Mix e Filtro finale
                    all_entities = active_this_frame + predicted_candidates
                    all_entities.sort(key=lambda x: x['conf'], reverse=True)
                    
                    for entity in all_entities[:MAX_ENTITIES]:
                        tid, (x, y, w, h), conf = entity['id'], entity['box'], entity['conf']
                        f.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,1,-1\n")

            print(f"Salvati: {res_file_path}")

if __name__ == "__main__":
    run_optimized_tracking()