import os
import sys
from ultralytics import YOLO
from pathlib import Path

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device

def run_optimized_tracking():
    # --- CONFIGURAZIONE ---
    # Usa il path assoluto o relativo corretto al file yaml appena creato
    CONF_THRES = 0.35
    MAX_ENTITIES = 28  # Il filtro "closed set" che abbiamo discusso
    # ----------------------

    DEVICE = get_best_device()
    model_name = 'best.pt'
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / model_name
    
    print(f"Caricamento modello: {weights_path}")
    print(f"Configurazione Tracker Custom: {TRACKER_CONFIG}")
    
    model = YOLO(weights_path)
    model.to(DEVICE)

    output_folder = TRACKER_OUTPUT_DIR / f"botsort_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_path = DATASET_TR23_TEST
    sequences = sorted(os.listdir(base_path))

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Tracking sequenza: {seq_name}")
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            # Reset del modello per pulire la memoria della GPU/Tracker tra sequenze
            model = YOLO(weights_path)
            model.to(DEVICE)

            results = model.track(
                source=img_source,
                conf=CONF_THRES,
                iou=0.6,
                imgsz=1280,
                device=0,
                classes=[0],
                tracker=str(TRACKER_CONFIG / "oc-sort.yaml"), # <--- QUI CARICHIAMO LA TUA CONFIGURAZIONE
                persist=True,
                verbose=False,
                stream=True
            )
            
            with open(res_file_path, 'w') as f:
                for result in results:
                    if result.boxes is None: continue
                    
                    frame_id = int(os.path.basename(result.path).split('.')[0])
                    boxes = result.boxes.cpu().numpy()
                    
                    # --- Raccolta e Filtro Top-K (Closed Set Real-time) ---
                    valid_candidates = []
                    
                    for box in boxes:
                        # Se il box non ha ID (perso dal tracker), lo scartiamo o mettiamo -1
                        if box.id is None: continue
                        
                        track_id = int(box.id[0])
                        x1, y1, x2, y2 = box.xyxy[0]
                        w = x2 - x1
                        h = y2 - y1
                        conf = box.conf[0]
                        
                        # Filtro geometrico (cartelloni bassi e larghi)
                        if (w / h) > 1.3: continue
                        
                        valid_candidates.append({
                            'line': f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,1,-1\n",
                            'conf': conf
                        })
                    
                    # Ordina per confidenza e tieni solo i top 24 (22 giocatori + arbitri)
                    valid_candidates.sort(key=lambda x: x['conf'], reverse=True)
                    final_candidates = valid_candidates[:MAX_ENTITIES]
                    
                    for cand in final_candidates:
                        f.write(cand['line'])

            print(f"Salvati: {res_file_path}")

if __name__ == "__main__":
    run_optimized_tracking()