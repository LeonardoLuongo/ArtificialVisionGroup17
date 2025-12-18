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

def run_decoupled_tracking():
    # --- PARAMETRI CRITICI ---
    # 1. Cosa vede il tracker (MOLTO BASSO per permettere a ByteTrack di lavorare)
    CONF_THRES_INPUT = 0.10  
    
    # 2. Cosa scriviamo nel file (ALTO per avere Precisione 99%)
    CONF_THRES_OUTPUT = 0.35 
    
    # 3. Quante entità massime (Closed Set)
    MAX_ENTITIES = 26 
    # -------------------------

    DEVICE = get_best_device()
    model_name = 'best.pt'
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / model_name
    
    # Percorso config tracker
    tracker_config_path = project_root / "src" / "Tracker" / "Yolo12m" / "soccer_tracker.yaml"

    print(f"Caricamento modello: {weights_path}")
    print(f"Tracker Config: {tracker_config_path}")
    
    model = YOLO(weights_path)
    model.to(DEVICE)

    output_folder = TRACKER_OUTPUT_DIR / f"{model_name.split('.')[0]}_decoupled"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_path = DATASET_TR23_TEST
    sequences = sorted(os.listdir(base_path))

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Tracking: {seq_name} (In: {CONF_THRES_INPUT}, Out: {CONF_THRES_OUTPUT})")
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")
            
            # Reset modello
            model = YOLO(weights_path)
            model.to(DEVICE)

            # INFERENZA: Passiamo confidenza bassa per alimentare il secondo stadio di ByteTrack
            results = model.track(
                source=img_source,
                conf=CONF_THRES_INPUT, # <--- QUI STA IL TRUCCO
                iou=0.6,
                imgsz=1980,
                device=0,
                classes=[0],
                tracker=str(tracker_config_path),
                persist=True,
                verbose=False,
                stream=True
            )
            
            with open(res_file_path, 'w') as f:
                for result in results:
                    if result.boxes is None: continue
                    
                    frame_id = int(os.path.basename(result.path).split('.')[0])
                    boxes = result.boxes.cpu().numpy()
                    
                    valid_candidates = []
                    
                    for box in boxes:
                        if box.id is None: continue # Box non associati
                        
                        track_id = int(box.id[0])
                        conf = box.conf[0]
                        
                        # --- FILTRO OUTPUT ---
                        # Qui scartiamo i box deboli.
                        # MA: il tracker li ha usati per calcolare la posizione!
                        # Se il box è sotto soglia, non lo scriviamo, ma l'ID è salvo nel tracker.
                        if conf < CONF_THRES_OUTPUT:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0]
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Filtro Aspect Ratio
                        if (w / h) > 1.3: continue
                        
                        valid_candidates.append({
                            'line': f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,1,-1\n",
                            'conf': conf
                        })
                    
                    # Filtro Closed Set (Top-K)
                    valid_candidates.sort(key=lambda x: x['conf'], reverse=True)
                    final_candidates = valid_candidates[:MAX_ENTITIES]
                    
                    for cand in final_candidates:
                        f.write(cand['line'])

            print(f"Salvati: {res_file_path}")

if __name__ == "__main__":
    run_decoupled_tracking()