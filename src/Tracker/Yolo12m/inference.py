import os
import sys
import shutil
from ultralytics import YOLO
from pathlib import Path

# Setup path per importare config
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device

def run_tracking_and_save_txt():
    # --- CONFIGURAZIONE ---
    CONF_THRES = 0.35
    IOU_THRES = 0.7
    # ----------------------

    DEVICE = get_best_device()
    model_name = 'best.pt'
    
    weights_path = MODEL_PATH / "training_runs" / "soccernet_high_precision" / "weights" / model_name
    
    print(f"Caricamento modello: {weights_path}")
    print(f"Device: {DEVICE}")
    
    # Inizializziamo il modello una volta
    model = YOLO(weights_path)
    model.to(DEVICE)

    base_path = DATASET_TR23_TEST
    output_folder = TRACKER_OUTPUT_DIR / f"{model_name.split('.')[0]}_tracker_results"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Errore pulizia: {e}")

    if not os.path.exists(base_path):
        print(f"Errore: Percorso {base_path} non trovato.")
        return

    sequences = sorted(os.listdir(base_path)) # Ordiniamo per pulizia

    for seq_name in sequences:
        seq_path = os.path.join(base_path, seq_name)
        img_source = os.path.join(seq_path, 'img1')

        if os.path.isdir(seq_path) and os.path.exists(img_source):
            print(f"--> Tracking sequenza: {seq_name}")
            
            res_file_path = os.path.join(output_folder, f"{seq_name}.txt")

            # --- RESET DEL TRACKER (FIX CORRETTO) ---
            model = YOLO(weights_path) 
            model.to(DEVICE)

            results = model.track(
                source=img_source,
                conf=CONF_THRES,
                iou=IOU_THRES,
                imgsz=1280,
                device=0,
                classes=[0],
                tracker="bytetrack.yaml", 
                persist=True,
                verbose=False,
                stream=True
            )
            
            with open(res_file_path, 'w') as f:
                for result in results:
                    path = result.path
                    frame_id = int(os.path.basename(path).split('.')[0])
                    
                    if result.boxes is None:
                        continue

                    boxes = result.boxes.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0]
                        w = x2 - x1
                        h = y2 - y1
                        conf = box.conf[0]
                        cls = box.cls[0]
                        
                        # --- LOGICA CORRETTA PER ID ---
                        if box.id is not None:
                            track_id = int(box.id[0])

                            if track_id < 0:
                                print(f"Attenzione: ID negativo ({track_id}) per box in frame {frame_id} della sequenza {seq_name}. Salto questo box.")
                                continue
                            
                            if int(cls) == 0: 
                                line = f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,1,-1\n"
                                f.write(line)
                        else:
                            # Saltiamo silenziosamente i box senza ID
                            pass
            
            del model

            print(f"Salvati risultati in: {res_file_path}")
        

if __name__ == "__main__":
    run_tracking_and_save_txt()