import os
import cv2
import glob
import sys
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
import gc  # <--- IMPORTANTE: Garbage Collector

# Configurazione Percorsi
DATASET_ROOT = Path("dataset/tracking-2023/test")
SEQUENCE_NAME = "SNMOT-116"
OUTPUT_DIR = Path("inference_results")

SEQ_DIR = DATASET_ROOT / SEQUENCE_NAME
IMG_DIR = SEQ_DIR / "img1"
GT_FILE = SEQ_DIR / "gt" / "gt.txt"
RESULT_FILE = OUTPUT_DIR / f"{SEQUENCE_NAME}.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_inference():
    # Pulisce la memoria prima di iniziare
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"--- Avvio Inference Zero-Shot con YOLOv8n su {SEQUENCE_NAME} ---")
    
    model = YOLO('yolov8n.pt')
    device = 0 if torch.cuda.is_available() else 'cpu' # Usa 0 (int) per GPU
    
    # Lista immagini
    img_files = sorted(glob.glob(str(IMG_DIR / "*.jpg")))
    
    if not img_files:
        print(f"Errore: Nessuna immagine trovata in {IMG_DIR}")
        return

    print(f"Trovate {len(img_files)} immagini. Inizio tracking...")
    
    with open(RESULT_FILE, 'w') as f_out:
        # stream=True crea un generatore
        results = model.track(
            source=img_files, 
            persist=True, 
            classes=[0], 
            conf=0.3, 
            iou=0.5,
            verbose=False,
            device=device,
            stream=True 
        )

        for i, result in enumerate(results):
            frame_id = i + 1
            
            # Estraiamo SUBITO i dati e spostiamo su CPU
            if result.boxes is not None and result.boxes.id is not None:
                # .numpy() su CPU è leggero, il tensore GPU è pesante
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    line = f"{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1"
                    f_out.write(line + "\n")
            
            # --- SALVAVITA PER LA RAM ---
            # L'oggetto result contiene l'immagine intera (orig_img).
            # Lo cancelliamo esplicitamente per liberare subito la RAM.
            del result 
            
            # Ogni 10 frame forziamo lo svuotamento della memoria
            if frame_id % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Processati {frame_id}/{len(img_files)} frame (Memoria pulita)")

    print(f"Inference completata. Risultati salvati in {RESULT_FILE}")
    
    # Pulizia finale pesante prima di passare alla valutazione
    del model
    del results
    gc.collect()
    torch.cuda.empty_cache()

def run_evaluation():
    print("\n--- Avvio Valutazione HOTA ---")
    
    if not GT_FILE.exists():
        print(f"ATTENZIONE: File GT non trovato. Skip.")
        return

    try:
        import trackeval
    except ImportError:
        print("TrackEval non installato.")
        return

    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

    class CustomDataset(trackeval.datasets.MotChallenge2DBox):
        def __init__(self, config):
            super().__init__(config)
        
        def _load_raw_file(self, tracker, seq, is_gt):
            file_path = str(GT_FILE) if is_gt else str(RESULT_FILE)
            try:
                data = np.loadtxt(file_path, delimiter=',')
            except:
                return {}, 0
                
            if len(data.shape) < 2:
                data = data.reshape(-1, 10)

            raw_data = {}
            num_timesteps = 0
            for row in data:
                frame = int(row[0])
                if frame > num_timesteps: num_timesteps = frame
                if frame not in raw_data: raw_data[frame] = []
                p_id = int(row[1])
                x, y, w, h = row[2:6]
                conf = row[6]
                raw_data[frame].append([p_id, x, y, w, h, conf, 1, 1])
            
            return raw_data, num_timesteps

        def get_seq_info(self):
            return [SEQUENCE_NAME]

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [CustomDataset(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    
    if len(dataset_list) > 0:
        return evaluator.evaluate(dataset_list, metrics_list)
    return None

if __name__ == "__main__":
    run_inference()
    run_evaluation()