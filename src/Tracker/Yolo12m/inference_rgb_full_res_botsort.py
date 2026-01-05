import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from collections import deque
from scipy.spatial.distance import cdist
from ultralytics import YOLO
import PIL.Image as Image
from torchvision import transforms
from torch import nn



# Aggiunta path progetto (Mantenuta come originale)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from config import *
from src.utils.get_best_device import get_best_device
import torchreid

def count_players_in_rois(boxes, roi_data, img_w, img_h):
    """
    Funzione placeholder/semplificata per il conteggio nelle ROI.
    Se hai già la tua logica complessa, ignora questo blocco.
    """
    counts = {1: 0, 2: 0} # Assumiamo ID ROI 1 e 2
    if roi_data is None or len(boxes) == 0:
        return counts
        
    # Esempio logica base: controlliamo se il centro del box è nella ROI
    # Qui dovresti implementare cv2.pointPolygonTest se le ROI sono poligoni complessi
    for box in boxes:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, y2  # Usiamo i piedi (y2) come riferimento
        
        # Logica fittizia: dividiamo il campo a metà se non c'è logica reale
        if cx < img_w / 2:
            counts[1] += 1
        else:
            counts[2] += 1
            
    return counts


def run_hybrid_tracking():
    # Setup Parametri
    TEAM_ID = "17"
    DEVICE = get_best_device()

    yolo = YOLO(MODEL_PATH / "bestyolo8.pt")
    
    # Setup Output
    output_folder = TRACKER_OUTPUT_DIR / "final_results2_fullbotsort"
    output_folder.mkdir(parents=True, exist_ok=True)

    base_path = DATASET_TR23_TEST
    sequences = sorted([s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s))])

    for seq_name in sequences:
        print(f"Processing Sequence: {seq_name}...")

        # Caricamento ROI (se presenti)
        roi_path = Path(base_path) / seq_name / "roi.json"
        roi_data = json.load(open(roi_path)) if roi_path.exists() else None
        
     
        # Inizializzazione Tracker YOLO (BotSORT)
        results_gen = yolo.track(
            source=os.path.join(base_path, seq_name, 'img1'),
            conf=0.1,
            iou=0.6,
            imgsz=1920,
            device=0,
            classes=[0], # Solo classe 'persona'
            tracker=str(TRACKER_CONFIG / "bot_sort_with_reid.yaml"),
            persist=True,
            stream=True,
            verbose=False
        )

        track_file = output_folder / f"tracking_{seq_name}_{TEAM_ID}.txt"
        behav_file = output_folder / f"behavior_{seq_name}_{TEAM_ID}.txt"
        
        with open(track_file, 'w') as f_t, open(behav_file, 'w') as f_b:
            for result in results_gen:
                
                # Estrazione Dati Frame
                frame_id = int(os.path.basename(result.path).split('.')[0])
                frame_img = cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2BGR)
                img_h, img_w = frame_img.shape[:2]

                boxes = result.boxes
                
                # È FONDAMENTALE controllare se ci sono ID. 
                # Se il tracker perde tutti i target, boxes.id è None e il codice crasherebbe.
                if boxes.id is not None:
                    # Convertiamo i tensori in numpy/liste pulite
                    # .int() serve perché gli ID sono interi
                    botsort_ids = boxes.id.int().cpu().tolist()
                    boxes_xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    
                    # Poiché il Re-ID è attivo nel YAML, questi ID sono già consistenti
                    global_ids = botsort_ids 
                else:
                    # Nessun oggetto tracciato in questo frame
                    botsort_ids = []
                    boxes_xyxy = []
                    global_ids = []

                # Scrittura Output (Tracking)
                for i, gid in enumerate(global_ids):
                    if gid == -1: continue 
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    # Formato: frame, ID, x, y, w, h
                    f_t.write(f"{frame_id},{gid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f}\n")

                # Scrittura Output (Behavior/Tactical)
                roi_counts = count_players_in_rois(boxes_xyxy, roi_data, img_w, img_h)
                f_b.write(f"{frame_id},1,{roi_counts[1]}\n")
                f_b.write(f"{frame_id},2,{roi_counts[2]}\n")

        print(f"Sequenza {seq_name} completata.")

if __name__ == "__main__":
    run_hybrid_tracking()