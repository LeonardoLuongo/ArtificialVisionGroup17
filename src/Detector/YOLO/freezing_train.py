# src/Detector/YOLO/train_tuned.py
from ultralytics import YOLO
from pathlib import Path
import sys

# Setup path
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))
from config import *

def train_yolo_tuned():
    dataset_yaml = YOLO_DATASET_PATH / "dataset.yaml"
    
    # Usa yolov8m o yolo11m (il più recente supportato)
    model_name = 'yolo12m.pt' 
    model = YOLO(MODEL_PATH / model_name) 

    print(f"Avvio Training Tuned (Loss Custom) di {model_name}...")
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        
        # --- TUNING DEI PESI DELLA LOSS (Il tuo tocco magico) ---
        # box=12.0,   # AUMENTATO (Default 7.5): Priorità massima alla precisione geometrica
        # dfl=2.0,    # AUMENTATO (Default 1.5): Migliora i bordi fini
        # cls=0.4,    # RIDOTTO (Default 0.5): Classificare è facile (poche classi), concentriamoci sui box
        
        # --- Altri parametri di stabilità ---
        optimizer='AdamW',
        lr0=0.0005, 
        freeze=10,  # Congeliamo il backbone per stabilità
        
        project=str(MODEL_PATH / "training_runs"),
        name='soccernet_high_precision',
        exist_ok=True,
        plots=True,
        device=0
    )
    
    print("Training completato. Controlla se mAP50-95 è migliorata!")

if __name__ == "__main__":
    train_yolo_tuned()