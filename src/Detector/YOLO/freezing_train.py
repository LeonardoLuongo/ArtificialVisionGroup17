# src/Detector/YOLO/train_tuned.py
from ultralytics import YOLO
from ultralytics.utils import metrics
from pathlib import Path
import sys

# Setup path
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))
from config import *

def custom_fitness(x):
    """
    x = [P, R, mAP@50, mAP@50-95]
    Di default YOLO usa: 0.1 * mAP@50 + 0.9 * mAP@50-95
    Noi usiamo: 1.0 * mAP@50 + 0.0 * mAP@50-95
    """
    return x[2] # Restituisce solo mAP@50

# Sovrascriviamo la funzione della libreria
metrics.fitness = custom_fitness

def train_yolo_tuned():
    dataset_yaml = YOLO_DATASET_PATH / "dataset.yaml"
    
    model_name = 'yolo12m.pt' 
    model = YOLO(MODEL_PATH / model_name) 

    print(f"Avvio Training Tuned (Loss Custom) di {model_name}...")
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=1280,
        batch=8,
        
        # --- TUNING DEI PESI DELLA LOSS (Il tuo tocco magico) ---
        box=7.5,    # (Default): Allows slightly looser boxes, increases recall
        cls=1.5,    # INCREASED: Penalize missing players (FN) heavily
        dfl=1.5,    # SLIGHTLY REDUCED: We don't need pixel-perfect distribution

        # --- Augmentation to help small objects ---
        scale=0.8,  # Allow significant zooming in during training
        degrees=0.0,
            
        # --- Altri parametri di stabilità ---
        optimizer='AdamW',
        lr0=0.0005, 
        freeze=10,  # Congeliamo il backbone per stabilità
        
        project=str(MODEL_PATH / "training_runs"),
        name='soccernet_high_precision_frame3',
        exist_ok=True,
        plots=True,
        device=0,
        workers=8
    )
    print("Training completato.")
    

if __name__ == "__main__":
    train_yolo_tuned()