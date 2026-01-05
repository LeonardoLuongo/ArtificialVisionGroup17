# src/Detector/YOLO/train.py

import sys
import os
import torch

# --- FIX PER RTX 5070 (BLACKWELL) ---
# Forza PyTorch a usare i kernel della serie 40
os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0"
# Forza il caricamento dinamico dei moduli CUDA
os.environ['CUDA_MODULE_LOADING'] = "LAZY"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# ------------------------------------

from ultralytics import YOLO
from pathlib import Path

# Aggiungi root al path per importare config
ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))
from config import *

def train_yolo():
    # Percorso del dataset convertito (generato dallo script precedente)
    dataset_yaml = YOLO_DATASET_PATH / "dataset.yaml"
    
    if not dataset_yaml.exists():
        print("Errore: Dataset YAML non trovato. Esegui prima convert_to_yolo.py")
        return
    
    model_name = 'yolov8m.pt' 
    model = YOLO(MODEL_PATH / model_name) 

    print(f"Avvio training di {model_name}...")
    
    # Parametri training
    results = model.train(
    data=str(dataset_yaml),
    epochs=100,             
    imgsz=1920,             # Se hai memoria, prova 1536
    batch=2,               # Auto-batch: YOLO cercherà il massimo possibile per i tuoi 12GB
    project=str(MODEL_PATH / "training_runs"),
    name='soccernet_v8m_ultra_1920',
    device=0,
    optimizer='AdamW',        # Proviamo SGD per una precisione maggiore
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    workers=2,
    
    # Augmentation "Aggressiva"
    mosaic=1.0,
    mixup=0.5,              # Aumentato per gestire occlusioni
    copy_paste=0.5,         # Aumentato per giocatori sovrapposti
    hsv_h=0.015,            # Variazioni di colore (maglie diverse)
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,           # Piccole rotazioni
    scale=0.5,
    shear=2.0,              # Deformazione prospettica (utile per i bordi del campo)
    perspective=0.0001,
    
    # Parametri di controllo
    patience=20,           
    close_mosaic=20,        # Più tempo per rifinire senza augmentation distruttiva
    val=True,
    amp=True,               # Automatic Mixed Precision (fondamentale su RTX 5070)
)
    
    print("Training completato.")
    # Il modello migliore sarà salvato in models/training_runs/soccernet_finetune/weights/best.pt

if __name__ == "__main__":
    train_yolo()