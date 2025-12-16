# src/Detector/YOLO/train.py
from ultralytics import YOLO
from pathlib import Path
import sys

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

    # Carica modello pre-addestrato (es. yolov8m.pt o quello che usi)
    # Nota: Assicurati che il nome del modello sia corretto (yolo12m non esiste ufficialmente, forse intendi v8 o v11?)
    model_name = 'yolo12m.pt' 
    model = YOLO(MODEL_PATH / model_name) 

    print(f"Avvio training di {model_name}...")
    
    # Parametri training
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,             # Regola in base al tempo (50-100 è buono)
        imgsz=640,             # O 1280 se hai GPU potente e vuoi vedere i piccoli dettagli
        batch=8,              # Regola in base alla VRAM
        project=str(MODEL_PATH / "training_runs"),
        name='soccernet_finetune',
        exist_ok=True,
        device=0,              # GPU id
        plots=True,
        patience=10,            # Early stopping se non migliora per 10 epoche
        cache=False,
        workers=4
    )
    
    print("Training completato.")
    # Il modello migliore sarà salvato in models/training_runs/soccernet_finetune/weights/best.pt

if __name__ == "__main__":
    train_yolo()