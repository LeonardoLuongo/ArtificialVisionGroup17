# =================================================
# FILE: src/BlurFineTuning/train_blur.py
# =================================================
from ultralytics import YOLO
from pathlib import Path
import sys

from config import *

def train_blur_finetune():
    # 1. PERCORSI
    # Il file YAML generato automaticamente da smart_extract.py
    blur_yaml = ROOT_DIR / "dataset/blur_dataset/dataset.yaml"
    
    # Il tuo modello MIGLIORE precedente (quello addestrato su tutto il dataset)
    # Assicurati che questo path sia corretto in base al tuo ultimo training!
    prev_model_path = MODEL_PATH / "training_runs/soccernet_high_precision/weights/best.pt"
    
    # Check esistenza file
    if not blur_yaml.exists():
        print(f"ERRORE: Non trovo {blur_yaml}")
        print("Esegui prima 'src/BlurFineTuning/smart_extract.py'!")
        return

    if not prev_model_path.exists():
        print(f"ERRORE: Non trovo i pesi precedenti in {prev_model_path}")
        print("Modifica 'prev_model_path' nello script con il percorso del tuo best.pt")
        return

    # 2. CARICAMENTO MODELLO
    print(f"--- Caricamento pesi: {prev_model_path.name} ---")
    model = YOLO(str(prev_model_path))
    
    # 3. AVVIO FINE-TUNING
    # Usiamo parametri molto conservativi perché vogliamo solo "adattare" 
    # il modello al blur, non insegnargli tutto da zero.
    print(f"--- Avvio Fine-Tuning su dataset Blur ---")
    
    results = model.train(
        data=str(blur_yaml),
        
        # Hyperparameters per Fine-Tuning
        epochs=20,              # Poche epoche bastano (avevi detto 15, 20 è ok con early stop)
        imgsz=1024,             # Stessa ris del training originale
        batch=4,                # Basso batch size per stabilità (o 8 se hai VRAM)
        
        # Learning Rate: IL SEGRETO
        # Usiamo un LR molto basso (1e-5) rispetto allo standard (1e-2)
        # Questo permette di modificare leggermente i pesi senza distruggere la conoscenza pregressa.
        optimizer='AdamW',
        lr0=1e-5,               
        lrf=0.01,               # Decay lento
        
        # Augmentation
        # Disabilitiamo mosaic perché confonde su dataset piccoli e specifici
        # Vogliamo che veda i frame sfocati "così come sono"
        mosaic=0.0,             
        mixup=0.0,
        
        # Loss Gains (I tuoi custom, manteniamoli coerenti)
        box=7.5,
        cls=1.5,
        dfl=1.5,
        
        # Salvataggio
        project=str(MODEL_PATH / "training_runs"),
        name='soccernet_blur_tuned',
        exist_ok=True,
        plots=True,
        device=0,               # Usa GPU 0
        workers=4,
        
        # Validation
        val=True,               # Valida ad ogni epoca (userà il val.txt originale definito nello yaml)
        save=True               # Salva i checkpoint
    )
    
    print("------------------------------------------------")
    print("Fine-tuning completato!")
    print(f"Nuovi pesi salvati in: {MODEL_PATH}/training_runs/soccernet_blur_tuned/weights/best.pt")
    print("Usa questi pesi per l'inferenza nel contest.")
    print("------------------------------------------------")

if __name__ == "__main__":
    train_blur_finetune()