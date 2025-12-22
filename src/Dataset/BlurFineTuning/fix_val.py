# =================================================
# FILE: src/BlurFineTuning/fix_val.py
# =================================================
from pathlib import Path
import sys
import yaml

from config import *

def fix_validation_split():
    print("--- Riparazione file splits/val.txt ---")
    
    # 1. Trova dove sono le vere immagini di validazione (create dal tuo convert_to_yolo.py)
    real_val_dir = YOLO_DATASET_PATH / "images/val"
    
    if not real_val_dir.exists():
        print(f"ERRORE CRITICO: Non trovo la cartella immagini di validazione: {real_val_dir}")
        print("Hai eseguito src/Dataset/convert_to_yolo.py?")
        return

    # 2. Ottieni la lista di tutti i file .jpg
    val_images = sorted(list(real_val_dir.glob("*.jpg")))
    print(f"Trovate {len(val_images)} immagini di validazione in {real_val_dir.name}")

    # 3. Sovrascrivi il file val.txt errato con i percorsi delle immagini
    blur_dataset_split_dir = ROOT_DIR / "dataset/blur_dataset/splits"
    blur_dataset_split_dir.mkdir(parents=True, exist_ok=True)
    
    new_val_txt = blur_dataset_split_dir / "val.txt"
    
    with open(new_val_txt, 'w') as f:
        for img in val_images:
            f.write(str(img.resolve()) + "\n")
            
    print(f"File corretto salvato in: {new_val_txt}")

    # 4. Aggiorna dataset.yaml per sicurezza
    # A volte YOLO preferisce puntare direttamente alla cartella invece che al txt
    # Ma se usiamo il txt, deve essere fatto come sopra.
    # Rigeneriamo lo YAML per essere sicuri al 100%
    
    yaml_path = ROOT_DIR / "dataset/blur_dataset/dataset.yaml"
    train_txt = blur_dataset_split_dir / "train.txt"
    
    if not train_txt.exists():
         print("ATTENZIONE: train.txt non trovato, ma rigenero lo yaml lo stesso.")

    yaml_content = {
        'path': str((ROOT_DIR / "dataset/blur_dataset").resolve()),
        'train': str(train_txt.resolve()),
        'val': str(new_val_txt.resolve()), # Ora punta al txt con i percorsi giusti
        'names': {0: 'player'}
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    print(f"YAML aggiornato: {yaml_path}")
    print("\n>>> ORA PUOI ESEGUIRE train_blur.py SENZA ERRORI <<<")

if __name__ == "__main__":
    fix_validation_split()