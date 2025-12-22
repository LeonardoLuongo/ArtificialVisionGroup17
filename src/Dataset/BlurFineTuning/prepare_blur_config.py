# =================================================
# FILE: src/BlurFineTuning/prepare_blur_config.py
# =================================================
import yaml
from pathlib import Path
import sys

# Setup path
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))
from config import YOLO_DATASET_PATH # Il path del dataset originale

def create_blur_yaml():
    blur_dataset_path = ROOT_DIR / "dataset/blur_dataset"
    
    # 1. Generiamo il train.txt per i nuovi dati sfocati
    train_imgs = sorted(list((blur_dataset_path / "images/train").glob("*.jpg")))
    
    if not train_imgs:
        print("Nessuna immagine trovata! Esegui prima smart_extract.py")
        return

    blur_train_txt = blur_dataset_path / "train_blur.txt"
    with open(blur_train_txt, 'w') as f:
        for img in train_imgs:
            f.write(str(img.resolve()) + "\n")
            
    print(f"File lista training creato con {len(train_imgs)} immagini sfocate.")

    # 2. Recuperiamo il val.txt ORIGINALE
    # Questo è fondamentale per confrontare le pere con le pere
    original_val_txt = YOLO_DATASET_PATH / "splits/val.txt" # O dove convert_to_yolo li mette
    # Se convert_to_yolo non ha creato splits/val.txt ma usa una cartella, puntiamo alla cartella
    # Dallo script convert_to_yolo vedo: val: images/val
    
    # Se vuoi usare le immagini di validazione originali:
    original_val_dir = YOLO_DATASET_PATH / "images/val"

    # 3. Creiamo lo YAML
    data = {
        'path': str(ROOT_DIR), # Root relativa per flessibilità
        
        # Train sui nuovi dati difficili
        'train': str(blur_train_txt.resolve()), 
        
        # Validation sui dati originali (per monitorare stabilità)
        'val': str(original_val_dir.resolve()), 
        
        # Manteniamo ESATTAMENTE le classi del tuo script convert_to_yolo
        'names': {
            0: 'player' 
        }
    }
    
    yaml_out = blur_dataset_path / "blur_finetune.yaml"
    with open(yaml_out, 'w') as f:
        yaml.dump(data, f)
        
    print(f"YAML creato: {yaml_out}")
    print("Usa questo file come 'data=...' nel comando di training.")

if __name__ == "__main__":
    create_blur_yaml()