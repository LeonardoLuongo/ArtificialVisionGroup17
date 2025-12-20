# =================================================
# FILE: src/Dataset/convert_to_yolo.py
# =================================================
import os
import shutil
import sys
import cv2
from pathlib import Path
from tqdm import tqdm

# Aggiungiamo la root al path per importare config
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))


# Definiamo il tasso di campionamento (1 frame ogni 25)
TRAIN_SUBSAMPLE_RATE = 3

# Importiamo le costanti definite in config
from config import DATASET_TR23_TRAIN, YOLO_DATASET_PATH, SPLIT_DIR

def convert_dataset_from_manifest():
    print("--- Inizio Conversione Dataset (Manifest Mode) ---")
    
    # 1. Verifica esistenza file manifest
    train_manifest = SPLIT_DIR / "train.txt"
    val_manifest = SPLIT_DIR / "val.txt"
    
    if not train_manifest.exists() or not val_manifest.exists():
        print(f"[ERRORE] File manifest non trovati in {SPLIT_DIR}")
        print("Esegui prima 'src/Dataset/create_splits.py' per generare le liste!")
        return

    # 2. Leggi le liste delle sequenze
    with open(train_manifest, 'r') as f:
        train_seqs = [line.strip() for line in f if line.strip()]
        
    with open(val_manifest, 'r') as f:
        val_seqs = [line.strip() for line in f if line.strip()]
        
    print(f"Sequenze caricate -> Train: {len(train_seqs)}, Val: {len(val_seqs)}")

    # 3. Pulisci e Prepara le cartelle di destinazione (train/val)
    # Nota: Non cancelliamo SPLIT_DIR perché è dentro YOLO_DATASET_PATH
    images_dir = YOLO_DATASET_PATH / "images"
    labels_dir = YOLO_DATASET_PATH / "labels"
    
    if images_dir.exists(): shutil.rmtree(images_dir)
    if labels_dir.exists(): shutil.rmtree(labels_dir)
    
    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Funzione interna per processare una lista di sequenze
    # ------------------------------------------------------------------
    def process_split(seq_list, split_name):
        dest_img_dir = images_dir / split_name
        dest_lbl_dir = labels_dir / split_name
        
        print(f"Elaborazione split: {split_name}...")
        
        for seq_name in tqdm(seq_list):
            seq_path = DATASET_TR23_TRAIN / seq_name
            gt_path = seq_path / "gt" / "gt.txt"
            img_source_dir = seq_path / "img1"
            
            # Controllo integrità
            if not seq_path.exists():
                print(f"[WARN] La sequenza '{seq_name}' è nel manifest ma non sul disco. Saltata.")
                continue
            if not gt_path.exists():
                # Alcune sequenze potrebbero non avere GT (raro nel train set), saltiamo
                continue

            # A. Otteniamo dimensioni immagine (necessario per normalizzare YOLO)
            # Leggiamo la prima immagine disponibile
            first_img = next(img_source_dir.glob("*.jpg"), None)
            if first_img is None:
                continue
                
            img_cv = cv2.imread(str(first_img))
            if img_cv is None: continue
            img_h, img_w, _ = img_cv.shape

            # B. Leggiamo e Parsiamo il GT (Ground Truth)
            # Formato MOT: frame, id, x_top, y_top, w, h, ...
            frames_data = {}
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6: continue
                    
                    frame_id = int(parts[0])
                    x_top, y_top, w_box, h_box = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    
                    # C. Convertiamo in formato YOLO (Normalizzato 0-1 e Centro)
                    # Class ID 0 = player
                    x_center = (x_top + w_box / 2) / img_w
                    y_center = (y_top + h_box / 2) / img_h
                    w_norm = w_box / img_w
                    h_norm = h_box / img_h
                    
                    # Clipping per sicurezza (valori devono stare tra 0 e 1)
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))
                    
                    if frame_id not in frames_data:
                        frames_data[frame_id] = []
                    
                    # Stringa YOLO: class x_c y_c w h
                    frames_data[frame_id].append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # D. Copiamo le immagini e creiamo i file label
            for img_file in img_source_dir.glob("*.jpg"):
                frame_idx = int(img_file.stem) # es. '000001' -> 1

                if split_name == 'train':
                    # Esempio: tiene frame 1, 11, 21... (se il resto della div è 1)
                    # Oppure frame 10, 20, 30... (se resto è 0). 
                    # Usiamo (frame_idx - 1) % RATE == 0 per prendere il primo frame sicuro.
                    if (frame_idx - 1) % TRAIN_SUBSAMPLE_RATE != 0:
                        continue 
                
                # Nome univoco: SequenceName_ImageName.jpg
                new_filename = f"{seq_name}_{img_file.name}"
                
                # Copia immagine
                shutil.copy(img_file, dest_img_dir / new_filename)
                
                # Crea file label (.txt)
                label_filename = f"{seq_name}_{img_file.stem}.txt"
                label_path = dest_lbl_dir / label_filename
                
                content = frames_data.get(frame_idx, [])
                
                with open(label_path, 'w') as lf:
                    if content:
                        lf.write("\n".join(content))
                    else:
                        # File vuoto = nessuna detection in questo frame (Background image)
                        # YOLO lo usa per capire dove NON c'è nulla
                        pass 

    # 4. Eseguiamo la conversione per entrambi gli split
    process_split(train_seqs, 'train')
    process_split(val_seqs, 'val')

    # 5. Creazione dataset.yaml per Ultralytics
    # Usiamo path assoluti o relativi al file yaml
    yaml_content = f"""
path: {YOLO_DATASET_PATH.as_posix()}
train: images/train
val: images/val
names:
  0: player
"""
    yaml_path = YOLO_DATASET_PATH / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print("------------------------------------------------")
    print("Conversione Completata!")
    print(f"Dataset YOLO pronto in: {YOLO_DATASET_PATH}")
    print(f"File configurazione: {yaml_path}")
    print("------------------------------------------------")

if __name__ == "__main__":
    convert_dataset_from_manifest()