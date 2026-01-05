# src/Dataset/convert_spm_to_yolo.py
import os
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm
import sys

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))

# Importiamo sia TRAIN che VAL dal config
from config import *
# Nota: se nel tuo config.py hai DATASET_SPM_VAL definito separatamente, usa quello.
# Altrimenti aggiungilo qui:
BASE_SPM = DATASET_SPM_TRAIN.parent
SPM_VAL_DIR = BASE_SPM / "val"

TRAIN_SUBSAMPLE_RATE = 3 

def convert_spm():
    # 1. Creazione struttura cartelle YOLO (Aggiunto TEST)
    for split in ["train", "val", "test"]:
        (YOLO_DATASET_SPM_PATH / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_SPM_PATH / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. Iteriamo su tutti e tre gli split
    for split in ["train", "val", "test"]:
        manifest = SPLIT_DIR / f"{split}.txt"
        if not manifest.exists():
            print(f"Manifest {manifest} non trovato, salto...")
            continue
            
        with open(manifest, 'r') as f:
            sequences = [l.strip() for l in f if l.strip()]

        print(f"Convertendo lo split {split} ({len(sequences)} sequenze)...")
        for seq in tqdm(sequences):
            # LOGICA DI RICERCA: Cerca la sequenza in train o in val
            seq_path = DATASET_SPM_TRAIN / seq
            if not seq_path.exists():
                seq_path = SPM_VAL_DIR / seq
            
            if not seq_path.exists():
                print(f"[ERROR] Sequenza {seq} non trovata in {DATASET_SPM_TRAIN} o {SPM_VAL_DIR}")
                continue

            gt_path = seq_path / "gt" / "gt.txt"
            img_dir = seq_path / "img1"

            # Prendiamo dimensioni immagine
            sample_img = next(img_dir.glob("*.jpg"))
            h, w = cv2.imread(str(sample_img)).shape[:2]

            # Parsing GT
            labels = {}
            with open(gt_path, 'r') as f:
                for line in f:
                    p = line.strip().split(',')
                    if len(p) < 6: continue
                    f_idx, x, y, bw, bh = int(p[0]), float(p[2]), float(p[3]), float(p[4]), float(p[5])
                    xc, yc, wn, hn = (x + bw/2)/w, (y + bh/2)/h, bw/w, bh/h
                    if f_idx not in labels: labels[f_idx] = []
                    labels[f_idx].append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            # Copia Immagini e Scrittura Label
            for img_path in img_dir.glob("*.jpg"):
                f_idx = int(img_path.stem)
                
                # Sottocampionamento SOLO per il train per velocizzare
                # Val e Test li teniamo completi per metriche accurate
                if split == "train" and (f_idx - 1) % TRAIN_SUBSAMPLE_RATE != 0:
                    continue

                new_name = f"{seq}_{img_path.name}"
                shutil.copy(img_path, YOLO_DATASET_SPM_PATH / "images" / split / new_name)
                
                label_file = YOLO_DATASET_SPM_PATH / "labels" / split / f"{seq}_{img_path.stem}.txt"
                with open(label_file, 'w') as lf:
                    if f_idx in labels:
                        lf.write("\n".join(labels[f_idx]))

    # 3. Generazione dataset.yaml (Aggiunta riga test)
    yaml_content = f"""path: {YOLO_DATASET_SPM_PATH.absolute()}
train: images/train
val: images/val
test: images/test
names:
  0: player
"""
    with open(YOLO_DATASET_SPM_PATH / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    print(f"Conversione completata. Dataset pronto in {YOLO_DATASET_SPM_PATH}")

if __name__ == "__main__":
    convert_spm()