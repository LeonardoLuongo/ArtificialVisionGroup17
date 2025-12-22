# =================================================
# FILE: src/BlurFineTuning/smart_extract.py
# =================================================
import cv2
import numpy as np
import shutil
import sys
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import config (Assicurati che config.py esporti queste variabili)
from config import *

# --- CONFIGURAZIONE ---
BLUR_DATASET_DIR = DATASET_BASIC_PATH / "blur_dataset"
OUT_IMG_DIR = BLUR_DATASET_DIR / "images/train"
OUT_LBL_DIR = BLUR_DATASET_DIR / "labels/train"
SPLIT_DIR = BLUR_DATASET_DIR / "splits"

# Parametri Selezione
PERCENTILE_MOTION = 90 
PERCENTILE_BLUR = 20   
MAX_FRAMES_PER_SEQ = 30  # Ridotto leggermente per bilanciare meglio

def calculate_metrics(frame, prev_small):
    """Calcola metriche su versione ridotta per velocità."""
    # Resize aggressivo per performance (480px width)
    h, w = frame.shape[:2]
    scale = 480 / w
    small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    # Blur: Varianza Laplaciano
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Motion: Optical Flow
    motion_score = 0.0
    if prev_small is not None:
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        # Farneback flow parametrizzato per velocità
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)
        
    return blur_score, motion_score, small

def get_gt_data(gt_path, img_w, img_h):
    """Legge il GT e lo converte in YOLO format in memoria."""
    gt_data = {}
    if not gt_path.exists(): return {}

    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            frame_id = int(parts[0])
            # Se ci sono più di 7 colonne, la classe è alla 7, altrimenti default 1
            cls_id_raw = int(parts[7]) if len(parts) > 7 else 1
            
            # FILTRO: Player(1), GK(2), Ref(3).
            if cls_id_raw in [1, 2, 3]: 
                x, y, w_box, h_box = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                
                # Normalizzazione YOLO
                x_c = max(0, min(1, (x + w_box / 2) / img_w))
                y_c = max(0, min(1, (y + h_box / 2) / img_h))
                w_n = max(0, min(1, w_box / img_w))
                h_n = max(0, min(1, h_box / img_h))
                
                if frame_id not in gt_data: gt_data[frame_id] = []
                # Classe 0 fissa per il detector
                gt_data[frame_id].append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
                
    return gt_data

def process_single_sequence(seq_path):
    """Funzione worker eseguita in parallelo."""
    try:
        img_dir = seq_path / "img1"
        gt_path = seq_path / "gt/gt.txt"
        
        if not img_dir.exists() or not gt_path.exists(): 
            return []

        frames = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        if not frames: return []
        
        # Init dimensioni
        first_img = cv2.imread(str(frames[0]))
        if first_img is None: return []
        real_h, real_w = first_img.shape[:2]
        
        # Parse GT
        gt_dict = get_gt_data(gt_path, real_w, real_h)
        if not gt_dict: return []

        frame_stats = []
        prev_small = None
        
        # 1. Calcolo Metriche Frame per Frame
        for f_path in frames:
            frame = cv2.imread(str(f_path))
            if frame is None: continue
            
            b_score, m_score, small = calculate_metrics(frame, prev_small)
            prev_small = small
            
            f_idx = int(f_path.stem)
            
            if f_idx in gt_dict: # Solo se ha label
                frame_stats.append({
                    "path": f_path,
                    "idx": f_idx,
                    "blur": b_score,
                    "motion": m_score
                })

        if not frame_stats: return []

        # 2. Soglie Locali
        b_vals = [x['blur'] for x in frame_stats]
        m_vals = [x['motion'] for x in frame_stats]
        thresh_blur = np.percentile(b_vals, PERCENTILE_BLUR)
        thresh_motion = np.percentile(m_vals, PERCENTILE_MOTION)
        
        # 3. Score e Selezione
        candidates = []
        for item in frame_stats:
            if item['blur'] <= thresh_blur or item['motion'] >= thresh_motion:
                norm_blur = 1.0 / (item['blur'] + 1e-5) 
                severity_score = norm_blur + (item['motion'] * 2) # Diamo peso al motion
                item['severity'] = severity_score
                candidates.append(item)
        
        candidates.sort(key=lambda x: x['severity'], reverse=True)
        selected = candidates[:MAX_FRAMES_PER_SEQ]
        
        # 4. Copia e Salvataggio (ritorna i percorsi salvati)
        saved_files = []
        for item in selected:
            new_name = f"{seq_path.name}_{item['idx']:06d}"
            
            dest_img = OUT_IMG_DIR / f"{new_name}.jpg"
            dest_lbl = OUT_LBL_DIR / f"{new_name}.txt"
            
            # Scrittura su disco
            shutil.copy(item['path'], dest_img)
            with open(dest_lbl, 'w') as f_lbl:
                f_lbl.write("\n".join(gt_dict[item['idx']]))
            
            saved_files.append(str(dest_img.resolve()))
            
        return saved_files
        
    except Exception as e:
        print(f"Errore su sequenza {seq_path.name}: {e}")
        return []

def main():
    # 1. Setup Cartelle
    if BLUR_DATASET_DIR.exists():
        print("Pulizia vecchia cartella blur...")
        shutil.rmtree(BLUR_DATASET_DIR)
    
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Raccogli Sequenze
    if not DATASET_TR23_TRAIN.exists():
        print(f"Errore path: {DATASET_TR23_TRAIN} non trovato.")
        return

    seqs = sorted([x for x in DATASET_TR23_TRAIN.iterdir() if x.is_dir()])
    print(f"Avvio estrazione parallela su {len(seqs)} sequenze...")

    all_train_images = []

    # 3. Esecuzione Parallela
    # Usa il numero di CPU - 1 per lasciare respiro al sistema
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_sequence, seq): seq for seq in seqs}
        
        for future in tqdm(as_completed(futures), total=len(seqs), desc="Estrazione"):
            result = future.result()
            all_train_images.extend(result)

    print(f"\nEstrazione completata. Totale immagini: {len(all_train_images)}")

    # 4. Generazione SPLITS (Train e Val)
    # Train: Le immagini appena estratte
    train_txt_path = SPLIT_DIR / "train.txt"
    with open(train_txt_path, 'w') as f:
        f.write("\n".join(all_train_images))

    real_val_dir = YOLO_DATASET_PATH / "images/val"
    val_txt_path = SPLIT_DIR / "val.txt"
    
    if real_val_dir.exists():
        # Scrivi i path delle immagini .jpg esistenti
        val_imgs = sorted(list(real_val_dir.glob("*.jpg")))
        with open(val_txt_path, 'w') as f:
            for img in val_imgs:
                f.write(str(img.resolve()) + "\n")
        print(f"Validation split linkato a {len(val_imgs)} immagini esistenti.")
    else:
        print("ATTENZIONE: Immagini di validazione (dataset_yolo/images/val) non trovate.")

    # 5. Generazione Dataset YAML
    yaml_content = f"""
path: {BLUR_DATASET_DIR.as_posix()}
train: splits/train.txt
val: splits/val.txt
names:
  0: player
"""
    yaml_path = BLUR_DATASET_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print("------------------------------------------------")
    print(f"Dataset pronto in: {BLUR_DATASET_DIR}")
    print(f"File YAML: {yaml_path}")
    print("Ora puoi lanciare il training passando questo YAML!")
    print("------------------------------------------------")

if __name__ == "__main__":
    multiprocessing.freeze_support() # Necessario per Windows
    main()