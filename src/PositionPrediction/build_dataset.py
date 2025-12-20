import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from config import DATASET_TR23_TRAIN, DATASET_TR23_TEST

# --- CONFIGURAZIONE ---
WINDOW_SIZE = 10
STEP = 2            # Saltiamo 2 frame tra una finestra e l'altra (riduce i dati duplicati del 50%)
IMG_W, IMG_H = 1920, 1080
VAL_RATIO = 0.2     # Percentuale di video del train da usare come validation

def process_sequences(seq_paths, desc="Processing"):
    """ Estrae X (storia) e Y (target) da una lista di cartelle video """
    all_x = []
    all_y = []
    
    for seq_path in tqdm(seq_paths, desc=desc):
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
        if not os.path.exists(gt_file): continue
        
        # Caricamento veloce con numpy
        try:
            data = np.loadtxt(gt_file, delimiter=',')
        except: continue

        unique_ids = np.unique(data[:, 1])
        for obj_id in unique_ids:
            player_data = data[data[:, 1] == obj_id]
            player_data = player_data[player_data[:, 0].argsort()]
            
            if len(player_data) < (WINDOW_SIZE + 1): continue
            
            # Estrazione e normalizzazione
            coords = player_data[:, 2:6].copy()
            coords[:, [0, 2]] /= IMG_W
            coords[:, [1, 3]] /= IMG_H
            
            # Sliding window con STEP per velocizzare
            for i in range(0, len(coords) - WINDOW_SIZE, STEP):
                all_x.append(coords[i : i + WINDOW_SIZE])
                all_y.append(coords[i + WINDOW_SIZE])
                
    return torch.FloatTensor(np.array(all_x)), torch.FloatTensor(np.array(all_y))

def build_and_save():
    # 1. Creazione cartelle
    base_out = Path(__file__).parent / "dataset"
    for folder in ["train", "val", "test"]:
        (base_out / folder).mkdir(parents=True, exist_ok=True)

    # 2. Split dei video di Training (Train vs Validation)
    all_train_seqs = [os.path.join(DATASET_TR23_TRAIN, d) for d in os.listdir(DATASET_TR23_TRAIN) 
                      if os.path.isdir(os.path.join(DATASET_TR23_TRAIN, d))]
    np.random.seed(42)
    np.random.shuffle(all_train_seqs)
    
    val_count = int(len(all_train_seqs) * VAL_RATIO)
    val_seqs = all_train_seqs[:val_count]
    train_seqs = all_train_seqs[val_count:]

    # 3. Elaborazione Train
    print("\n--- Elaborazione Dataset di Training ---")
    train_x, train_y = process_sequences(train_seqs, desc="Train Set")
    torch.save(train_x, base_out / "train" / "x.pth")
    torch.save(train_y, base_out / "train" / "y.pth")
    print(f"Salvato Train: {train_x.shape[0]} campioni")

    # 4. Elaborazione Val
    print("\n--- Elaborazione Dataset di Validation ---")
    val_x, val_y = process_sequences(val_seqs, desc="Val Set")
    torch.save(val_x, base_out / "val" / "x.pth")
    torch.save(val_y, base_out / "val" / "y.pth")
    print(f"Salvato Val: {val_x.shape[0]} campioni")

    # 5. Elaborazione Test (GT ufficiale del test set)
    print("\n--- Elaborazione Dataset di Test ---")
    test_seqs = [os.path.join(DATASET_TR23_TEST, d) for d in os.listdir(DATASET_TR23_TEST) 
                 if os.path.isdir(os.path.join(DATASET_TR23_TEST, d))]
    test_x, test_y = process_sequences(test_seqs, desc="Test Set")
    torch.save(test_x, base_out / "test" / "x.pth")
    torch.save(test_y, base_out / "test" / "y.pth")
    print(f"Salvato Test: {test_x.shape[0]} campioni")

if __name__ == "__main__":
    build_and_save()