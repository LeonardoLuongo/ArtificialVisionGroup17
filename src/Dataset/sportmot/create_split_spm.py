# src/Dataset/create_splits_spm.py
import random
import os
from pathlib import Path
import sys

# Setup path
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))
from config import *


def generate_spm_football_splits():
    football_txt_path = DATASET_SPM / "splits_txt" / "football.txt" 
    
    if not football_txt_path.exists():
        print(f"Errore: {football_txt_path} non trovato!")
        return

    # 1. Leggi tutte le sequenze di calcio
    with open(football_txt_path, 'r') as f:
        all_football_seqs = [line.strip() for line in f if line.strip()]
    
    # 2. Creiamo un pool di sequenze che HANNO i Ground Truth (per poter fare il test locale)
    labeled_pool = []
    for seq in all_football_seqs:
        # Controlliamo se la sequenza è in una cartella che solitamente ha i GT
        if (DATASET_SPM_TRAIN / seq).exists() or (DATASET_SPM_VAL / seq).exists():
            labeled_pool.append(seq)

    # 3. Shuffle deterministico
    random.seed(42)
    random.shuffle(labeled_pool)
    
    # 4. Calcolo degli indici per lo split 70% / 20% / 10%
    total = len(labeled_pool)
    train_end = int(total * 0.7)
    val_end   = int(total * 0.9) # 0.7 + 0.2

    train_seqs = labeled_pool[:train_end]
    val_seqs   = labeled_pool[train_end:val_end]
    test_seqs  = labeled_pool[val_end:]
    
    # 5. Salvataggio manifest
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    manifests = {
        "train.txt": train_seqs,
        "val.txt": val_seqs,
        "test.txt": test_seqs
    }
    
    for filename, seq_list in manifests.items():
        with open(SPLIT_DIR / filename, "w") as f:
            f.write("\n".join(seq_list))
            
    print("------------------------------------------------")
    print(f"Split 70/20/10 creato con successo in {SPLIT_DIR}:")
    print(f" - Train (70%): {len(train_seqs)} sequenze (usate per l'addestramento)")
    print(f" - Val   (20%): {len(val_seqs)} sequenze (usate per fitness/early stopping)")
    print(f" - Test  (10%): {len(test_seqs)} sequenze (usate per calcolare mAP finale)")
    print("------------------------------------------------")

if __name__ == "__main__":
    generate_spm_football_splits()