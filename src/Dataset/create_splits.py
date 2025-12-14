# =================================================
# FILE: src/Dataset/create_splits.py
# =================================================
import os
import random
from config import DATASET_TR23_TRAIN, SPLIT_DIR

# --- PARAMETRI NUOVI (Da mettere in config.py se vuoi, o lasciare qui) ---
VAL_RATIO = 0.2  # Percentuale di dati da usare per la validazione (20%)
SEED = 42        # Numero magico per la riproducibilità
# -------------------------------------------------------------------------

def generate_manifests():
    print(f"Generazione manifest partendo da: {DATASET_TR23_TRAIN}")
    
    if not DATASET_TR23_TRAIN.exists():
        print(f"ERRORE: La cartella dataset {DATASET_TR23_TRAIN} non esiste.")
        return

    # 1. Leggi tutte le cartelle (sequenze)
    sequences = [s for s in os.listdir(DATASET_TR23_TRAIN) if (DATASET_TR23_TRAIN / s).is_dir()]
    
    # 2. Ordina la lista (FONDAMENTALE per avere coerenza tra sistemi diversi)
    sequences = sorted(sequences)
    
    # 3. Mescola in modo deterministico
    random.seed(SEED)
    random.shuffle(sequences)
    
    # 4. Calcola il punto di taglio
    split_idx = int(len(sequences) * (1 - VAL_RATIO))
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]
    
    # 5. Prepara cartella output
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 6. Scrivi i file
    train_file = SPLIT_DIR / "train.txt"
    val_file = SPLIT_DIR / "val.txt"
    
    with open(train_file, "w") as f:
        f.write("\n".join(train_seqs))
        
    with open(val_file, "w") as f:
        f.write("\n".join(val_seqs))
        
    print("------------------------------------------------")
    print(f"Manifest creati con successo in: {SPLIT_DIR}")
    print(f"Train sequences ({len(train_seqs)}): salvate in train.txt")
    print(f"Val sequences   ({len(val_seqs)}): salvate in val.txt")
    print("------------------------------------------------")

if __name__ == "__main__":
    generate_manifests()