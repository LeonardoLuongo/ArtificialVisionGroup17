import os
import csv
from typing import Dict, Tuple
from config import *

def read_behavior(path: str) -> Dict[Tuple[int, int], int]:
    """Legge i file behavior (frame, region_id, n_players)"""
    out: Dict[Tuple[int, int], int] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                fr = int(row[0])
                rid = int(row[1])
                n = int(float(row[2]))
                out[(fr, rid)] = n
            except ValueError:
                continue
    return out

def evaluate_nmae(dataset_root: str, predictions_root: str, team_id: str):
    """
    Calcola MAE e nMAE globale su tutti i video trovati nel dataset.
    """
    abs_err_sum = 0.0
    total_entries = 0
    
    # Trova tutte le cartelle numeriche nel dataset (es. 116, 117...)
    video_ids = [d for d in os.listdir(dataset_root) 
                 if os.path.isdir(os.path.join(dataset_root, d)) and d.isdigit()]
    video_ids.sort(key=lambda s: int(s))

    print(f"--- Valutazione nMAE per il Team {team_id} ---")

    for vid in video_ids:
        gt_path = os.path.join(dataset_root, vid, "gt", "behavior_gt.txt")
        pr_path = os.path.join(predictions_root, f"behavior_{vid}_{team_id}.txt")

        if not os.path.isfile(gt_path):
            print(f"Attenzione: Ground Truth mancante per video {vid} ({gt_path})")
            continue
        
        if not os.path.isfile(pr_path):
            print(f"Attenzione: Predizione mancante per video {vid} ({pr_path})")
            # Se la predizione manca, il professore considera 0 giocatori (penalità)
            pr_b = {} 
        else:
            pr_b = read_behavior(pr_path)
            
        gt_b = read_behavior(gt_path)

        # Calcolo errore per ogni entry nel Ground Truth
        video_err = 0.0
        video_count = 0
        for key, gt_val in gt_b.items():
            pred_val = pr_b.get(key, 0) # Default a 0 se manca il frame/roi nelle predizioni
            err = abs(pred_val - gt_val)
            abs_err_sum += err
            video_err += err
            total_entries += 1
            video_count += 1
            
        if video_count > 0:
            print(f"Video {vid}: MAE = {video_err/video_count:.4f}")

    if total_entries == 0:
        print("Errore: Nessun dato trovato per la valutazione.")
        return

    # Formule ufficiali 
    mae = abs_err_sum / total_entries
    nmae = (10.0 - min(10.0, max(0.0, mae))) / 10.0

    print("-" * 40)
    print(f"MAE GLOBALE:  {mae:.6f}")
    print(f"nMAE FINALE:  {nmae:.6f}")
    print("-" * 40)

if __name__ == "__main__":
    # Configura i tuoi percorsi qui
    DATASET_DIR =  DATASET_TR23_TEST # Dopo aver rinominato le cartelle in 116, 117...
    PREDS_DIR = TRACKER_OUTPUT_DIR / "final_robust_tracker"
    MY_TEAM_ID = "17" # Il tuo XX

    evaluate_nmae(DATASET_DIR, PREDS_DIR, MY_TEAM_ID)