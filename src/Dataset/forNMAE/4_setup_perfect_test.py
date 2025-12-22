import shutil
import os
from pathlib import Path
from config import * # Assicurati che contenga i percorsi corretti

def setup_perfect_simulation(video_id, team_id, dataset_dir, preds_dir):
    """
    Copia GT Tracking e GT Behavior nella cartella delle predizioni
    per simulare un punteggio perfetto.
    """
    video_path = Path(dataset_dir) / str(video_id)
    dest_path = Path(preds_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # 1. Copia Behavior (Il test principale che vuoi fare)
    src_beh = video_path / "gt" / "behavior_gt.txt"
    dst_beh = dest_path / f"behavior_{video_id}_{team_id}.txt"

    if src_beh.exists():
        shutil.copy(src_beh, dst_beh)
        print(f"[OK] Behavior GT copiato come predizione: {dst_beh.name}")
    else:
        print(f"[ERR] Manca behavior_gt.txt in {video_path}/gt/. Hai lanciato lo script di generazione?")
        return

    # 2. Copia Tracking (Necessario per vedere i box nel simulatore)
    # Nota: Il simulatore legge le prime 6 colonne, quindi il formato a 10 colonne del GT va bene lo stesso.
    src_track = video_path / "gt" / "gt.txt"
    dst_track = dest_path / f"tracking_{video_id}_{team_id}.txt"

    if src_track.exists():
        shutil.copy(src_track, dst_track)
        print(f"[OK] Tracking GT copiato come predizione: {dst_track.name}")
    else:
        print(f"[ERR] Manca gt.txt in {video_path}/gt/")

if __name__ == "__main__":
    # --- CONFIGURAZIONE ---
    VIDEO_TO_TEST = "116"   # Il video che vuoi visualizzare
    MY_TEAM_ID = "17"       # Il tuo ID
    
    # Percorsi (Adattali se diversi nel tuo config)
    DATASET_DIR = DATASET_TR23_TEST 
    PREDS_DIR = "lecture_example_from_training/Predictions_folder"

    print(f"--- Configurazione Test Perfetto per Video {VIDEO_TO_TEST} ---")
    setup_perfect_simulation(VIDEO_TO_TEST, MY_TEAM_ID, DATASET_DIR, PREDS_DIR)
    
    print("\nOra esegui il simulatore del professore!")
    print(f"Comando suggerito (adatta i percorsi):")
    print(f'python simulator.py')