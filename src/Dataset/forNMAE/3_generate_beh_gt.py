import os
import json
import csv
from pathlib import Path
from collections import defaultdict
from config import * # Assicurati che DATASET_TR23_TEST sia definito qui

def generate_behavior_gt(video_folder):
    video_path = Path(video_folder)
    gt_file = video_path / "gt" / "gt.txt"
    roi_file = video_path / "roi.json"
    output_file = video_path / "gt" / "behavior_gt.txt"

    # Controllo esistenza file necessari
    if not (gt_file.exists() and roi_file.exists()):
        print(f"SKIP: File mancanti in {video_folder.name}")
        return

    # 1. Carica configurazione ROI
    with open(roi_file, 'r') as f:
        rois = json.load(f)

    W, H = 1920, 1080 # Risoluzione standard SoccerNet

    # 2. Struttura dati per raggruppare i giocatori per frame
    # frame_players[frame_id] = lista di tuple (cx, cy)
    frame_players = defaultdict(list)
    max_frame = 0

    print(f"Leggo GT per {video_folder.name}...")

    # Leggiamo il file GT che è ordinato per ID (non per frame)
    with open(gt_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            try:
                frame = int(row[0])
                # Parsing coordinate
                x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                
                # Calcolo punto "Centro della Base"
                cx = x + (w / 2.0)
                cy = y + h

                # Salviamo il giocatore nel frame corrispondente
                frame_players[frame].append((cx, cy))

                # Teniamo traccia del frame più alto visto finora
                if frame > max_frame:
                    max_frame = frame
            except ValueError:
                continue

    # 3. Pre-calcolo limiti ROI in pixel (per velocità)
    roi_limits = {}
    for rid in [1, 2]:
        r = rois[f"roi{rid}"]
        roi_limits[rid] = {
            'x_min': r["x"] * W,
            'x_max': (r["x"] + r["width"]) * W,
            'y_min': r["y"] * H,
            'y_max': (r["y"] + r["height"]) * H
        }

    # 4. Scrittura sequenziale ordinata (1 -> max_frame)
    # Questo risolve il problema dell'ordine del file di input
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Ciclo forzato dal frame 1 all'ultimo frame
        for fr in range(1, max_frame + 1):
            
            # Conta giocatori per questo frame specifico
            counts = {1: 0, 2: 0}
            
            # Recupera la lista di giocatori per questo frame (o lista vuota se nessuno)
            players_in_frame = frame_players.get(fr, [])
            
            for (cx, cy) in players_in_frame:
                for rid in [1, 2]:
                    lim = roi_limits[rid]
                    # Check se il punto è nel rettangolo
                    if (lim['x_min'] <= cx <= lim['x_max']) and \
                       (lim['y_min'] <= cy <= lim['y_max']):
                        counts[rid] += 1
            
            # Scrittura delle due righe per il frame corrente
            writer.writerow([fr, 1, counts[1]])
            writer.writerow([fr, 2, counts[2]])

    print(f" -> Generato: {output_file.name} (Max Frame: {max_frame})")

if __name__ == "__main__":
    # Sostituisci con il percorso corretto se non usi config.py
    base_dataset_dir = DATASET_TR23_TEST
    
    # Filtra solo cartelle numeriche (116, 117...)
    all_folders = sorted(
        [d for d in base_dataset_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )

    if not all_folders:
        print("Nessuna cartella numerica trovata!")
    else:
        for folder in all_folders:
            generate_behavior_gt(folder)