import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# Setup path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from config import TRACKER_OUTPUT_DIR, OUTPUT_DIR

def interpolate_tracks(max_gap=20):
    """
    max_gap: Numero massimo di frame mancanti da riempire. 
             20 frame = 0.8 secondi (a 25fps). Oltre questo è rischioso inventare.
    """
    
    # 1. Trova l'ultima cartella dei risultati
    if not TRACKER_OUTPUT_DIR.exists():
        print("Nessuna cartella risultati trovata.")
        return

    all_subdirs = [d for d in TRACKER_OUTPUT_DIR.iterdir() if d.is_dir()]
    if not all_subdirs:
        print("Nessuna sottocartella trovata.")
        return
        
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    print(f"Interpolazione sui file in: {latest_subdir.name}")
    
    # 2. Crea cartella di output per i file interpolati
    interpolated_dir = TRACKER_OUTPUT_DIR / f"{latest_subdir.name}_interpolated"
    interpolated_dir.mkdir(parents=True, exist_ok=True)
    
    txt_files = sorted(list(latest_subdir.glob("*.txt")))
    
    total_added_boxes = 0

    for txt_file in txt_files:
        seq_name = txt_file.stem
        print(f"--> Elaborazione: {seq_name}")
        
        # --- LETTURA ---
        # Struttura dati: tracks[track_id] = [ (frame, x, y, w, h, conf), ... ]
        tracks = defaultdict(list)
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            frame = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            
            tracks[track_id].append((frame, x, y, w, h, conf))
            
        # --- INTERPOLAZIONE ---
        new_lines = []
        seq_added = 0
        
        for track_id, data in tracks.items():
            # Ordina per frame (fondamentale)
            data.sort(key=lambda x: x[0])
            
            # Aggiungi i dati originali alla lista finale
            for d in data:
                # MOT Format: frame, id, x, y, w, h, conf, -1, 1, -1
                line = f"{d[0]},{track_id},{d[1]:.2f},{d[2]:.2f},{d[3]:.2f},{d[4]:.2f},{d[5]:.4f},-1,1,-1\n"
                new_lines.append(line)
            
            # Cerca buchi
            for i in range(len(data) - 1):
                curr_frame, curr_x, curr_y, curr_w, curr_h, _ = data[i]
                next_frame, next_x, next_y, next_w, next_h, _ = data[i+1]
                
                gap = next_frame - curr_frame
                
                # Se c'è un buco e non è troppo grande (es. giocatore uscito dal campo)
                if 1 < gap <= max_gap:
                    # Riempiamo i frame mancanti
                    for step in range(1, gap):
                        interp_frame = curr_frame + step
                        
                        # Interpolazione Lineare (Lerp)
                        alpha = step / gap
                        interp_x = curr_x + (next_x - curr_x) * alpha
                        interp_y = curr_y + (next_y - curr_y) * alpha
                        interp_w = curr_w + (next_w - curr_w) * alpha
                        interp_h = curr_h + (next_h - curr_h) * alpha
                        
                        # Confidenza? Diamogli una media o un valore fisso basso
                        # (Non importa molto per TrackEval se non si usano soglie strane)
                        interp_conf = 0.5 
                        
                        line = f"{interp_frame},{track_id},{interp_x:.2f},{interp_y:.2f},{interp_w:.2f},{interp_h:.2f},{interp_conf:.4f},-1,1,-1\n"
                        new_lines.append(line)
                        seq_added += 1

        # --- SALVATAGGIO ---
        # Riordiniamo tutto il file per frame (MOT format lo richiede spesso)
        # Parse temporaneo per ordinare
        final_data = []
        for line in new_lines:
            parts = line.split(',')
            final_data.append((int(parts[0]), line))
            
        final_data.sort(key=lambda x: x[0])
        
        out_path = interpolated_dir / f"{seq_name}.txt"
        with open(out_path, 'w') as f:
            for _, line in final_data:
                f.write(line)
                
        print(f"    + Aggiunti {seq_added} box interpolati.")
        total_added_boxes += seq_added

    print(f"\nInterpolazione conclusa. Totale box recuperati: {total_added_boxes}")
    print(f"Nuovi risultati salvati in: {interpolated_dir}")
    print("ORA: Modifica run_eval.py per puntare a questa cartella e vedi quanto hai guadagnato!")

if __name__ == "__main__":
    interpolate_tracks(max_gap=20) # 20 frame = 0.8 secondi