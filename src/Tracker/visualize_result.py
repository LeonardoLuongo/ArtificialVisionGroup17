import cv2
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- SETUP PATH per importare config ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from config import *

def visualize_tracking():
    # --- CONFIGURAZIONE VISIVA ---
    # BGR format (Blue, Green, Red)
    # Magenta/Fucsia è ottimo contro il verde
    BOX_COLOR = (255, 0, 255) 
    TEXT_COLOR = (255, 0, 255)
    THICKNESS = 2
    FONT_SCALE = 0.6
    
    # --- CONFIGURAZIONE CARTELLE ---
    
    # Cerchiamo la cartella corretta in TRACKER_OUTPUT_DIR
    # Se non sei sicuro del nome, questo snippet prende l'ultima modificata
    all_subdirs = [d for d in TRACKER_OUTPUT_DIR.iterdir() if d.is_dir()]
    if not all_subdirs:
        print("Nessuna cartella risultati trovata in output.")
        return
    
    # Ordina per ultima modifica (così prende l'ultimo run fatto)
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    print(f"Userò i risultati dalla cartella: {latest_subdir.name}")
    
    trackers_folder = latest_subdir
    
    # 2. Cartella dove salvare i video
    video_output_dir = OUTPUT_DIR / "visualization_videos"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Cartella delle immagini originali
    base_data_path = DATASET_TR23_TEST

    # Ottieni lista sequenze dai risultati txt
    txt_files = sorted(list(trackers_folder.glob("*.txt")))
    
    if not txt_files:
        print(f"Nessun file .txt trovato in {trackers_folder}")
        return

    print(f"Trovate {len(txt_files)} sequenze da visualizzare.")

    for txt_file in txt_files:
        seq_name = txt_file.stem # es. "SNMOT-060"
        print(f"\n--> Generazione video per: {seq_name}")
        
        # Percorso immagini originali
        img_dir = base_data_path / seq_name / "img1"
        if not img_dir.exists():
            print(f"ATTENZIONE: Immagini non trovate per {seq_name}. Salto.")
            continue
            
        # Carica le detection/tracking dal txt
        # Formato MOT: frame, id, x, y, w, h, ...
        # Creiamo un dizionario: frame_id -> lista di [id, x, y, w, h]
        tracks = {}
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6: continue
                
                frame_idx = int(parts[0])
                obj_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                
                if frame_idx not in tracks:
                    tracks[frame_idx] = []
                tracks[frame_idx].append((obj_id, x, y, w, h))

        # Lista immagini
        img_files = sorted([f for f in img_dir.glob("*.jpg")])
        if not img_files:
            continue

        # Setup Video Writer
        # Leggiamo la prima immagine per le dimensioni
        first_img = cv2.imread(str(img_files[0]))
        height, width, _ = first_img.shape
        
        save_path = video_output_dir / f"{seq_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec mp4
        out = cv2.VideoWriter(str(save_path), fourcc, 25.0, (width, height)) # 25 FPS standard

        for img_path in tqdm(img_files, desc="Rendering frames"):
            frame_id = int(img_path.stem) # es 00001 -> 1
            
            frame = cv2.imread(str(img_path))
            
            # Se ci sono detection per questo frame, disegnale
            if frame_id in tracks:
                for (obj_id, x, y, w, h) in tracks[frame_id]:
                    # Coordinate intere per OpenCV
                    x1 = int(x)
                    y1 = int(y)
                    x2 = int(x + w)
                    y2 = int(y + h)
                    
                    # 1. Disegna Box Sottile
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
                    
                    # 2. Testo ID (Sopra il box)
                    label = f"{obj_id}"
                    
                    # Calcola dimensione testo per posizionarlo bene
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
                    
                    # Posizione testo: appena sopra il box
                    text_x = x1
                    text_y = max(y1 - 5, text_h + 5) # Evita di uscire dall'immagine in alto
                    
                    # Disegna solo il testo (senza sfondo per pulizia, come richiesto)
                    # Usiamo lo stesso colore del box per coerenza
                    cv2.putText(frame, label, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS)

            out.write(frame)

        out.release()
        print(f"Video salvato in: {save_path}")

    print("\nVisualizzazione completata!")

if __name__ == "__main__":
    visualize_tracking()