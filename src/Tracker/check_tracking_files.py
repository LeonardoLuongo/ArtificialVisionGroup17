import os
import csv
import re

def check_tracking_files(folder_path):
    # Regex per trovare file che corrispondono al pattern tracking_xxx_17.txt
    pattern = re.compile(r'^tracking_.*_17\.txt$')
    
    # Lista di tutti i file nella cartella
    files = [f for f in os.listdir(folder_path) if pattern.match(f)]
    
    if not files:
        print(f"Nessun file trovato in '{folder_path}' che segue il pattern.")
        return

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        issues_found = False
        
        # Dizionario per memorizzare gli ID visti per ogni frame {frame_id: set(object_ids)}
        frames_data = {}

        with open(file_path, mode='r') as f:
            reader = csv.reader(f)
            for line_num, row in enumerate(reader, 1):
                if not row: continue # Salta righe vuote
                
                try:
                    frame_id = row[0].strip()
                    object_id = row[1].strip()
                    
                    if frame_id not in frames_data:
                        frames_data[frame_id] = set()
                    
                    if object_id in frames_data[frame_id]:
                        print(f"[ERRORE] {filename} (linea {line_num}): ID {object_id} duplicato nel Frame {frame_id}")
                        issues_found = True
                    else:
                        frames_data[frame_id].add(object_id)
                        
                except IndexError:
                    print(f"[AVVISO] {filename} (linea {line_num}): Formato riga non valido.")

        if not issues_found:
            print(f"[OK] {filename}: Nessun duplicato trovato.")

# Inserisci qui il percorso della tua cartella 'final_provisional_tracker'
path = './out/tracker_results/final_provisional_tracker'
check_tracking_files(path)