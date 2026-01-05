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

    print(f"Trovati {len(files)} file da analizzare.\n")

    # Variabile per il totale globale (somma degli ID trovati in tutti i file)
    total_global_ids = 0

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        issues_found = False
        
        # Dizionario per memorizzare gli ID visti per ogni frame {frame_id: set(object_ids)}
        frames_data = {}
        
        # Set per contare gli ID unici specifici di QUESTO file
        unique_ids_in_file = set()

        with open(file_path, mode='r') as f:
            reader = csv.reader(f)
            for line_num, row in enumerate(reader, 1):
                if not row: continue # Salta righe vuote
                
                try:
                    frame_id = row[0].strip()
                    object_id = row[1].strip()
                    
                    # Aggiungo l'ID al set degli unici per il conteggio
                    unique_ids_in_file.add(object_id)
                    
                    # --- Controllo duplicati nello stesso frame ---
                    if frame_id not in frames_data:
                        frames_data[frame_id] = set()
                    
                    if object_id in frames_data[frame_id]:
                        print(f"[ERRORE] {filename} (linea {line_num}): ID {object_id} duplicato nel Frame {frame_id}")
                        issues_found = True
                    else:
                        frames_data[frame_id].add(object_id)
                        
                except IndexError:
                    print(f"[AVVISO] {filename} (linea {line_num}): Formato riga non valido.")

        # Calcoli finali per il singolo file
        count_current_file = len(unique_ids_in_file)
        total_global_ids += count_current_file

        if not issues_found:
            print(f"[OK] {filename}: Nessun duplicato. -> IDs unici: {count_current_file}")
        else:
            print(f"[!] {filename}: Errori trovati. -> IDs unici rilevati: {count_current_file}")

    # Stampa del riepilogo totale
    print("-" * 50)
    print(f"ANALISI COMPLETATA")
    print(f"Numero file processati: {len(files)}")
    print(f"Totale cumulativo ID unici trovati: {total_global_ids}")

# Inserisci qui il percorso della tua cartella
path = './out/tracker_results/final_results2_rgb_norm_FR_VIT'
check_tracking_files(path)