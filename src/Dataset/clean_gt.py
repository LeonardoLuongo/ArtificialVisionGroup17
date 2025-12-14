import os
import configparser
import shutil
from config import *

def clean_gt_for_soccer():
    # Percorso ai tuoi dati di test (o train se stai facendo prove)
    base_paths = [DATASET_TR23_TEST, DATASET_TR23_TRAIN]
    
    # Classi che consideriamo "TARGET" da tracciare
    # In base al PDF, il task è "Player tracking". 
    # Solitamente include portieri. Gli arbitri a volte sono esclusi o inclusi.
    # Se il gameinfo distingue "referee", decidi tu se tenerlo. 
    # Qui sotto tengo players, goalkeepers e referee.
    valid_keywords = ['player', 'goalkeeper', 'referee'] 

    
    for base_path in base_paths:
        if not os.path.exists(base_path):
            print(f"Errore: {base_path} non trovato.")
            return
        
        for seq_name in os.listdir(base_path):
            seq_path = os.path.join(base_path, seq_name)
            gameinfo_path = os.path.join(seq_path, 'gameinfo.ini')
            gt_folder = os.path.join(seq_path, 'gt')
            gt_path = os.path.join(gt_folder, 'gt.txt')
            
            # Facciamo un backup del gt originale se non esiste
            gt_backup = os.path.join(gt_folder, 'gt_backup.txt')

            if os.path.exists(gameinfo_path) and os.path.exists(gt_path):
                if not os.path.exists(gt_backup):
                    shutil.copy(gt_path, gt_backup)
                
                # Leggiamo gli ID da rimuovere (es. la palla)
                config = configparser.ConfigParser()
                config.read(gameinfo_path)
                
                ids_to_keep = []
                if 'Sequence' in config:
                    for key, value in config['Sequence'].items():
                        if key.startswith('trackletid_'):
                            # Se la descrizione contiene player/goalkeeper/referee, teniamo l'ID
                            if any(k in value.lower() for k in valid_keywords):
                                try:
                                    # trackletID_20 -> prendiamo 20
                                    obj_id = int(key.split('_')[1])
                                    ids_to_keep.append(obj_id)
                                except:
                                    pass

                # Riscriviamo il GT pulito
                with open(gt_backup, 'r') as f_in, open(gt_path, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split(',')
                        # Formato GT: frame, ID, x, y, w, h, conf, class, vis
                        if len(parts) < 6: continue
                        
                        obj_id = int(parts[1])
                        
                        if obj_id in ids_to_keep:
                            # TRUCCO TECNICO:
                            # Assicuriamoci che ci siano 10 colonne (standard MOT)
                            # E impostiamo la colonna 8 (classe) a 1 ("pedestrian" per TrackEval)
                            # Anche se sono calciatori, TrackEval vuole vedere "1" lì.
                            
                            # Recuperiamo i dati essenziali
                            frame = parts[0]
                            pid = parts[1]
                            x, y, w, h = parts[2:6]
                            
                            # Creiamo la riga perfetta per TrackEval:
                            # conf=1, class=1, visibility=1
                            new_line = f"{frame},{pid},{x},{y},{w},{h},1,1,1\n"
                            f_out.write(new_line)
                
                print(f"Pulito GT per sequenza: {seq_name}")

if __name__ == "__main__":
    clean_gt_for_soccer()