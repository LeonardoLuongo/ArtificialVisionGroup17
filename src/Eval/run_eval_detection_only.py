import sys
import os
import numpy as np
from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import HOTA
from config import *
# from config import * # Se usi config, assicurati che i path siano assoluti, altrimenti sovrascrivili qui sotto


def fix_ids_for_detection_only(trackers_folder):
    """
    Scansiona i file .txt dei risultati e assegna un ID univoco a ogni riga.
    Questo serve perché TrackEval crasha se vede ID duplicati (es. -1) nello stesso frame.
    """
    import glob
    
    # Cerca tutti i file .txt ricorsivamente nella cartella dei tracker
    txt_files = glob.glob(os.path.join(trackers_folder, '**', '*.txt'), recursive=True)
    
    if not txt_files:
        print(f"Nessun file .txt trovato in {trackers_folder} da correggere.")
        return

    print(f"Correzione ID per {len(txt_files)} file (Detection only mode)...")
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        # Usa un contatore globale o per riga per garantire unicità assoluta
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            # --- TRUCCO ---
            # Sostituisci la colonna ID (indice 1) con un numero progressivo (i+1)
            # Formato MOT: <frame>, <id>, <bb_left>, <bb_top>, <width>, <height>, ...
            parts[1] = str(i + 1) 
            
            new_lines.append(','.join(parts) + '\n')
            
        # Sovrascrivi il file (o salvane una copia se preferisci)
        with open(txt_file, 'w') as f:
            f.writelines(new_lines)
    
    print("Correzione ID completata.")

def run_evaluation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = ROOT_DIR
    
    # --- CORREZIONE QUI SOTTO ---
    # GT_FOLDER deve puntare alla cartella che CONTIENE le sequenze (SNMOT-XXX)
    # Non aggiungere "gt" qui.
    gt_folder_abs = DATASET_TR23_TEST 
    
    trackers_folder_abs = DETECTOR_OUTPUT_DIR 
    seqmap_folder_abs = SEQMAP_DIR 
    
    print(f"Cercando GT in: {gt_folder_abs}") 
    # Dovrebbe stampare: .../dataset/tracking-2023/test
    print(f"Cercando Seqmap in: {seqmap_folder_abs}")

    # Verifica veloce
    if not os.path.exists(os.path.join(gt_folder_abs, 'SNMOT-116', 'seqinfo.ini')):
        print("ATTENZIONE: seqinfo.ini non trovato nemmeno nel path assoluto per 116!")

    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,
    }

    dataset_config = {
        'GT_FOLDER': gt_folder_abs, 
        'TRACKERS_FOLDER': trackers_folder_abs,
        'OUTPUT_FOLDER': None,
        'TRACKERS_TO_EVAL': None, 
        'CLASSES_TO_EVAL': ['pedestrian'], 
        'BENCHMARK': 'MOT17', 
        'SPLIT_TO_EVAL': 'test',
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': False,
        'DO_PREPROC': False,
        'TRACKER_SUB_FOLDER': '', 
        'OUTPUT_SUB_FOLDER': '',
        "OUTPUT_SUMMARY": True, 
        
        # --- MODIFICA QUI ---
        # Imposta a True perché il tuo gt_folder_abs include già "/test"
        'SKIP_SPLIT_FOL': True, 
        
        'SEQ_INFO': None,
        'SUB_FOLDER': 'gt', 
        'SEQMAP_FOLDER': seqmap_folder_abs, 
        'SEQMAP_FILE': SEQMAP_FILE, 
    }
    fix_ids_for_detection_only(trackers_folder_abs)
    dataset = MotChallenge2DBox(dataset_config)
    hota = HOTA()
    
    import trackeval
    evaluator = trackeval.Evaluator(eval_config)
    try:
        np.float = float
        np.int = int   # Spesso serve anche questo
    except AttributeError:
        pass
    results = evaluator.evaluate([dataset], [hota])
    
    if results:
        # 1. Ottieni il dizionario del Dataset (livello 'MotChallenge2DBox')
        dataset_res = results[0]
        dataset_name = list(dataset_res.keys())[0]
        
        # 2. Ottieni il dizionario del Tracker (livello 'yolo')
        tracker_res = dataset_res[dataset_name]
        tracker_name = list(tracker_res.keys())[0]
        
        # 3. Ottieni il dizionario delle Sequenze
        sequences_res = tracker_res[tracker_name]
        
        # Qui hai una scelta: vuoi la media totale o una sequenza specifica?
        # TrackEval solitamente crea una chiave 'COMBINED_SEQ' per la media.
        # Se non c'è, prendiamo la prima sequenza disponibile (es. SNMOT-117).
        seq_name = 'COMBINED_SEQ'
        if seq_name not in sequences_res:
            seq_name = list(sequences_res.keys())[0]
            print(f"Attenzione: 'COMBINED_SEQ' non trovato. Mostro i risultati per: {seq_name}")
        
        # 4. Finalmente accediamo alla classe 'pedestrian' e alle metriche 'HOTA'
        # Percorso: Dataset -> Tracker -> Sequence -> Class -> Metric
        stats = sequences_res[seq_name]['pedestrian']['HOTA']
        
        deta_score = stats['DetA'][0] * 100 
        hota_score = stats['HOTA'][0] * 100

        print(f"\nMetriche finali ({tracker_name} - {seq_name}):")
        print(f"DetA: {deta_score:.2f}%")
        print(f"HOTA: {hota_score:.2f}%")
        
        # Se vuoi vedere anche DetRe (Recall) e DetPr (Precision) dato che ti interessano
        det_re = stats['DetRe'][0] * 100
        det_pr = stats['DetPr'][0] * 100
        print(f"DetRe (Recall): {det_re:.2f}%")
        print(f"DetPr (Precision): {det_pr:.2f}%")

if __name__ == "__main__":
    run_evaluation()