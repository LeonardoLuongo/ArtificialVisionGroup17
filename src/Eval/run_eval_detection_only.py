import sys
import os
import numpy as np
from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import HOTA
import trackeval
from config import * # Assicurati che config.py sia accessibile

def fix_ids_for_detection_only(trackers_folder):
    """
    Scansiona i file .txt dei risultati e assegna un ID univoco a ogni riga.
    NOTA: Questo distrugge le informazioni di tracking temporale (AssA sarà ~0),
    ma è perfetto se ti interessa calcolare SOLO la DetA (Detection Accuracy).
    """
    import glob
    
    txt_files = glob.glob(os.path.join(trackers_folder, '**', '*.txt'), recursive=True)
    
    if not txt_files:
        print(f"Nessun file .txt trovato in {trackers_folder} da correggere.")
        return

    print(f"Correzione ID per {len(txt_files)} file (Detection only mode)...")
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            # Assegna un ID progressivo unico per riga (rompe il tracking, ok per detection)
            parts[1] = str(i + 1) 
            
            new_lines.append(','.join(parts) + '\n')
            
        with open(txt_file, 'w') as f:
            f.writelines(new_lines)
    
    print("Correzione ID completata.")

def run_evaluation():
    # --- CONFIGURAZIONE PATH ---
    gt_folder_abs = DATASET_TR23_TEST 
    trackers_folder_abs = DETECTOR_OUTPUT_DIR 
    seqmap_folder_abs = SEQMAP_DIR 
    
    print(f"GT Folder: {gt_folder_abs}") 
    print(f"Seqmap Folder: {seqmap_folder_abs}")

    # Configurazione di base per l'evaluator
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'PRINT_RESULTS': False,  # Mettiamo False per non intasare la console, stampiamo noi i dati custom
        'PRINT_ONLY_COMBINED': False,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': False,
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': True,
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
        'OUTPUT_SUMMARY': True, 
        'SKIP_SPLIT_FOL': True, 
        'SEQ_INFO': None,
        'SUB_FOLDER': 'gt', 
        'SEQMAP_FOLDER': seqmap_folder_abs, 
        'SEQMAP_FILE': SEQMAP_FILE, 
    }

    # Eseguiamo la correzione degli ID per calcolare la Detection pura
    fix_ids_for_detection_only(trackers_folder_abs)
    
    dataset = MotChallenge2DBox(dataset_config)
    hota = HOTA()
    
    evaluator = trackeval.Evaluator(eval_config)
    
    # Fix per compatibilità numpy
    try:
        np.float = float
        np.int = int
    except AttributeError:
        pass

    # Esecuzione
    results = evaluator.evaluate([dataset], [hota])
    
    if results:
        dataset_res = results[0]
        dataset_name = list(dataset_res.keys())[0]
        tracker_res = dataset_res[dataset_name]
        tracker_name = list(tracker_res.keys())[0]
        sequences_res = tracker_res[tracker_name]
        
        seq_name = 'COMBINED_SEQ'
        if seq_name not in sequences_res:
            seq_name = list(sequences_res.keys())[0]
            print(f"Attenzione: COMBINED_SEQ non trovato. Uso: {seq_name}")
        
        # Estrazione metriche HOTA
        stats = sequences_res[seq_name]['pedestrian']['HOTA']
        
        # --- MODIFICA CRUCIALE PER ESTRARRE SOLO 0.5 ---
        # L'array 'DetA' ha 19 valori corrispondenti alle soglie:
        # 0.05, 0.10, ..., 0.50, ..., 0.95
        # L'indice 9 corrisponde esattamente alla soglia 0.50
        index_05 = 9 
        
        # Estrazione valori specifici all'indice 9
        deta_05 = stats['DetA'][index_05] * 100
        det_re_05 = stats['DetRe'][index_05] * 100
        det_pr_05 = stats['DetPr'][index_05] * 100
        
        # NOTA: Calcoliamo anche HOTA e AssA a 0.5, ma attenzione:
        # Poiché usi "fix_ids_for_detection_only", AssA sarà bassissima (~0),
        # e quindi anche HOTA sarà bassissima. 
        # Tuttavia, DetA sarà corretta.
        hota_05 = stats['HOTA'][index_05] * 100
        assa_05 = stats['AssA'][index_05] * 100

        print("-" * 40)
        print(f"RISULTATI SPECIFICI PER SOGLIA IoU = 0.5 ({tracker_name})")
        print("-" * 40)
        print(f"DetA @ 0.5 (Detection Accuracy):  {deta_05:.2f}%")
        print(f"DetRe @ 0.5 (Recall):             {det_re_05:.2f}%")
        print(f"DetPr @ 0.5 (Precision):          {det_pr_05:.2f}%")
        print("-" * 40)
        print("Metriche di Tracking (saranno basse se hai resettato gli ID):")
        print(f"AssA @ 0.5 (Association Acc):     {assa_05:.2f}%")
        print(f"HOTA @ 0.5:                       {hota_05:.2f}%")
        print("-" * 40)

if __name__ == "__main__":
    run_evaluation()