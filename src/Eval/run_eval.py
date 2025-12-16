import sys
import os
import numpy as np
from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import HOTA
import trackeval
from config import *

def run_evaluation():
    # --- CONFIGURAZIONE PATH ---
    # GT_FOLDER deve puntare alla cartella che CONTIENE le sequenze (SNMOT-XXX)
    gt_folder_abs = DATASET_TR23_TEST 
    trackers_folder_abs = DETECTOR_OUTPUT_DIR 
    seqmap_folder_abs = SEQMAP_DIR 
    
    print(f"Cercando GT in: {gt_folder_abs}") 
    print(f"Cercando Seqmap in: {seqmap_folder_abs}")

    # Verifica veloce
    if not os.path.exists(os.path.join(gt_folder_abs, 'SNMOT-116', 'seqinfo.ini')):
        print("ATTENZIONE: seqinfo.ini non trovato nemmeno nel path assoluto per 116!")

    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'PRINT_RESULTS': False, # Mettiamo False per stampare solo i nostri dati puliti
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
        'OUTPUT_SUMMARY': True, 
        'SKIP_SPLIT_FOL': True, 
        'SEQ_INFO': None,
        'SUB_FOLDER': 'gt', 
        'SEQMAP_FOLDER': seqmap_folder_abs, 
        'SEQMAP_FILE': SEQMAP_FILE, 
    }
    
    # NOTA: Qui NON chiamiamo più fix_ids_for_detection_only()
    # perché per il tracking servono gli ID originali del tuo algoritmo!
    
    dataset = MotChallenge2DBox(dataset_config)
    hota = HOTA()
    
    evaluator = trackeval.Evaluator(eval_config)
    try:
        np.float = float
        np.int = int
    except AttributeError:
        pass
        
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
            print(f"Attenzione: 'COMBINED_SEQ' non trovato. Mostro i risultati per: {seq_name}")
        
        # Percorso: Dataset -> Tracker -> Sequence -> Class -> Metric
        stats = sequences_res[seq_name]['pedestrian']['HOTA']
        
        # --- CORREZIONE CRUCIALE: INDICE 9 ---
        # TrackEval calcola 19 soglie (0.05 ... 0.95).
        # index 0 = 0.05
        # index 9 = 0.50 (Quello che vuole la competizione)
        idx = 9

        deta_score = stats['DetA'][idx] * 100 
        assa_score = stats['AssA'][idx] * 100
        hota_score = stats['HOTA'][idx] * 100

        print(f"\nMetriche Tracking @ IoU 0.5 ({tracker_name} - {seq_name}):")
        print(f"HOTA: {hota_score:.2f}%  <-- QUESTO è il valore per la classifica")
        print(f"DetA: {deta_score:.2f}%")
        print(f"AssA: {assa_score:.2f}%")
        
        det_re = stats['DetRe'][idx] * 100
        det_pr = stats['DetPr'][idx] * 100
        print(f"DetRe (Recall): {det_re:.2f}%")
        print(f"DetPr (Precision): {det_pr:.2f}%")

if __name__ == "__main__":
    run_evaluation()