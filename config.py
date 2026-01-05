from pathlib import Path
# Questo trova automaticamente la cartella dove si trova QUESTO file (la root)
ROOT_DIR = Path(__file__).parent.resolve()

MODEL_PATH = ROOT_DIR / "models"

DATASET_BASIC_PATH = ROOT_DIR / "dataset"
DATASET_TR23 = DATASET_BASIC_PATH / "tracking-2023"
DATASET_TR23_TEST = DATASET_TR23 / "test"
DATASET_TR23_TRAIN = DATASET_TR23 / "train"


DATASET_SPM = DATASET_BASIC_PATH / "sportsmot"
DATASET_SPM_TEST = DATASET_SPM / "test"
DATASET_SPM_TRAIN = DATASET_SPM / "train"
DATASET_SPM_VAL = DATASET_SPM / "val"

YOLO_DATASET_PATH = DATASET_TR23 / "dataset_yolo"
YOLO_DATASET_SPM_PATH = DATASET_SPM / "dataset_yolo"
SPLIT_DIR = YOLO_DATASET_PATH / "splits"
SPLIT_SPM_DIR = YOLO_DATASET_SPM_PATH / "splits"

OUTPUT_DIR = ROOT_DIR/ "out"
DETECTOR_OUTPUT_DIR = OUTPUT_DIR / "detector_results"
TRACKER_OUTPUT_DIR = OUTPUT_DIR / "tracker_results"

SEQMAP_DIR = OUTPUT_DIR / "seqmap"
SEQMAP_FILE = SEQMAP_DIR / "soccer_test.txt"

TRACKER_CONFIG = ROOT_DIR / "src" / "Tracker" / "Yolo12m" 