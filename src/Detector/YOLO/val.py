from ultralytics import YOLO

# 1. Carica il modello che hai appena addestrato
model = YOLO(r'C:\Users\cesco\Documents\GitHub\ArtificialVisionGroup17\models\training_runs\soccernet_v8m_ultra_19208\weights\best.pt')

# 2. Esegui la validazione
metrics = model.val(
    data=r'C:\Users\cesco\Documents\GitHub\ArtificialVisionGroup17\dataset\tracking-2023\dataset_yolo\dataset.yaml',
    split='val', # o 'test' se vuoi usare il test set
    imgsz=1920,   # usa la stessa dimensione usata in training
)

# 3. Stampa i risultati principali
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")