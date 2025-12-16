from ultralytics import YOLO
from config import MODEL_PATH

# Il blocco if __name__ == '__main__': è OBBLIGATORIO su Windows
if __name__ == '__main__':
    # Carica il peso 'last.pt' invece di 'yolov8n.pt'
    model = YOLO(MODEL_PATH / "training_runs" / "soccernet_finetune" / "weights" / "last.pt")

    # Il parametro resume=True è fondamentale
    results = model.train(resume=True)