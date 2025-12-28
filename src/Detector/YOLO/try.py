from ultralytics import YOLO
import cv2

# 1. Carica il modello
model = YOLO('yolov8n.pt') 

# 2. Percorso immagine
path_immagine = 'calciatori.jpg'

# 3. Esegui il rilevamento
results = model.predict(source=path_immagine, classes=0, conf=0.5)

# Carichiamo l'immagine originale con OpenCV per disegnarci sopra
img = cv2.imread(path_immagine)

# 4. Iteriamo sui risultati
for r in results:
    boxes = r.boxes
    for i, box in enumerate(boxes):
        # Ottieni le coordinate del box (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Crea l'etichetta personalizzata (id:1, id:2, ecc.)
        # i + 1 perché l'indice parte da 0
        label = f"id:{i+1}"
        
        # --- DISEGNO MANUALE ---
        # Disegna il rettangolo (Verde, spessore 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Disegna lo sfondo per il testo (piccolo rettangolo pieno sopra il box)
        cv2.rectangle(img, (x1, y1 - 35), (x1 + 80, y1), (0, 255, 0), -1)
        
        # Scrivi il testo "id:X" (Bianco)
        cv2.putText(img, label, (x1 + 5, y1 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# 5. Mostra e salva il risultato
cv2.imshow('Calciatori con ID', img)
cv2.imwrite('risultato_id.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()