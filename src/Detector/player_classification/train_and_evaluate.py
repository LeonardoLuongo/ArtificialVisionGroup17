import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Setup path per importare config
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR))
from config import MODEL_PATH, OUTPUT_DIR

def plot_confusion_matrix(y_true, y_pred, target_names, title, filename):
    """Genera e salva una Confusion Matrix elegante usando Seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalizzazione per vedere le percentuali
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(title)
    plt.ylabel('Ground Truth (Reale)')
    plt.xlabel('Predizione KNN')
    
    save_path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix salvata in: {save_path}")

def apply_reject_option(probabilities, threshold=0.65):
    """
    Se la confidenza è bassa, restituisce -1 (Rejected).
    Altrimenti restituisce la classe con probabilità massima.
    """
    preds = []
    confidences = []
    for prob in probabilities:
        max_p = np.max(prob)
        confidences.append(max_p)
        if max_p < threshold:
            preds.append(-1) # Scartato
        else:
            preds.append(np.argmax(prob))
    return np.array(preds), np.array(confidences)

def evaluate_set(name, X, y_true, model, target_names, threshold=0.65):
    """Funzione di valutazione completa con Reject Option e Matrix."""
    print(f"\n--- VALUTAZIONE: {name.upper()} ---")
    
    # Otteniamo le probabilità dal KNN
    probs = model.predict_proba(X)
    preds_with_reject, _ = apply_reject_option(probs, threshold=threshold)
    
    # Calcolo statistiche di scarto
    mask_accepted = preds_with_reject != -1
    n_total = len(y_true)
    n_accepted = np.sum(mask_accepted)
    n_rejected = n_total - n_accepted
    
    print(f"Campioni totali: {n_total}")
    print(f"Accettati: {n_accepted} ({n_accepted/n_total:.2%})")
    print(f"Scartati (Reject Option): {n_rejected} ({n_rejected/n_total:.2%})")
    
    if n_accepted > 0:
        y_true_acc = y_true[mask_accepted]
        y_pred_acc = preds_with_reject[mask_accepted]
        
        print(f"\nAccuracy sui campioni accettati: {accuracy_score(y_true_acc, y_pred_acc):.2%}")
        print(classification_report(y_true_acc, y_pred_acc, target_names=target_names))
        
        # Genera Confusion Matrix per i campioni accettati
        plot_confusion_matrix(y_true_acc, y_pred_acc, target_names, 
                             f"Confusion Matrix {name} (Threshold {threshold})", 
                             f"cm_{name.lower()}.png")
    else:
        print("Soglia troppo alta: nessun campione accettato.")

def main():
    target_names = ['Team L', 'Team R', 'Referee', 'GK L', 'GK R']
    REJECT_THRESHOLD = 0.65 # (0.0 - 1.0)

    print("Caricamento feature...")
    try:
        X_train_full = np.load(MODEL_PATH / "knn_X_train.npy")
        y_train_full = np.load(MODEL_PATH / "knn_y_train.npy")
    except FileNotFoundError:
        print("Errore: file .npy non trovati. Esegui prepare_knn_data.py")
        return

    # Split interno 80/20 per validazione
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print(f"Training KNN (K=7, Spatial HSV)...")
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)

    evaluate_set("Validation", X_val, y_val, knn, target_names, threshold=REJECT_THRESHOLD)

    X_test_path = MODEL_PATH / "knn_X_test.npy"
    y_test_path = MODEL_PATH / "knn_y_test.npy"

    if X_test_path.exists():
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        evaluate_set("Test_Set", X_test, y_test, knn, target_names, threshold=REJECT_THRESHOLD)

    model_file = MODEL_PATH / "soccer_knn_advanced.pkl"
    joblib.dump(knn, model_file)
    print(f"\nModello salvato: {model_file}")

if __name__ == "__main__":
    main()
