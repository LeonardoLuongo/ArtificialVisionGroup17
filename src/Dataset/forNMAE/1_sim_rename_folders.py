import os
from pathlib import Path
from config import *

def rename_dataset_folders(base_path):
    # Converte in oggetto Path per una gestione più semplice
    data_dir = Path(base_path)
    
    if not data_dir.exists():
        print(f"Errore: Il percorso {base_path} non esiste.")
        return

    print(f"Analisi della cartella: {data_dir.absolute()}")
    
    count = 0
    # Itera su tutti gli elementi nella cartella
    for folder in data_dir.iterdir():
        # Controlla se è una directory e se segue il pattern SNMOT-XXX
        if folder.is_dir() and folder.name.startswith("SNMOT-"):
            try:
                # Estrae la parte numerica dopo il trattino
                new_name = folder.name.split("-")[1]
                
                # Crea il nuovo percorso
                new_path = folder.parent / new_name
                
                # Rinoma la cartella
                folder.rename(new_path)
                print(f"Rinominata: {folder.name} -> {new_name}")
                count += 1
            except IndexError:
                print(f"Salto {folder.name}: formato non riconosciuto.")
            except FileExistsError:
                print(f"Errore: La cartella {new_name} esiste già.")

    print(f"\nOperazione completata. Cartelle rinominate: {count}")

if __name__ == "__main__":
    rename_dataset_folders(DATASET_TR23_TEST)