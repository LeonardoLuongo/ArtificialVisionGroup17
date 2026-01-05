import os
from pathlib import Path
from config import DETECTOR_OUTPUT_DIR

def rinomina_file_txt(target_dir):
    if not target_dir.exists():
        print(f"Errore: Il percorso {target_dir} non esiste.")
        return

    print(f"Analisi file in: {target_dir}")
    print("Scegli l'operazione:")
    print("1: SNMOT-XXX.txt        -> tracking_XXX_17.txt")
    print("2: tracking_XXX_17.txt  -> SNMOT-XXX.txt")
    print("3: XXX.txt              -> SNMOT-XXX.txt")
    print("4: SNMOT-XXX.txt        -> XXX.txt")       # Nuova opzione
    print("5: tracking_XXX_17.txt  -> XXX.txt")       # Nuova opzione
    
    scelta = input("Inserisci un numero da 1 a 5: ")
    
    if scelta not in ['1', '2', '3', '4', '5']:
        print("Scelta non valida.")
        return

    count = 0

    # Iteriamo su tutti i file .txt nella cartella
    for file_path in target_dir.glob("*.txt"):
        vecchio_nome = file_path.name
        vecchio_stem = file_path.stem  # Nome senza estensione
        nuovo_nome = None

        if scelta == '1':
            # SNMOT-116.txt -> tracking_116_17.txt
            if vecchio_stem.startswith("SNMOT-"):
                num_part = vecchio_stem.replace("SNMOT-", "")
                if num_part.isdigit():
                    nuovo_nome = f"tracking_{num_part}_17.txt"

        elif scelta == '2':
            # tracking_116_17.txt -> SNMOT-116.txt
            if vecchio_stem.startswith("tracking_") and vecchio_stem.endswith("_17"):
                num_part = vecchio_stem.replace("tracking_", "").replace("_17", "")
                if num_part.isdigit():
                    nuovo_nome = f"SNMOT-{num_part}.txt"
        
        elif scelta == '3':
            # 116.txt -> SNMOT-116.txt
            if vecchio_stem.isdigit():
                nuovo_nome = f"SNMOT-{vecchio_stem}.txt"

        elif scelta == '4':
            # SNMOT-116.txt -> 116.txt
            if vecchio_stem.startswith("SNMOT-"):
                num_part = vecchio_stem.replace("SNMOT-", "")
                if num_part.isdigit():
                    nuovo_nome = f"{num_part}.txt"

        elif scelta == '5':
            # tracking_116_17.txt -> 116.txt
            if vecchio_stem.startswith("tracking_") and vecchio_stem.endswith("_17"):
                num_part = vecchio_stem.replace("tracking_", "").replace("_17", "")
                if num_part.isdigit():
                    nuovo_nome = f"{num_part}.txt"

        if nuovo_nome:
            nuovo_path = file_path.parent / nuovo_nome
            
            try:
                if not nuovo_path.exists():
                    file_path.rename(nuovo_path)
                    print(f"Rinominato: {vecchio_nome} -> {nuovo_nome}")
                    count += 1
                else:
                    if vecchio_nome != nuovo_nome:
                        print(f"Salto {vecchio_nome}: {nuovo_nome} esiste già.")
            except Exception as e:
                print(f"Errore durante la rinomina di {vecchio_nome}: {e}")

    print(f"\nOperazione completata. File rinominati: {count}")

# Esecuzione
if __name__ == "__main__":
    try:
        # Applichiamo alla sottocartella "best"
        percorso_target = DETECTOR_OUTPUT_DIR / "best"
        rinomina_file_txt(percorso_target)
    except NameError:
        print("Errore: La variabile DETECTOR_OUTPUT_DIR non è definita.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")