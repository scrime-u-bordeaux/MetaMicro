import librosa
import numpy as np
import yaml
import os
import wave

##########################################################################################
# CHARGEMENTS
# Fichier yaml a compléter
yaml_path = "parametre.yaml"

with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

# Fonction d'assistance
def ensure_path(section, key, path):
    section[key] = path
    return section[key]

# Chargement du fichier wave 
file_path = config["calcul_mfcc"]["file_path_audio_non_concat"]
y, sr = librosa.load(file_path, sr=None)

##########################################################################################
# VERIDICATION DE L'EXISTANCE DES FICHIERS
def check_overwrite_or_rename(filepath: str) -> str:
    while os.path.exists(filepath):
        response = input(f"Le fichier '{filepath}' existe déjà. Voulez-vous l'écraser ? (o/n) : ").strip().lower()
        if response in ["o", "oui", "y", "yes"]:
            return filepath
        elif response in ["n", "non", "no"]:
            new_path = input("Entrez un nouveau nom de fichier (ou appuyez sur Entrée pour annuler) : ").strip()
            if new_path == "":
                print("Écriture annulée.")
                return None
            filepath = new_path
        else:
            print("Réponse non reconnue. Répondez par 'o' ou 'n'.")
    return filepath

##########################################################################################
# ECRITURE DES MARQUEURS

# Paramètres de fenêtrage
frame_length = int(0.02 * sr)  # 20 ms
hop_length = int(0.01 * sr)    # 10 ms

# Calcul RMS et time vector
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
mean_rms = np.mean(rms)

# Fenêtres d'origine
timestamps = [
    (4, 10),
    (15, 20),
    (24, 28),
    (33, 37),
    (42, 46),
    (50, 55),
    (60, 65),
    (69, 73),
    (78, None),
]

# Lettres associées
letters = config["calcul_mfcc"]["letters"] * 3
windows = [
    (start, end, letters[i]) for i, (start, end) in enumerate(timestamps)
    if i < len(letters)
]

# Création des marqueurs
markers = []
active = rms > mean_rms 

# Durée de ta.wav = offset pour décaler les fenêtres
file_path_ta = config["calcul_mfcc"]["file_path_ta"]
with wave.open(file_path_ta, "rb") as wav_file:
    offset = wav_file.getnframes() / wav_file.getframerate()

for start_sec, end_sec, label in windows:
    # Masque temporel de la fenêtre
    mask = (times >= start_sec) & (times <= end_sec)

    active_in_window = mask & active
    if not np.any(active_in_window):
        continue

    # Regroupement des blocs consécutifs actifs
    indices = np.where(active_in_window)[0]
    groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

    for g in groups:
        t_start = float(times[g[0]]) + offset
        t_end = float(times[g[-1]]) + offset
        if t_end - t_start > 1.0:
            markers.append((t_start, t_end, label))


##########################################################################################
# ENREGISTREMENT DU FICHIER TEXTE
while True:
    filename = check_overwrite_or_rename(input("Où voulez-vous enregistrer le fichier ? (ex: audio/mon_fichier.txt) : ").strip())
    if filename:
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            try:
                os.makedirs(folder)
                print(f"Dossier '{folder}' créé.")
            except Exception as e:
                print(f"Impossible de créer le dossier : {e}")
                continue
    else:
        print("Aucune sauvegarde effectuée.")
    
    break

# Ajout du fichier wav dans le yaml
calcul = config["calcul_mfcc"]
ensure_path(calcul, "file_path_txt_non_concat", os.path.join(filename))

# Réécriture du fichier YAML avec les chemins complétés
with open(yaml_path, "w") as file:
    yaml.dump(config, file, sort_keys=False, allow_unicode=True)

# Écriture du fichier marqueurs
with open(filename, "w") as f:
    for start, end, label in markers:
        f.write(f"{start:.3f}\t{end:.3f}\t{label}\n")

print(f"Fichier {filename} généré.")
