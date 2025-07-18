import pandas as pd
import librosa
import numpy as np
import yaml
from itertools import combinations
import os

##########################################################################################
# CHARGER YAML
with open("parametre.yaml", "r") as file:
    config = yaml.safe_load(file)

# Parametres d'entrée
letters = config["calcul_mfcc"]["letters"]
file_path_txt = config["calcul_mfcc"]["file_path_txt"]
file_path_audio = config["calcul_mfcc"]["file_path_audio"]
time_of_block = config["calcul_mfcc"]["time_of_block"]
recouvrement_block = config["calcul_mfcc"]["recouvrement_block"]

# Paramètres MFCC
mfcc_params = config["calcul_mfcc"]["pamatres_mfcc"]
n_mfcc = mfcc_params["n_mfcc"]
n_fft = mfcc_params["n_fft"]
n_mels = mfcc_params["n_mels"]

# Autres paramètres
other_params = config["calcul_mfcc"]["other_params"]
use_delta = other_params.get("delta_mfcc", False)
use_zcr = other_params.get("zero_crossing", False)
use_centroid = other_params.get("centroid", False)
use_slope = other_params.get("slope", False)

# Chemin de sortie
output_path = config["calcul_mfcc"]["output_path"]
file_path_txt_non_concat = config["calcul_mfcc"]["file_path_txt_non_concat"]

# Chemin du YAML
yaml_path = "parametre.yaml"

##########################################################################################
# VERIFICATION DE L'EXISTANCE DES FICHIERS
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
            print(f"Le fichier sera sauvegardé sous le nom '{filepath}'.")
        else:
            print("Réponse non reconnue. Répondez par 'o' ou 'n'.")
    return filepath

# Fonction pour créer un chemin dans le YAML
def ensure_path(section, key, path):
    section[key] = path
    return section[key]

##########################################################################################
# CHARGEMENT DES DONNÉES
# Charger les marqueurs
markers_df = pd.read_csv(file_path_txt, sep="\t", header=None, names=["start", "end", "label"])

# Charger l'audio
signal, fs = librosa.load(file_path_audio, sr=None)

##########################################################################################
# CALCUL DES MFCC
# axe temporel
duration = len(signal) / fs
time = np.linspace(0, duration, num=len(signal))

# Parametres MFCC
block_size = int(time_of_block * fs)
recouvrement = int(block_size * recouvrement_block)
features_per_block = []
block_labels = []  
block_start_times = []
letters_with_st = letters + [l for l in ["s", "t"] if l not in letters]

# Identifier les segments pour chaque bloc
for i, start in enumerate(range(0, len(signal), recouvrement)): 
    print(f"i: {i} / {len(signal) // recouvrement}", end="\r", flush=True)

    block = signal[start:start+block_size]
    block_start_time = start / fs  
    block_end_time = (start + block_size) / fs 

    # Trouver la classe dominante dans le bloc
    segment_durations = {label: 0 for label in letters_with_st}
    for _, row in markers_df.iterrows():
        segment_start, segment_end, segment_label = row["start"], row["end"], row["label"]
        if segment_label not in letters_with_st:
            continue
        if segment_end > block_start_time and segment_start < block_end_time:
            overlap_start = max(block_start_time, segment_start)
            overlap_end = min(block_end_time, segment_end)
            segment_durations[segment_label] += (overlap_end - overlap_start) # duree du chevauchement

    # Trouver les lettres actives (celles qui ont un label dans le bloc)
    active_segments = [k for k, v in segment_durations.items() if v > 0]
    excluded_pairs = [set(pair) for pair in combinations(letters_with_st, 2)]

    # Cas 1 — Aucune lettre active (il y a un silence)
    if len(active_segments) == 0:
        block_label = "s"

    # Cas 2 — Exactement deux lettres actives, la bloc est donc exclu
    elif len(active_segments) == 2 and set(active_segments) in excluded_pairs:
        continue  # On ignore ce bloc

    # Cas 3 — Une seule lettre active
    elif len(active_segments) == 1:
        block_label = active_segments[0]

    # Calcul MFCC
    n_fft=min(n_fft, block_size)
    mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, fmax=fs/2, n_mels=n_mels)

    if len(block) == block_size:
        # 1. Calculer les MFCC
        mfcc = librosa.feature.mfcc(y=block.astype(float), sr=fs, n_mfcc=n_mfcc,
                                    n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 10,
                                    fmax=fs/2, mel_basis=mel_basis)
        features = [mfcc.flatten()]

        # 2. Delta MFCC (dérivées)
        if use_delta:
            delta = librosa.feature.delta(mfcc, order=1)
            features.append(delta.flatten())

        # 3. Pente moyenne du signal
        if use_slope:
            slope = np.diff(block)
            slope_rms = np.sqrt(np.mean(slope ** 2))
            features.append(np.array([slope_rms]))

        # 4. Zero-crossing rate
        if use_zcr:
            zcr = np.mean(librosa.feature.zero_crossing_rate(block, frame_length=len(block), hop_length=len(block)))
            features.append(np.array([zcr]))

        # 5. Spectral centroid
        if use_centroid:
            centroid = np.mean(librosa.feature.spectral_centroid(y=block, sr=fs))
            features.append(np.array([centroid]))

        # Enregistrement des caractéristiques
        feature_vector = np.concatenate(features)
        features_per_block.append(feature_vector)
        block_labels.append(block_label)
        block_start_times.append(block_start_time)

##########################################################################################
# SAUVEGARDE
# Sauvegarder le fichier CSV
mfcc_matrix = np.array(features_per_block)
df_mfcc = pd.DataFrame(mfcc_matrix)
df_mfcc["start_time"] = block_start_times
df_mfcc["label"] = block_labels

# Dossier cible pour la sauvegarde
folder = os.path.dirname(file_path_txt_non_concat)  # Utilisation du dossier de 'file_path_txt_non_concat'

# Crée le dossier si nécessaire
if not os.path.exists(folder):
    os.makedirs(folder)
    print(f"Le dossier '{folder}' a été créé avec succès.")
else:
    print(f"Le dossier '{folder}' existe déjà.")

# Définir le chemin de sauvegarde pour le fichier CSV
final_path = os.path.join(folder, "mfcc_features.csv")

# Vérifier si le fichier existe déjà et demander à l'utilisateur s'il veut l'écraser
final_path = check_overwrite_or_rename(final_path)

# Si le chemin final est valide, sauvegarder le fichier CSV
if final_path:
    df_mfcc.to_csv(final_path, index=False)
    print(f"Fichier sauvegardé dans {final_path}")
    
    # Mise à jour du chemin dans le YAML
    ensure_path(config["calcul_mfcc"], "output_path", final_path)
    
    # Sauvegarder le fichier YAML avec le nouveau chemin
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"Le chemin de sortie a été mis à jour dans le fichier YAML : {final_path}")
else:
    print("Aucune sauvegarde effectuée.")

# Enregistre le chemin du fichier CSV dans le YAML
config["calcul_mfcc"]["output_path"] = final_path