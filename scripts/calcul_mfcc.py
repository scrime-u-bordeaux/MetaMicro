import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import modif_libro.spectral as spectral

##########################################################################################
## CHARGEMENTS
# Charger les marqueurs
file_path = "audio/tatata_suite_add.txt"
markers_df = pd.read_csv(file_path, sep="\t", header=None, names=["start", "end", "label"])

# Charger l'audio
audio_file = "audio/tatata_suite_add_precis.wav"
signal, fs = librosa.load(audio_file, sr=None)

##########################################################################################
# CALCUL DES MFCC
# axe temporel
duration = len(signal) / fs
time = np.linspace(0, duration, num=len(signal))

# Parametres MFCC
block_size = int(0.005 * fs)
mfcc_per_block = []
block_labels = []  
block_start_times = []

# Identifier les segments pour chaque bloc
for i, start in enumerate(range(0, len(signal), block_size)): 
    print(f"i: {i} / {len(signal) // block_size}", end="\r", flush=True)
 
    block = signal[start:start+block_size]
    block_start_time = start / fs  
    block_end_time = (start + block_size) / fs 

    # Trouver la classe dominante
    segment_durations = {"t": 0, "a": 0}
    

    for _, row in markers_df.iterrows():
        segment_start, segment_end, segment_label = row["start"], row["end"], row["label"]
        
        if segment_end > block_start_time and segment_start < block_end_time:
            overlap_start = max(block_start_time, segment_start)
            overlap_end = min(block_end_time, segment_end)
            segment_durations[segment_label] += (overlap_end - overlap_start) 

    # Si le bloc chevauche plusieurs classes, on l'ignore
    if (segment_durations["t"] > 0) and (segment_durations["a"] > 0):
        continue  

    # Attribuer le segment dominant
    if segment_durations["t"] > 0:
        block_label = "t"
    elif segment_durations["a"] > 0:
        block_label = "a"
    else:
        block_label = "s" 

    n_fft=min(512, block_size)
    window = np.ones(min(512, block_size))
    win_length = n_fft

    # Build a Mel filter
    mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, fmax=fs/2, n_mels=10)

    # Calculer les MFCCs
    if len(block) == block_size:
        mfcc = spectral.mfcc(y=block.astype(float), sr=fs ,n_mfcc=13,
                            n_fft=min(512, len(block)), hop_length=32,
                            n_mels=10, fmax=fs/2, mel_basis = mel_basis)

        mfcc_vector = mfcc.flatten() 

        mfcc_per_block.append(mfcc_vector)
        block_labels.append(block_label)
        block_start_times.append(block_start_time)


# Convertir en tableau NumPy
mfcc_matrix = np.array(mfcc_per_block)

# Sauvegarder les données avec toutes les valeurs MFCCs
df_mfcc = pd.DataFrame(mfcc_matrix)
df_mfcc["start_time"] = block_start_times
df_mfcc["label"] = block_labels

df_mfcc.to_csv("mfcc_features.csv", index=False)
print("MFCCs calculés et sauvegardés dans 'mfcc_features.csv'")

##########################################################################################
# AFFICHAGE PCA
# Verification que toutes les classes sont bien attribuees
assert all(label in ["t", "a", "s"] for label in block_labels), "Erreur: Un bloc n'est pas classifie !"

# PCA pour visualiser en 2D
pca = PCA(n_components=2)
mfcc_2d = pca.fit_transform(mfcc_matrix)

# Associer les couleurs aux classes
colors = {"t": "blue", "a": "green", "s": "red"}  
class_colors = [colors[label] for label in block_labels] 

plt.figure(figsize=(8, 6))
scatter = plt.scatter(mfcc_2d[:, 0], mfcc_2d[:, 1], c=class_colors, edgecolors='k')
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection des MFCCs en 2D avec Classes")

# Ajouter une légende
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors.values()]
plt.legend(handles, ["t", "a", "s"], title="Classes")

plt.grid()
plt.show()

##########################################################################################
# AFFICHAGE DU SIGNAL AVEC COULEURS PAR BLOCS
plt.figure(figsize=(12, 4))
for i, (start_time, label) in enumerate(zip(block_start_times, block_labels)):
    color = colors[label]
    plt.plot(time[int(start_time * fs):int((start_time + 0.03) * fs)], 
             signal[int(start_time * fs):int((start_time + 0.03) * fs)], 
             color=color)

plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Representation du signal audio avec classification des blocs")
plt.grid()
plt.show()
