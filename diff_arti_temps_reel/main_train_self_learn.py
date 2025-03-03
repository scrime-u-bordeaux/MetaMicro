import numpy as np
import matplotlib.pyplot as plt
import wave
import librosa
import librosa.feature
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import scipy.spatial.distance as dist

##########################################################################################""
# OUVRIR LE FICHIER AUDIO
with wave.open("audio/tatata.wav", "r") as wav_file:
    sample_rate = wav_file.getframerate()
    n_frames = wav_file.getnframes()
    n_channels = wav_file.getnchannels()
    signal = np.frombuffer(wav_file.readframes(n_frames), dtype=np.int16)
    
    # Convertir en mono si nécessaire
    if n_channels > 1:
        signal = signal.reshape(-1, n_channels)[:, 0]

##########################################################################################""
# CALCUL DES MFCC
# Créer un axe temporel
duration = len(signal) / sample_rate
time = np.linspace(0, duration, num=len(signal))

# Découper le signal en blocs de 0.01 s et calculer les MFCCs
block_size = int(0.005 * sample_rate)  # Nombre d'échantillons par bloc
mfcc_per_block = []
block_ids = []
block_start_times = []

for i, start in enumerate(range(0, len(signal), block_size)):
    print(f"i: {i} / {len(signal) // block_size}", end="\r", flush=True)
    block = signal[start:start+block_size]
    if len(block) == block_size: 
        mfcc = librosa.feature.mfcc(y=block.astype(float), sr=sample_rate, n_mfcc=13, n_fft=min(512, len(block)), n_mels=40, fmax=sample_rate/2)
        mfcc_per_block.append(mfcc.mean(axis=1))  # Moyenne des coefficients MFCC
        block_ids.append(i)
        block_start_times.append(start / sample_rate)

# Conversion des MFCCs en matrice
mfcc_matrix = np.array(mfcc_per_block)

##########################################################################################""
# CLASSIFICATION
# Clustering en trois classes
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(mfcc_matrix)

# Mapping des labels aux catégories
categories = np.array(["a", "t", "s"])
predicted_labels = categories[kmeans_labels]

# Identifier les points dont la distance aux deux clusters les plus proches est similaire
cluster_centers = kmeans.cluster_centers_
distances = np.array([dist.cdist([mfcc], cluster_centers, metric='euclidean')[0] for mfcc in mfcc_matrix])
min_distances = np.sort(distances, axis=1)[:, :2]  # Prendre les deux plus proches distances
ambiguity_threshold = 15  # Seuil de similarité entre les deux distances
ambiguous_points = np.abs(min_distances[:, 0] - min_distances[:, 1]) < ambiguity_threshold
predicted_labels[ambiguous_points] = ""

# Création du tableau
results_df = pd.DataFrame({
    "Bloc ID": block_ids,
    "MFCC Moyenne": list(mfcc_matrix),
    "Classe": predicted_labels
})

##########################################################################################""
# AFFICHAGE
# Afficher les points des MFCCs projetés en 2D
pca = PCA(n_components=2)
mfcc_matrix_cut = mfcc_matrix[:, :2]
mfcc_2d = pca.fit_transform(mfcc_matrix)

# Associer les couleurs aux classes
colors = {"t": "blue", "a": "green", "s": "red", "": "gray"}  # Gray pour les non-classifiés
class_colors = [colors[label] if label in colors else "gray" for label in predicted_labels]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(mfcc_2d[:, 0], mfcc_2d[:, 1], c=class_colors, edgecolors='k')
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection des MFCCs en 2D avec Classes")

# Ajouter une légende
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors.values()]
plt.legend(handles, ["a", "t", "s", "non classé"], title="Classes")

plt.grid()
plt.show()

# Afficher le signal audio avec coloration des blocs
plt.figure(figsize=(12, 4))
for i, (start_time, label) in enumerate(zip(block_start_times, predicted_labels)):
    if label:  # Ne pas afficher les blocs non classifiés
        color = colors[label]
        plt.plot(time[int(start_time * sample_rate):int((start_time + 0.01) * sample_rate)], 
                 signal[int(start_time * sample_rate):int((start_time + 0.01) * sample_rate)], 
                 color=color)

plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Représentation du signal audio avec classification des blocs")
plt.grid()
plt.show()


#ocenaudio