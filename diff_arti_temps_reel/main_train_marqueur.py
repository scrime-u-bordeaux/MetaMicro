import pandas as pd
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

##########################################################################################
## CHARGEMENTS
# Chargement des marqueurs
file_path = "audio/tatata.txt"
markers_df = pd.read_csv(file_path, sep="\t", header=None, names=["start", "end", "label"])
print(markers_df.to_string(index=False))

# Charger l'audio
audio_file = "audio/tatata.wav"
audio = AudioSegment.from_wav(audio_file)
signal, fs = librosa.load(audio_file, sr=None)

##########################################################################################
# CALCUL DES MFCC
# Créer un axe temporel
duration = len(signal) / fs
time = np.linspace(0, duration, num=len(signal))

block_size = int(0.005 * fs)  # Taille d'un bloc en secondes
mfcc_per_block = []
block_ids = []
block_start_times = []
block_labels = []  # Ajout pour stocker les labels

for i, start in enumerate(range(0, len(signal), block_size)):  
    print(f"i: {i} / {len(signal) // block_size}", end="\r", flush=True)
    
    block = signal[start:start+block_size]
    block_start_time = start / fs  # Début du bloc en secondes
    block_end_time = (start + block_size) / fs  # Fin du bloc en secondes

    # Liste des segments qui chevauchent le bloc
    segment_durations = {"t": 0, "a": 0}  # Temps de recouvrement pour chaque segment
    
    for _, row in markers_df.iterrows():
        segment_start, segment_end, segment_label = row["start"], row["end"], row["label"]
        
        # Verifier s'il y a chevauchement
        if segment_end > block_start_time and segment_start < block_end_time:
            overlap_start = max(block_start_time, segment_start)  
            overlap_end = min(block_end_time, segment_end) 
            overlap_duration = overlap_end - overlap_start
            
            # Ajouter la duree de chevauchement au bon segment
            if segment_label in segment_durations:
                segment_durations[segment_label] += overlap_duration

    # Determiner le segment dominant
    if segment_durations["t"] > segment_durations["a"]:
        block_label = "t"
    elif segment_durations["a"] > segment_durations["t"]:
        block_label = "a"
    else:
        block_label = "s"  

    # Calcul des MFCCs uniquement pour les blocs complets
    if len(block) == block_size:
        mfcc = librosa.feature.mfcc(y=block.astype(float), sr=fs, n_mfcc=13, 
                                    n_fft=min(512, len(block)), n_mels=40, fmax=fs/2)
        
        mfcc_per_block.append(mfcc.mean(axis=1))  
        block_ids.append(i)
        block_start_times.append(block_start_time)
        block_labels.append(block_label) 

# Conversion en matrice numpy
mfcc_matrix = np.array(mfcc_per_block)

##########################################################################################
# CLASSIFIEUR
# Conversion des labels en valeurs numeriques
label_mapping = {"t": 0, "a": 1, "s": 2} 
y = np.array([label_mapping[label] for label in block_labels]) 
X = mfcc_matrix 

# Division des donnEes en train/test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, stratify=y, random_state=42)

# TRAIN : CrEation et entrainement du classifieur KNN (k=5)
knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean")  # Distance euclidienne
knn.fit(X_train, y_train)

# PREDICTION ET EVALUATION
y_pred = knn.predict(X_test)

# Calcul de l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle KNN : {accuracy * 100:.2f}%")

##########################################################################################
# AFFICHAGE PCA
# Verification que toutes les classes sont bien attribuees
assert all(label in ["t", "a", "s"] for label in block_labels), "Erreur: Un bloc n'est pas classifie !"

# PCA pour visualiser en 2D
pca = PCA(n_components=2)
mfcc_2d = pca.fit_transform(mfcc_matrix)

# Associer les couleurs aux classes
colors = {"t": "blue", "a": "green", "s": "red"}  
class_colors = [colors[label] for label in block_labels]  # On enlève les "non classés"

plt.figure(figsize=(8, 6))
scatter = plt.scatter(mfcc_2d[:, 0], mfcc_2d[:, 1], c=class_colors, edgecolors='k')
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection des MFCCs en 2D avec Classes")

# Ajouter une légende
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors.values()]
plt.legend(handles, ["a", "t", "s"], title="Classes")

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

##########################################################################################
## AFFICHAGE DE LA MATRICE DE CONFUSION
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["t", "a", "s"])
disp.plot(cmap="Blues")
plt.title("Matrice de Confusion du modèle KNN")
plt.show()
