import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
# CHARGEMENTS
# Charger les marqueurs
file_path = "audio/u_ta_la_ti_li_i_n.txt"

markers_df = pd.read_csv(file_path, sep="\t", header=None, names=["start", "end", "label"])

# Charger l'audio
audio_file = "audio/u_ta_la_ti_li_i_n.wav"
signal, fs = librosa.load(audio_file, sr=None)

##########################################################################################
# CALCUL DES MFCC
# axe temporel
duration = len(signal) / fs
time = np.linspace(0, duration, num=len(signal))

# Parametres MFCC
block_size = int(0.005 * fs)  # 30 ms en echantillons
recouvrement = int(block_size/2)
mfcc_per_block = []
block_labels = []  
block_start_times = []

# Identifier les segments pour chaque bloc
for i, start in enumerate(range(0, len(signal), recouvrement)): 
    print(f"i: {i} / {len(signal) // recouvrement}", end="\r", flush=True)
 
    block = signal[start:start+block_size]
    block_start_time = start / fs  
    block_end_time = (start + block_size) / fs 

    # Trouver la classe dominante
    segment_durations = {"a": 0, "t": 0, "i": 0, "u": 0, "n": 0}
    for _, row in markers_df.iterrows():
        segment_start, segment_end, segment_label = row["start"], row["end"], row["label"]
        
        if segment_end > block_start_time and segment_start < block_end_time:
            overlap_start = max(block_start_time, segment_start)
            overlap_end = min(block_end_time, segment_end)
            segment_durations[segment_label] += (overlap_end - overlap_start) # duree du chevauchement

    # Trouver les lettres actives (celles qui ont un label dans le bloc)
    active_segments = [k for k, v in segment_durations.items() if v > 0]
    excluded_pairs = [
        {"t", "a"},
        {"s", "t"},
        {"a", "i"},
        {"s", "i"},
        {"t", "i"},
        {"t", "u"},
        {"a", "u"},
        {"s", "u"},
        {"i", "u"},
        {"n", "a"},
        {"n", "i"},
        {"n", "s"},
        {"n", "u"},
    ]

    # Cas 1 — Aucune lettre active (il y a un silence
    if len(active_segments) == 0:
        block_label = "s"

    # Cas 2 — Exactement deux lettres actives, la bloc est donc exclu
    elif len(active_segments) == 2 and set(active_segments) in excluded_pairs:
        continue  # On ignore ce bloc

    # Cas 3 — Une seule lettre active
    elif len(active_segments) == 1:
        block_label = active_segments[0]

    # Paramètres MFCC
    n_fft=min(512, block_size)
    win_length = n_fft

    # Construire les filtres de Mel
    mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, fmax=fs/2, n_mels=40)

    if len(block) == block_size:

        # 1. Calculer les MFCC
        mfcc = librosa.feature.mfcc(y=block.astype(float), sr=fs ,n_mfcc=13, 
                            n_fft=min(512, len(block)), win_length = n_fft, hop_length= win_length // 10,
                            fmax=fs/2, mel_basis = mel_basis)
        
        # 1. Delta MFCC (dérivées)
        delta = librosa.feature.delta(mfcc, order=1)

        # 2. Zero-crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(block, frame_length=len(block), hop_length=len(block)))

        # 3. Spectral centroid
        centroid = np.mean(librosa.feature.spectral_centroid(y=block, sr=fs))

        # 4. Pente moyenne du signal
        slope = np.diff(block)
        slope_rms = np.sqrt(np.mean(slope**2))

        # Combiner toutes les features
        mfcc_vector = mfcc.flatten()
        delta_vector = delta.flatten()
        feature_vector = np.concatenate([mfcc_vector, [slope_rms, zcr]])

        # Stocker les features
        mfcc_per_block.append(feature_vector)
        block_labels.append(block_label)
        block_start_times.append(block_start_time)

        
# Convertir en tableau NumPy
mfcc_matrix = np.array(mfcc_per_block)

# Sauvegarder les données avec toutes les valeurs MFCCs
df_mfcc = pd.DataFrame(mfcc_matrix)
df_mfcc["start_time"] = block_start_times
df_mfcc["label"] = block_labels

df_mfcc.to_csv("data/mfcc_features.csv", index=False) # mfcc_features_ta_la_r_avant ou mfcc_features_ta_la_r_plus_court ou  mfcc_features_ta_la_r_fonctionne
print("MFCCs calculés et sauvegardés dans 'mfcc_features.csv'")

##########################################################################################
# AFFICHAGE DU SIGNAL AVEC COULEURS PAR BLOCS
colors = {"l": "blue", "a": "green", "s": "red", "t": "orange", "r": "purple"}  
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
