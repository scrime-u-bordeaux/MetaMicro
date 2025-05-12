import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import modif_libro.spectral as spectral   

##########################################################################################
## CHARGEMENTS
# Charger les marqueurs
file_path = "audio/tata_lala_sans_r_parallel.txt"

markers_df = pd.read_csv(file_path, sep="\t", header=None, names=["start", "end", "label"])

# # Charger l'audio
audio_file = "audio/tata_lala_sans_r_parallel.wav"
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
    segment_durations = {"l": 0, "a": 0, "t": 0, "r": 0}
    

    for _, row in markers_df.iterrows():
        segment_start, segment_end, segment_label = row["start"], row["end"], row["label"]
        
        if segment_end > block_start_time and segment_start < block_end_time:
            overlap_start = max(block_start_time, segment_start)
            overlap_end = min(block_end_time, segment_end)
            segment_durations[segment_label] += (overlap_end - overlap_start) # duree du chevauchement

    # Trouver les lettres actives (durée > 0)
    active_segments = [k for k, v in segment_durations.items() if v > 0]

    excluded_pairs = [
        {"l", "a"},
        {"t", "a"},
        {"s", "t"},
        {"l", "r"},
        {"a", "r"},
        {"s", "r"},
        {"l", "s"},
        {"t", "r"},
    ]

    # Cas 1 — Aucun segment actif
    if len(active_segments) == 0:
        # print("cas1")
        block_label = "s"

    # Cas 2 — Exactement deux lettres actives, et la paire est dans les exclus
    elif len(active_segments) == 2 and set(active_segments) in excluded_pairs:
        continue  # On ignore ce bloc

    # Cas 3 — Un seul segment actif 
    elif len(active_segments) == 1:
        block_label = active_segments[0]


    n_fft=min(512, block_size)
    window = np.ones(min(512, block_size))
    win_length = n_fft

    # Build a Mel filter
    mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, fmax=fs/2, n_mels=40)

    # Calculer les MFCCs
    if len(block) == block_size:
        mfcc = spectral.mfcc(y=block.astype(float), sr=fs ,n_mfcc=7, 
                            n_fft=min(512, len(block)), win_length = n_fft, hop_length= win_length // 10, 
                            fmax=fs/2, mel_basis = mel_basis)
        
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

df_mfcc.to_csv("ta_la/mfcc_features_ta_la_sans_r_main_plus_court.csv", index=False) # mfcc_features_ta_la_r_avant ou mfcc_features_ta_la_r_plus_court ou  mfcc_features_ta_la_r_fonctionne
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
