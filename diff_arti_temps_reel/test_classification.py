import pandas as pd
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import time 
import pdb
import scipy
##########################################################################################
# CHARGER LES DONNNEES 
# Charger le modèle KNN
knn = joblib.load("knn_model.pkl")

# Charger l'audio
audio_file = "audio/test_myriam.wav"
signal, fs = librosa.load(audio_file, sr=None)

##########################################################################################
# EXTRAIRE LES MFCC
# Paramètres MFCC 
t_size = 0.005
block_size = int(t_size * fs)
mfcc_features = []
time_values = []
n_fft=min(512, block_size)
window = np.ones(min(512, block_size))
win_length = n_fft

# Build a Mel filter
mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, fmax=fs/2, n_mels=10)

# Calcul fft
fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

# Pad the window out to n_fft size
fft_window = librosa.util.pad_center(fft_window, size=n_fft)

# Extraire les MFCCs pour chaque bloc
start_time = time.time()
for start in range(0, len(signal), block_size):
    block = signal[start:start+block_size]
    
    if len(block) == block_size:
        mfcc = librosa.feature.mfcc(y=block.astype(float), sr=fs ,n_mfcc=13,
                            n_fft=min(512, len(block)), hop_length=32,
                            n_mels=10, fmax=fs/2, mel_basis = mel_basis, fft_window=fft_window)
        mfcc_vector = mfcc.flatten()  
        mfcc_features.append(mfcc_vector)
        time_values.append(start / fs) 
print(f"Temps MFCC: {time.time() - start_time:.4f} sec")
# pdb.set_trace()

start_time = time.time()    
# Convertir en tableau NumPy
mfcc_matrix = np.array(mfcc_features)
df_mfcc = pd.DataFrame(mfcc_matrix)
print("df_mfcc size:", df_mfcc.shape) 
print(f"Temps Convertion numpy: {time.time() - start_time:.4f} sec")

##########################################################################################
# MODELE KNN
# Prédictions du modèle KNN
start_time = time.time()
predictions = knn.predict(df_mfcc)
print(f"Temps prediction: {time.time() - start_time:.4f} sec")

# Mapping 
start_time = time.time()
label_mapping = {0: "t", 1: "a", 2: "s"}
predictions_text = [label_mapping[p] for p in predictions]

df_predictions = pd.DataFrame({"Time (s)": time_values, "Label": predictions_text})
df_predictions.to_csv("predictions_recorded .csv", index=False)
print("Prédictions sauvegardées dans 'predictions_recorded.csv'")
print(f"Temps mapping: {time.time() - start_time:.4f} sec")

##########################################################################################
# AFFICHAGE
# Affichage des segments audio avec leurs prédictions
plt.figure(figsize=(12, 4))
time = np.linspace(0, len(signal) / fs, num=len(signal))

# Associer chaque segment à une couleur selon la prédiction
colors = {"t": "blue", "a": "green", "s": "red"}
for i, (start, pred) in enumerate(zip(range(0, len(signal), block_size), predictions_text)):
    plt.plot(time[start:start+block_size], signal[start:start+block_size], color=colors[pred])

plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Classification des segments audio avec le modèle KNN")
plt.grid()
plt.show()
