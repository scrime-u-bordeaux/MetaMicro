import pyaudio
import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
import wave
import librosa
import pdb
import time
from sklearn.neighbors import KNeighborsClassifier
from inspect import getsource

print(getsource(KNeighborsClassifier.predict))


##########################################################################################
# INITIALISATION ET CHARGEMENT DES DONNEES
# Charger le modèle KNN
knn = joblib.load("knn_model.pkl")

# Parametres audio
filename = "audio/recorded.wav"
FORMAT = pyaudio.paInt16 
CHANNELS = 1  
RATE = 44100
CHUNK = int(RATE * 0.005) 

# Initialisation de PyAudio
p = pyaudio.PyAudio()

##########################################################################################
# PRISE DU FLUX AUDIO EN TEMPS REEL
# Ouverture du flux audio
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)
audio_frames = []
frames = []

# Initialisation des parametres mfcc
mfcc_features = []
time_values = []
start = 0
n_fft=min(512, CHUNK)
window = np.ones(min(512, CHUNK))
win_length = n_fft

# Build a Mel filter
mel_basis = librosa.filters.mel(sr=RATE, n_fft=n_fft, fmax=RATE/2, n_mels=10)

# Calcul fft
fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

# Pad the window out to n_fft size
fft_window = librosa.util.pad_center(fft_window, size=n_fft)

# Extraire les MFCCs sur le signal temps reel
try:
    while True:
        # Enregistrement des donnees pour affichage et fichier wav
        # start_time = time.time()
        data = stream.read(CHUNK)  
        audio_frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)  
        frames.extend(audio_data)

        # Calcul des mfcc
        block = audio_data.astype(np.float32) / 32768.0
        fs = RATE
        start += len(audio_data)
        
        mfcc = librosa.feature.mfcc(y=block.astype(float), sr=fs ,n_mfcc=13,
                            n_fft=min(512, len(block)), hop_length=32, window = np.ones(min(512, len(block))), # mettre la moitie du bloc
                            n_mels=10, fmax=fs/2, mel_basis = mel_basis, fft_window=fft_window)
        mfcc_vector = mfcc.flatten() 
        time_values.append(start / fs) 

        # Convertir en tableau NumPy
        df_mfcc = pd.DataFrame([mfcc_vector])

        # Prédictions du modèle KNN
        predictions = knn.predict(df_mfcc)
        # print("predictions :", predictions, "time value:", start / fs) 
        # print(f"Temps MFCC: {time.time() - start_time:.4f} sec")
        
except KeyboardInterrupt:
    print("\nArrêt de l'enregistrement.")
    
# Fermeture du flux et de PyAudio
stream.stop_stream()
stream.close()
p.terminate()

# Affichage du signal audio final
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(frames) / RATE, len(frames)), frames)
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signal Audio Capturé")
plt.show()

# Enregistrer l'audio
wf = wave.open(filename, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(audio_frames))
wf.close()