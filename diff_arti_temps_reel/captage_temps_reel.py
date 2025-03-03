import pyaudio
import numpy as np

# Parametres audio
filename = "recorded.wav"
FORMAT = pyaudio.paInt16  # Format des données audio 
CHANNELS = 1  # Nombre de canaux 
RATE = 44100  # Frequence d'echantillonnage 
CHUNK = 1024  # Taille du buffer

# Initialisation de PyAudio
p = pyaudio.PyAudio()

# Ouverture du flux audio
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

try:
    while True:
        data = stream.read(CHUNK)  # Lecture du buffer
        audio_data = np.frombuffer(data, dtype=np.int16)  # Conversion en tableau numpy
        print(f"Amplitude moyenne: {np.mean(np.abs(audio_data))}")
except KeyboardInterrupt:
    print("\nArrêt de l'enregistrement.")
    
# Fermeture du flux et de PyAudio
stream.stop_stream()
stream.close()
p.terminate()
