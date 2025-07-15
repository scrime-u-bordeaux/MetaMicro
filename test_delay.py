import pyaudio
import time
import numpy as np
from scipy.signal import correlate
import wave
import matplotlib.pyplot as plt

# Paramètres audio
FORMAT = pyaudio.paInt16 
CHANNELS = 2  
RATE = 44100
CHUNK = int((RATE * 0.005)) 
RECORD_SECONDS = 3    

# Noms des fichiers de sortie
INPUT_WAV_FILE = "input_signal.wav"
OUTPUT_WAV_FILE = "output_signal.wav"

p = pyaudio.PyAudio()

# Ouverture de l'entrée micro
stream_in = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)

# Ouverture de la sortie audio (casque)
stream_out = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    output_device_index=5,
                    frames_per_buffer=CHUNK)

print("en cours... (Ctrl+C pour quitter)")

try:
    while True:
        data = stream_in.read(CHUNK, exception_on_overflow=False)
        stream_out.write(data)
except KeyboardInterrupt:
    print("\n Arrêt ")
finally:
    stream_in.stop_stream()
    stream_in.close()
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()

# input_frames = []
# output_frames = []

# for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream_in.read(CHUNK, exception_on_overflow=False)
#     input_frames.append(data)
#     output_frames.append(data)
#     stream_out.write(data)

# # Fermer
# stream_in.stop_stream()
# stream_in.close()
# stream_out.stop_stream()
# stream_out.close()
# p.terminate()

# # Sauvegarde des signaux dans des fichiers WAV
# def save_wave(filename, frames):
#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()

# save_wave(INPUT_WAV_FILE, input_frames)
# save_wave(OUTPUT_WAV_FILE, output_frames)

# print(f"Fichiers enregistrés : {INPUT_WAV_FILE}, {OUTPUT_WAV_FILE}")