# venv
import pyaudio # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import time
import joblib # type: ignore
import matplotlib.pyplot as plt # type: ignore
import wave
import librosa # type: ignore
import pdb
import time
import xgboost as xgb # type: ignore
from scipy.signal import butter, lfilter # type: ignore
import pygame # type: ignore
import mido # type: ignore
import fluidsynth # type: ignore

##########################################################################################
# CHANGER LA SOURCE ET LA SORTIE 
# pactl set-default-source alsa_input.usb-Focusrite_Scarlett_4i4_USB_D89YRE90C17918-00.multichannel-input
# pactl set-default-sink alsa_output.pci-0000_00_1f.3.analog-stereo
# pactl set-default-sink alsa_output.pci-0000_00_1f.3.analog-stereo


##########################################################################################
# INITIALISATION ET CHARGEMENT DES DONNEES

# Initialiser FluidSynth avec une soundfont
fluid = fluidsynth.Synth()
fluid.start()
sfid = fluid.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")  
fluid.program_select(0, sfid, 0, 73)

# Charger le fichier MIDI
midi_file = mido.MidiFile("midi/potter.mid")

# Initialisation de pygame
pygame.mixer.init()
pygame.mixer.music.load("audio/flute_son.wav")

# Charger le modèle XGBoost
knn_model = joblib.load("knn_model.pkl")

# Paramètres audio
filename = "audio/recorded.wav"
FORMAT = pyaudio.paInt16 
CHANNELS = 1  
RATE = 44100
CHUNK = int(RATE * 0.005) 

# Initialisation de PyAudio
p = pyaudio.PyAudio()

##########################################################################################
# PARALETRES POUR PRISE DU FLUX AUDIO EN TEMPS REEL
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

# Initialisation des paramètres MFCC
mfcc_features = []
time_values = []
start = 0
n_fft = min(512, CHUNK)
window = np.ones(min(512, CHUNK))
win_length = n_fft

# Build a Mel filter
mel_basis = librosa.filters.mel(sr=RATE, n_fft=n_fft, fmax=RATE/2, n_mels=10)

# Calcul FFT
fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

# Pad the window out to n_fft size
fft_window = librosa.util.pad_center(fft_window, size=n_fft)

batch_size = 10  # Nombre de blocs MFCC à accumuler avant prédiction
mfcc_buffer = []
tab_pred = []
events = []

# Stocker la dernière prédiction pour comparaison
last_prediction = None

##########################################################################################
# DEFINITION DU FILTRE PASSE BAS
nyquist = 0.5 * RATE
cutoff = 2000
order=1
normal_cutoff = cutoff / nyquist
b, a = butter(order, normal_cutoff, btype="low", analog=False)


##########################################################################################
# PRISE DU FLUX AUDIO EN TEMPS REEL
prev_event = None
midi_notes = [msg for msg in midi_file if msg.type == 'note_on' and msg.velocity !=0]  # Extraire les notes MIDI
# print(midi_notes)
note_id = -1

try:
    while True:
        data = stream.read(CHUNK)  
        audio_data = np.frombuffer(data, dtype=np.int16) 
        # audio_frames.append(data) # Decommenter pour l'affichage

        # Appliquer le filtre passe-bas
        filtered_audio = lfilter(b, a, audio_data)
        # frames.extend(filtered_audio) # Decommenter pour l'affichage

        # Calcul des MFCCs
        block = filtered_audio.astype(np.float32) / 32768.0
        fs = RATE
        start += len(filtered_audio)
        
        mfcc = librosa.feature.mfcc(y=block.astype(float), sr=fs, n_mfcc=13,
                                    n_fft=min(512, len(block)), hop_length=32,
                                    n_mels=10, fmax=fs/2, mel_basis=mel_basis)
        mfcc_vector = mfcc.flatten()

        # Ajouter à la mémoire tampon
        mfcc_buffer.append(mfcc_vector)

        # Prédire quand on a accumulé assez de blocs
        if len(mfcc_buffer) >= batch_size:
            df_mfcc = pd.DataFrame(mfcc_buffer)
            predictions = knn_model.predict(df_mfcc)  
            # tab_pred.extend(predictions) # Decommenter pour l'affichage

            for pred in predictions:
                if pred != last_prediction:
                    # label_mapping = {0: "t", 1: "a", 2: "s"}
                    # print(f"Prédiction : {label_mapping[pred]}, Time : {t:.2f} sec")
                    if pred == 0 and prev_event != 0:  # Afficher uniquement le premier 0
                        # pygame.mixer.music.play()
                        print("ON")
                        # events.append(("ON", start / fs)) # Decommenter pour l'affichage
                        prev_event = 0

                        # Lancer les notes midi 
                        # for msg in midi_notes:
                        #     if msg.type == 'note_on':
                        # start_time = time.time()
                        fluid.noteon(0, midi_notes[note_id].note, midi_notes[note_id].velocity)
                        # print(f"Temps noteon: {time.time() - start_time:.4f} sec")

                    elif pred == 2 and prev_event != 2:  # Afficher uniquement le premier 2
                        # pygame.mixer.music.stop()
                        print("OFF")
                        # events.append(("OFF", start / fs)) # Decommenter pour l'affichage
                        prev_event = 2

                        # # Arrêter toutes les notes en cours
                        # for msg in midi_notes:
                        #     if msg.type == 'note_on':
                        fluid.noteoff(0,  midi_notes[note_id].note)
                        note_id += 1

                    last_prediction = pred

            # Réinitialiser les buffers
            mfcc_buffer = []

except KeyboardInterrupt:
    print("\nArrêt de l'enregistrement.")

    
# Fermeture du flux et de PyAudio
stream.stop_stream()
stream.close()
p.terminate()

# ##########################################################################################
# # AFFICHAGE COULEUR

# # # Affichage du signal audio final
# # plt.figure(figsize=(10, 4))
# # plt.plot(np.linspace(0, len(frames) / RATE, len(frames)), frames)

# # Générer l'axe des temps
# time = np.linspace(0, len(frames) / fs, num=len(frames))

# # Dictionnaire de correspondance entre les prédictions et les couleurs
# label_mapping = {0: "t", 1: "a", 2: "s"}
# predictions_text = [label_mapping[p] for p in tab_pred]
# colors = {"t": "blue", "a": "green", "s": "red"}

# plt.figure(figsize=(10, 4))

# # Tracer le signal par segments en fonction des prédictions
# for i, (start, pred) in enumerate(zip(range(0, len(frames), CHUNK), predictions_text)):
#     plt.plot(time[start:start+CHUNK], frames[start:start+CHUNK], color=colors[pred])


# plt.xlabel("Temps (s)")
# plt.ylabel("Amplitude")
# plt.title("Signal Audio Capturé")
# plt.show()

# # Enregistrer l'audio
# wf = wave.open(filename, "wb")
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b"".join(audio_frames))
# wf.close()

# ##########################################################################################
# # AFFICHAGE ON/OFF

# # Créer l'axe temporel pour l'affichage du signal
# time = np.linspace(0, len(frames) / fs, num=len(frames))

# # Affichage du signal audio
# plt.figure(figsize=(12, 4))
# plt.plot(np.linspace(0, len(frames) / RATE, len(frames)), frames)
# df_events = pd.DataFrame(events, columns=["Event", "Time (s)"])

# # Ajouter les barres pour les événements ON et OFF
# for _, row in df_events.iterrows():
#     event_type = row["Event"]
#     event_time = row["Time (s)"]
    
#     if event_type == "ON":
#         plt.axvline(x=event_time, color="blue", linestyle="--")
#     elif event_type == "OFF":
#         plt.axvline(x=event_time, color="red", linestyle="--")

# # Ajouter la légende et les labels
# plt.xlabel("Temps (s)")
# plt.ylabel("Amplitude")
# plt.title("Signal Audio avec Indicateurs ON/OFF")
# plt.legend()
# plt.grid()
# plt.show()
