# venv
import pyaudio # type: ignore
import pandas as pd # type: ignore
import numpy as np 
import joblib # type: ignore
import matplotlib.pyplot as plt 
import librosa 
from scipy.signal import butter, lfilter 
import mido # type: ignore
import fluidsynth # type: ignore
import scripts.modif_libro.spectral as spectral     

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
sfid = fluid.sfload("FluidR3_GM.sf2")  
fluid.program_select(0, sfid, 0, 73)

# Charger le fichier MIDI
midi_file = mido.MidiFile("midi/potter.mid")

# Charger le modèle XGBoost
knn_model = joblib.load("scripts/knn_model.pkl")

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
    output=True,
    # input_device_index=1,
    # output_device_index=1,
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
midi_notes = [msg for msg in midi_file if msg.type == 'note_on' and msg.velocity !=0]   
note_id = -1

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False) 
        audio_data = np.frombuffer(data, dtype=np.int16) 

        # Appliquer le filtre passe-bas
        filtered_audio = lfilter(b, a, audio_data)

        # Calcul des MFCCs
        block = filtered_audio.astype(np.float32) / 32768.0
        fs = RATE
        start += len(filtered_audio)
        
        mfcc = spectral.mfcc(y=block.astype(float), sr=fs, n_mfcc=13,
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
                    if pred == 0 and prev_event != 0:  # Afficher uniquement le premier 0
                        print("ON")
                        prev_event = 0
                        fluid.noteon(0, midi_notes[note_id].note, midi_notes[note_id].velocity)

                    elif pred == 2 and prev_event != 2:  # Afficher uniquement le premier 2
                        print("OFF")
                        prev_event = 2
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