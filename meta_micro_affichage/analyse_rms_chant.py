# jouer sur les parametres de threshold de temps reel et correction
import pyaudio 
import pandas as pd  
import numpy as np  
import joblib  
import librosa 
from scipy.signal import butter
import mido  
# import fluidsynth  
from enum import Enum
import scripts.modif_libro.spectral as spectral     
import sys
import wave
from sklearn.neighbors import BallTree   
import matplotlib.pyplot as plt
import threading
from collections import Counter
import csv
import plotly.graph_objects as go
import matplotlib

##########################################################################################
# CHANGER LA SOURCE ET LA SORTIE 
# pactl set-default-source alsa_input.usb-Focusrite_Scarlett_4i4_USB_D89YRE90C17918-00.multichannel-input
# pactl set-default-sink alsa_output.pci-0000_00_1f.3.analog-stereo

##########################################################################################
# INITIALISATION ET CHARGEMENT DES DONNEES

# Parametres pour respiro
port_name = "IAC Driver Bus 1 Bus 1" 
midi_out = mido.open_output(port_name)

# Port pour midifile
port2 = mido.open_output("IAC Driver Bus 1 Bus 2")

# Parametres pour les CC    
rms_max = 500 # Augmenter la valeur pour plus de variation de rms
CHANNEL_RESPIRO = 0
CC_rms = 2 # Channel 2: Breath Control
CC_i = 9   # Channel 9: changement de timbre i
CC_u = 14  # Channel 14: changment de timbre u

# Charger le fichier MIDI
# midi_file = mido.MidiFile("midi/Trombone Etude #2_ TMEA 2022 - 2023.mid")
# midi_file = mido.MidiFile("midi/F. A. Belcke - 2020 All-State Trombone Etude 1.mid")
midi_file = mido.MidiFile("midi/Etude 1.mid")

# Charger le modele et les vecteurs propres
knn_model = joblib.load("scripts/ici/knn_model_test.pkl")  
eigenvectors_thresholded = joblib.load("scripts/ici/eigenvectors_test.pkl") 
mean_X = joblib.load("scripts/ici/mfcc_mean_test.pkl")
std_X = joblib.load("scripts/ici/mfcc_std_test.pkl")

# Tronquer mean et std
block_size = 11
## actions = ['keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'drop', 'keep', 'drop', 'drop', 'keep', 'keep', 'drop']
# actions = ['keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep']
# n_blocks = len(actions)
# rows_to_keep = []

# for i, action in enumerate(actions):
#     start = i * block_size
#     end = start + block_size
#     if action == 'keep':
#         rows_to_keep.extend(range(start, end))

# # Supprimer aussi les 22 dernières lignes
# rows_to_keep = [i for i in rows_to_keep if i < mean_X.shape[0]]

# # Appliquer le filtrage
# mean_X_truncated = mean_X[rows_to_keep]
# std_X_truncated = std_X[rows_to_keep]

mean_X_truncated = mean_X
std_X_truncated = std_X

# Paramètres audio
filename = "audio/recorded.wav"
taux_recouvrement = 1
FORMAT = pyaudio.paInt16 
CHANNELS = 1  
RATE = 44100
CHUNK = int((RATE * 0.005))

# Initialisation de PyAudio
p = pyaudio.PyAudio()

# Initialisation des thread
lock = threading.Lock()

##########################################################################################
# PARALETRES POUR PRISE DU FLUX AUDIO EN TEMPS REEL
# Def de la fonction callback
latest_audio = np.zeros(CHUNK, dtype=np.int16)

def callback(in_data, frame_count, time_info, status):
    global latest_audio
    # with lock:
    latest_audio = np.frombuffer(in_data, dtype=np.int16)
    return (in_data, pyaudio.paContinue)

# Ouverture du flux audio
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    # input_device_index=6,
    output_device_index=4,
    frames_per_buffer=CHUNK,
    # stream_callback=callback
)
audio_frames = []
frames = []

# Initialisation des paramètres MFCC
mfcc_features = []
time_values = []
start = 0
n_fft = min(512, int(CHUNK * taux_recouvrement))
window = np.ones(min(512, int(CHUNK * taux_recouvrement))) # changer
win_length = n_fft

# Build a Mel filter
mel_basis = librosa.filters.mel(sr=RATE, n_fft=n_fft, fmax=RATE/2, n_mels=40)

# Calcul FFT
fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

# Pad the window out to n_fft size
fft_window = librosa.util.pad_center(fft_window, size=n_fft)

batch_size = 6 # Nombre de blocs MFCC à accumuler avant prédiction
mfcc_buffer = []
tab_pred = []
events = []
proba_list = []
majority_labels_seq = []
recouvrement = 3

# Stocker la dernière prédiction pour comparaison
last_prediction = None
last_prediction_boucle = None

##########################################################################################
# DEFINITION DU FILTRE PASSE BAS
nyquist = 0.5 * RATE
cutoff = 2000
order=1
normal_cutoff = cutoff / nyquist
b, a = butter(order, normal_cutoff, btype="low", analog=False)

##########################################################################################
# AUTOMATE
# # Dictionnaire des transitions

# transition avec i sans le l avant
# transitions = { 
#       ("1","l") :"5"  , ("1","a") :"4"  , ("1","t") :"3"  , ("1","s") :"2"  , ("1","i") :"11" ,
#       ("2","l") :"5"  , ("2","a") :"4"  , ("2","t") :"3"  , ("2","s") :"10" , ("2","i") :"11" ,
#       ("3","l") :"6"  , ("3","a") :"9"  , ("3","t") :"6"  , ("3","s") :"2"  , ("3","i") :"13" ,
#       ("4","l") :"7"  , ("4","a") :"9"  , ("4","t") :"6"  , ("4","s") :"2"  , ("4","i") :"12" ,
#       ("5","l") :"5"  , ("5","a") :"4"  , ("5","t") :"3"  , ("5","s") :"10" , ("5","i") :"5"  , # ("5","i") :"11" 
#       ("6","l") :"6"  , ("6","a") :"9"  , ("6","t") :"6"  , ("6","s") :"2"  , ("6","i") :"13" , 
#       ("7","l") :"7"  , ("7","a") :"8"  , ("7","t") :"7"  , ("7","s") :"2"  , ("7","i") :"7"  , # ("7","i") :"12"
#       ("8","l") :"7"  , ("8","a") :"9"  , ("8","t") :"9"  , ("8","s") :"2"  , ("8","i") :"12" ,
#       ("9","l") :"7"  , ("9","a") :"9"  , ("9","t") :"9"  , ("9","s") :"2"  , ("9","i") :"12" ,
#       ("10","l"):"5"  , ("10","a"):"4"  , ("10","t"):"3"  , ("10","s"):"10" , ("10","i"):"11" ,
#       ("11","l"):"7"  , ("11","a"):"8"  , ("11","t"):"6"  , ("11","s"):"2"  , ("11","i"):"13" ,
#       ("12","l"):"7"  , ("12","a"):"8"  , ("12","t"):"13" , ("12","s"):"2"  , ("12","i"):"13" ,
#       ("13","l"):"7"  , ("13","a"):"8"  , ("13","t"):"13" , ("13","s"):"2"  , ("13","i"):"13" 
# }

# # transition avec t sans avoir forcement le s avant
# transitions = { 
#       ("1","l") :"5"  , ("1","a") :"4"  , ("1","t") :"3"  , ("1","s") :"2"  , ("1","i") :"11" ,
#       ("2","l") :"5"  , ("2","a") :"4"  , ("2","t") :"3"  , ("2","s") :"10" , ("2","i") :"11" ,
#       ("3","l") :"6"  , ("3","a") :"9"  , ("3","t") :"6"  , ("3","s") :"2"  , ("3","i") :"13" ,
#       ("4","l") :"7"  , ("4","a") :"9"  , ("4","t") :"6"  , ("4","s") :"2"  , ("4","i") :"12" , # ("4","t") :"6" 
#       ("5","l") :"5"  , ("5","a") :"4"  , ("5","t") :"3"  , ("5","s") :"10" , ("5","i") :"11" , 
#       ("6","l") :"6"  , ("6","a") :"9"  , ("6","t") :"6"  , ("6","s") :"2"  , ("6","i") :"13" , 
#       ("7","l") :"7"  , ("7","a") :"8"  , ("7","t") :"7"  , ("7","s") :"2"  , ("7","i") :"12" ,
#       ("8","l") :"7"  , ("8","a") :"9"  , ("8","t") :"9"  , ("8","s") :"2"  , ("8","i") :"12" , # ("8","t") :"9"
#       ("9","l") :"7"  , ("9","a") :"9"  , ("9","t") :"3"  , ("9","s") :"2"  , ("9","i") :"12" , # ("9","t") :"9"  
#       ("10","l"):"5"  , ("10","a"):"4"  , ("10","t"):"3"  , ("10","s"):"10" , ("10","i"):"11" ,
#       ("11","l"):"7"  , ("11","a"):"8"  , ("11","t"):"6"  , ("11","s"):"2"  , ("11","i"):"13" , # ("11","t"):"6"
#       ("12","l"):"7"  , ("12","a"):"8"  , ("12","t"):"13" , ("12","s"):"2"  , ("12","i"):"13" ,  # ("12","t"):"13"
#       ("13","l"):"7"  , ("13","a"):"8"  , ("13","t"):"13" , ("13","s"):"2"  , ("13","i"):"13"    # ("13","t"):"13"
# }

# transition normale comme sur la feuille
transitions = { 
      ("1","n") :"5"  , ("1","a") :"4"  , ("1","t") :"3"  , ("1","s") :"2"  , ("1","i") :"11" , ("1", "u"): "14",
      ("2","n") :"5"  , ("2","a") :"4"  , ("2","t") :"3"  , ("2","s") :"10" , ("2","i") :"11" , ("2", "u"): "14",
      ("3","n") :"6"  , ("3","a") :"9"  , ("3","t") :"6"  , ("3","s") :"2"  , ("3","i") :"13" , ("3", "u"): "16",
      ("4","n") :"7"  , ("4","a") :"9"  , ("4","t") :"6"  , ("4","s") :"2"  , ("4","i") :"12" , ("4", "u"): "15",
      ("5","n") :"5"  , ("5","a") :"4"  , ("5","t") :"3"  , ("5","s") :"10" , ("5","i") :"11" , ("5", "u"): "14",
      ("6","n") :"6"  , ("6","a") :"9"  , ("6","t") :"6"  , ("6","s") :"2"  , ("6","i") :"13" , ("6", "u"): "16",
      ("7","n") :"7"  , ("7","a") :"8"  , ("7","t") :"7"  , ("7","s") :"2"  , ("7","i") :"12" , ("7", "u"): "15",
      ("8","n") :"7"  , ("8","a") :"9"  , ("8","t") :"9"  , ("8","s") :"2"  , ("8","i") :"12" , ("8", "u"): "15",
      ("9","n") :"7"  , ("9","a") :"9"  , ("9","t") :"9"  , ("9","s") :"2"  , ("9","i") :"12" , ("9", "u"): "15",
      ("10","n"):"5"  , ("10","a"):"4"  , ("10","t"):"3"  , ("10","s"):"10" , ("10","i"):"11" , ("10", "u"): "14",
      ("11","n"):"7"  , ("11","a"):"8"  , ("11","t"):"6"  , ("11","s"):"2"  , ("11","i"):"13" , ("11", "u"): "15",
      ("12","n"):"7"  , ("12","a"):"8"  , ("12","t"):"13" , ("12","s"):"2"  , ("12","i"):"13" , ("12", "u"): "15",
      ("13","n"):"7"  , ("13","a"):"8"  , ("13","t"):"13" , ("13","s"):"2"  , ("13","i"):"13" , ("13", "u"): "15",
      ("14","n"):"7"  , ("14","a"):"8"  , ("14","t"):"6"  , ("14","s"):"2"  , ("14","i"):"12" , ("14", "u"): "16",
      ("15","n"):"7"  , ("15","a"):"8"  , ("15","t"):"16" , ("15","s"):"2"  , ("15","i"):"12" , ("15", "u"): "16",
      ("16","n"):"7"  , ("16","a"):"8"  , ("16","t"):"16" , ("16","s"):"2"  , ("16","i"):"12" , ("16", "u"): "16"
}


# Fonction de transition qui retourne le nouvel état et l'action associée
def transition(etat, char):
    new_state = transitions.get((etat, char), etat) 
    action = None

    if etat == "2":
        action = "OFF"
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))       # timbre i
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))       # timbre u
    elif etat in {"3", "4", "11", "14"}:
        action = "ON"
        if etat in {"11"}:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=127)) # timbre i
        if etat in {"14"}:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=127)) # timbre u
        else:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))   # timbre i
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))   # timbre u
    elif etat in {"8", "12", "15"}:
        action = "ON_OFF"
        if etat == "12":
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=127)) # timbre i
        if etat == "15":
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=127)) # timbre u
        else:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))   # timbre i
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))   # timbre u
    elif etat == "13":
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=127))     # timbre i
    elif etat == "16":
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=127))     # timbre u
    elif etat == "9":
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))       # timbre i
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))       # timbre u
    return new_state, action

# NOTE_ON AND NOTE_OFF
# Initialisation de la pile
note_stack = []
note_pointer = 0  # Pour suivre la position dans midi_notes

# Fonction pour traiter un événement ON ou OFF
def handle_event(event_type):
    global note_pointer
    if event_type == "ON":
        if note_pointer < len(midi_notes):
            note = midi_notes[note_pointer]
            note_stack.append(note)
            midi_out.send(mido.Message('note_on', note = note.note, velocity=note.velocity))
            port2.send(mido.Message('note_on', note = note.note, velocity=note.velocity))
            note_pointer += 1
    elif event_type == "OFF":
        if note_stack:
            note_to_off = note_stack.pop(0)
            midi_out.send(mido.Message('note_off', note = note_to_off.note))
            port2.send(mido.Message('note_off', note = note_to_off.note))

##########################################################################################
# PRISE DU FLUX AUDIO EN TEMPS REEL
prev_event = None
midi_notes = [msg for msg in midi_file if msg.type == 'note_on' and msg.velocity !=0]  # Extraire les notes MIDI
note_id = -1
majority_label_prec = []
majority_label_no_vide = []
audio_buffer = np.array([], dtype=np.int16)

# Parametres pour respiro
alpha = 0.5
smoothed_rms = 0
velocity_base = 100

class Label(Enum):
    A = 1
    I = 2
    S = 3
    T = 4
    VIDE = 99

class Event(Enum):
    ON_NOTE_T = 0
    OFF_S = 1
    ON_L = 2
    ON_NOTE_L = 3
    OFF_I = 4
    
# Chargement du jeu d'entraînement pour obtenir les points 'l'
label_mapping = {"a": 1, "i": 2, "s": 3, "t": 4}

df_ent = pd.read_csv("scripts/ici/X_proj_scaled_test.csv")

df_ent["label"] = df_ent["label"].map(label_mapping)

X_ent = df_ent.iloc[:, :-1].values
y_ent = df_ent["label"].values.astype(int)

centroids = {}
thresholds = {}

# Coefficients spécifiques par label
sigma_factors = {
    "F": (5, 5, 5),
    "T": (5, 4, 4),
    "S": (5, 5, 5),
    "A": (5, 4, 4),
    "I": (5, 5, 5),
    "U": (5, 5, 5),
    "N": (5, 5, 5),
}

for label in list(Label)[:-1]:  # On exclut Label.VIDE
    label_name = label.name
    X_class = X_ent[y_ent == label.value]

    # Calcul du centroïde
    center = np.mean(X_class, axis=0)
    centroids[label_name] = center

    # Seuils 3-sigma (ou autres) par axe selon le label
    factors = sigma_factors[label_name]
    thresholds[label_name] = {
        "axis0": factors[0] * np.std(X_class[:, 0]),
        "axis1": factors[1] * np.std(X_class[:, 1]),
        "axis2": factors[2] * np.std(X_class[:, 2])
    }

# Initialisation de l'arbre de recherche
tree = BallTree(X_ent, metric='euclidean')
# radius = 1.7 # avec i
# radius = 10  # avec aoui
# radius = 10  # avec  aouin
# radius = 8.5 # avec aouim
radius = 10

# k = 4 # avec aoui
# k = 9 # avec aouin
# k = 7  #; avec auinm
k = 3

# Creation du seuil majorité
seuil_majorite = int(k * batch_size / 2)
print(seuil_majorite)

# Importer les données de la séquence
log_file = open("predictions_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp (ms)", "predictions_labels", "label_counts", "majority_label", "rms", "midi_val_rms"])

# Initialisation de l'état de l'automate
etat = "1"
rms_max = 0
rms_sqrt_max = 0
try:
    sequence_labels = []
    last_action_count = 0
    print("Go")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        # data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # with lock:
        # audio_data = latest_audio.copy()
        audio_frames.append(data) # décommenter pour l'enregistrement

        # Ajouter au tampon
        audio_buffer = np.concatenate((audio_buffer, audio_data))

        # Calcul RMS
        # rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        # print("rms:", rms)
        # rms_sqrt = np.sqrt(rms) 
        # print("rms_sqrt:", rms_sqrt)
        # rms_min = 0
        # rms_sqrt_clip = (rms_sqrt - rms_min) / (rms_max - rms_min)
        # print("rms_sqrt_clip:", rms_sqrt_clip)
        # # smoothed_rms = alpha * rms_sqrt + (1 - alpha) * smoothed_rms
        # # print("rms:", rms, "smoothed_rms:", smoothed_rms)
        # # if smoothed_rms>rms_max:
        # #     smoothed_rms = rms_max
        # # midi_val_rms = int((smoothed_rms/rms_max)*126)
        # midi_min, midi_max = 0, 126
        # midi_val_rms = (midi_min + (midi_max - midi_min) * rms_sqrt_clip)
        # print("midi_val_rms:",midi_val_rms)
        # # print(midi_val_rms)
        # bar_length = int(midi_val_rms*0.2) 

        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        
        if rms > rms_max:
            rms_max = rms
            print(rms_max)

except KeyboardInterrupt:
    print("\nArrêt de l'enregistrement.")
    
# Fermeture du flux et de PyAudio
stream.stop_stream()
stream.close()
p.terminate()