# jouer sur les parametres de threshold de temps reel et correction
import pyaudio 
import pandas as pd  
import numpy as np  
import joblib  
import librosa 
from scipy.signal import butter
import mido  
from enum import Enum
import modif_libro.spectral as spectral     
import sys
import wave
from sklearn.neighbors import BallTree   
import matplotlib.pyplot as plt
import threading
from collections import Counter
import csv
import plotly.graph_objects as go
import tkinter as tk
from tkinter import ttk
import threading
import yaml
import matplotlib
import time

##########################################################################################
## LECTURE DES PARAMÈTRES YAML
with open("parametre.yaml", "r") as file:
    config = yaml.safe_load(file)

main_respiro_param = config["main_respiro"]

# Lettres
letters = config["calcul_mfcc"]["letters"]
letters_with_st = letters + [l for l in ["s", "t"] if l not in letters]

# Parametres mfcc
mfcc_params = config["calcul_mfcc"]["pamatres_mfcc"]
n_mfcc = mfcc_params["n_mfcc"]
n_fft = mfcc_params["n_fft"]
n_mels = mfcc_params["n_mels"]

# Parametres pour respiro
port_name_respiro = main_respiro_param["port_name_respiro"]
midi_out = mido.open_output(port_name_respiro)

# Port pour midifile
port_name_midifile = main_respiro_param["port_name_midifile"]
port2 = mido.open_output(port_name_midifile)

# Parametres pour les CC    
rms_max_souffle = main_respiro_param["rms_max"] # Augmenter la valeur pour plus de variation de rms
rms_max = rms_max_souffle / 0.9
CHANNEL_RESPIRO = main_respiro_param["CHANNEL_RESPIRO"]
CC_rms = main_respiro_param["CC_rms"] # Channel 2: Breath Control
CC_i = main_respiro_param["CC_i"]   # Channel 9: changement de timbre i
CC_u = main_respiro_param["CC_u"]  # Channel 14: changment de timbre u
CC_a = main_respiro_param["CC_a"]  # changment de timbre a

# Charger le fichier MIDI
midi_file = mido.MidiFile(main_respiro_param["midi_file"])

# Extraire les chemins depuis le YAML
outputs = config["classification"]["outputs"]

knn_model_path = outputs["knn_model_output"]
eigenvectors_path = outputs["eigenvectors_output"]
mean_X_path = outputs["mean_X_output"]
std_X_path = outputs["std_X_output"]
proj_pca_path = outputs["proj_pca_output"]

# Charger les fichiers
knn_model = joblib.load(knn_model_path)
eigenvectors_thresholded = joblib.load(eigenvectors_path)
mean_X = joblib.load(mean_X_path)
std_X = joblib.load(std_X_path)
df_ent = pd.read_csv(proj_pca_path)
df_loaded = pd.read_csv(proj_pca_path)

other_params_calcul = config["calcul_mfcc"]["other_params"]
use_delta = other_params_calcul.get("delta_mfcc", False)
use_zcr = other_params_calcul.get("zero_crossing", False)
use_centroid = other_params_calcul.get("centroid", False)
use_slope = other_params_calcul.get("slope", False)

other_params_main_respiro = config["main_respiro"]["other_params"]
use_vide_if_n_label = other_params_main_respiro["vide_if_n_label"].get("value", False)
n_label_for_if_vide = other_params_main_respiro["vide_if_n_label"].get("n", False)
use_remplacer_t_par_i = other_params_main_respiro["remplacer_t_par_i"].get("value", False)
n_label_for_use_remplacer_t_par_i = other_params_main_respiro["remplacer_t_par_i"].get("n", False)

##########################################################################################
# INITIALISATION ET CHARGEMENT DES DONNEES
# Tronquer mean et std
block_size = 11

mean_X_truncated = mean_X
std_X_truncated = std_X

# Paramètres audio
filename = "audio/recorded.wav"
taux_recouvrement = 1
FORMAT = pyaudio.paInt16 
CHANNELS = main_respiro_param["CHANNELS"]  
RATE = main_respiro_param["RATE"]
fs = RATE
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

batch_size = main_respiro_param["batch_size"] # Nombre de blocs MFCC à accumuler avant prédiction
mfcc_buffer = []
tab_pred = []
events = []
proba_list = []
majority_labels_seq = []
recouvrement = main_respiro_param["recouvrement"]

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

#  Fonction pour adapter les transitions selon si la personne choisit le "n" ou le "l" ou les 2
def adapt_transitions(transitions, letters):
    # Cas 1 : uniquement "n" ➝ ne rien changer
    if "n" in letters and "l" not in letters:
        return transitions

    # Cas 2 : uniquement "l" ➝ remplacer "n" par "l"
    if "l" in letters and "n" not in letters:
        new_transitions = {}
        for (state, symbol), next_state in transitions.items():
            new_symbol = "l" if symbol == "n" else symbol
            new_transitions[(state, new_symbol)] = next_state
        return new_transitions

    # Cas 3 : "n" et "l" ➝ ne rien changer
    return transitions

new_transitions = adapt_transitions(transitions, letters)

# Fonction de transition qui retourne le nouvel état et l'action associée
def transition(etat, char):
    new_state = new_transitions.get((etat, char), etat) 
    action = None

    if etat == "2":
        action = "OFF"
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))       # timbre i
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))       # timbre u
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=0))       # timbre a
    elif etat in {"3", "4", "11", "14"}:
        action = "ON"
        if etat in {"11"}:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=0))   # timbre a
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))   # timbre u
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=127)) # timbre i
        if etat in {"14"}:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))   # timbre i
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=0))   # timbre a
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=127)) # timbre u
        if etat in {"4"}:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))   # timbre i
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))   # timbre u
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=127)) # timbre a
        else:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))   # timbre i
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))   # timbre u
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=0))   # timbre a
    elif etat in {"8", "12", "15"}:
        action = "ON_OFF"
        if etat == "12":
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=0))   # timbre a
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))   # timbre u
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=127)) # timbre i
        if etat == "15":
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))   # timbre i
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=0))   # timbre a
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=127)) # timbre u
        if etat == "8":
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))   # timbre i
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))   # timbre u
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=127)) # timbre a
    elif etat == "13":
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=0))       # timbre a
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))       # timbre u
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=127))     # timbre i
    elif etat == "16":
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))       # timbre i
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=0))       # timbre a
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=127))     # timbre u
    elif etat == "9":
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))       # timbre i
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_u, value=0))       # timbre u
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_a, value=127))     # timbre a
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
alpha = 0.05
smoothed_midi_val_rms = 0
velocity_base = 100

# Générer un dictionnaire pour l'Enum
enum_dict = {letter.upper(): i + 1 for i, letter in enumerate(letters_with_st)}
enum_dict["VIDE"] = 99  # Ajouter VIDE=99

# Créer l'Enum dynamiquement
Label = Enum("Label", enum_dict)

# Voici ce qu'on obtient avec A,I,S,T
# class Label(Enum):
#     A = 1
#     I = 2
#     S = 3
#     T = 4
#     VIDE = 99

class Event(Enum):
    ON_NOTE_T = 0
    OFF_S = 1
    ON_L = 2
    ON_NOTE_L = 3
    OFF_I = 4

class Event(Enum):
    ON_NOTE_T = 0
    OFF_S = 1
    ON_L = 2
    ON_NOTE_L = 3
    OFF_I = 4
    
# Chargement du jeu d'entraînement 
label_mapping = {label: idx + 1 for idx, label in enumerate(letters_with_st)}
label_mapping_reverse = {v: k for k, v in label_mapping.items()}
label_mapping_reverse[99] = "vide"

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
radius = main_respiro_param["radius"]
k = main_respiro_param["k"]

# Creation du seuil majorité
seuil_majorite = int(k * batch_size / 2)

# Importer les données de la séquence
log_file = open("predictions_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp (ms)", "predictions_labels", "label_counts", "majority_label", "rms", "midi_val_rms"])

stop_flag = threading.Event()

##########################################################################################
# FENÊTRE TKINTER 
root = tk.Tk()
root.title("Contrôle en temps réel")
root.geometry("500x250")
root.configure(bg="#2c3e50")

# Texte principal
label_title = tk.Label(
    root,
    text="Soufflez pour commencer",
    font=("Arial", 16, "bold"),
    bg="#2c3e50",
    fg="#ecf0f1"
)
label_title.pack(pady=10)

# Bouton STOP
def stop_program():
    stop_flag.set()
    label_title.config(text="Arrêt en cours...")
    root.after(500, root.destroy)  # Ferme la fenêtre tkinter proprement

stop_button = tk.Button(
    root,
    text="Stop",
    command=stop_program,
    font=("Arial", 12),
    bg="#e74c3c",
    fg="black",
    activebackground="#c0392b",
    activeforeground="black",
    padx=15,
    pady=5
)
stop_button.pack(pady=15)

##########################################################################################
# PARAMETRE POUR L'AFFICHAGE GRAPHIQUE
# Mode interactif
plt.ion()

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1, 1)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')  # Pas d'axes visibles

# Positions horizontales : gauche et droite
positions = [
    (-0.8, 0.5),    # Gauche
    (0.8, 0.5),     # Droite
]

text_objects = []
# Ajouter une zone de texte en bas pour l'historique
history_text = ax.text(
    0, -1.3,  # position sous les lettres
    "",       # texte vide au départ
    fontsize=20,
    ha='center',
    va='center'
)

# Créer deux emplacements pour les lettres
for pos in positions:
    txt = ax.text(
        pos[0], pos[1],
        "",              # Vide au départ
        fontsize=50,
        ha='center',
        va='center'
    )
    text_objects.append(txt)

# Créer l’aiguille (ligne)
# Elle partira de (0, -1) et bougera son extrémité
needle, = ax.plot([0, 0], [-1, -0.5], lw=3, color='red')

plt.show()

cycle_duration = 2
last_update_time = 0

##########################################################################################
# BOUCLE AUDIO DANS UN THREAD 
historique_lettres = []
etat = "1"
def audio_loop():
    global rms_max
    global audio_buffer
    global smoothed_midi_val_rms
    global start
    global mfcc_buffer
    global etat
    global last_update_time
    try:
        sequence_labels = []
        last_action_count = 0
        print("Go")
        label_title.config(text="Go")
        while not stop_flag.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # with lock:
            audio_frames.append(data) 

            # Ajouter au tampon
            audio_buffer = np.concatenate((audio_buffer, audio_data))

            # Calcul RMS
            rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
            if rms > rms_max:
                rms_max = rms
            rms_clip = (rms - 0) / (rms_max - 0)
            rms_sqrt = np.tanh(1.6*rms_clip)
            midi_val_rms = rms_sqrt * 126
            smoothed_midi_val_rms = alpha * midi_val_rms + (1 - alpha) * smoothed_midi_val_rms
            rms_max = main_respiro_param["rms_max"]

            rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
            
            while len(audio_buffer) >= int(CHUNK * taux_recouvrement):
                
                # Envoie de la valeur rms pour les breath control
                midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_rms, value=int(smoothed_midi_val_rms)))

                # Normalisation de l'amplitude du signal
                if rms > 0:
                    audio_data = audio_data / rms  # normalise le signal par sa propre puissance RMS
                    audio_data = np.clip(audio_data, -1.0, 1.0)  # évite les dépassements d'amplitude
                    audio_data = (audio_data * 32767).astype(np.int16)

                start += len(audio_buffer)
                frames.extend(audio_data) # Decommenter pour l'affichage

                # Parametres pour le calcul des MFCC
                all_audio_buffer = audio_buffer[:int(CHUNK * taux_recouvrement)] 
                audio_buffer = audio_buffer[CHUNK:] 
                block = all_audio_buffer.astype(np.float32) / 32768.0
                
                # 1. Calcule des MFCC
                mfcc = spectral.mfcc(y=block.astype(float), sr=fs, n_mfcc=n_mfcc,
                                        n_fft=n_fft, win_length=n_fft, hop_length=n_fft // 10,
                                        fmax=fs/2, mel_basis=mel_basis)

                features = [mfcc.flatten()]

                # 2. Delta MFCC (dérivées)
                if use_delta:
                    delta = librosa.feature.delta(mfcc, order=1)
                    features.append(delta.flatten())

                # 3. Pente moyenne du signal
                if use_slope:
                    slope = np.diff(block)
                    slope_rms = np.sqrt(np.mean(slope ** 2))
                    features.append(np.array([slope_rms]))

                # 4. Zero-crossing rate
                if use_zcr:
                    zcr = np.mean(librosa.feature.zero_crossing_rate(block, frame_length=len(block), hop_length=len(block)))
                    features.append(np.array([zcr]))

                # 5. Spectral centroid
                if use_centroid:
                    centroid = np.mean(librosa.feature.spectral_centroid(y=block, sr=fs))
                    features.append(np.array([centroid]))

                mfcc_vector = mfcc.flatten()
                feature_vector = np.concatenate([mfcc_vector, [slope_rms, zcr]])
                feature_centered = feature_vector - mean_X_truncated
                feature_scaled = feature_centered / std_X_truncated

                # Ajouter à la mémoire tampon
                mfcc_buffer.append(feature_scaled)
                time_values.append(start / fs) # decommenter pour l'affichage
                
                # Prédire quand on a accumulé assez de blocs
                if len(mfcc_buffer) >= batch_size:
                    df_mfcc = pd.DataFrame(mfcc_buffer)
                    df_mfcc_pca = df_mfcc @ eigenvectors_thresholded

                    # Recherche des voisinsu niquement dans le rayon
                    indices_radius, distances_radius = tree.query_radius(df_mfcc_pca, r=radius, return_distance=True, sort_results=True)

                    predictions = []
                    neighbors_per_point = []
                    
                    for dists, idxs in zip(distances_radius, indices_radius):
                        if len(idxs) == 0:
                            predictions.append(Label.VIDE.value)
                            tab_pred.append(Label.VIDE.value)
                            neighbors_per_point.append(0)
                        else:
                        # Prendre les k plus proches dans le rayon
                            top_idxs = idxs[:k]  
                            labels = y_ent[top_idxs]
                            pred = np.bincount(labels).argmax() # renvoie le label majoritaire                     
                            predictions_labels = label_mapping_reverse[pred]
                            if predictions_labels in letters:
                                # Moyenne du batch projeté                       
                                batch_center = df_mfcc_pca.mean(axis=0)
                                # Calcul des distances aux centroïdes
                                distances_to_centroids = {                            
                                    label: np.linalg.norm(batch_center - centroids[label])
                                    for label in [l.upper() for l in letters_with_st]                        
                                    }

                                min_dist = min(distances_to_centroids.values())
                                max_dist = max(distances_to_centroids.values())

                                # Trier les lettres par distance croissante
                                sorted_labels = sorted(distances_to_centroids.items(), key=lambda x: x[1])

                                # Ne garder que les 2 plus proches
                                closest_labels = sorted_labels[:2]

                                # Extraire uniquement les lettres
                                displayed_letters = [lbl for lbl, _ in closest_labels]

                                # Mettre à jour l’affichage des lettres
                                for i, txt in enumerate(text_objects):
                                    txt.set_text(displayed_letters[i])

                                # Si assez de temps s’est écoulé depuis la dernière mise à jour
                                current_time = time.time()
                                if current_time - last_update_time >= 0.01:
                                    elapsed = (current_time % cycle_duration)

                                    # Récupérer les deux lettres les plus proches
                                    (label1, dist1), (label2, dist2) = closest_labels

                                    # plus dist1 est petit, plus weight est proche de 1
                                    weight = dist2 / (dist1 + dist2 + 1e-6)  # Normalisation pour éviter division par 0
                                    weight = np.clip(weight, 0, 1)  # S’assurer que weight est entre 0 et 1

                                    # Récupérer les positions des lettres
                                    pos1 = positions[0]  # gauche
                                    pos2 = positions[1]  # droite

                                    # Calculer la position pondérée de l’aiguille
                                    x_tip = pos1[0] * weight + pos2[0] * (1 - weight)
                                    y_tip = pos1[1] * weight + pos2[1] * (1 - weight)

                                    base_x = 0
                                    base_y = -1.2

                                    # Mettre à jour la ligne (aiguille)
                                    needle.set_data([base_x, x_tip], [base_y, y_tip])

                                    # Mettre à jour les lettres
                                    for i, txt in enumerate(text_objects):
                                        txt.set_text(displayed_letters[i])

                                    plt.draw()
                                    # plt.pause(0.001)  #  pour rester fluide
                                    last_update_time = current_time

                            tab_pred.append(pred) # Decommenter pour l'affichage
                            predictions.extend(labels.tolist())
                            neighbors_per_point.append(len(top_idxs))

                    predictions = np.array(predictions)
                    predictions_indices = predictions 

                    predictions_labels = [label_mapping_reverse[idx] for idx in predictions_indices]
                    label_counts = Counter(predictions_labels)

                    # Supprimer les 'vide' du comptage
                    non_vide_counts = {label: count for label, count in label_counts.items() if label != "vide"}
                            
                    # Maintenant on peut appeler np.bincount en toute sécurité
                    majority_label = int(np.bincount(predictions_indices).argmax())

                    # Si plus de 3 labels différents (non vide), forcer majority_label à VIDE
                    if use_vide_if_n_label: 
                        if len(non_vide_counts) > n_label_for_if_vide:
                            majority_label = Label.VIDE.value
                        else:
                            majority_label_name = label_mapping_reverse.get(majority_label, "vide")
                            count_majority_label = label_counts.get(majority_label_name, 0)

                            if count_majority_label < seuil_majorite:
                                majority_label = Label.VIDE.value

                    # On associe au VIDE les labels trop loin du centre
                    df_mfcc_pca_np = df_mfcc_pca.values if hasattr(df_mfcc_pca, 'values') else df_mfcc_pca

                    # Ajouter le label courant dans l'historique (en gardant seulement les 4 derniers)
                    majority_label_prec.append(majority_label)
                    if len(majority_label_prec) > 5:
                        majority_label_prec.pop(0)

                    # Ajouter le dernier label majoritaire qui n'est pas vide
                    if majority_label != Label.VIDE.value:
                        majority_label_no_vide = majority_label

                        if majority_label != Label.S.value:
                            # Ajouter la lettre détectée à l’historique si elle est différente de la précédente
                            label_char = Label(majority_label).name.lower()
                            if not majority_labels_seq or majority_labels_seq[-1] != majority_label:
                                majority_labels_seq.append(majority_label)
                                # Mettre à jour l’historique des lettres (éviter doublons consécutifs)
                                # if not 'historique_lettres' in globals():
                                #     historique_lettres = []
                                if not historique_lettres or historique_lettres[-1] != label_char:
                                    historique_lettres.append(label_char)
                                    if len(historique_lettres) > 10:
                                        historique_lettres.pop(0)

                            # Afficher l’historique sous forme de texte
                            history_text.set_text("".join(historique_lettres))

                    # Vérification : 4 précédents + celui actuel = 5 't' → remplacer par 'i'
                    if use_remplacer_t_par_i:
                        if (
                            majority_label == label_mapping["t"]
                            and len(majority_label_prec) == n_label_for_use_remplacer_t_par_i
                            and all(lbl == label_mapping["t"] for lbl in majority_label_prec)
                        ):  
                            if "i" in letters:
                                majority_label = Label.I.value

                    mfcc_features.extend(df_mfcc.values.tolist())  # decommanter pour correction
                    time_values.extend([start / fs] * len(predictions)) # Decommenter pour correction

                    if majority_label != Label.VIDE.value:
                        
                        predictions[:] = majority_label
                        majority_labels_seq.append(majority_label)

                        # Enregistrement dans fichier secondaire
                        timestamp = (start / fs)  # temps en secondes
                        label_name = Label(majority_label).name.lower()
                        csv_writer.writerow([
                            f"{timestamp:.3f}",
                            str(predictions_labels),
                            str(dict(label_counts)),
                            label_name,
                            rms,
                            midi_val_rms
                        ])
                        log_file.flush()

                        etat, action = transition(etat, Label(majority_label).name.lower())
                        if action == "ON_OFF":
                            handle_event("ON")
                            handle_event("OFF")
                        elif action:
                            handle_event(action) 
        
                    # Réinitialiser les buffers avec fenetre glissante
                    mfcc_buffer = mfcc_buffer[recouvrement:]

    except Exception as e:
            print(f"Erreur audio : {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Flux audio arrêté.")

audio_thread = threading.Thread(target=audio_loop)
audio_thread.start()

##########################################################################################
# LANCEMENT INTERFACE 
root.mainloop()
audio_thread.join()
    
# Fermeture du flux et de PyAudio
stream.stop_stream()
stream.close()
p.terminate()

########################################################################################
# REAFFICHAGE PCA best comb
# Calcul PCA 2D features
X_selected_pca = df_loaded.iloc[:, :-1].values

# Mapping couleur → label (inversé du dico original)
block_labels_balanced = df_loaded["label"].values

# Couleurs
colors = {"f": "blue", "a": "green", "i": "red", "s": "orange", "t": "purple", "u": "cyan", "n": "brown"}
class_colors = [colors[l] for l in block_labels_balanced]

# Prédictions en temps réel
predictions_array = np.array(tab_pred)
pred_l_mask = predictions_array == "f"

mfcc_matrix = np.array(mfcc_features)
mfcc_pca_temp_reel = mfcc_matrix @ eigenvectors_thresholded

fig3d = go.Figure()

# 4. Nuage d'entraînement (PCA de base)
fig3d.add_trace(go.Scatter3d(
    x=X_selected_pca[:, 0],
    y=X_selected_pca[:, 1],
    z=X_selected_pca[:, 2],
    mode='markers',
    marker=dict(size=3, color=class_colors, opacity=0.12),
    name="Base PCA"
))

# 5. Prédictions temps réel autres que 'l'
for int_label, color in  {1: "green", 2: "red", 3: "orange", 4: "purple", 99: "black"}.items():
    # if int_label == 0 or int_label == 3:
    mask = predictions_array == int_label
    fig3d.add_trace(go.Scatter3d(
        x=mfcc_pca_temp_reel[mask ,0],
        y=mfcc_pca_temp_reel[mask ,1],
        z=mfcc_pca_temp_reel[mask ,2],
        mode='markers',
        marker=dict(size=6, color=color, line=dict(color='black', width=0.5)),
        name=f"Prédits '{Label(int_label).name.lower()}'",
        text=[f"t = {t:.2f}s" for t in (np.array(time_values)[mask] * 0.1)],
        hoverinfo='text'
    ))

# 7. Sphères 3D + centroïdes
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 30)
u, v = np.meshgrid(u, v)

for label, center in centroids.items():
    r0 = thresholds[label]["axis0"]
    r1 = thresholds[label]["axis1"]
    r2 = thresholds[label]["axis2"]

    xs = center[0] + r0 * np.cos(u) * np.sin(v)
    ys = center[1] + r1 * np.sin(u) * np.sin(v)
    zs = center[2] + r2 * np.cos(v)

    color_label = label.lower()
    fig3d.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.12,
        colorscale=[[0, colors[color_label]], [1, colors[color_label]]],
        showscale=False,
        name=f"Sphère seuil '{label}'"
    ))

    fig3d.add_trace(go.Scatter3d(
        x=[center[0]],
        y=[center[1]],
        z=[center[2]],
        mode="markers+text",
        marker=dict(size=10, color="black", symbol="x"),
        text=[f"Centroïde '{label}'"],
        textposition="top center",
        name=f"Centroïde '{label}'"
    ))

# 8. Layout final
fig3d.update_layout(
    title="PCA 3D — Prédictions temps réel + sphères seuils et centroïdes",
    scene=dict(
        xaxis=dict(title="PCA 1"),
        yaxis=dict(title="PCA 2"),
        zaxis=dict(title="PCA 3"),
        aspectmode="cube"
    ),
    legend_title="Légende",
    margin=dict(l=0, r=0, b=0, t=50)
)

fig3d.show()