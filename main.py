# jouer sur les parametres de threshold de temps reel et correction
import pyaudio 
import pandas as pd  
import numpy as np  
import joblib  
import librosa 
from scipy.signal import butter
import mido  
import fluidsynth  
from enum import Enum
##########################################################################################
# CHANGER LA SOURCE ET LA SORTIE 
# pactl set-default-source alsa_input.usb-Focusrite_Scarlett_4i4_USB_D89YRE90C17918-00.multichannel-input
# pactl set-default-sink alsa_output.pci-0000_00_1f.3.analog-stereo

##########################################################################################
# INITIALISATION ET CHARGEMENT DES DONNEES

# Initialiser FluidSynth avec une soundfont
fluid = fluidsynth.Synth()
fluid.start()
sfid = fluid.sfload("FluidR3_GM/FluidR3_GM.sf2")  
fluid.program_select(0, sfid, 0, 73)

print("ici222")
# Charger le fichier MIDI
midi_file = mido.MidiFile("midi/potter.mid")

# Charger le modele et les vecteurs propres
knn_model = joblib.load("scripts/knn_model_db_sans_r_opt_main_corrige_avant.pkl")  
eigenvectors_thresholded = joblib.load("scripts/eigenvectors_thresholded_corrige_avant_tronque.pkl") 
mean_X = joblib.load("scripts/mfcc_mean.pkl")
std_X = joblib.load("scripts/mfcc_std.pkl")

# Paramètres audio
taux_recouvrement = 1
FORMAT = pyaudio.paInt16 
CHANNELS = 1  
RATE = 44100
CHUNK = int((RATE * 0.005) // taux_recouvrement)

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
    # input_device_index=6,
    # output_device_index=1,
    frames_per_buffer=2048
)
audio_frames = []
frames = []

# Initialisation des paramètres MFCC
mfcc_features = []
time_values = []
start = 0
n_fft = min(512, int(CHUNK * taux_recouvrement))
window = np.ones(min(512, int(CHUNK * taux_recouvrement)))
win_length = n_fft

# Build a Mel filter
mel_basis = librosa.filters.mel(sr=RATE, n_fft=n_fft, fmax=RATE/2, n_mels=40)

# Calcul FFT
fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

# Pad the window out to n_fft size
fft_window = librosa.util.pad_center(fft_window, size=n_fft)

batch_size = 6  # Nombre de blocs MFCC à accumuler avant prédiction
mfcc_buffer = []
tab_pred = []
events = []
proba_list = []

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
# Dictionnaire des transitions
transitions = {
    ("1", "l"): "5",  ("1", "a"): "4",  ("1", "t"): "3",  ("1", "s"): "2",
    ("2", "l"): "5",  ("2", "a"): "4",  ("2", "t"): "3",  ("2", "s"): "10",
    ("3", "l"): "6",  ("3", "a"): "9",  ("3", "t"): "6",  ("3", "s"): "2",
    ("4", "l"): "7",  ("4", "a"): "9",  ("4", "t"): "6",  ("4", "s"): "2",
    ("5", "l"): "5",  ("5", "a"): "4",  ("5", "t"): "3",  ("5", "s"): "10",
    ("6", "l"): "6",  ("6", "a"): "9",  ("6", "t"): "6",  ("6", "s"): "2",
    ("7", "l"): "7",  ("7", "a"): "8",  ("7", "t"): "7",  ("7", "s"): "2",
    ("8", "l"): "7",  ("8", "a"): "9",  ("8", "t"): "9",  ("8", "s"): "2",
    ("9", "l"): "7",  ("9", "a"): "9",  ("9", "t"): "9",  ("9", "s"): "2",
    ("10", "l"): "5", ("10", "a"): "4", ("10", "t"): "3", ("10", "s"): "10"
}

# Fonction de transition qui retourne le nouvel état et l'action associée
def transition(etat, char):
    new_state = transitions.get((etat, char), etat) 
    action = None

    if etat == "2":
        action = "OFF"
    elif etat in {"3", "4"}:
        action = "ON"
    elif etat == "8":
        action = "ON_OFF"

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
            fluid.noteon(0, note.note, note.velocity)
            # print(f"ON → note {note.note} (pile = {[n.note for n in note_stack]})")
            note_pointer += 1
    elif event_type == "OFF":
        if note_stack:
            note_to_off = note_stack.pop(0)
            fluid.noteoff(0, note_to_off.note)
            # print(f"OFF → note {note_to_off.note} (pile = {[n.note for n in note_stack]})")

##########################################################################################
# PRISE DU FLUX AUDIO EN TEMPS REEL
prev_event = None
midi_notes = [msg for msg in midi_file if msg.type == 'note_on' and msg.velocity !=0]  # Extraire les notes MIDI
note_id = -1
buffer_sensibilite_l = []
buffer_sensibilite_a = []
audio_buffer = np.array([], dtype=np.int16)

class Label(Enum):
    L = 0
    A = 1
    S = 2
    T = 3
    R = 4
    VIDE = 5

class Event(Enum):
    ON_NOTE_T = 0
    OFF_S = 1
    ON_L = 2
    ON_NOTE_L = 3
    OFF_R = 4
    
# Chargement du jeu d'entraînement pour obtenir les points 'l'
label_mapping = {"l": 0, "a": 1, "s": 2, "t": 3, "r": 4}

df_ent = pd.read_csv("scripts/X_proj_scaled_avec_labels_corrige_avant.csv")

df_ent["label"] = df_ent["label"].map(label_mapping)

X_ent = df_ent.iloc[:, :-1].values
y_ent = df_ent["label"].values.astype(int)

centroids = {}
thresholds = {}

# Coefficients spécifiques par label
sigma_factors = {
    "L": (5, 5, 5),
    "T": (5, 4, 4),
    "S": (5, 5, 5),
    "A": (5, 4, 4),
    "R": (5, 5, 5),
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

# Initialisation de l'état de l'automate
etat = "1"
try:
    sequence_labels = []
    last_action_count = 0
    print("Go")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Ajouter au tampon
        audio_buffer = np.concatenate((audio_buffer, audio_data))

        while len(audio_buffer) >= int(CHUNK * taux_recouvrement):

            # Parametres pour le calcul des MFCC
            all_audio_buffer = audio_buffer[:int(CHUNK * taux_recouvrement)] 
            audio_buffer = audio_buffer[CHUNK:] 
            block = all_audio_buffer.astype(np.float32) / 32768.0
            fs = RATE
            start += len(audio_buffer)
            
            # Calcule des MFCC
            mfcc = librosa.feature.mfcc(y=block.astype(float), sr=fs ,n_mfcc=8,
                                n_fft=min(512, len(block)), win_length = n_fft, hop_length= win_length // 10,
                                fmax=fs/2, mel_basis = mel_basis)
            
            mfcc_vector = mfcc.flatten()
            mfcc_centered = mfcc_vector - mean_X[:-55]
            mfcc_scaled = mfcc_centered / std_X[:-55] 

            # Ajouter à la mémoire tampon
            mfcc_buffer.append(mfcc_scaled)
            
            # Prédire quand on a accumulé assez de blocs
            if len(mfcc_buffer) >= batch_size:

                # Projeter les MFCCs sur les vecteurs propres
                df_mfcc = pd.DataFrame(mfcc_buffer)
                df_mfcc_pca = df_mfcc @ eigenvectors_thresholded

                # Predire avec le modèle KNN
                predictions = knn_model.predict(df_mfcc_pca)
                predictions_indices = predictions 
                        
                # Calcule les labels majoritaires
                majority_label = int(np.bincount(predictions_indices).argmax())
                predictions[:] = majority_label

                # Calcul des etats et actions
                etat, action = transition(etat, Label(majority_label).name.lower())
                if action == "ON_OFF":
                    handle_event("ON")
                    handle_event("OFF")
                elif action:
                    handle_event(action)
    
                # Réinitialiser les buffers
                mfcc_buffer = []

except KeyboardInterrupt:
    print("\nArrêt de l'enregistrement.")

    
# Fermeture du flux et de PyAudio
stream.stop_stream()
stream.close()
p.terminate()
