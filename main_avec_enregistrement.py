# Adaptation du traitement temps réel pour un fichier audio en mode batch avec rythme conservé
import numpy as np
import pandas as pd
import joblib
import librosa
import mido
import fluidsynth
from enum import Enum
from scipy.signal import butter
from sklearn.neighbors import BallTree
import csv
from collections import Counter
import scripts.modif_libro.spectral as spectral  

##########################################################################################
# INITIALISATION ET CHARGEMENT DES DONNEES
# Audio
RATE = 44100
CHUNK = int(RATE * 0.005)
filename = "audio/recorded.wav"
y, sr = librosa.load(filename, sr=RATE)
y_pcm = (y * 32768).astype(np.int16)
n_chunks = len(y_pcm) // CHUNK
audio_frames = []
frames = []

# Initialiser FluidSynth avec une soundfont
fluid = fluidsynth.Synth()
fluid.start(driver="coreaudio")
sfid = fluid.sfload("FluidR3_GM/FluidR3_GM.sf2")  
fluid.program_select(0, sfid, 0, 73)

# Parametres pour respiro
port_name = "IAC Driver Bus 1 Bus 1" 
midi_out = mido.open_output(port_name)

# Parametres pour les CC    
rms_max = 150 # Augmenter la valeur pour plus de variation de rms
CHANNEL_RESPIRO = 0
CC_rms = 2 # Channel 2: Breath Control
CC_i = 9 # Channel 9: changement de timbre

# Charger le fichier MIDI
midi_file = mido.MidiFile("midi/Etude 1.mid")

# Charger le modele et les vecteurs propres
knn_model = joblib.load("scripts/ta_la_ti_li/knn_model_db_sans_r_opt_main_corrige_avant_ta_la_ti_li_i.pkl")  
eigenvectors_thresholded = joblib.load("scripts/ta_la_ti_li/eigenvectors_thresholded_corrige_avant_tronque_ta_la_ti_li_i.pkl") 
mean_X = joblib.load("scripts/ta_la_ti_li/mfcc_mean_ta_la_ti_li_i.pkl")
std_X = joblib.load("scripts/ta_la_ti_li/mfcc_std_ta_la_ti_li_i.pkl")

# Tronquer mean et std
block_size = 11
actions = ['keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'drop', 'keep', 'drop', 'drop', 'keep', 'keep', 'drop']
n_blocks = len(actions)
rows_to_keep = []

for i, action in enumerate(actions):
    start = i * block_size
    end = start + block_size
    if action == 'keep':
        rows_to_keep.extend(range(start, end))

# Supprimer aussi les 22 dernières lignes
rows_to_keep = [i for i in rows_to_keep if i < mean_X.shape[0]]

# Appliquer le filtrage
mean_X_truncated = mean_X[rows_to_keep]
std_X_truncated = std_X[rows_to_keep]


# Initialisation des paramètres MFCC
mfcc_features = []
time_values = []
start = 0
n_fft = min(512, int(CHUNK))
window = np.ones(min(512, int(CHUNK)))
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
# transitions = { 
#       ("1","l") :"5"  , ("1","a") :"4"  , ("1","t") :"3"  , ("1","s") :"2"  , ("1","i") :"11" ,
#       ("2","l") :"5"  , ("2","a") :"4"  , ("2","t") :"3"  , ("2","s") :"10" , ("2","i") :"11" ,
#       ("3","l") :"6"  , ("3","a") :"9"  , ("3","t") :"6"  , ("3","s") :"2"  , ("3","i") :"13" ,
#       ("4","l") :"7"  , ("4","a") :"9"  , ("4","t") :"6"  , ("4","s") :"2"  , ("4","i") :"12" ,
#       ("5","l") :"5"  , ("5","a") :"4"  , ("5","t") :"3"  , ("5","s") :"10" , ("5","i") :"5"  , # ("5","i") :"11" 
#       ("6","l") :"6"  , ("6","a") :"9"  , ("6","t") :"6"  , ("6","s") :"2"  , ("6","i") :"13" , 
#       ("7","l") :"7"  , ("7","a") :"8"  , ("7","t") :"7"  , ("7","s") :"2"  , ("7","i") :"7" , # ("7","i") :"12"
#       ("8","l") :"7"  , ("8","a") :"9"  , ("8","t") :"9"  , ("8","s") :"2"  , ("8","i") :"12" ,
#       ("9","l") :"7"  , ("9","a") :"9"  , ("9","t") :"9"  , ("9","s") :"2"  , ("9","i") :"12" ,
#       ("10","l"):"5"  , ("10","a"):"4"  , ("10","t"):"3"  , ("10","s"):"10" , ("10","i"):"11" ,
#       ("11","l"):"7"  , ("11","a"):"8"  , ("11","t"):"6"  , ("11","s"):"2"  , ("11","i"):"13" ,
#       ("12","l"):"7"  , ("12","a"):"8"  , ("12","t"):"13" , ("12","s"):"2"  , ("12","i"):"13" ,
#       ("13","l"):"7"  , ("13","a"):"8"  , ("13","t"):"13" , ("13","s"):"2"  , ("13","i"):"13" 
# }

transitions = { 
      ("1","l") :"5"  , ("1","a") :"4"  , ("1","t") :"3"  , ("1","s") :"2"  , ("1","i") :"11" ,
      ("2","l") :"5"  , ("2","a") :"4"  , ("2","t") :"3"  , ("2","s") :"10" , ("2","i") :"11" ,
      ("3","l") :"6"  , ("3","a") :"9"  , ("3","t") :"6"  , ("3","s") :"2"  , ("3","i") :"13" ,
      ("4","l") :"7"  , ("4","a") :"9"  , ("4","t") :"6"  , ("4","s") :"2"  , ("4","i") :"12" ,
      ("5","l") :"5"  , ("5","a") :"4"  , ("5","t") :"3"  , ("5","s") :"10" , ("5","i") :"11" , 
      ("6","l") :"6"  , ("6","a") :"9"  , ("6","t") :"6"  , ("6","s") :"2"  , ("6","i") :"13" , 
      ("7","l") :"7"  , ("7","a") :"8"  , ("7","t") :"7"  , ("7","s") :"2"  , ("7","i") :"12" ,
      ("8","l") :"7"  , ("8","a") :"9"  , ("8","t") :"9"  , ("8","s") :"2"  , ("8","i") :"12" ,
      ("9","l") :"7"  , ("9","a") :"9"  , ("9","t") :"9"  , ("9","s") :"2"  , ("9","i") :"12" ,
      ("10","l"):"5"  , ("10","a"):"4"  , ("10","t"):"3"  , ("10","s"):"10" , ("10","i"):"11" ,
      ("11","l"):"7"  , ("11","a"):"8"  , ("11","t"):"6"  , ("11","s"):"2"  , ("11","i"):"13" ,
      ("12","l"):"7"  , ("12","a"):"8"  , ("12","t"):"13" , ("12","s"):"2"  , ("12","i"):"13" ,
      ("13","l"):"7"  , ("13","a"):"8"  , ("13","t"):"13" , ("13","s"):"2"  , ("13","i"):"13" 
}


# Fonction de transition qui retourne le nouvel état et l'action associée
def transition(etat, char):
    new_state = transitions.get((etat, char), etat) 
    action = None

    if etat == "2":
        action = "OFF"
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))       # timbre i
    elif etat in {"3", "4", "11"}:
        action = "ON"
        if etat in {"11"}:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=127)) # timbre i
        else:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))   # timbre i
    elif etat in {"8", "12"}:
        action = "ON_OFF"
        if etat == "12":
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=127)) # timbre i
        else:
            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))   # timbre i
    elif etat == "13":
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=127))     # timbre i
    elif etat == "9":
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_i, value=0))       # timbre i
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
            note_pointer += 1
    elif event_type == "OFF":
        if note_stack:
            note_to_off = note_stack.pop(0)
            midi_out.send(mido.Message('note_off', note = note_to_off.note))

##########################################################################################
# PRISE DU FLUX AUDIO EN TEMPS REEL
prev_event = None
midi_notes = [msg for msg in midi_file if msg.type == 'note_on' and msg.velocity !=0]  # Extraire les notes MIDI
note_id = -1
majority_label_prec = []
audio_buffer = np.array([], dtype=np.int16)

# Parametres pour respiro
alpha = 0.05
smoothed_rms = 0
velocity_base = 100

class Label(Enum):
    L = 0
    A = 1
    S = 2
    T = 3
    I = 4
    VIDE = 99

class Event(Enum):
    ON_NOTE_T = 0
    OFF_S = 1
    ON_L = 2
    ON_NOTE_L = 3
    OFF_I = 4
    
# Chargement du jeu d'entraînement pour obtenir les points 'l'
label_mapping = {"l": 0, "a": 1, "s": 2, "t": 3, "i": 4}

df_ent = pd.read_csv("scripts/ta_la_ti_li/X_proj_scaled_avec_labels_corrige_avant_ta_la_ti_li_i.csv")

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
    "I": (5, 5, 5),
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
radius = 1.7
k = 5

# Creation du seuil majorité
seuil_majorite = int(k * batch_size / 2)
print(seuil_majorite)

# Importer les données de la séquence
log_file = open("predictions_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp (ms)", "predictions_labels", "label_counts", "majority_label"])

# Boucle principale sur l'enregistrement
etat = "1"
timestamp = 0
for i in range(n_chunks):
    data = y_pcm[i * CHUNK:(i + 1) * CHUNK]
    audio_data = np.frombuffer(data, dtype=np.int16)

    # with lock:
    # audio_data = latest_audio.copy()
    audio_frames.append(data) # décommenter pour l'enregistrement

    # Ajouter au tampon
    audio_buffer = np.concatenate((audio_buffer, audio_data))

    # Calcul RMS
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    smoothed_rms = alpha * rms + (1 - alpha) * smoothed_rms
    if smoothed_rms>rms_max:
        smoothed_rms = rms_max
    midi_val_rms = int((smoothed_rms/rms_max)*126)
    bar_length = int(midi_val_rms*0.2) 
    # bar = "-" * bar_length
    # sys.stdout.write(f"\rRMS: {bar}")
    # sys.stdout.flush()

    while len(audio_buffer) >= int(CHUNK):
        
        # Envoie de la valeur rms pour les breath control
        midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_rms, value=midi_val_rms))

        # Normalisation de l'amplitude du signal
        if rms > 0:
            audio_data = audio_data / rms  # normalise le signal par sa propre puissance RMS
            audio_data = np.clip(audio_data, -1.0, 1.0)  # évite les dépassements d'amplitude
            audio_data = (audio_data * 32767).astype(np.int16)

        start += len(audio_buffer)
        frames.extend(audio_data) # Decommenter pour l'affichage

        # Parametres pour le calcul des MFCC
        all_audio_buffer = audio_buffer[:int(CHUNK)] 
        audio_buffer = audio_buffer[CHUNK:] 
        block = all_audio_buffer.astype(np.float32) / 32768.0
        fs = RATE
        
        # Calcule des MFCC
        mfcc = spectral.mfcc(y=block.astype(float), sr=fs ,n_mfcc=11,
                            n_fft=min(512, len(block)), win_length = n_fft, hop_length= win_length // 10,
                            fmax=fs/2, mel_basis = mel_basis)
        
        mfcc_vector = mfcc.flatten()
        mfcc_centered = mfcc_vector - mean_X_truncated
        mfcc_scaled = mfcc_centered / std_X_truncated

        # Ajouter à la mémoire tampon
        mfcc_buffer.append(mfcc_scaled)
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
                    tab_pred.append(pred) # Decommenter pour l'affichage
                    predictions.extend(labels.tolist())
                    neighbors_per_point.append(len(top_idxs))

            predictions = np.array(predictions)
            label_mapping = {"l": 0, "a": 1, "s": 2, "t": 3, "i": 4}
            predictions_indices = predictions 

            # print
            label_mapping_reverse = {0: "l", 1: "a", 2: "s", 3: "t", 4: "i", 99: "vide"}
            predictions_labels = [label_mapping_reverse[idx] for idx in predictions_indices]
            label_counts = Counter(predictions_labels)
            # print(predictions_labels)
            # print("Occurrences:", label_counts)

            # Supprimer les 'vide' du comptage
            non_vide_counts = {label: count for label, count in label_counts.items() if label != "vide"}

                    
            # Maintenant on peut appeler np.bincount en toute sécurité
            majority_label = int(np.bincount(predictions_indices).argmax())

            # Si plus de 3 labels différents (non vide), forcer majority_label à VIDE
            if len(non_vide_counts) > 2:
                majority_label = Label.VIDE.value
            else:
                majority_label_name = label_mapping_reverse.get(majority_label, "vide")
                count_majority_label = label_counts.get(majority_label_name, 0)

                if count_majority_label < seuil_majorite:
                    # print((start / fs) , "ici")
                    majority_label = Label.VIDE.value

            # On associe au VIDE les labels trop loin du centre
            df_mfcc_pca_np = df_mfcc_pca.values if hasattr(df_mfcc_pca, 'values') else df_mfcc_pca

            # Ajouter le label courant dans l'historique (en gardant seulement les 4 derniers)
            majority_label_prec.append(majority_label)
            if len(majority_label_prec) > 5:
                majority_label_prec.pop(0)

            # Vérification : 4 précédents + celui actuel = 5 't' → remplacer par 'i'
            if (
                majority_label == label_mapping["t"]
                and len(majority_label_prec) == 5
                and all(lbl == label_mapping["t"] for lbl in majority_label_prec)
            ):
                majority_label = Label.VIDE.value
                # majority_label = Label.I.value

            if majority_label != Label.VIDE.value:
                mfcc_features.extend(df_mfcc.values.tolist())  # decommanter pour correction
                time_values.extend([start / fs] * len(predictions)) # Decommenter pour correction
                
                predictions[:] = majority_label
                majority_labels_seq.append(majority_label)

                # Enregistrement dans fichier secondaire
                timestamp = (start / fs)  # temps en secondes
                label_name = Label(majority_label).name.lower()
                csv_writer.writerow([
                    f"{timestamp:.3f}",
                    str(predictions_labels),
                    str(dict(label_counts)),
                    label_name
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
