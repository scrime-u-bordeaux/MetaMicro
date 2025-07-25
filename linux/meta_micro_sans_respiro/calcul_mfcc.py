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
import scripts.modif_libro.spectral as spectral  
from sklearn.neighbors import BallTree   
from collections import Counter
import csv
import threading
import yaml
import tkinter as tk

##########################################################################################
# CHANGER LA SOURCE ET LA SORTIE 
# pactl set-default-source alsa_input.usb-Focusrite_Scarlett_4i4_USB_D89YRE90C17918-00.multichannel-input
# pactl set-default-sink alsa_output.pci-0000_00_1f.3.analog-stereo

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

# Charger le fichier MIDI
midi_file = mido.MidiFile(main_respiro_param["midi_file"])

# Port pour midifile
port_name_midifile = main_respiro_param["port_name_midifile"]
port2 = mido.open_output(port_name_midifile)

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

# Initialiser FluidSynth avec une soundfont
fluid = fluidsynth.Synth()
fluid.start()
sfid = fluid.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")  
fluid.program_select(0, sfid, 0, 73)

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
# PARAMETRES POUR PRISE DU FLUX AUDIO EN TEMPS REEL
# Def de la fonction callback
latest_audio = np.zeros(CHUNK, dtype=np.int16)

def callback(in_data, frame_count, time_info, status):
    global latest_audio
    latest_audio = np.frombuffer(in_data, dtype=np.int16)
    return (in_data, pyaudio.paContinue)

# Ouverture du flux audio
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output_device_index=4,
    frames_per_buffer=CHUNK,
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
# Dictionnaire des transitions
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
# Fonction de transition qui retourne le nouvel état et l'action associée
def transition(etat, char):
    new_state = new_transitions.get((etat, char), etat) 
    action = None

    if etat == "2":
        action = "OFF"
    elif etat in {"3", "4", "11", "14"}:
        action = "ON"
    elif etat in {"8", "12", "15"}:
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
            note_pointer += 1
    elif event_type == "OFF":
        if note_stack:
            note_to_off = note_stack.pop(0)
            fluid.noteoff(0, note_to_off.note)

##########################################################################################
# PRISE DU FLUX AUDIO EN TEMPS REEL
prev_event = None
midi_notes = [msg for msg in midi_file if msg.type == 'note_on' and msg.velocity !=0]  # Extraire les notes MIDI
note_id = -1
majority_label_prec = []
audio_buffer = np.array([], dtype=np.int16)

# Générer un dictionnaire pour l'Enum
enum_dict = {letter.upper(): i + 1 for i, letter in enumerate(letters_with_st)}
enum_dict["VIDE"] = 99  # Ajouter VIDE=99

# Créer l'Enum dynamiquement
Label = Enum("Label", enum_dict)

class Event(Enum):
    ON_NOTE_T = 0
    OFF_S = 1
    ON_L = 2
    ON_NOTE_L = 3
    OFF_I = 4

# Chargement du jeu d'entraînement 
label_mapping = {label: idx + 1 for idx, label in enumerate(letters_with_st)}
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
    "L": (5, 5, 5)
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
csv_writer.writerow(["timestamp (ms)", "predictions_labels", "label_counts", "majority_label"])

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
# BOUCLE AUDIO DANS UN THREAD 
etat = "1"
def audio_loop():
    global audio_buffer
    global start
    global mfcc_buffer
    global etat
    try:
        sequence_labels = []
        last_action_count = 0
        print("Go")
        label_title.config(text="Go")
        while not stop_flag.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Ajouter au tampon
            audio_buffer = np.concatenate((audio_buffer, audio_data))

            while len(audio_buffer) >= int(CHUNK * taux_recouvrement):

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
                            tab_pred.append(pred) # Decommenter pour l'affichage
                            predictions.extend(labels.tolist())
                            neighbors_per_point.append(len(top_idxs))

                    predictions = np.array(predictions)
                    label_mapping = {label: idx + 1 for idx, label in enumerate(letters_with_st)}
                    predictions_indices = predictions 

                    label_mapping_reverse = {v: k for k, v in label_mapping.items()}
                    label_mapping_reverse[99] = "vide"
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

                    # Vérification : 4 précédents + celui actuel = 5 't' → remplacer par 'i'
                    if use_remplacer_t_par_i:
                        if (
                            majority_label == label_mapping["t"]
                            and len(majority_label_prec) == n_label_for_use_remplacer_t_par_i
                            and all(lbl == label_mapping["t"] for lbl in majority_label_prec)
                        ):
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