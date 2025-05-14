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
import sys
import wave

##########################################################################################
# CHANGER LA SOURCE ET LA SORTIE 
# pactl set-default-source alsa_input.usb-Focusrite_Scarlett_4i4_USB_D89YRE90C17918-00.multichannel-input
# pactl set-default-sink alsa_output.pci-0000_00_1f.3.analog-stereo

##########################################################################################
# INITIALISATION ET CHARGEMENT DES DONNEES
# Initialiser FluidSynth avec une soundfont
fluid = fluidsynth.Synth()
fluid.start(driver="coreaudio")
sfid = fluid.sfload("FluidR3_GM/FluidR3_GM.sf2")  
fluid.program_select(0, sfid, 0, 73)

# Parametres pour respiro
port_name = "IAC Driver Bus 1" 
midi_out = mido.open_output(port_name)

# Parametres pour les CC    
rms_max = 150 # Augmenter la valeur pour plus de variation de rms
CHANNEL_RESPIRO = 0
CC_rms = 2 # Channel 2: Breath Control

# Charger le fichier MIDI
midi_file = mido.MidiFile("midi/potter.mid")

# Charger le modele et les vecteurs propres
knn_model = joblib.load("scripts/ta_la_ti_li/knn_model_db_sans_r_opt_main_corrige_avant_ta_la_ti_li_i.pkl")  
eigenvectors_thresholded = joblib.load("scripts/ta_la_ti_li/eigenvectors_thresholded_corrige_avant_tronque_ta_la_ti_li_i.pkl") 
mean_X = joblib.load("scripts/ta_la_ti_li/mfcc_mean_ta_la_ti_li_i.pkl")
std_X = joblib.load("scripts/ta_la_ti_li/mfcc_std_ta_la_ti_li_i.pkl")

# Tronquer mean et std
block_size = 11
actions = ['keep', 'drop', 'keep', 'keep', 'drop', 'keep', 'drop', 'keep', 'drop', 'drop', 'keep']
n_blocks = len(actions)
rows_to_keep = []

for i, action in enumerate(actions):
    start = i * block_size
    end = start + block_size
    if action == 'keep':
        rows_to_keep.extend(range(start, end))

# Supprimer aussi les 22 dernières lignes
rows_to_keep = [i for i in rows_to_keep if i < mean_X.shape[0] - 22]

# Appliquer le filtrage
mean_X_truncated = mean_X[rows_to_keep]
std_X_truncated = std_X[rows_to_keep]

# Paramètres audio
filename = "audio/recorded.wav"
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
window = np.ones(min(512, int(CHUNK * taux_recouvrement)))
win_length = n_fft

# Build a Mel filter
mel_basis = librosa.filters.mel(sr=RATE, n_fft=n_fft, fmax=RATE/2, n_mels=40)

# Calcul FFT
fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

# Pad the window out to n_fft size
fft_window = librosa.util.pad_center(fft_window, size=n_fft)

batch_size = 10 # Nombre de blocs MFCC à accumuler avant prédiction
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
    elif etat in {"3", "4", "11"}:
        action = "ON"
    elif etat in {"8", "12"}:
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
buffer_sensibilite_l = []
buffer_sensibilite_a = []
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
    VIDE = 5

class Event(Enum):
    ON_NOTE_T = 0
    OFF_S = 1
    ON_L = 2
    ON_NOTE_L = 3
    OFF_I = 4
    
# Chargement du jeu d'entraînement pour obtenir les points 'l'
label_mapping = {"l": 0, "a": 1, "s": 2, "t": 3, "i": 4}

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

# Initialisation de l'état de l'automate
etat = "1"
try:
    sequence_labels = []
    last_action_count = 0
    print("Go")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
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

        while len(audio_buffer) >= int(CHUNK * taux_recouvrement):

            # Parametres pour le calcul des MFCC
            all_audio_buffer = audio_buffer[:int(CHUNK * taux_recouvrement)] 
            audio_buffer = audio_buffer[CHUNK:] 
            block = all_audio_buffer.astype(np.float32) / 32768.0
            fs = RATE
            start += len(audio_buffer)

            midi_out.send(mido.Message('control_change', channel=CHANNEL_RESPIRO, control=CC_rms, value=midi_val_rms))
            
            # Calcule des MFCC
            mfcc = spectral.mfcc(y=block.astype(float), sr=fs ,n_mfcc=11,
                                n_fft=min(512, len(block)), win_length = n_fft, hop_length= win_length // 10,
                                fmax=fs/2, mel_basis = mel_basis)
            
            mfcc_vector = mfcc.flatten()
            mfcc_centered = mfcc_vector - mean_X_truncated
            mfcc_scaled = mfcc_centered / std_X_truncated

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

# try:
#     sequence_labels = []
#     last_action_count = 0
#     print("Go")
#     while True:
#         data = stream.read(CHUNK, exception_on_overflow=False)
#         audio_data = np.frombuffer(data, dtype=np.int16)
#         audio_frames.append(data) # Decommenter pour l'affichage

#         # Ajouter au tampon
#         audio_buffer = np.concatenate((audio_buffer, audio_data))

#         while len(audio_buffer) >= int(CHUNK * taux_recouvrement):

#             frames.extend(audio_data) # Decommenter pour l'affichage

#             # Calcul des MFCCs
#             all_audio_buffer = audio_buffer[:int(CHUNK * taux_recouvrement)] 
#             audio_buffer = audio_buffer[CHUNK:] 
#             block = all_audio_buffer.astype(np.float32) / 32768.0
#             fs = RATE
#             start += len(audio_buffer)
            
#             mfcc = spectral.mfcc(y=block.astype(float), sr=fs ,n_mfcc=8,
#                                 n_fft=min(512, len(block)), win_length = n_fft, hop_length= win_length // 10,
#                                 fmax=fs/2, mel_basis = mel_basis)
            
#             # mfcc_vector = np.concatenate(selected_blocks)
#             mfcc_vector = mfcc.flatten()
#             # print(mean_X , std_X)
#             mfcc_centered = mfcc_vector - mean_X[:-55]
#             mfcc_scaled = mfcc_centered / std_X[:-55] 

#             # Ajouter à la mémoire tampon
#             mfcc_buffer.append(mfcc_scaled)
#             time_values.append(start / fs) # decommenter pour l'affichage

#             # Prédire quand on a accumulé assez de blocs
#             if len(mfcc_buffer) >= batch_size:
#                 df_mfcc = pd.DataFrame(mfcc_buffer)
#                 df_mfcc_pca = df_mfcc @ eigenvectors_thresholded
#                 predictions = knn_model.predict(df_mfcc_pca)

#                 mfcc_features.extend(df_mfcc.values.tolist())  # decommanter pour correction
#                 time_values.extend([start / fs] * len(predictions)) # Decommenter pour correction

#                 label_mapping = {"l": 0, "a": 1, "s": 2, "t": 3, "r": 4}
#                 # predictions_indices = np.array([label_mapping[p] for p in predictions]) # si on utilise la correction apres
#                 predictions_indices = predictions # si on utilise la correction avant 

#                 # On associe au VIDE les labels trop loin du centre
#                 df_mfcc_pca_np = df_mfcc_pca.values if hasattr(df_mfcc_pca, 'values') else df_mfcc_pca
                        
#                 # Maintenant on peut appeler np.bincount en toute sécurité
#                 majority_label = int(np.bincount(predictions_indices).argmax())

#                 # Remplacer toutes les prédictions par la classe majoritaire
#                 tab_pred.extend(predictions) # Decommenter pour l'affichage
#                 predictions[:] = majority_label
#                 # print(f"majority_label : {Label(majority_label).name.lower()}")

#                 etat, action = transition(etat, Label(majority_label).name.lower())
#                 if action == "ON_OFF":
#                     handle_event("ON")
#                     handle_event("OFF")
#                 elif action:
#                     handle_event(action)
    
#                 # Réinitialiser les buffers
#                 # mfcc_buffer = mfcc_buffer[-buffer_recouvrement:]
#                 mfcc_buffer = []

# except KeyboardInterrupt:
#     print("\nArrêt de l'enregistrement.")

    
# # Fermeture du flux et de PyAudio
# stream.stop_stream()
# stream.close()
# p.terminate()

# ##########################################################################################
# # AFFICHAGE COULEUR
# # Générer l'axe des temps
# time = np.linspace(0, len(frames) / fs, num=len(frames))

# # Dictionnaire de correspondance entre les prédictions et les couleurs
# # colors = {"l": "blue", "a": "green", "s": "red", "t": "orange", "r": "purple"} # Si on utilise la correction apres
# colors = {0: "blue", 1: "green", 2: "red", 3: "orange", 4: "purple"} # Si on utilise la correction avant

# # Création des timestamps cohérents avec le nombre de prédictions
# pas = CHUNK * batch_size / fs  
# time_values = np.arange(len(tab_pred)) * pas

# # Création du data aFrame principal avec labels
# df_predictions = pd.DataFrame({
#     "Time (s)": time_values,
#     "Label": tab_pred
# })
# plt.figure(figsize=(10, 4))

# # Tracer le signal par segments en fonction des prédictions
# for i, (start, pred) in enumerate(zip(range(0, len(frames), CHUNK), tab_pred)):
#     plt.plot(time[start:start+CHUNK], frames[start:start+CHUNK], color=colors[pred])

# plt.xlabel("Temps (s)")
# plt.ylabel("Amplitude")
# plt.title("Signal Audio Capturé")
# plt.show()

# Enregistrer l'audio
wf = wave.open(filename, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(audio_frames))
wf.close()

# ########################################################################################
# # REAFFICHAGE PCA best comb
# # Calcul PCA 2D features
# df_loaded = pd.read_csv("scripts/X_proj_scaled_avec_labels_corrige_avant.csv") 
# # df_loaded = pd.read_csv("script/X_balanced_sans_r_opti_main_corrige.csv") 

# X_selected_pca = df_loaded.iloc[:, :-1].values

# # Mapping couleur → label (inversé du dico original)
# block_labels_balanced = df_loaded["label"].values

# # Couleurs
# colors = {"l": "blue", "a": "green", "s": "red", "t": "orange", "r": "purple"}
# class_colors = [colors[l] for l in block_labels_balanced]

# # Prédictions en temps réel
# predictions_array = np.array(tab_pred)
# pred_l_mask = predictions_array == "l"

# mfcc_matrix = np.array(mfcc_features)
# mfcc_pca_temp_reel = mfcc_matrix @ eigenvectors_thresholded
 
# ##########################################################################################
# # FIGURE 1 — AFFICHAGE DU SEUIL ET SUPPERPOSITION DE LA CLASSE 'l'
# # 1. Filtrer les points de la classe 'l' (entraînement)
# mask_l = [color == 'blue' for color in class_colors]  # 'blue' correspond à la classe 'l'
# X_l_pca = X_selected_pca[mask_l]

# # 2. Calculer centroïde + seuil
# center_l = np.mean(X_l_pca, axis=0)
# distances = np.linalg.norm(X_l_pca - center_l, axis=1)
# # threshold = 2 * np.std(distances)

# # 3. Masque prédictions 'l'
# pred_l_mask = predictions_array == "l"

# # === FIGURE FUSIONNÉE ===
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Nuage d'entraînement
# ax.scatter(
#     X_selected_pca[:, 0],
#     X_selected_pca[:, 1],
#     X_selected_pca[:, 2],
#     c=class_colors,
#     alpha=0.15,
#     s=10
# )

# # Prédictions 'l'
# ax.scatter(
#     mfcc_pca_temp_reel[pred_l_mask, 0],
#     mfcc_pca_temp_reel[pred_l_mask, 1],
#     mfcc_pca_temp_reel[pred_l_mask, 2],
#     c="deeppink",
#     edgecolors="k",
#     s=50,
#     label="Prédits 'l'"
# )

# # Sphères pour chaque centroïde
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# for label, center in centroids.items():
#     r0 = thresholds[label]["axis0"]
#     r1 = thresholds[label]["axis1"]
#     r2 = thresholds[label]["axis2"]
    
#     x = center[0] + r0 * np.cos(u) * np.sin(v)
#     y = center[1] + r1 * np.sin(u) * np.sin(v)
#     z = center[2] + r2 * np.cos(v)

#     ax.plot_surface(x, y, z, color='red', alpha=0.12)
#     ax.scatter(center[0], center[1], center[2], c='black', marker='x', s=100, label=f"Centroïde {label}")

# ax.set_xlabel("PCA 1")
# ax.set_ylabel("PCA 2")
# ax.set_zlabel("PCA 3")
# ax.set_title("Projection PCA 3D — Centroïdes + sphères seuil")

# plt.tight_layout()
# plt.show()

# ##########################################################################################
# # FIGURE 2 — PCA D'ENTRAÎNEMENT + PRÉDICTIONS TEMPS RÉEL SUPERPOSÉES

# # 1. S'assurer que les prédictions sont bien sous forme d'entiers
# predictions_array = np.array(tab_pred)

# # 2. Masque des prédictions 'l'
# pred_l_mask = predictions_array == Label.L.value  # donc == 0

# # 3. Normaliser les temps (uniquement si on a des points 'l')
# if np.any(pred_l_mask):
#     time_norm = (time_values - np.min(time_values)) / (np.max(time_values) - np.min(time_values))
#     cmap = matplotlib.colormaps.get_cmap("Blues")
#     colors_l = [f"rgba({int(255*r)},{int(255*g)},{int(255*b)},{a:.2f})"
#                 for r, g, b, a in cmap(time_norm[pred_l_mask])]
# else:
#     colors_l = []

# fig3d = go.Figure()

# # 4. Nuage d'entraînement (PCA de base)
# fig3d.add_trace(go.Scatter3d(
#     x=X_selected_pca[:, 0],
#     y=X_selected_pca[:, 1],
#     z=X_selected_pca[:, 2],
#     mode='markers',
#     marker=dict(size=3, color=class_colors, opacity=0.12),
#     name="Base PCA"
# ))

# # 5. Prédictions temps réel autres que 'l'
# for int_label, color in {1: "green", 2: "red", 3: "orange", 4: "purple"}.items():
#     mask = predictions_array == int_label
#     fig3d.add_trace(go.Scatter3d(
#         x=mfcc_pca_temp_reel[mask, 0],
#         y=mfcc_pca_temp_reel[mask, 1],
#         z=mfcc_pca_temp_reel[mask, 2],
#         mode='markers',
#         marker=dict(size=6, color=color, line=dict(color='black', width=0.5)),
#         name=f"Prédits '{Label(int_label).name.lower()}'",
#         text=[f"t = {t:.2f}s" for t in (np.array(time_values)[mask] * 0.1)],
#         hoverinfo='text'
#     ))

# # 6. Prédictions 'l' avec dégradé de bleu
# if np.any(pred_l_mask):
#     fig3d.add_trace(go.Scatter3d(
#         x=mfcc_pca_temp_reel[pred_l_mask, 0],
#         y=mfcc_pca_temp_reel[pred_l_mask, 1],
#         z=mfcc_pca_temp_reel[pred_l_mask, 2],
#         mode='markers',
#         marker=dict(
#             size=6,
#             color=colors_l,
#             line=dict(color='black', width=0.5)
#         ),
#         name="Prédits 'l'",
#         text=[f"t = {t:.2f}s" for t in (np.array(time_values)[pred_l_mask] * 0.1)],
#         hoverinfo='text'
#     ))

# # 7. Sphères 3D + centroïdes
# u = np.linspace(0, 2*np.pi, 30)
# v = np.linspace(0, np.pi, 30)
# u, v = np.meshgrid(u, v)

# for label, center in centroids.items():
#     r0 = thresholds[label]["axis0"]
#     r1 = thresholds[label]["axis1"]
#     r2 = thresholds[label]["axis2"]

#     xs = center[0] + r0 * np.cos(u) * np.sin(v)
#     ys = center[1] + r1 * np.sin(u) * np.sin(v)
#     zs = center[2] + r2 * np.cos(v)

#     color_label = label.lower()
#     fig3d.add_trace(go.Surface(
#         x=xs, y=ys, z=zs,
#         opacity=0.12,
#         colorscale=[[0, colors[color_label]], [1, colors[color_label]]],
#         showscale=False,
#         name=f"Sphère seuil '{label}'"
#     ))

#     fig3d.add_trace(go.Scatter3d(
#         x=[center[0]],
#         y=[center[1]],
#         z=[center[2]],
#         mode="markers+text",
#         marker=dict(size=10, color="black", symbol="x"),
#         text=[f"Centroïde '{label}'"],
#         textposition="top center",
#         name=f"Centroïde '{label}'"
#     ))

# # 8. Layout final
# fig3d.update_layout(
#     title="PCA 3D — Prédictions temps réel + sphères seuils et centroïdes",
#     scene=dict(
#         xaxis=dict(title="PCA 1"),
#         yaxis=dict(title="PCA 2"),
#         zaxis=dict(title="PCA 3"),
#         aspectmode="cube"
#     ),
#     legend_title="Légende",
#     margin=dict(l=0, r=0, b=0, t=50)
# )

# fig3d.show()