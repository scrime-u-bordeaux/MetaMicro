import pyaudio
import wave
import os
import time
import yaml

##########################################################################################
# CHARGEMENTS
# Fichier yaml a compléter
yaml_path = "parametre.yaml"

with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

# Fonction d'assistance
def ensure_path(section, key, path):
    section[key] = path
    return section[key]

##########################################################################################
# VERIDICATION DE L'EXISTANCE DES FICHIERS
def check_overwrite_or_rename(filepath: str) -> str:
    while os.path.exists(filepath):
        response = input(f"Le fichier '{filepath}' existe déjà. Voulez-vous l'écraser ? (o/n) : ").strip().lower()
        if response in ["o", "oui", "y", "yes"]:
            return filepath
        elif response in ["n", "non", "no"]:
            new_path = input("Entrez un nouveau nom de fichier (ou appuyez sur Entrée pour annuler) : ").strip()
            if new_path == "":
                print("Écriture annulée.")
                return None
            filepath = new_path
        else:
            print("Réponse non reconnue. Répondez par 'o' ou 'n'.")
    return filepath

##########################################################################################
# ENREGISTREMENTS
# Paramètres audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = int(RATE * 0.005)

# Instructions pour l'utilisateur
print("Bonjour !")
time.sleep(4)
print("Je vais vous demander de prononcer 3 voyelles en soufflant.")
time.sleep(4)
print("Prononcez 'a', 'i' et 'ou' à tour de rôle quand vous verrez 'go' jusqu'à ce que je vous dise 'stop'.")
time.sleep(5)
print("Vous êtes prêt ?")
time.sleep(2)

# Initialisation de PyAudio
p = pyaudio.PyAudio()

# Ouverture du flux
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)
audio_frames = []

# Enregistrement en continu avec consignes
letters = config["calcul_mfcc"]["letters"] * 3
for label in letters:
    print(f"Fais un '{label}' jusqu'au stop dans :")
    time.sleep(1)
    for n in [3, 2, 1]:
        print(n)
        time.sleep(1)
    print("go")

    duration = 9  # secondes d'enregistrement pour cette voyelle
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frames.append(data)

    print("stop")
    time.sleep(2)

print("Merci !")
print("Enregistrement terminé.")

# Fermeture
stream.stop_stream()
stream.close()
p.terminate()

##########################################################################################
# ENREGISTREMENT DU FICHIER
# Demande du nom de fichier
while True:
    filename = check_overwrite_or_rename(
        input("Où voulez-vous enregistrer le fichier ? (ex: script/mon_fichier.wav) : ").strip()
    )

    if filename:
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            try:
                os.makedirs(folder)
                print(f"Dossier '{folder}' créé.")
            except Exception as e:
                print(f"Impossible de créer le dossier : {e}")
                continue
        break  # On sort de la boucle si le chemin est valide
    else:
        print("Aucune sauvegarde effectuée.")

# Ajout du fichier wav dans le fichier de configuration YAML
calcul = config["calcul_mfcc"]
ensure_path(calcul, "file_path_audio_non_concat", filename)

# Réécriture du fichier YAML avec les chemins complétés
with open(yaml_path, "w") as file:
    yaml.dump(config, file, sort_keys=False, allow_unicode=True)

# Sauvegarde du fichier WAV
with wave.open(filename, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(audio_frames))

print(f"Fichier audio sauvegardé sous : {filename}")
