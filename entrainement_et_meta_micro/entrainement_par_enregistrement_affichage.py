import tkinter as tk
from tkinter import filedialog
import pyaudio
import wave
import yaml
import os
import time
import threading
from tkinter import font, ttk

##########################################################################################
# CHARGER YAML
yaml_path = "parametre.yaml"
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

# Chemin de sortie
output_path = config["calcul_mfcc"]["file_path_audio_non_concat"]

##########################################################################################
# PARAMÈTRES AUDIO
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = int(RATE * 0.005)

##########################################################################################
# FONCTIONS
def log(message):
    text_log.insert(tk.END, message + "\n")
    text_log.see(tk.END)
    root.update()

# Fonction pour sauvegarder le YAML
def save_yaml():
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, sort_keys=False, allow_unicode=True)

# Fonction pour démarrer l'enregistrement
def start_recording():
    threading.Thread(target=record_audio).start()

# Fonction pour enregistrer l'audio
def record_audio():
    # Initialisation
    log("Initialisation de l’enregistrement…")
    record_button.config(state="disabled")
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

    # # Instructions pour l'utilisateur
    # log("Bonjour !")
    # time.sleep(4)
    # log("Je vais vous demander de prononcer 3 voyelles en soufflant.")
    # time.sleep(4)
    # log("Prononcez 'a', 'i' et 'ou' à tour de rôle quand vous verrez 'go' jusqu'à ce que je vous dise 'stop'.")
    # time.sleep(5)
    # log("Vous êtes prêt ?")
    # time.sleep(2)

    # Enregistrement en continu avec consignes
    letters = config["calcul_mfcc"]["letters"] * 3
    for label in letters:
        # Enregistrement de chaque voyelle
        log(f"Prononcez : {label}")
        countdown(3)
        log("Enregistrement en cours…")

        duration = 5  # secondes
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_frames.append(data)
            progress_var.set((i + 1) / (RATE / CHUNK * duration) * 100)
            root.update_idletasks()

        log("Pause.")
        time.sleep(2)

    # Fin de l’enregistrement
    stream.stop_stream()
    stream.close()
    p.terminate()
    log("Enregistrement terminé.")
    ask_save_file(audio_frames, p, output_path)

# Fonction pour le compte à rebours
def countdown(seconds):
    for i in range(seconds, 0, -1):
        log(f"{i}…")
        time.sleep(1)

# Fonction pour demander où sauvegarder le fichier audio
def ask_save_file(audio_frames, p, default_path):
    # Boîte de dialogue pour choisir le fichier
    file_path = filedialog.asksaveasfilename(
        defaultextension=".wav" if default_path.endswith(".wav") else ".pkl",
        filetypes=[("Fichiers WAV", "*.wav"), ("Fichiers Joblib", "*.pkl")],
        initialfile=os.path.basename(default_path),
        title="Sauvegarder le fichier sous…"
    )
    if not file_path:
        log("Sauvegarde annulée.")
        record_button.config(state="normal")
        return

    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    # Ajout au YAML
    config["calcul_mfcc"]["file_path_audio_non_concat"] = file_path
    save_yaml()
    log(f"YAML mis à jour : {file_path}")

    # Sauvegarde du fichier
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(audio_frames))
    log(f"Fichier audio sauvegardé : {file_path}")

    record_button.config(state="normal")

##########################################################################################
# INTERFACE
root = tk.Tk()
root.title("Enregistreur Vocal")
root.configure(bg="#2c3e50")

title_font = font.Font(family="Arial", size=20, weight="bold")
button_font = font.Font(family="Arial", size=14)
log_font = font.Font(family="Courier", size=11)

# Titre
title_label = tk.Label(
    root,
    text="Enregistrement Audio des Voyelles",
    font=title_font,
    bg="#2c3e50",
    fg="#ecf0f1",
    pady=10
)
title_label.pack()

# Cadre principal
frame = tk.Frame(root, padx=15, pady=15, bg="#34495e", bd=2, relief="groove")
frame.pack(pady=10)

# Bouton enregistrer
record_button = tk.Button(
    frame,
    text="Démarrer l’enregistrement",
    command=start_recording,
    font=button_font,
    bg="#6A4878",
    fg="black",
    activebackground="#8e44ad",
    activeforeground="black",
    bd=0,
    padx=10,
    pady=5
)
record_button.pack(fill="x", pady=10)

# Barre de progression stylée
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, length=300, mode="determinate")
progress_bar.pack(pady=10)

# Zone de log
text_log = tk.Text(
    root,
    height=10,
    width=60,
    bg="#1e272e",
    fg="#b086c0",
    insertbackground="black",
    font=log_font,
    bd=2,
    relief="sunken"
)
text_log.pack(pady=10)

log("Prêt pour l’enregistrement.")

# Lancement de l'interface
root.mainloop()
