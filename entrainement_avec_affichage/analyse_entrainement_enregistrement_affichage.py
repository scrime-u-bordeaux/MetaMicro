import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import yaml
import os
import wave
import threading
from tkinter import font, ttk

##########################################################################################
# CHARGER YAML
yaml_path = "parametre.yaml"
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

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

# Fonctions pour lancer l'analyse de l'audio dans un thread
def process_audio():
    threading.Thread(target=analyse_audio).start()

def analyse_audio():
    try:
        log("Chargement du fichier audio…")
        file_path = config["calcul_mfcc"]["file_path_audio_non_concat"]
        y, sr = librosa.load(file_path, sr=None)

        # Paramètres de fenêtrage
        frame_length = int(0.02 * sr)  # 20 ms
        hop_length = int(0.01 * sr)    # 10 ms

        # Calcul RMS pour détecter exctement les marqueurs, cad ceux qui sont au-dessus du seuil = mean_rms
        log("Calcul du RMS…")
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        mean_rms = np.mean(rms)

        # Fenêtres d'origine
        timestamps = [
            (4, 10), (15, 20), (24, 28), (33, 37),
            (42, 46), (50, 55), (60, 65), (69, 73), (78, None)
        ]
        letters = config["calcul_mfcc"]["letters"] * 3
        windows = [
            (start, end, letters[i]) for i, (start, end) in enumerate(timestamps)
            if i < len(letters)
        ]

        # Création des marqueurs
        log("Détection des marqueurs…")
        markers = []
        active = rms > mean_rms

        file_path_ta = config["calcul_mfcc"]["file_path_ta"]
        with wave.open(file_path_ta, "rb") as wav_file:
            offset = wav_file.getnframes() / wav_file.getframerate()

        for start_sec, end_sec, label in windows:
            mask = (times >= start_sec) & (times <= end_sec)
            active_in_window = mask & active
            if not np.any(active_in_window):
                continue
            indices = np.where(active_in_window)[0]
            groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
            for g in groups:
                t_start = float(times[g[0]]) + offset
                t_end = float(times[g[-1]]) + offset
                if t_end - t_start > 1.0:
                    markers.append((t_start, t_end, label))

        log(f"{len(markers)} marqueurs détectés.")

        # Demander où sauvegarder
        file_path_txt = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt")],
            title="Sauvegarder les marqueurs sous…"
        )
        if not file_path_txt:
            log("Sauvegarde annulée.")
            return

        # Sauvegarde YAML
        config["calcul_mfcc"]["file_path_txt_non_concat"] = file_path_txt
        save_yaml()
        log(f"Chemin ajouté au YAML : {file_path_txt}")

        # Sauvegarde des marqueurs
        with open(file_path_txt, "w") as f:
            for start, end, label in markers:
                f.write(f"{start:.3f}\t{end:.3f}\t{label}\n")

        log(f"Fichier {file_path_txt} sauvegardé.")
    except Exception as e:
        log(f"Erreur : {e}")

##########################################################################################
# INTERFACE
root = tk.Tk()
root.title("Détection des Marqueurs Audio")
root.configure(bg="#2c3e50")

title_font = font.Font(family="Arial", size=20, weight="bold")
button_font = font.Font(family="Arial", size=14)
log_font = font.Font(family="Courier", size=11)

# Titre
title_label = tk.Label(
    root,
    text="Analyse et Marqueurs Audio",
    font=title_font,
    bg="#2c3e50",
    fg="#ecf0f1",
    pady=10
)
title_label.pack()

# Cadre principal
frame = tk.Frame(root, padx=15, pady=15, bg="#34495e", bd=2, relief="groove")
frame.pack(pady=10)

# Bouton lancer
start_button = tk.Button(
    frame,
    text="Lancer l'analyse",
    command=process_audio,
    font=button_font,
    bg="#6A4878",
    fg="white",
    activebackground="#8e44ad",
    activeforeground="white",
    bd=0,
    padx=10,
    pady=5
)
start_button.pack(fill="x", pady=10)

# Zone de log
text_log = tk.Text(
    root,
    height=12,
    width=70,
    bg="#1e272e",
    fg="#b086c0",
    insertbackground="white",
    font=log_font,
    bd=2,
    relief="sunken"
)
text_log.pack(pady=10)

log("Prêt pour l’analyse.")

# Lancement de l'interface
root.mainloop()
