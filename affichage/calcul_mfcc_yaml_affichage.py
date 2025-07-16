import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import librosa
import numpy as np
import yaml
from itertools import combinations
import os
import threading
from tkinter import font, ttk

# ----------- CHARGER YAML -----------
yaml_path = "parametre.yaml"
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

letters = config["calcul_mfcc"]["letters"]
file_path_txt = config["calcul_mfcc"]["file_path_txt"]
file_path_audio = config["calcul_mfcc"]["file_path_audio"]
time_of_block = config["calcul_mfcc"]["time_of_block"]
recouvrement_block = config["calcul_mfcc"]["recouvrement_block"]

mfcc_params = config["calcul_mfcc"]["pamatres_mfcc"]
n_mfcc = mfcc_params["n_mfcc"]
n_fft = mfcc_params["n_fft"]
n_mels = mfcc_params["n_mels"]

other_params = config["calcul_mfcc"]["other_params"]
use_delta = other_params.get("delta_mfcc", False)
use_zcr = other_params.get("zero_crossing", False)
use_centroid = other_params.get("centroid", False)
use_slope = other_params.get("slope", False)

output_path = config["calcul_mfcc"]["output_path"]

# ----------- FONCTIONS UTILITAIRES -----------
def log(message):
    text_log.insert(tk.END, message + "\n")
    text_log.see(tk.END)
    root.update()

def save_yaml():
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, sort_keys=False, allow_unicode=True)

def check_overwrite_or_rename(filepath: str) -> str:
    while os.path.exists(filepath):
        response = messagebox.askyesnocancel("Fichier existant",
            f"Le fichier '{filepath}' existe déjà.\nVoulez-vous l'écraser ?")
        if response is None:
            return None  # Annuler
        elif response:
            return filepath
        else:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("Fichiers CSV", "*.csv")],
                title="Sauvegarder le fichier sous…"
            )
            if not filepath:
                return None
    return filepath

# ----------- CALCUL MFCC -----------
def start_mfcc():
    threading.Thread(target=compute_mfcc).start()

def compute_mfcc():
    try:
        log("Chargement des fichiers…")
        markers_df = pd.read_csv(file_path_txt, sep="\t", header=None, names=["start", "end", "label"])
        signal, fs = librosa.load(file_path_audio, sr=None)

        log("Calcul des caractéristiques…")
        block_size = int(time_of_block * fs)
        recouvrement = int(block_size * recouvrement_block)
        features_per_block = []
        block_labels = []
        block_start_times = []
        letters_with_st = letters + [l for l in ["s", "t"] if l not in letters]
        excluded_pairs = [set(pair) for pair in combinations(letters_with_st, 2)]

        total_blocks = len(range(0, len(signal), recouvrement))
        for i, start in enumerate(range(0, len(signal), recouvrement)):
            block = signal[start:start+block_size]
            block_start_time = start / fs  
            block_end_time = (start + block_size) / fs 

            # Trouver la classe dominante   
            segment_durations = {label: 0 for label in letters_with_st}
            for _, row in markers_df.iterrows():
                segment_start, segment_end, segment_label = row["start"], row["end"], row["label"]
                if segment_label not in letters_with_st:
                    continue
                if segment_end > block_start_time and segment_start < block_end_time:
                    overlap_start = max(block_start_time, segment_start)
                    overlap_end = min(block_end_time, segment_end)
                    segment_durations[segment_label] += (overlap_end - overlap_start)

            active_segments = [k for k, v in segment_durations.items() if v > 0]

            # Cas 1 — Aucun segment actif
            if len(active_segments) == 0:
                block_label = "s"
            # Cas 2 — Deux lettres actives exclues
            elif len(active_segments) == 2 and set(active_segments) in excluded_pairs:
                continue
            # Cas 3 — Une seule lettre active
            elif len(active_segments) == 1:
                block_label = active_segments[0]

            # Calcul MFCC
            n_fft_local = min(n_fft, block_size)
            mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft_local, fmax=fs/2, n_mels=n_mels)

            if len(block) == block_size:
                mfcc = librosa.feature.mfcc(y=block.astype(float), sr=fs, n_mfcc=n_mfcc,
                                            n_fft=n_fft_local, win_length=n_fft_local, hop_length=n_fft_local // 10,
                                            fmax=fs/2, mel_basis=mel_basis)
                features = [mfcc.flatten()]

                if use_delta:
                    delta = librosa.feature.delta(mfcc, order=1)
                    features.append(delta.flatten())
                if use_slope:
                    slope = np.diff(block)
                    slope_rms = np.sqrt(np.mean(slope ** 2))
                    features.append(np.array([slope_rms]))
                if use_zcr:
                    zcr = np.mean(librosa.feature.zero_crossing_rate(block, frame_length=len(block), hop_length=len(block)))
                    features.append(np.array([zcr]))
                if use_centroid:
                    centroid = np.mean(librosa.feature.spectral_centroid(y=block, sr=fs))
                    features.append(np.array([centroid]))

                feature_vector = np.concatenate(features)
                features_per_block.append(feature_vector)
                block_labels.append(block_label)
                block_start_times.append(block_start_time)

            progress_var.set((i+1)/total_blocks*100)
            root.update_idletasks()

        log("Calcul terminé.")
        save_results(features_per_block, block_labels, block_start_times)
    except Exception as e:
        log(f"Erreur : {e}")
        messagebox.showerror("Erreur", str(e))

def save_results(features_per_block, block_labels, block_start_times):
    mfcc_matrix = np.array(features_per_block)
    df_mfcc = pd.DataFrame(mfcc_matrix)
    df_mfcc["start_time"] = block_start_times
    df_mfcc["label"] = block_labels

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("Fichiers CSV", "*.csv")],
        title="Sauvegarder les MFCC sous…"
    )
    if not file_path:
        log("Sauvegarde annulée.")
        return

    config["calcul_mfcc"]["output_path"] = file_path
    save_yaml()
    log(f"YAML mis à jour : {file_path}")

    try:
        df_mfcc.to_csv(file_path, index=False)
        log(f"Fichier MFCC sauvegardé : {file_path}")
    except Exception as e:
        log(f"Erreur lors de la sauvegarde : {e}")

# ----------- INTERFACE -----------

root = tk.Tk()
root.title("Calcul MFCC Audio")
root.configure(bg="#2c3e50")

title_font = font.Font(family="Arial", size=20, weight="bold")
button_font = font.Font(family="Arial", size=14)
log_font = font.Font(family="Courier", size=11)

title_label = tk.Label(
    root,
    text="Extraction des MFCC et caractéristiques audio",
    font=title_font,
    bg="#2c3e50",
    fg="#ecf0f1",
    pady=10
)
title_label.pack()

frame = tk.Frame(root, padx=15, pady=15, bg="#34495e", bd=2, relief="groove")
frame.pack(pady=10)

start_button = tk.Button(
    frame,
    text="Lancer le calcul MFCC",
    command=start_mfcc,
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

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, length=300, mode="determinate")
progress_bar.pack(pady=10)

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

log("Prêt pour le calcul.")

root.mainloop()
