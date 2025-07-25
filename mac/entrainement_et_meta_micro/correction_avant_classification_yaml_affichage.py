import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import yaml
import os
import threading
from tkinter import font, ttk

##########################################################################################
# CHARGER YAML
yaml_path = "mac/entrainement_et_meta_micro/parametre.yaml"
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

input_path = config["calcul_mfcc"]["output_path"]
letters = config["calcul_mfcc"]["letters"]
seuil_de_correction = config["correction_avant_classification"]["seuil_de_correction"]
output_path = config["correction_avant_classification"]["output_path"]

##########################################################################################
# FONCTIONS UTILITAIRES
def log(message):
    text_log.insert(tk.END, message + "\n")
    text_log.see(tk.END)
    root.update()

# Fonction pour sauvegarder le YAML
def save_yaml():
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, sort_keys=False, allow_unicode=True)

# Fonction pour demander où sauvegarder le fichier
def ask_save_file(default_path):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv" if default_path.endswith(".csv") else ".pkl",
        filetypes=[
            ("Fichiers CSV", "*.csv"),
            ("Fichiers Joblib", "*.pkl"),
            ("Tous les fichiers", "*.*")
        ],
        initialfile=os.path.basename(default_path),
        initialdir=os.path.dirname(default_path),
        title="Sauvegarder le fichier sous…"
    )
    if file_path:
        launch_dir = os.getcwd()  # Répertoire d'où le script est lancé
        rel_path = os.path.relpath(file_path, start=launch_dir)
        return rel_path
    return None

##########################################################################################
# CORRECTION DES DONNÉES
# Fonction pour faire la correction
def start_correction():
    threading.Thread(target=correct_data).start()

def correct_data():
    try:
        log("Chargement des données…")
        df_loaded = pd.read_csv(input_path)
        X = df_loaded.iloc[:, :-1].values
        block_labels_balanced = df_loaded["label"].values

        # Couleurs
        color_palette = ["green", "red", "orange", "purple", "cyan", "brown", "blue", "olive", "pink", "gray"]
        letters_with_st = letters + [l for l in ["s", "t"] if l not in letters]
        colors = {label: color_palette[i % len(color_palette)] for i, label in enumerate(letters_with_st)}

        # Correction
        log("Correction des classes…")
        df_pca = pd.DataFrame(X)
        df_pca["label"] = block_labels_balanced
        filtered_points = []

        feature_cols = df_pca.columns[:-1]
        seuil = seuil_var.get()
        total_classes = len(df_pca["label"].unique())

        for idx, label in enumerate(df_pca["label"].unique()):

            # Calculer les distances au centres des classes
            class_points = df_pca[df_pca["label"] == label]
            centroid = class_points[feature_cols].mean()
            distances = np.linalg.norm(class_points[feature_cols].values - centroid.values, axis=1)

            # Filtrer les points au dessus du seuil choisi
            threshold = seuil * distances.mean()
            class_points_filtered = class_points[distances < threshold]
            filtered_points.append(class_points_filtered)

            # Afficher une barre de progression
            progress_var.set((idx + 1) / total_classes * 100)
            root.update_idletasks()

        df_filtered = pd.concat(filtered_points)

        # Sauvegarde
        save_path = ask_save_file(output_path)
        if save_path:
            df_filtered.to_csv(save_path, index=False)
            config["correction_avant_classification"]["output_path"] = save_path
            save_yaml()
            log(f"Fichier corrigé sauvegardé : {save_path}")
        else:
            log("Sauvegarde annulée.")

    except Exception as e:
        log(f"Erreur : {e}")
        messagebox.showerror("Erreur", str(e))

def update_seuil_label(value):
    seuil_value_label.config(text=f"Seuil : {float(value):.2f}")

##########################################################################################
# INTERFACE
root = tk.Tk()
root.title("Correction des données avant classification")
root.configure(bg="#2c3e50")

# Polices
title_font = font.Font(family="Arial", size=20, weight="bold")
button_font = font.Font(family="Arial", size=14)
log_font = font.Font(family="Courier", size=11)

# Titre
title_label = tk.Label(
    root,
    text="Correction N-dimensionnelle des données",
    font=title_font,
    bg="#2c3e50",
    fg="#ecf0f1",
    pady=10
)
title_label.pack()

# Cadre principal
frame = tk.Frame(root, padx=15, pady=15, bg="#34495e", bd=2, relief="groove")
frame.pack(pady=10)

# Slider pour le seuil de correction
seuil_var = tk.DoubleVar(value=seuil_de_correction)

# Label dynamique pour la valeur du seuil
seuil_value_label = tk.Label(frame, text=f"Seuil : {seuil_de_correction:.2f}", font=button_font, bg="#34495e", fg="black")
seuil_value_label.pack(pady=(0, 5))

seuil_slider = ttk.Scale(
    frame,
    from_=0.4, to=1.0,
    variable=seuil_var,
    orient="horizontal",
    length=300,
    command=update_seuil_label
)
seuil_slider.pack(pady=5)

# Bouton lancer
start_button = tk.Button(
    frame,
    text="Lancer la correction",
    command=start_correction,
    font=button_font,
    bg="#6A4878",
    fg="black",
    activebackground="#8e44ad",
    activeforeground="black",
    bd=0,
    padx=10,
    pady=5
)
start_button.pack(fill="x", pady=10)

# Barre de progression
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, length=300, mode="determinate")
progress_bar.pack(pady=10)

# Zone de log
text_log = tk.Text(
    root,
    height=10,
    width=70,
    bg="#1e272e",
    fg="#b086c0",
    insertbackground="black",
    font=log_font,
    bd=2,
    relief="sunken"
)
text_log.pack(pady=10)

log("Prêt pour la correction.")

# Lancement de l'interface
root.mainloop()
