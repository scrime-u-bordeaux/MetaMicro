import tkinter as tk
from tkinter import filedialog, messagebox
import yaml
import os
import subprocess
from tkinter import font
import mido

##########################################################################################
# CHARGER YAML
yaml_path = "linux/entrainement_et_meta_micro/parametre.yaml"
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

##########################################################################################
# DESCRIPTIONS DES PARAMETRES
param_descriptions = {
    "rms_max": "Valeur maximale du RMS utilisée pour normaliser le volume. A modifier à la main ou avec l'option 'calculer souffle maximum'",
    "port_name_respiro": "Nom du port MIDI pour Respiro.",
    "port_name_midifile": "Nom du port MIDI pour le fichier MIDI.",
    "batch_size": "Nombre de blocs de taille 'time_of_block' à accumuler avant de faire la prédictions. A augmenter si il y a trop d'erreur, à baisser sur l'outil est trop lent",
    "recouvrement": "Chevauchement entre deux batchs consécutifs. A augmenter si il y a trop d'erreur, à baisser sur l'outil est trop lent",
    "radius": "Distance maximale pour trouver les voisins en classification.",
    "k": "Nombre de voisins pris en compte pour le KNN.",
    "midi_file": "Fichier MIDI à jouer si MidfilePerformer n'est pas utiliser.",
    "enregistrement_audio": "Chemin à utiliser pour enregistrer l'audio.",
    "other_params.vide_if_n_label.value": "Active la création de label vide si il y a plus de n lettres différentes en résulats de classification.",
    "other_params.vide_if_n_label.n": "Nombre maximal de lettres différentes acceptées.",
    "other_params.remplacer_t_par_i.value": "Active la correction des 't' mal détectés en 'i'.",
    "other_params.remplacer_t_par_i.n": "Nombre d’occurrences nécessaires pour appliquer la correction."
}

##########################################################################################
# FONCTIONS UTILITAIRES
def log(message):
    text_log.insert(tk.END, message + "\n")
    text_log.see(tk.END)
    root.update()

def save_yaml():
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, sort_keys=False, allow_unicode=True)
    log(f"YAML mis à jour : {yaml_path}")

def update_paths_in_yaml(base_dir):
    def update_path(path):
        if path is None or not isinstance(path, str):
            return path
        filename = os.path.basename(path)
        return os.path.join(base_dir, filename)

    # Mettre à jour les chemins
    for section in ["correction_avant_classification", "classification"]:
        for key in config[section]:
            if isinstance(config[section][key], str):
                config[section][key] = update_path(config[section][key])
            elif isinstance(config[section][key], dict):
                for subkey in config[section][key]:
                    if isinstance(config[section][key][subkey], str):
                        config[section][key][subkey] = update_path(config[section][key][subkey])

    save_yaml()

def choose_directory():
    folder_selected = filedialog.askdirectory(title="Choisir le dossier d’entraînement")
    if folder_selected:
        log(f"Dossier sélectionné : {folder_selected}")
        update_paths_in_yaml(folder_selected)
        log("Tous les chemins ont été mis à jour.")
    else:
        log("Aucun dossier sélectionné.")

def show_description(param_name):
    description = param_descriptions.get(param_name, "Aucune description disponible.")
    messagebox.showinfo("Explication du paramètre", description)

def create_info_button(parent, param_name):
    return tk.Button(
        parent,
        text="info",
        command=lambda: show_description(param_name),
        font=("Arial", 10, "bold"),
        bg="#3498db",
        fg="black",
        activebackground="#2980b9",
        activeforeground="black",
        bd=1,
        relief="ridge",
        width=2,
        height=1
    )

##########################################################################################
# FONCTION POUR LANCER LES SCRIPTS
def run_script(script_name):
    log(f"Lancement de {script_name} ...")
    try:
        subprocess.run(["python3", script_name], check=True)
        log(f"{script_name} terminé avec succès.")
    except subprocess.CalledProcessError as e:
        log(f"Erreur lors de l’exécution de {script_name} : {e}")

##########################################################################################
# FENÊTRE D’EDITION DES PARAMETRES
def open_param_window():
    param_window = tk.Toplevel(root)
    param_window.title("Modifier les paramètres main_respiro")
    param_window.configure(bg="#34495e")
    param_window.geometry("800x700")

    entries = {}
    midi_vars = {}

    title_label = tk.Label(
        param_window,
        text="Modifier les paramètres main_respiro",
        font=button_font,
        bg="#34495e",
        fg="#ecf0f1",
        pady=10
    )
    title_label.pack()

    param_frame = tk.Frame(param_window, padx=15, pady=15, bg="#34495e")
    param_frame.pack(pady=10, fill="both", expand=True)

    # Liste des ports MIDI disponibles
    try:
        midi_ports = mido.get_output_names()
        if not midi_ports:
            midi_ports = ["Aucun port détecté"]
    except Exception as e:
        midi_ports = ["Erreur détection MIDI"]
        log(f"Erreur lors de la détection des ports MIDI : {e}")

    def choose_file_and_update(key):
        file_path = filedialog.askopenfilename(title=f"Sélectionner un fichier pour {key}")
        if file_path:
            config["main_respiro"][key] = file_path
            save_yaml()
            log(f"{key} mis à jour : {file_path}")

    # Créer un champ pour chaque paramètre
    for param, value in config["main_respiro"].items():
        if isinstance(value, dict):
            for subparam, subvalue in value.items():
                if isinstance(subvalue, dict):
                    for subsubparam, subsubvalue in subvalue.items():
                        row = tk.Frame(param_frame, bg="#34495e")
                        row.pack(fill="x", pady=2)

                        label = tk.Label(row, text=f"{param}.{subparam}.{subsubparam}", width=30, anchor="w", bg="#34495e", fg="white")
                        label.pack(side="left")

                        info_button = create_info_button(row, f"{param}.{subparam}.{subsubparam}")
                        info_button.pack(side="left", padx=5)

                        entry = tk.Entry(row, width=40)
                        entry.insert(0, str(subsubvalue))
                        entry.pack(side="right", expand=True, fill="x")
                        entries[f"{param}.{subparam}.{subsubparam}"] = entry
                else:
                    row = tk.Frame(param_frame, bg="#34495e")
                    row.pack(fill="x", pady=2)

                    label = tk.Label(row, text=f"{param}.{subparam}", width=30, anchor="w", bg="#34495e", fg="white")
                    label.pack(side="left")

                    info_button = create_info_button(row, f"{param}.{subparam}")
                    info_button.pack(side="left", padx=5)

                    entry = tk.Entry(row, width=40)
                    entry.insert(0, str(subvalue))
                    entry.pack(side="right", expand=True, fill="x")
                    entries[f"{param}.{subparam}"] = entry
        else:
            row = tk.Frame(param_frame, bg="#34495e")
            row.pack(fill="x", pady=2)

            label = tk.Label(row, text=param, width=30, anchor="w", bg="#34495e", fg="white")
            label.pack(side="left")

            info_button = create_info_button(row, param)
            info_button.pack(side="left", padx=5)

            if param in ["midi_file", "enregistrement_audio"]:
                file_button = tk.Button(
                    row,
                    text="Choisir un fichier",
                    command=lambda p=param: choose_file_and_update(p),
                    font=("Arial", 10),
                    bg="#2980b9",
                    fg="black",
                    activebackground="#3498db",
                    activeforeground="black",
                    bd=0,
                    padx=5,
                    pady=5
                )
                file_button.pack(side="right")
            elif param in ["port_name_respiro", "port_name_midifile"]:
                midi_var = tk.StringVar()
                midi_var.set(value if value in midi_ports else midi_ports[0])
                dropdown = tk.OptionMenu(row, midi_var, *midi_ports)
                dropdown.config(font=("Arial", 10), bg="#34495e", fg="black", width=40)
                dropdown.pack(side="right", fill="x", expand=True)
                midi_vars[param] = midi_var
            else:
                entry = tk.Entry(row, width=50)
                entry.insert(0, str(value))
                entry.pack(side="right", expand=True, fill="x")
                entries[param] = entry

    # Sauvegarder les changements
    def save_main_respiro_changes():
        for key, entry in entries.items():
            value = entry.get()
            value_lower = value.strip().lower()
            if value_lower == "true":
                value = True
            elif value_lower == "false":
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

            parts = key.split(".")
            ref = config["main_respiro"]
            for part in parts[:-1]:
                ref = ref[part]
            ref[parts[-1]] = value

        for key, var in midi_vars.items():
            config["main_respiro"][key] = var.get()
        save_yaml()
        log("Paramètres main_respiro mis à jour.")
        messagebox.showinfo("Sauvegarde", "Les paramètres ont été enregistrés.")
        param_window.destroy()

    button_frame = tk.Frame(param_window, bg="#34495e")
    button_frame.pack(pady=10)

    save_button = tk.Button(
        button_frame,
        text="Enregistrer les modifications",
        command=save_main_respiro_changes,
        font=button_font,
        bg="#27ae60",
        fg="black",
        activebackground="#2ecc71",
        activeforeground="black",
        bd=0,
        padx=15,
        pady=5
    )
    save_button.pack()

##########################################################################################
# INTERFACE PRINCIPALE
    
root = tk.Tk()
root.title("Méta-Micro")
root.configure(bg="#2c3e50")
root.geometry("600x700")

# Polices
title_font = font.Font(family="Arial", size=20, weight="bold")
button_font = font.Font(family="Arial", size=14)
log_font = font.Font(family="Courier", size=11)

# Titre
title_label = tk.Label(
    root,
    text="Méta-Micro",
    font=title_font,
    bg="#2c3e50",
    fg="#ecf0f1",
    pady=10
)
title_label.pack()

# Cadre principal
frame = tk.Frame(root, padx=15, pady=15, bg="#34495e", bd=2, relief="groove")
frame.pack(pady=10)

# Bouton choisir dossier
select_button = tk.Button(
    frame,
    text="Choisir le dossier d’entraînement",
    command=choose_directory,
    font=button_font,
    bg="#6A4878",
    fg="black",
    activebackground="#8e44ad",
    activeforeground="black",
    bd=0,
    padx=10,
    pady=5
)
select_button.pack(fill="x", pady=10)

# Bouton ouvrir paramètres
param_button = tk.Button(
    frame,
    text="Modifier les paramètres",
    command=open_param_window,
    font=button_font,
    bg="#f39c12",
    fg="black",
    activebackground="#f1c40f",
    activeforeground="black",
    bd=0,
    padx=10,
    pady=5
)
param_button.pack(fill="x", pady=10)

##########################################################################################
# BOUTONS POUR LANCER LES SCRIPTS
script_frame = tk.Frame(root, padx=15, pady=15, bg="#34495e", bd=2, relief="groove")
script_frame.pack(pady=10)

script_title = tk.Label(
    script_frame,
    text="Lancer les scripts",
    font=button_font,
    bg="#34495e",
    fg="#ecf0f1",
    pady=5
)
script_title.pack()

btn_rms = tk.Button(
    script_frame,
    text="Jouer avec l'accordeur de voyelle",
    command=lambda: run_script("linux/entrainement_et_meta_micro/accordage_de_voyelle.py"),
    font=button_font,
    bg="#2980b9",
    fg="black",
    activebackground="#3498db",
    activeforeground="black",
    bd=0,
    padx=10,
    pady=5
)
btn_rms.pack(fill="x", pady=5)

btn_accordeur = tk.Button(
    script_frame,
    text="Jouer sans accordeur avec midifile",
    command=lambda: run_script("linux/entrainement_et_meta_micro/calcul_mfcc_midifile.py"),
    font=button_font,
    bg="#8e44ad",
    fg="black",
    activebackground="#9b59b6",
    activeforeground="black",
    bd=0,
    padx=10,
    pady=5
)
btn_accordeur.pack(fill="x", pady=5)

btn_respiro = tk.Button(
    script_frame,
    text="Jouer sans accordage",
    command=lambda: run_script("linux/entrainement_et_meta_micro/calcul_mfcc.py"),
    font=button_font,
    bg="#16a085",
    fg="black",
    activebackground="#1abc9c",
    activeforeground="black",
    bd=0,
    padx=10,
    pady=5
)
btn_respiro.pack(fill="x", pady=5)

##########################################################################################
# ZONE LOG
text_log = tk.Text(
    root,
    height=10,
    width=70,
    bg="#1e272e",
    fg="#b086c0",
    insertbackground="white",
    font=log_font,
    bd=2,
    relief="sunken"
)
text_log.pack(pady=10)

log("Prêt à mettre à jour les chemins YAML.")

root.mainloop()