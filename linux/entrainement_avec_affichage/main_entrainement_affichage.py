import tkinter as tk
from tkinter import filedialog
import subprocess
import yaml
import os
import wave
from pydub import AudioSegment
from tkinter import font
import shutil

##########################################################################################
# FONCTIONS 
def log(message):
    text_log.insert(tk.END, message + "\n")
    text_log.see(tk.END)
    root.update()

# Fonction pour lancer les scripts Python
def run_script(script_name):
    log(f"-> Lancement de : {script_name}")
    result = subprocess.run(["python3", script_name])
    if result.returncode != 0:
        log(f"Le script {script_name} a échoué (code {result.returncode}).")
        show_error(f"{script_name} a échoué.")
        return False
    log(f"Script {script_name} terminé avec succès.\n")
    return True

# Fonction pour créer un chemin si manquant
def ensure_path(section, key, path):
    section[key] = path
    return section[key]

# Fonction pour le popup de choix des lettres
def show_input_popup(title, prompt):
    popup = tk.Toplevel(root)
    popup.title(title)
    popup.configure(bg="#2c3e50")
    popup.geometry("400x150")
    popup.grab_set()  # bloque la fenêtre principale

    label = tk.Label(popup, text=prompt, font=button_font, bg="#2c3e50", fg="white")
    label.pack(pady=10)

    entry = tk.Entry(popup, font=button_font, bg="#34495e", fg="white", insertbackground="white")
    entry.pack(pady=5, fill="x", padx=20)

    result = {}

    # Fonction pour valider l'entrée
    def on_ok():
        value = entry.get().strip()
        if value:
            result["value"] = value
            popup.destroy()
        else:
            show_warning("Vous devez entrer une valeur.")

    btn_ok = tk.Button(popup, text="Valider", font=button_font,
                        bg="#6A4878", fg="white",
                        activebackground="#8e44ad", activeforeground="white",
                        command=on_ok)
    btn_ok.pack(pady=10)

    popup.wait_window()
    return result.get("value")

# Fonction pour afficher un message d'avertissement
def show_warning(message):
    popup = tk.Toplevel(root)
    popup.title("Attention")
    popup.configure(bg="#2c3e50")
    popup.geometry("400x100")
    popup.grab_set()

    label = tk.Label(popup, text=message, font=button_font, bg="#2c3e50", fg="yellow")
    label.pack(pady=20)

    btn_ok = tk.Button(popup, text="OK", font=button_font,
                       bg="#6A4878", fg="white",
                       activebackground="#8e44ad", activeforeground="white",
                       command=popup.destroy)
    btn_ok.pack(pady=5)

    popup.wait_window()

# Fonction pour afficher un message d'erreur
def show_error(message):
    popup = tk.Toplevel(root)
    popup.title("Erreur")
    popup.configure(bg="#2c3e50")
    popup.geometry("400x100")
    popup.grab_set()

    label = tk.Label(popup, text=message, font=button_font, bg="#2c3e50", fg="red")
    label.pack(pady=20)

    btn_ok = tk.Button(popup, text="OK", font=button_font,
                       bg="#6A4878", fg="white",
                       activebackground="#8e44ad", activeforeground="white",
                       command=popup.destroy)
    btn_ok.pack(pady=5)

    popup.wait_window()

# Fonction Oui/Non pour le choix de l'enregistrement ou du fichier
def show_yes_no(title, message):
    popup = tk.Toplevel(root)
    popup.title(title)
    popup.configure(bg="#2c3e50")
    popup.geometry("400x150")
    popup.grab_set()

    label = tk.Label(popup, text=message, font=button_font, bg="#2c3e50", fg="white")
    label.pack(pady=20)

    result = {"choice": None}

    # Fonctions et boutons Oui et Non
    def on_yes():
        result["choice"] = True
        popup.destroy()

    def on_no():
        result["choice"] = False
        popup.destroy()

    btn_yes = tk.Button(popup, text="Oui", font=button_font,
                        bg="#27ae60", fg="white",
                        activebackground="#229954", activeforeground="white",
                        command=on_yes)
    btn_yes.pack(side="left", padx=30, pady=10, expand=True)

    btn_no = tk.Button(popup, text="Non", font=button_font,
                       bg="#c0392b", fg="white",
                       activebackground="#a93226", activeforeground="white",
                       command=on_no)
    btn_no.pack(side="right", padx=30, pady=10, expand=True)

    popup.wait_window()
    return result["choice"]

# Fonction pour selectrionner les lettres
def select_letters():
    global letters
    letters_str = show_input_popup("Choix des lettres", "Entrez les lettres (ex: a,i,u)")
    if letters_str:
        letters = letters_str.strip().split(",")
        log(f"Lettres choisies : {letters}")
        config["calcul_mfcc"]["letters"] = letters
        save_yaml()
    else:
        show_warning("Vous devez entrer au moins une lettre.")

# Fonction pour l'affichage du choix de l'enregistrement ou du fichier
def choose_audio_or_record():
    choice = show_yes_no("Choix audio", "Voulez-vous enregistrer un fichier audio ?")
    if choice:
        if run_script("entrainement_par_enregistrement_affichage.py"):
            reload_yaml()
        if run_script("analyse_entrainement_enregistrement_affichage.py"):
            reload_yaml()
    else:
        audio_path = filedialog.askopenfilename(title="Sélectionnez un fichier audio")
        txt_path = filedialog.askopenfilename(title="Sélectionnez le fichier texte des marqueurs")
        if audio_path and txt_path:
            config["calcul_mfcc"]["file_path_audio_non_concat"] = audio_path
            config["calcul_mfcc"]["file_path_txt_non_concat"] = txt_path
            save_yaml()
            log(f"Fichier audio : {audio_path}\nFichier texte : {txt_path}")

# Fonction pour concaténer les fichiers audio et texte avec l'entrainement de référence et le nouveau
def concatenate_files():
    log("Concaténation des fichiers audio et texte…")
    try:
        file_audio_ta = config["calcul_mfcc"]["file_path_ta"]
        file_audio_non_concat = config["calcul_mfcc"]["file_path_audio_non_concat"]
        file_text_ta = config["calcul_mfcc"]["file_path_ta_text"]
        file_text_non_concat = config["calcul_mfcc"]["file_path_txt_non_concat"]

        output_audio = "audio.wav"
        output_text = "text.txt"

        # Créer le fichier audio.wav si il n'existe pas
        log(f"Création du fichier audio : {output_audio}")

        audio1 = AudioSegment.from_file(file_audio_ta)
        audio2 = AudioSegment.from_file(file_audio_non_concat)

        # Vérifier la compatibilité des paramètres
        if (audio1.frame_rate != audio2.frame_rate or
            audio1.sample_width != audio2.sample_width or
            audio1.channels != audio2.channels):
            log("Conversion du second fichier audio pour correspondre au format du premier…")
            audio2 = audio2.set_frame_rate(audio1.frame_rate)\
                            .set_sample_width(audio1.sample_width)\
                            .set_channels(audio1.channels)

        # Concaténer les deux fichiers audio
        concatenated_audio = audio1 + audio2
        concatenated_audio.export(output_audio, format="wav")

        # Créer le fichier text.txt seulement s'il n'existe pas
        log(f"Création du fichier texte : {output_text}")

        # Lire et concaténer les fichiers texte
        with open(file_text_non_concat, "r", encoding="utf-8") as f1, \
                open(file_text_ta, "r", encoding="utf-8") as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

        with open(output_text, "w", encoding="utf-8") as out_txt:
            out_txt.writelines(lines1 + lines2)

        log("Concaténation terminée.")
    except Exception as e:
        show_error(f"Erreur : {e}")
        log(f"Erreur : {e}")

# Foction pour sauvegarder le YAML
def save_yaml():
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

# Fonction pour recharger le YAML
def reload_yaml():
    global config
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

# Fonction pour lancer les fonctions python
def launch_scripts():
    if run_script("calcul_mfcc_yaml_affichage.py"):
        if run_script("correction_avant_classification_yaml_affichage.py"):
            run_script("classification_yaml_affichage.py")

# Fonction pour ajouter le YAML au dossier
def add_yaml_to_folder():
    try:
        folder = os.path.dirname(config["calcul_mfcc"]["output_path"]) 
        destination = os.path.join(folder, "parametre.yaml")
        shutil.copy(yaml_path, destination)
        
        log(f"Le fichier {yaml_path} a été copié dans {destination}")

    except Exception as e:
        log(f"Erreur lors de la copie : {e}")

##########################################################################################
# INTERFACE
root = tk.Tk()
root.title("Entrainement")
root.configure(bg="#2c3e50")

# Styles
title_font = font.Font(family="Arial", size=20, weight="bold")
button_font = font.Font(family="Arial", size=14)
log_font = font.Font(family="Courier", size=11)

# Titre
title_label = tk.Label(root, text="Entrainement", font=title_font, bg="#2c3e50", fg="#ecf0f1", pady=10)
title_label.pack()

# Frame pour les boutons
frame = tk.Frame(root, padx=15, pady=15, bg="#34495e", bd=2, relief="groove")
frame.pack(pady=10)

# Bouton pour choisir les lettres
btn_letters = tk.Button(frame, text="Choisir les lettres", command=select_letters, font=button_font,
                        bg="#6A4878", fg="white", activebackground="#8e44ad", activeforeground="white",
                        bd=0, padx=10, pady=5)
btn_letters.pack(fill="x", pady=5)

# Bouton pour choisir l'audio ou l'enregistrement
btn_audio = tk.Button(frame, text="Choisir Audio ou Enregistrement", command=choose_audio_or_record, font=button_font,
                      bg="#6A4878", fg="white", activebackground="#8e44ad", activeforeground="white",
                      bd=0, padx=10, pady=5)
btn_audio.pack(fill="x", pady=5)

# Bouton pour lancer la concaténation des fichiers
btn_concat = tk.Button(frame, text="Concaténer les fichiers", command=concatenate_files, font=button_font,
                       bg="#6A4878", fg="white", activebackground="#8e44ad", activeforeground="white",
                       bd=0, padx=10, pady=5)
btn_concat.pack(fill="x", pady=5)

# Bouton pour lancer les scripts Python
btn_scripts = tk.Button(frame, text="Lancer le scripts complet", command=launch_scripts, font=button_font,
                         bg="#6A4878", fg="white", activebackground="#8e44ad", activeforeground="white",
                         bd=0, padx=10, pady=5)
btn_scripts.pack(fill="x", pady=5)

# Bouton pour ajouter le YAML au dossier
btn_add_yaml = tk.Button(frame, text="Ajouter le YAML au dossier", command=add_yaml_to_folder, font=button_font,
                         bg="#6A4878", fg="white", activebackground="#8e44ad", activeforeground="white",
                         bd=0, padx=10, pady=5)
btn_add_yaml.pack(fill="x", pady=5)

# Zone de log
text_log = tk.Text(
    root, 
    height=15, 
    width=70, 
    bg="#1e272e", 
    fg="#b086c0",
    insertbackground="white", 
    font=log_font, 
    bd=2, 
    relief="sunken"
)
text_log.pack(pady=10)

# Charger YAML
yaml_path = "parametre.yaml"
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

log("Interface prête. Veuillez choisir une étape.")

# Lancement de l'interface
root.mainloop()