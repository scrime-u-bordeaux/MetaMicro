import yaml
import os
import subprocess
import time
import wave
from pydub import AudioSegment
import shutil

##########################################################################################
# FONCTION DE LANCEMENT DES SCRIPTS PYTHON
def run_script(script_name):
    print(f"\n-> Lancement de : {script_name}")
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        print(f"-> Le script {script_name} a échoué avec le code {result.returncode}. Arrêt du pipeline.")
        exit(result.returncode)
    print(f"Script {script_name} terminé avec succès.")

##########################################################################################
# LECTURE DES PARAMÈTRES YAML ET CREATION SI MANQUANT
yaml_path = "parametre.yaml"

with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

# Fonction pour créer un chemin dans le YAML
def ensure_path(section, key, path):
    section[key] = path
    return section[key]

# choix des lettres à ajouter dans le YAML
letters = input("Veuillez entrer les lettres à utiliser pour l'entraînement (ex: a,i,u) : ").strip().split(",")
calcul = config["calcul_mfcc"]
ensure_path(calcul, "letters", letters)

# Sauvegarde 
with open(yaml_path, "w") as file:
    yaml.dump(config, file, sort_keys=False, allow_unicode=True)

# choix de l'enregistrement ou du fichier audio
condition = input("Voulez-vous créer un enregistrement ou utiliser un fichier audio existant ? (répondez 'e' ou 'f') : ").strip()
if condition.lower() == "e":

    # Lancement du script d'enregistrement
    run_script("entrainement_par_enregistrement.py")

    # Recharge du YAML mis à jour par le script
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    file_path_audio_non_concat = config["calcul_mfcc"]["file_path_audio_non_concat"]

    # Lancement du script d'analyse pour ecricre les marqueurs
    run_script("analyse_entrainement_enregistrement.py")

    # Recharge du YAML mis à jour par le script
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    file_path_txt_non_concat = config["calcul_mfcc"]["file_path_txt_non_concat"]

else:
    file_path_txt_non_concat = input("Veuillez entrer le chemin du fichier texte contenant les marqueurs (ex: ici/marqueur_test.txt) : ").strip()
    file_path_audio_non_concat = input("Veuillez entrer le chemin du fichier audio (ex: ici/test.wav) : ").strip()

##########################################################################################
# CONCATENATION DES FICHIERS AUDIO
file_audio_ta = config["calcul_mfcc"]["file_path_ta"]
file_path_audio = "audio.wav"

# Mettre les fichiers aux mêmes param
with wave.open(file_audio_ta, "rb") as ref, wave.open(file_path_audio_non_concat, "rb") as src:
    ref_params = ref.getparams()
    src_params = src.getparams()

    if (ref_params.nchannels != src_params.nchannels or
        ref_params.sampwidth != src_params.sampwidth or
        ref_params.framerate != src_params.framerate or
        ref_params.comptype != src_params.comptype):

        audio = AudioSegment.from_file(file_path_audio_non_concat)
        audio = audio.set_channels(ref_params.nchannels)
        audio = audio.set_frame_rate(ref_params.framerate)
        audio = audio.set_sample_width(ref_params.sampwidth)

        file_path_audio_non_concat_converted = "converted_audio.wav"
        audio.export(file_path_audio_non_concat_converted, format="wav")

# Concatenation du fichier audio avec le fichier audio de référence
with wave.open(file_audio_ta, "rb") as wav1, wave.open(file_path_audio_non_concat_converted, "rb") as wav2:

    # Vérification que les paramètres sont identiques
    params1 = wav1.getparams()
    params2 = wav2.getparams()

    # Vérifier la compatibilité des paramètres 
    if (params1.nchannels != params2.nchannels or
        params1.sampwidth != params2.sampwidth or
        params1.framerate != params2.framerate or
        params1.comptype != params2.comptype):
        raise ValueError("Les fichiers WAV ont des formats incompatibles (canaux, taux, compression, etc.).")

    # Lire les données audio brutes
    frames1 = wav1.readframes(wav1.getnframes())
    frames2 = wav2.readframes(wav2.getnframes())

# Écriture dans un nouveau fichier
with wave.open(file_path_audio, "wb") as out:
    out.setparams(wav1.getparams())
    out.writeframes(frames1 + frames2)

# Mise à jour du YAML
ensure_path(config["calcul_mfcc"], "file_path_audio", file_path_audio)

# Réécriture du fichier YAML avec les chemins complétés
with open(yaml_path, "w") as file:
    yaml.dump(config, file, sort_keys=False, allow_unicode=True)

##########################################################################################
# CONCATENATION DES FICHIERS TEXTE
file_text_ta = config["calcul_mfcc"]["file_path_ta_text"]
file_path_txt = "text.txt"

# Concatenation du fichier texte avec le fichier texte de référence
with open(file_path_txt_non_concat, "r", encoding="utf-8") as f1, open(file_text_ta, "r", encoding="utf-8") as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

# Concaténation
with open(file_path_txt, "w", encoding="utf-8") as out:
    out.writelines(lines1 + lines2)

# Mise à jour du YAML si besoin
ensure_path(config["calcul_mfcc"], "file_path_txt", file_path_txt)

# Réécriture du fichier YAML avec les chemins complétés
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(config, f, sort_keys=False, allow_unicode=True)

##########################################################################################
# EXTRACTION DES CHEMINS ET COMPLETION DU YAML
# Extraction du dossier et du nom de base
print(file_path_txt_non_concat)
folder = os.path.dirname(file_path_txt_non_concat)
base_name = os.path.splitext(os.path.basename(file_path_txt_non_concat))[0]

# Compléter les chemins manquants
ensure_path(calcul, "output_path", os.path.join(folder, f"mfcc_features.csv"))
correction = config.get("correction_avant_classification", {})

ensure_path(correction, "output_path", os.path.join(folder, f"mfcc_features_{base_name}_corrige.csv"))
config["correction_avant_classification"] = correction

outputs = config["classification"]["outputs"]
if_suppression_mfcc = config["classification"]["if_suppression_mfcc"]
ensure_path(outputs, "mean_X_output", os.path.join(folder, f"mfcc_mean_{base_name}.pkl"))
ensure_path(outputs, "std_X_output", os.path.join(folder, f"mfcc_std_{base_name}.pkl"))
ensure_path(outputs, "proj_pca_output", os.path.join(folder, f"X_proj_scaled_{base_name}.csv"))
ensure_path(outputs, "eigenvectors_output", os.path.join(folder, f"eigenvectors_{base_name}.pkl"))
ensure_path(outputs, "knn_model_output", os.path.join(folder, f"knn_model_{base_name}.pkl"))
ensure_path(if_suppression_mfcc, "eigenvectors_output_tronque", os.path.join(folder, f"eigenvectors_tronque_{base_name}.pkl"))

# Réécriture du fichier YAML avec les chemins complétés
with open(yaml_path, "w") as file:
    yaml.dump(config, file, sort_keys=False, allow_unicode=True)

print(f"Fichier YAML mis à jour avec les chemins complétés : {yaml_path}")

##########################################################################################
# LANCEMENT DES SCRIPTS PYTHON
# Ordre d'exécution
time.sleep(1)
# run_script("calcul_mfcc_yaml.py")
run_script("correction_avant_classification_yaml.py")
run_script("classification_yaml.py")

print("\nPipeline terminé avec succès.")

##########################################################################################
# COPIER LE FICHIER YAML DANS LE DOSSIER
# Copier le fichier parametre.yaml dans le dossier
destination = os.path.join(folder, "parametre.yaml")

# Copie du fichier
shutil.copy(yaml_path, destination)

print(f"Le fichier {yaml_path} a été copié dans {destination}")