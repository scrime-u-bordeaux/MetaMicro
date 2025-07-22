import pandas as pd
import numpy as np
import yaml
import os

##########################################################################################
## LECTURE DES PARAMÈTRES YAML
with open("parametre.yaml", "r") as file:
    config = yaml.safe_load(file)

input_path = config["calcul_mfcc"]["output_path"]
letters = config["calcul_mfcc"]["letters"]
seuil_de_correction = config["correction_avant_classification"]["seuil_de_correction"]
output_path = config["correction_avant_classification"]["output_path"]

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
# CHARGEMENT DES DONNEES
df_loaded = pd.read_csv(input_path)
X = df_loaded.iloc[:, :-1].values
block_labels_balanced = df_loaded["label"].values

# Couleurs
color_palette = ["green", "red", "orange", "purple", "cyan", "brown", "blue", "olive", "pink", "gray"]
letters_with_st = letters + [l for l in ["s", "t"] if l not in letters]
colors = {label: color_palette[i % len(color_palette)] for i, label in enumerate(letters_with_st)}
class_colors = [colors[l] for l in block_labels_balanced]

##########################################################################################
# CORRECTION N-DIMENSIONNELLE

# Création d’un DataFrame avec les features + label texte
df_pca = pd.DataFrame(X)
df_pca["label"] = block_labels_balanced

# Nouveau DataFrame filtré
filtered_points = []

# Colonnes features
feature_cols = df_pca.columns[:-1]

# Parcours de chaque classe
for label in df_pca["label"].unique():
    class_points = df_pca[df_pca["label"] == label]

    # Calcul du centroïde
    centroid = class_points[feature_cols].mean()

    # Distances euclidiennes au centroïde
    distances = np.linalg.norm(class_points[feature_cols].values - centroid.values, axis=1)

    # Seuil de distance 
    threshold = seuil_de_correction * distances.mean() 

    # Filtrage
    class_points_filtered = class_points[distances < threshold]
    filtered_points.append(class_points_filtered)

# Fusion
df_filtered = pd.concat(filtered_points)

# Mise à jour des données et couleurs
X_selected_pca_filtered = df_filtered[feature_cols].values
block_labels_filtered = df_filtered["label"].values
class_colors_filtered = [colors[label] for label in block_labels_filtered]

# Sauvegarde du nouveau DataFrame filtré
final_path = check_overwrite_or_rename(output_path)
if final_path:
    df_filtered.to_csv(output_path, index=False)
    print(f"Fichier sauvegardé dans {final_path}")
else:
    print("Aucune sauvegarde effectuée.")
print(f"Fichier corrigé sauvegardé dans '{output_path}'")
