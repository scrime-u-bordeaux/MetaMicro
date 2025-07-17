import pandas as pd
import numpy as np

##########################################################################################
# CHARGEMENT DES DONNEES
df_loaded = pd.read_csv("data_ta_la_ti_li_i/mfcc_features.csv")
X = df_loaded.iloc[:, :-1].values
block_labels_balanced = df_loaded["label"].values

# Couleurs
colors = {"a": "green", "s": "red", "t": "orange", "n": "brown", "u": "blue", "i": "purple", "l": "deeppink"}
class_colors = [colors[l] for l in block_labels_balanced]

##########################################################################################
# CORRECTION 
# Création d’un DataFrame avec les features et label texte
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
    threshold = 0.7 * distances.mean() 

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
df_filtered.to_csv("new_data/mfcc_features_corrige.csv", index=False)
print("Fichier corrigé sauvegardé dans 'mfcc_features_corrige.csv'")
