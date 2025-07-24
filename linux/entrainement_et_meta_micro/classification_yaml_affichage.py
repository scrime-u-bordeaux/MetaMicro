import tkinter as tk
from tkinter import font
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.metrics import davies_bouldin_score
from joblib import Parallel, delayed
import yaml
import os
import threading

##########################################################################################
# CHARGER YAML
with open("linux/entrainement_et_meta_micro/parametre.yaml", "r") as file:
    config = yaml.safe_load(file)

# Paramètres d'entrée
file_path_txt = config["calcul_mfcc"]["file_path_txt"]
input_path = config["correction_avant_classification"]["output_path"]
letters = config["calcul_mfcc"]["letters"]

suppression_mfcc = config["classification"]["if_suppression_mfcc"]["suppression_mfcc"]
seuil_pour_mfcc_max = config["classification"]["if_suppression_mfcc"]["seuil_pour_mfcc_max"]

dimension_affichage = config["classification"]["dimension_affichage"]

test_size = config["classification"]["parametres_knn"]["test_size"]
random_state = config["classification"]["parametres_knn"]["random_state"]
n_neighbors_max = config["classification"]["parametres_knn"]["n_neighbors_max"]

# Extraire dossier et base_name
folder = os.path.dirname(file_path_txt)
base_name = os.path.splitext(os.path.basename(file_path_txt))[0]
classification = config["classification"]

# Chemins de sortie
mean_X_output = config["classification"]["outputs"]["mean_X_output"]
std_X_output = config["classification"]["outputs"]["std_X_output"]
proj_pca_output = config["classification"]["outputs"]["proj_pca_output"]
knn_model_output = config["classification"]["outputs"]["knn_model_output"]

if suppression_mfcc:
    eigenvectors_output = config["classification"]["if_suppression_mfcc"]["eigenvectors_output_tronque"]
else:
    eigenvectors_output = config["classification"]["outputs"]["eigenvectors_output"]

##########################################################################################
# FONCTIONS UTILITAIRES
def log(message):
    text_log.insert(tk.END, message + "\n")
    text_log.see(tk.END)
    root.update()

# Fonction pour sauvegarder le fichier
def ask_save_file(default_path):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv" if default_path.endswith(".csv") else ".pkl",
        filetypes=[("Fichiers CSV", "*.csv"), ("Fichiers Joblib", "*.pkl")],
        initialfile=os.path.basename(default_path),
        title="Sauvegarder le fichier sous…"
    )
    return file_path if file_path else None

##########################################################################################
# SCRIPT CLASSIFICATION
# Fonction pour lancer le script de classification
def start_script():
    threading.Thread(target=full_classification_script).start()

def full_classification_script():
    try:
        log("Chargement des données…")
        mfcc_matrix = pd.read_csv(input_path)

        # Ajout des lettres "s" et "t"
        letters_with_st = letters + [l for l in ["s", "t"] if l not in letters]
        label_mapping = {label: idx + 1 for idx, label in enumerate(letters_with_st)}

        # Récupération des blocs et de leurs labels dans la matrice MFCC
        mfcc_matrix["label"] = mfcc_matrix["label"].map(label_mapping)
        X = mfcc_matrix.iloc[:, :-2].values
        block_labels = mfcc_matrix["label"].values.astype(int)

        log("Equilibrage du dataset…")
        # Trouver la classe la moins représentée pour avoir un échantillonnage équilibré
        class_counts = Counter(block_labels)
        min_class_size = min(class_counts.values())

        # Fonction pour détecter des points les plus isolés avec KNN
        def get_density_scores(X, k=5):
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(X)
            distances, _ = knn.kneighbors(X)
            return distances.mean(axis=1)

        # Rééquilibrage du dataset pour éviter les points aberrants selon leur densité
        X_balanced, block_labels_balanced = [], []
        for label in np.unique(block_labels):
            idx = np.where(block_labels == label)[0]
            X_class = X[idx]
            density_scores = get_density_scores(X_class, k=8)
            sorted_idx = np.argsort(density_scores)
            selected_idx = sorted_idx[:min_class_size]

            X_balanced.append(X_class[selected_idx])
            block_labels_balanced.extend([label] * min_class_size)

        X_balanced = np.vstack(X_balanced)
        block_labels_balanced = np.array(block_labels_balanced).astype(int)

        # Enregistrement de la moyenne et de l'écart-type
        log("Calcul et sauvegarde des statistiques (moyenne, écart-type)…")
        mean_X = np.mean(X_balanced, axis=0)
        std_X = np.std(X_balanced, axis=0)

        # Sauvegarde de la moyenne et de l'écart-type
        save_path = ask_save_file(mean_X_output)
        if save_path:
            joblib.dump(mean_X, save_path)
            config["classification"]["outputs"]["mean_X_output"] = save_path
            with open("linux/entrainement_et_meta_micro/parametre.yaml", "w") as file:
                yaml.dump(config, file, sort_keys=False, allow_unicode=True)
            log(f"Moyenne sauvegardée : {save_path}")
        else:
            log("Sauvegarde de la moyenne annulée.")

        save_path = ask_save_file(std_X_output)
        if save_path:
            joblib.dump(std_X, save_path)
            config["classification"]["outputs"]["std_X_output"] = save_path
            with open("linux/entrainement_et_meta_micro/parametre.yaml", "w") as file:
                yaml.dump(config, file, sort_keys=False, allow_unicode=True)
            log(f"Ecart-type sauvegardé : {save_path}")
        else:
            log("Sauvegarde de l'écart-type annulée.")


        # Calcul de la PCA
        log("Calcul de la PCA et vecteurs propres…")

        # Centrage et normalisation des données
        X_centered = X_balanced - mean_X
        X_scaled = X_centered / std_X

        # Calcul de la matrice de covariance
        cov_matrix = np.cov(X_scaled.T)

        # Calcul des valeurs et vecteurs propres
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Affichage des vecteurs propres
        plt.figure(figsize=(10, 8))
        plt.imshow(np.abs(eigenvectors), aspect='auto', cmap='viridis_r')
        plt.colorbar(label='Valeurs absolues du vecteur propre')
        plt.title('Vecteurs propres (colonnes)')
        plt.show()

        # Calcul du score de Davies-Bouldin
        log("Recherche du meilleur nombre de dimensions projetées…")
        scores = Parallel(n_jobs=-1)(
            delayed(lambda j: davies_bouldin_score(X_scaled @ eigenvectors[:, :j], block_labels_balanced) if j > 0 else np.nan)(j)
            for j in range(1, len(eigenvalues) + 1)
        )

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(scores) + 1), scores, marker='o')
        plt.xlabel("Nombre de dimensions projetées")
        plt.ylabel("Davies-Bouldin score (plus bas = meilleur)")
        plt.title("Qualité de séparation des groupes selon la projection")
        plt.grid()
        plt.show()

        # Sélection du meilleur nombre de dimensions selon le score de Davies-Bouldin
        best_n = np.nanargmin(scores) + 1 # +1 car l'index 0 correspond à n=1
        best_score = scores[best_n - 1]
        log(f"Meilleur score DB : {best_score:.4f} atteint pour n = {best_n} dimensions.")

        # Changement de best score pour l'affichage sur le score vaut 2
        if best_n <= 2:
            best_n = 3

        # Réduction des vecteurs propres aux nombres de dimensions sélectionnés
        best_eigenvectors = eigenvectors[:, :best_n]

        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(best_eigenvectors), aspect='auto', cmap='viridis_r')
        plt.colorbar(label='Valeurs absolues du vecteur propre')
        plt.title(f'{best_n} vecteurs propres correspondant au meilleur score DB')
        plt.show()

        # Application du seuillage de certaines mfcc si nécessaire
        if suppression_mfcc:
            log("Application du seuillage MFCC…")
            abs_eigenvectors = np.abs(best_eigenvectors)
            max_per_vector = abs_eigenvectors.max(axis=0)
            above_avg_mask = abs_eigenvectors > seuil_pour_mfcc_max * max_per_vector
            eigenvectors_thresholded = np.where(above_avg_mask, best_eigenvectors, 0)
        else:
            eigenvectors_thresholded = best_eigenvectors

        plt.figure(figsize=(12, 6))
        plt.imshow(np.abs(eigenvectors_thresholded), aspect='auto', cmap='magma')
        plt.colorbar(label="Valeurs des vecteurs propres (seuillés)")
        plt.title("Vecteurs propres après seuillage")
        plt.show()

        # Projection des données sur les vecteurs propres
        X_proj_thresh = X_scaled @ eigenvectors_thresholded[:, :best_n]

        # Affichage de la projection 3D
        log("Affichage de la projection 3D sur les vecteurs propres seuillés…")

        # Mapping label
        label_mapping_inv = {v: k for k, v in label_mapping.items()}
        block_labels_text = [label_mapping_inv[label] for label in block_labels_balanced]

        # Couleurs
        color_palette = ["green", "red", "orange", "purple", "cyan", "brown", "blue", "olive", "pink", "gray"]
        colors = {label: color_palette[i % len(color_palette)] for i, label in enumerate(letters_with_st)}
        class_colors = [colors[label_mapping_inv[l]] for l in block_labels_balanced]

        # Projection 3D sur les vecteurs propres seuillés
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        x = X_proj_thresh[:, dimension_affichage[0]]
        y = X_proj_thresh[:, dimension_affichage[1]]
        z = X_proj_thresh[:, dimension_affichage[2]]

        # Affichage des points avec couleurs personnalisées
        scatter = ax.scatter(
            x,
            y,
            z,
            c=class_colors,
            alpha=0.6,
            s=20
        )

        # Axes et titre
        ax.set_xlabel("Comp. 1 (seuillée)")
        ax.set_ylabel("Comp. 2 (seuillée)")
        ax.set_zlabel("Comp. 3 (seuillée)")
        ax.set_title("Projection 3D sur vecteurs propres seuillés")

        # Légende cohérente avec tes couleurs
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=color, markersize=10, 
                       label=f"Classe '{label}'")
                   for label, color in colors.items()
        ]
        ax.legend(handles=handles, title="Légende")
        plt.tight_layout()
        plt.show()

        # Sauvegarde de la projection seuillée
        save_path = ask_save_file(proj_pca_output)
        if save_path:
            df_proj_thresh = pd.DataFrame(X_proj_thresh, columns=[f"comp_{i+1}" for i in range(X_proj_thresh.shape[1])])
            df_proj_thresh["label"] = block_labels_text
            df_proj_thresh.to_csv(save_path, index=False)
            config["classification"]["outputs"]["proj_pca_output"] = save_path
            with open("linux/entrainement_et_meta_micro/parametre.yaml", "w") as file:
                yaml.dump(config, file, sort_keys=False, allow_unicode=True)
            log(f"Projection seuillée sauvegardée : {save_path}")
        else:
            log("Sauvegarde de la projection seuillée annulée.")

        # Sauvegarde des vecteurs propres threshold seuillés 
        save_path = ask_save_file(eigenvectors_output)
        if save_path:
            joblib.dump(eigenvectors_thresholded[:, :best_n], save_path)
            config["classification"]["outputs"]["eigenvectors_output"] = save_path
            with open("linux/entrainement_et_meta_micro/parametre.yaml", "w") as file:
                yaml.dump(config, file, sort_keys=False, allow_unicode=True)
            log(f"Vecteurs propres seuillés sauvegardés : {save_path}")
        else:
            log("Sauvegarde des vecteurs propres seuillés annulée.")

        # Sauvegarde des vecteurs propres seuillés
        save_path = ask_save_file(eigenvectors_output)
        if save_path:
            joblib.dump(eigenvectors_thresholded, save_path)
            log(f"Vecteurs propres seuillés sauvegardés : {save_path}")

        # Entraînement du modèle KNN
        log("Entraînement du modèle KNN…")

        # Séparation des données en train et test
        X_train, X_test, y_train, y_test = train_test_split(
            X_proj_thresh, block_labels_balanced, test_size=test_size, stratify=block_labels_balanced, random_state=random_state
        )

        # Recherche des meilleurs paramètres KNN
        best_k, best_metric, best_acc = None, None, 0
        for k in range(1, n_neighbors_max + 1):
            for metric in ["euclidean", "manhattan", "minkowski"]:
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                scores = cross_val_score(knn, X_train, y_train, cv=5)
                avg_score = np.mean(scores)
                if avg_score > best_acc:
                    best_k, best_metric, best_acc = k, metric, avg_score

        log(f"Meilleur KNN trouvé : k={best_k}, distance={best_metric}, précision={best_acc:.2%}")

        # Entraînement final
        knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)

        # Fit du modèle KNN
        knn.fit(X_train, y_train)

        # Prédiction knn pour la matrice de confusion
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        log(f"Précision finale sur le jeu de test : {accuracy:.2%}")

        # Affichage de la matrice de confusion
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=letters_with_st,
            cmap="Blues"
        )
        plt.title("Matrice de confusion")
        plt.show()

        # Sauvegarde du modèle KNN
        save_path = ask_save_file(knn_model_output)
        if save_path:
            joblib.dump(knn, save_path)
            config["classification"]["outputs"]["knn_model_output"] = save_path
            with open("linux/entrainement_et_meta_micro/parametre.yaml", "w") as file:
                yaml.dump(config, file, sort_keys=False, allow_unicode=True)
            log(f"Moldèle KNN sauvegardée : {save_path}")
        else:
            log("Sauvegarde du modèle KNN annulée.")

    except Exception as e:
        log(f"Erreur : {e}")
        messagebox.showerror("Erreur", str(e))

##########################################################################################
# INTERFACE
root = tk.Tk()
root.title("script Classification et PCA")
root.configure(bg="#2c3e50")

# Polices
title_font = font.Font(family="Arial", size=20, weight="bold")
button_font = font.Font(family="Arial", size=14)
log_font = font.Font(family="Courier", size=11)

# Titre
title_label = tk.Label(
    root,
    text="script de Classification et PCA",
    font=title_font,
    bg="#2c3e50",
    fg="#ecf0f1",
    pady=10
)
title_label.pack()

# Cadre principal
frame = tk.Frame(root, padx=15, pady=15, bg="#34495e", bd=2, relief="groove")
frame.pack(pady=10)

# Bouton pour lancer le script de classification
start_button = tk.Button(
    frame,
    text="Lancer le script",
    command=start_script,
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
    height=15,
    width=80,
    bg="#1e272e",
    fg="#b086c0",
    insertbackground="white",
    font=log_font,
    bd=2,
    relief="sunken"
)
text_log.pack(pady=10)

log("Prêt pour le script.")

# Lancement de l'interface
root.mainloop()