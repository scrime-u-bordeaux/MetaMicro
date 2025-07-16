import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score
from joblib import Parallel, delayed
import yaml
import os

##########################################################################################
## LECTURE DES PARAMÈTRES YAML
with open("parametre.yaml", "r") as file:
    config = yaml.safe_load(file)

input_path = config["correction_avant_classification"]["output_path"]
letters = config["calcul_mfcc"]["letters"]

suppression_mfcc = config["classification"]["if_suppression_mfcc"]["suppression_mfcc"]
seuil_pour_mfcc_max = config["classification"]["if_suppression_mfcc"]["seuil_pour_mfcc_max"]

dimension_affichage = config["classification"]["dimension_affichage"]

test_size    = config["classification"]["parametres_knn"]["test_size"] 
random_state = config["classification"]["parametres_knn"]["random_state"]
n_neighbors_max = config["classification"]["parametres_knn"]["n_neighbors_max"]

# Entrée et sortie
file_path_txt = config["calcul_mfcc"]["file_path_txt"]

# Extraire dossier et base_name
folder = os.path.dirname(file_path_txt)                  
base_name = os.path.splitext(os.path.basename(file_path_txt))[0] 
classification = config["classification"]

mean_X_output = config["classification"]["outputs"]["mean_X_output"]
std_X_output  = config["classification"]["outputs"]["std_X_output"]
proj_pca_output = config["classification"]["outputs"]["proj_pca_output"]
knn_model_output = config["classification"]["outputs"]["knn_model_output"]

if suppression_mfcc:
    eigenvectors_output_tronque = config["classification"]["if_suppression_mfcc"]["eigenvectors_output_tronque"]
else:
    eigenvectors_output = config["classification"]["outputs"]["eigenvectors_output"]

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
# CHARGER LES DONNNEES 
mfcc_matrix = pd.read_csv(input_path)

# Encoder les labels (t=0, a=1, s=2)
letters_with_st = letters + [l for l in ["s", "t"] if l not in letters]
label_mapping = {label: idx + 1 for idx, label in enumerate(letters_with_st)}
mfcc_matrix["label"] = mfcc_matrix["label"].map(label_mapping)

# Séparer les features et labels
X = mfcc_matrix.iloc[:, :-2].values  
block_labels = mfcc_matrix["label"].values.astype(int) 

##########################################################################################
# TROUVER LES POINTS LES PLUS ELOIGNE
# Trouver la classe la moins représentée
class_counts = Counter(block_labels)
min_class_size = min(class_counts.values()) 

# Détection des points les plus isolés avec KNN
def get_density_scores(X, k=5):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    return distances.mean(axis=1)

# Undersampling basé sur la densité des points
X_balanced = []
block_labels_balanced = []

for label in np.unique(block_labels):
    idx = np.where(block_labels == label)[0] 
    X_class = X[idx]  
    density_scores = get_density_scores(X_class, k=8)  
    sorted_idx = np.argsort(density_scores) # np.argsort() retourne les indices tries par ordre croissant des valeurs de density_scores
    selected_idx = sorted_idx[:min_class_size]  

    X_balanced.append(X_class[selected_idx])
    block_labels_balanced.extend([label] * min_class_size)

# Conversion en numpy array
X_balanced = np.vstack(X_balanced)
block_labels_balanced = np.array(block_labels_balanced).astype(int) 

# Création du DataFrame avec les colonnes MFCC
df_balanced = pd.DataFrame(X_balanced)
df_balanced["label"] = block_labels_balanced

##########################################################################################
# CALCUL DE PCA
X_centered = X_balanced - np.mean(X_balanced, axis=0)
X_scaled = X_centered / np.std(X_balanced, axis=0)

# Sauvegarde de la moyenne et de l'écart-type
mean_X = np.mean(X_balanced, axis=0)
final_path = check_overwrite_or_rename(mean_X_output)
if final_path:
    joblib.dump(mean_X, mean_X_output)
    print(f"Fichier sauvegardé dans {final_path}")
else:
    print("Aucune sauvegarde effectuée.")
print(f"Fichier corrigé sauvegardé dans '{mean_X_output}'")

std_X = np.std(X_balanced, axis=0)
final_path = check_overwrite_or_rename(std_X_output)
if final_path:
    joblib.dump(std_X, std_X_output)
    print(f"Fichier sauvegardé dans {final_path}")
else:
    print("Aucune sauvegarde effectuée.")
print(f"Fichier corrigé sauvegardé dans '{std_X_output}'")

# Matrice de var-covar
cov_matrix = np.cov(X_scaled.T)

# Val et vect propres
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Affichage des vecteurs propres sous forme d'image
fig = plt.figure(figsize=(10, 8))
plt.imshow(np.abs(eigenvectors), aspect='auto', cmap='viridis_r')  # meilleure lisibilité
plt.colorbar(label='Valeurs absolues du vecteur propre')
plt.title('Vecteurs propres (colonnes)')
plt.xlabel('Index du vecteur propre')
plt.ylabel('Valeurs absolues du vecteur')
plt.tight_layout()
plt.show()  

##########################################################################################
# PROJECTION ET ÉVALUATION PAR DB INDEX
# Recherche du meilleur nombre de dimensions projetées
scores = []

# Fonction pour une projection + score pour un j donné
def compute_score(j, X_scaled, eigenvectors, labels):
    try:
        X_proj = X_scaled @ eigenvectors[:, :j]
        score = davies_bouldin_score(X_proj, labels)
    except Exception as e:
        print(f"Erreur pour j={j}: {e}")
        score = np.nan
    return score

# Parallélisation
scores = Parallel(n_jobs=-1)(  # -1 = utilise tous les cœurs dispo
    delayed(compute_score)(j, X_scaled, eigenvectors, block_labels_balanced)
    for j in range(1, len(eigenvalues) + 1)
)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(scores) + 1), scores, marker='o')
plt.xlabel("Nombre de dimensions projetées")
plt.ylabel("Davies-Bouldin score (plus bas = meilleur)")
plt.title("Qualité de séparation des groupes selon la projection")
plt.grid()
plt.tight_layout()
plt.show()

# Trouver la meilleure valeur de n (le score le plus bas)
best_n = np.nanargmin(scores) + 1   # +1 car l'index 0 correspond à n=1
best_score = scores[best_n - 1]

print(f"Meilleur score DB : {best_score:.4f} atteint pour n = {best_n} dimensions.")

# Changement de best score pour l'affichage sur le score vaut 2
if best_n == 2:
    best_n = 3

# Conserver les n meilleurs vecteurs propres
best_eigenvectors = eigenvectors[:, :best_n]

# Affichage
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(best_eigenvectors), aspect='auto', cmap='viridis_r')
plt.colorbar(label='Valeurs absolues du vecteur propre')
plt.title(f'{best_n} vecteurs propres correspondant au meilleur score DB')
plt.xlabel('Composante principale')
plt.ylabel('Valeurs absolues du vecteur')
plt.tight_layout()
plt.show()

##########################################################################################
# RECUPERER LES VARIABLE AU DESSUS DU SEUIL DE LA VALEUR MAXIMUM MFCC PAR COLONNE
# Valeurs absolues des vecteurs propres (pour éviter les effets de signe)
abs_eigenvectors = np.abs(best_eigenvectors)

# Moyenne des valeurs par vecteur propre 
max_per_vector = abs_eigenvectors.max(axis=0)

# On crée une matrice booléenne : True là où la valeur dépasse la moyenne de sa colonne
if suppression_mfcc:
    above_avg_mask = abs_eigenvectors > seuil_pour_mfcc_max * max_per_vector

    # On garde uniquement les valeurs qui dépassent cette moyenne (autres mises à 0)
    eigenvectors_thresholded = np.where(above_avg_mask, best_eigenvectors, 0)
else:
    # Si on ne supprime pas, on garde les vecteurs propres tels quels
    eigenvectors_thresholded = best_eigenvectors


# Affichage des vecteurs propres "seuillés"
plt.figure(figsize=(12, 6))
plt.imshow(np.abs(eigenvectors_thresholded), aspect='auto', cmap='magma')
plt.colorbar(label="Valeurs des vecteurs propres (seuillées)")
plt.title("Valeurs des vecteurs propres au-dessus de la moyenne de leur colonne")
plt.xlabel("Index du vecteur propre")
plt.ylabel("Dimensions (features)")
plt.tight_layout()
plt.show()

##########################################################################################
# REPROJETER SUR LES NOUVEAUX VECTEUR PROPRES ET NOUVEAU SCORE
j = best_n
X_proj_thresh = X_scaled @ eigenvectors_thresholded[:, :j]

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

assert len(dimension_affichage) >= 3, "dimension_affichage doit contenir au moins 3 indices"
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
ax.set_xlabel("Comp. 1 (seuilée)")
ax.set_ylabel("Comp. 2 (seuilée)")
ax.set_zlabel("Comp. 3 (seuilée)")
ax.set_title("Projection 3D sur vecteurs propres seuillés")

# Légende cohérente avec tes couleurs
handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color, markersize=10,
               label=f"Classe '{label}'")
    for label, color in colors.items()
]
handles.append(
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor="deeppink", markeredgecolor="black",
               markersize=10, label="Prédits 'l'")
)
ax.legend(handles=handles, title="Légende")

plt.tight_layout()
plt.show()

# Nouveau score
score = davies_bouldin_score(X_proj_thresh, block_labels_balanced)
print(f"Nouveau score DB : {score:.4f}")

##########################################################################################
# SAUVEGARDE
# Créer un DataFrame avec la projection seuillée et les labels *textes*
df_proj_thresh = pd.DataFrame(X_proj_thresh, columns=[f"comp_{i+1}" for i in range(X_proj_thresh.shape[1])])
df_proj_thresh["label"] = block_labels_text

# Sauvegarde en CSV
final_path = check_overwrite_or_rename(proj_pca_output)
if final_path:
    df_proj_thresh.to_csv(proj_pca_output, index=False)
    print(f"Fichier sauvegardé dans {final_path}")
else:
    print("Aucune sauvegarde effectuée.")
print(f"Fichier corrigé sauvegardé dans '{proj_pca_output}'")
print(f"Fichier '{proj_pca_output}' sauvegardé avec la projection seuillée.")

# Sauvegarde des vecteurs propres seuillés
final_path = check_overwrite_or_rename(eigenvectors_output)
if final_path:
    joblib.dump(eigenvectors_thresholded[:, :j], eigenvectors_output)
    print(f"Fichier sauvegardé dans {final_path}")
else:
    print("Aucune sauvegarde effectuée.")
print(f"Fichier corrigé sauvegardé dans '{eigenvectors_output}'")

# SI ON VEUT SUPPRIMER DES VALEURS MFCC INUTILES
if suppression_mfcc:
    block_size = 11
    actions  = ['keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep', 'keep']

    n_blocks = len(actions)
    rows_to_keep = []

    for i, action in enumerate(actions):
        start = i * block_size
        end = start + block_size
        if action == 'keep':
            rows_to_keep.extend(range(start, end))

    # Si on veut supprimer des colonnes, on peut le faire ici
    rows_to_keep = [i for i in rows_to_keep if i < eigenvectors_thresholded.shape[0]]

    # Appliquer le filtrage
    eigenvectors_truncated = eigenvectors_thresholded[rows_to_keep, :j]

    # Sauvegarde de la version troncaturée
    
    final_path = check_overwrite_or_rename(eigenvectors_output_tronque)
    if final_path:
        joblib.dump(eigenvectors_truncated, eigenvectors_output_tronque)
        print(f"Fichier sauvegardé dans {final_path}")
    else:
        print("Aucune sauvegarde effectuée.")
    print(f"Fichier corrigé sauvegardé dans '{eigenvectors_output_tronque}'")

    print("eigenvectors_truncated:", eigenvectors_truncated)

# print("eigen_vectors:", eigenvectors_thresholded[:, :j])

##########################################################################################
# ENTRAINEMENT
# Reconvertir les labels texte en entiers pour l'entraînement
block_labels_int = [k for l in block_labels_text for k, v in label_mapping_inv.items() if v == l]

# Split des données pour classification
X_train, X_test, y_train, y_test = train_test_split(
    X_proj_thresh,
    block_labels_int,
    test_size=0.4,
    stratify=block_labels_int,
    random_state=42
)

# Recherche des meilleurs paramètres
best_k = None
best_distance = None
best_score = 0

k_values = range(1, 50)
distances = ["euclidean", "manhattan", "minkowski", "cosine"]

for k in k_values:
    for dist in distances:
        knn = KNeighborsClassifier(n_neighbors=k, metric=dist)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        avg_score = np.mean(scores)

        if avg_score > best_score:
            best_k = k
            best_distance = dist
            best_score = avg_score

print(f" Meilleur k : {best_k}, distance : {best_distance}, précision (CV) : {best_score:.2f}")

# Entraînement final
knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_distance)

# Fit du modèle KNN
knn.fit(X_train, y_train)

# Évaluation
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision finale du modèle KNN sur test : {accuracy * 100:.2f}%")

# Sauvegarde du modèle KNN
final_path = check_overwrite_or_rename(knn_model_output)
if final_path:
    joblib.dump(knn, knn_model_output)
    print(f"Fichier sauvegardé dans {final_path}")
else:
    print("Aucune sauvegarde effectuée.")
print(f"Fichier corrigé sauvegardé dans '{knn_model_output}'")

# MATRICE DE CONFUSION
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=letters_with_st,
    cmap="Blues"
)
plt.title("Matrice de Confusion du modèle KNN (avec PCA)")
plt.show()