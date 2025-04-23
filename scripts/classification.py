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
from collections import defaultdict

##########################################################################################
# CHARGER LES DONNNEES 
mfcc_matrix = pd.read_csv("mfcc_features_ta_la_sans_r_main_plus_corrige.csv")

# Encoder les labels (t=0, a=1, s=2)
label_mapping = {"l": 0, "a": 1, "s": 2, "t": 3, "r": 4}
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

# Matrice de var-covar
cov_matrix = np.cov(X_scaled.T)

# Val et vect propres
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Affichage des vecteurs propres sous forme d'image
fig = plt.figure(figsize=(10, 8))
plt.imshow(np.abs(eigenvectors), aspect='auto', cmap='viridis_r')  # meilleure lisibilité
plt.colorbar(label='Valeur du vecteur propre')
plt.title('Vecteurs propres (colonnes)')
plt.xlabel('Index du vecteur propre')
plt.ylabel('Valeurs du vecteur')
plt.tight_layout()
plt.show()  

# Affichage des vecteurs propres sous forme d'image sans val absolue
fig = plt.figure(figsize=(10, 8))
plt.imshow(eigenvectors, aspect='auto', cmap='viridis_r')  # meilleure lisibilité
plt.colorbar(label='Valeur du vecteur propre')
plt.title('Vecteurs propres (colonnes)')
plt.xlabel('Index du vecteur propre')
plt.ylabel('Valeurs du vecteur')
plt.tight_layout()
plt.show()  


##########################################################################################
# PROJECTION ET ÉVALUATION PAR DB INDEX
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
best_n = np.nanargmin(scores) + 1  # +1 car l'index 0 correspond à n=1
best_score = scores[best_n - 1]

print(f"Meilleur score DB : {best_score:.4f} atteint pour n = {best_n} dimensions.")

# Réafficher les n meilleurs vecteurs propres
best_eigenvectors = eigenvectors[:, :best_n]

# Affichage
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(best_eigenvectors), aspect='auto', cmap='viridis_r')
plt.colorbar(label='Valeur du vecteur propre')
plt.title(f'{best_n} vecteurs propres correspondant au meilleur score DB')
plt.xlabel('Composante principale')
plt.ylabel('Valeurs du vecteur')
plt.tight_layout()
plt.show()

# Affichage sans les valeurs absolues
plt.figure(figsize=(10, 6))
plt.imshow(best_eigenvectors, aspect='auto', cmap='viridis_r')
plt.colorbar(label='Valeur du vecteur propre')
plt.title(f'{best_n} vecteurs propres correspondant au meilleur score DB')
plt.xlabel('Composante principale')
plt.ylabel('Valeurs du vecteur')
plt.tight_layout()
plt.show()

##########################################################################################
# PRECUPERER LES VARIABLE AU DESSUS DU SEUIL
# Valeurs absolues des vecteurs propres (pour éviter les effets de signe)
abs_eigenvectors = np.abs(best_eigenvectors)

# Moyenne des valeurs par vecteur propre (i.e., moyenne par colonne)
max_per_vector = abs_eigenvectors.max(axis=0)

# On crée une matrice booléenne : True là où la valeur dépasse la moyenne de sa colonne
above_avg_mask = abs_eigenvectors > 0.85 * max_per_vector

# On garde uniquement les valeurs qui dépassent cette moyenne (autres mises à 0)
eigenvectors_thresholded = np.where(above_avg_mask, best_eigenvectors, 0)

# Affichage des vecteurs propres "seuillés"
plt.figure(figsize=(12, 6))
plt.imshow(np.abs(eigenvectors_thresholded), aspect='auto', cmap='magma')
plt.colorbar(label="Valeurs des vecteurs propres (seuillées)")
plt.title("Valeurs des vecteurs propres au-dessus de la moyenne de leur colonne")
plt.xlabel("Index du vecteur propre")
plt.ylabel("Dimensions (features)")
plt.tight_layout()
plt.show()

# Affichage des indices retenus
# Récupérer tous les indices non nuls dans la matrice seuillée
all_indices = set()

for i in range(eigenvectors_thresholded.shape[1]):
    indices = np.where(eigenvectors_thresholded[:, i] != 0)[0]
    all_indices.update(indices)

# Conversion en liste triée
all_indices_sorted = sorted(all_indices)

# Grouper les indices par tranche de 10 (ou 11 si tu veux vraiment des onzaines)
grouped_indices = defaultdict(list)

# Taille d'un "bloc"
group_size = 11

# Grouper les indices par tranche de 11 et centrer dans leur tranche
grouped_indices = defaultdict(list)
for idx in all_indices_sorted:
    group_key = (idx // group_size) * group_size
    grouped_indices[group_key].append(idx - group_key)

# Générer une "matrice" pour un affichage aligné
matrix = []
for group_start in sorted(grouped_indices.keys()):
    group = grouped_indices[group_start]
    row = ['  '] * group_size
    for val in group:
        if val < group_size:
            row[val] = str(val).rjust(2)
    matrix.append(row)

# Affichage avec légende à gauche
print("Composantes conservées groupées :\n")
for i, row in enumerate(matrix):
    line = " ".join(row)
    print(f"mfcc_{i} = {line}")

##########################################################################################
# REPROJETER SUR LES NOUVEAUX VECTEUR PROPRES ET NOUVEAU SCORE
j = 3  
X_proj_thresh = X_scaled @ eigenvectors_thresholded[:, :j]

# Mapping label
label_mapping = {0: "l", 1: "a", 2: "s", 3: "t", 4: "r"}
block_labels_text = [label_mapping[label] for label in block_labels_balanced]

# Couleurs
colors = {"l": "blue", "a": "green", "s": "red", "t": "orange", "r": "purple"}
class_colors = [colors[label_mapping[l]] for l in block_labels_balanced]

# Projection 3D sur les vecteurs propres seuillés
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Affichage des points avec couleurs personnalisées
scatter = ax.scatter(
    X_proj_thresh[:, 0],
    X_proj_thresh[:, 1],
    X_proj_thresh[:, 2],
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
               markerfacecolor="deeppink", markeredgecolor="k",
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
df_proj_thresh.to_csv("X_proj_scaled_avec_labels_corrige_avant.csv", index=False)
print("Fichier 'X_proj_scaled_avec_labels_corrige_avant.csv' sauvegardé.")

# Version troncaturée : on enlève les 55 dernières lignes (i.e. dimensions MFCC)
eigenvectors_truncated = eigenvectors_thresholded[:-55, :j]

# Sauvegarde de la version troncaturée
joblib.dump(eigenvectors_truncated, "eigenvectors_thresholded_corrige_avant_tronque.pkl")
print("Vecteurs propres seuillés tronqués sauvegardés dans 'eigenvectors_thresholded_corrige_avant_tronque.pkl'")

##########################################################################################
# ENTRAINEMENT
# Reconvertir les labels texte en entiers pour l'entraînement
block_labels_int = [k for l in block_labels_text for k, v in label_mapping.items() if v == l]

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
knn.fit(X_train, y_train)

# Évaluation
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision finale du modèle KNN sur test : {accuracy * 100:.2f}%")

# Sauvegarde du modèle KNN
joblib.dump(knn, "knn_model_db_sans_r_opt_main_corrige_avant.pkl")
print(" Modèle KNN sauvegardé dans 'knn_model_db_sans_r_opt_main_corrige_avant.pkl'")

# MATRICE DE CONFUSION
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["l", "a", "s", "t"],
    cmap="Blues"
)
plt.title("Matrice de Confusion du modèle KNN (avec PCA)")
plt.show()