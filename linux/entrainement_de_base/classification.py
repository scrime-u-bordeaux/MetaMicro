import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import davies_bouldin_score
from joblib import Parallel, delayed
from collections import defaultdict

##########################################################################################
# CHARGER LES DONNNEES 
dossier = "data_u_ta_la_ti_li_i_n/"
mfcc_matrix = pd.read_csv(dossier + "/mfcc_features_corrige.csv")

# Encoder les labels (t=0, a=1, s=2)
label_mapping = {"a": 1, "n": 2, "u": 3, "i":4, "t": 5, "s": 6, "l": 7}
mfcc_matrix["label"] = mfcc_matrix["label"].map(label_mapping)

# Séparer les features et labels
X = mfcc_matrix.iloc[:, :-2].values  
block_labels = mfcc_matrix["label"].values.astype(int) 

##########################################################################################
# TROUVER LES POINTS LES PLUS ELOIGNE
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
# Centrage et normalisation des données
X_centered = X_balanced - np.mean(X_balanced, axis=0)
X_scaled = X_centered / np.std(X_balanced, axis=0)

# Sauvegarde de la moyenne et de l'écart-type
mean_X = np.mean(X_balanced, axis=0)
std_X = np.std(X_balanced, axis=0)
joblib.dump(mean_X, dossier + "/mfcc_mean.pkl")
joblib.dump(std_X, dossier + "/mfcc_std.pkl")

# Calcul de la matrice de covariance
cov_matrix = np.cov(X_scaled.T)

# Calcul des valeurs et vecteurs propres
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Affichage des vecteurs propres
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

# Fonction pour une projection et le calcul du score de Davies-Bouldinné
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

# Sélection du meilleur nombre de dimensions selon le score de Davies-Bouldin
best_n = np.nanargmin(scores) + 1   # +1 car l'index 0 correspond à n=1
best_score = scores[best_n - 1]

print(f"Meilleur score DB : {best_score:.4f} atteint pour n = {best_n} dimensions.")

# Changement de best score pour l'affichage sur le score vaut 2
if best_n == 2:
    best_n = 3

# Réduction des vecteurs propres aux nombres de dimensions sélectionnés
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

# Moyenne des valeurs par vecteur propre (i.e., moyenne par colonne)
max_per_vector = abs_eigenvectors.max(axis=0)

# On crée une matrice booléenne : True là où la valeur dépasse la moyenne de sa colonne
suppression_mfcc = False
if suppression_mfcc:
    above_avg_mask = abs_eigenvectors > 0.45 * max_per_vector

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
j = best_n
X_proj_thresh = X_scaled @ eigenvectors_thresholded[:, :j]

# Mapping label
label_mapping_reverse = {1: "a", 2: "n", 3: "u", 4: "i", 5: "t", 6: "s", 7: "l"}
block_labels_text = [label_mapping_reverse[label] for label in block_labels_balanced]

# Couleurs
colors = {"a": "green", "s": "red", "t": "orange", "n": "brown", "u": "blue", "i": "purple", "l": "deeppink"}
class_colors = [colors[label_mapping_reverse[l]] for l in block_labels_balanced]

# Projection 3D sur les vecteurs propres seuillés
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = X_proj_thresh[:, 0]
y = X_proj_thresh[:, 1]
z = X_proj_thresh[:, 2]

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
# Créer un DataFrame avec la projection seuillée 
df_proj_thresh = pd.DataFrame(X_proj_thresh, columns=[f"comp_{i+1}" for i in range(X_proj_thresh.shape[1])])
df_proj_thresh["label"] = block_labels_text

# Sauvegarde en CSV
df_proj_thresh.to_csv(dossier + "X_proj_scaled_avec_labels_corrige_avant.csv", index=False)
print("Fichier 'X_proj_scaled_avec_labels_corrige_avant.csv' sauvegardé.")

# Sauvegarde des vecteurs propres seuillés
joblib.dump(eigenvectors_thresholded[:, :j], dossier + "eigenvectors_thresholded_corrige_avant.pkl")
print("Vecteurs propres seuillés sauvegardés dans 'eigenvectors_thresholded_corrige_avant.pkl'")

##########################################################################################
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
    joblib.dump(eigenvectors_truncated, dossier + "eigenvectors_thresholded_corrige_avant_tronque.pkl")
    print("Vecteurs propres seuillés tronqués sauvegardés dans 'eigenvectors_thresholded_corrige_avant_tronque.pkl'")

    print("eigenvectors_truncated:", eigenvectors_truncated)

##########################################################################################
# ENTRAINEMENT
# Reconvertir les labels texte en entiers pour l'entraînement
block_labels_int = [k for l in block_labels_text for k, v in label_mapping_reverse.items() if v == l]

# Séparation des données en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X_proj_thresh,
    block_labels_int,
    test_size=0.4,
    stratify=block_labels_int,
    random_state=42
)

# Recherche des meilleurs paramètres KNN
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

# Prédiction knn pour la matrice de confusion
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Précision finale du modèle KNN sur test : {accuracy * 100:.2f}%")

joblib.dump(knn, dossier + "knn_model_db_sans_r_opt_main_corrige_avant.pkl")
print(" Modèle KNN sauvegardé dans 'knn_model_db_sans_r_opt_main_corrige_avant.pkl'")

# Affichage de la matrice de confusion <-- à modfier selon les lettrs choisies
if dossier == "data_u_ta_la_ti_li_i_n/": 
    label_for_display = ["a", "n", "u", "i", "t", "s"] 
else:
    label_for_display = ["a", "i", "t", "s", "l"]

disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=label_for_display,
    cmap="Blues"
)
plt.title("Matrice de Confusion du modèle KNN (avec PCA)")
plt.show()
