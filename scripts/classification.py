import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

##########################################################################################
# CHARGER LES DONNNEES 
mfcc_matrix = pd.read_csv("mfcc_features.csv")

# Encoder les labels (t=0, a=1, s=2)
label_mapping = {"t": 0, "a": 1, "s": 2}
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
    density_scores = get_density_scores(X_class, k=5)  
    sorted_idx = np.argsort(density_scores) # np.argsort() retourne les indices tries par ordre croissant des valeurs de density_scores
    selected_idx = sorted_idx[:min_class_size]  

    X_balanced.append(X_class[selected_idx])
    block_labels_balanced.extend([label] * min_class_size)

# Conversion en numpy array
X_balanced = np.vstack(X_balanced)
block_labels_balanced = np.array(block_labels_balanced).astype(int) 

##########################################################################################
# CLASSIFICATION AVEC KNN
X_train, X_test, y_train, y_test = train_test_split(X_balanced, block_labels_balanced, test_size=0.4, stratify=block_labels_balanced, random_state=42)

# Sélection du meilleur k et de la meilleure distance
best_k = None
best_distance = None
best_score = 0

k_values = range(5, 50) 
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

print(f" Meilleur k : {best_k}, Meilleure distance : {best_distance} avec précision de {best_score:.2f}")

knn.fit(X_train, y_train)

# Prédiction et évaluation
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Précision du modèle KNN : {accuracy * 100:.2f}%")

# Sauvegarde du modèle entraîné
joblib.dump(knn, "knn_model.pkl")

##########################################################################################
# AFFICHAGE MATRICE DE CONFUSION
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["t", "a", "s"])
disp.plot(cmap="Blues")
plt.title("Matrice de Confusion du modèle KNN")
plt.show()
