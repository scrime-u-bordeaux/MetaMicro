import matplotlib.pyplot as plt
import random
import numpy as np
import time

# Mode interactif
plt.ion()

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1, 1)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')  # Pas d'axes visibles

# Positions horizontales : gauche et droite
positions = [
    (-0.8, 0.5),    # Gauche
    (0.8, 0.5),     # Droite
]

letters = ["A", "I", "T", "S"]
text_objects = []

# Créer deux emplacements pour les lettres
for pos in positions:
    txt = ax.text(
        pos[0], pos[1],
        "",              # Vide au départ
        fontsize=50,
        ha='center',
        va='center'
    )
    text_objects.append(txt)

# Créer l’aiguille (ligne)
# Elle partira de (0, -1) et bougera son extrémité
needle, = ax.plot([0, 0], [-1, -0.5], lw=3, color='red')

plt.show()

# Durée d’un cycle d’oscillation (en secondes)
cycle_duration = 2

while True:
    # Tirer 2 lettres aléatoires
    displayed_letters = random.sample(letters, 2)
    print(displayed_letters)

    # Mettre à jour l’affichage des lettres
    for i, txt in enumerate(text_objects):
        txt.set_text(displayed_letters[i])

    # Animer l’aiguille pendant un cycle complet
    t_start = time.time()
    while (time.time() - t_start) < cycle_duration:
        elapsed = time.time() - t_start

        # Calculer la position de l’extrémité de l’aiguille
        x_tip = np.cos(elapsed * np.pi / cycle_duration * np.pi) * 0.8
        y_tip = -0.2  # hauteur de l’extrémité
        # Mettre à jour la ligne (aiguille)
        needle.set_data([0, x_tip], [-1, y_tip])

        plt.draw()
        plt.pause(0.01)  # Pause courte pour animation fluide