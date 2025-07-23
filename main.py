import tkinter as tk
from tkinter import font
import subprocess

# Lancement d'un script Python externe
def lancer_linux():
    subprocess.run(["python3", "linux/entrainement_et_meta_micro/main.py"])

def lancer_mac():
    subprocess.run(["python3", "mac/entrainement_et_meta_micro/main.py"])

# Création de la fenêtre principale
root = tk.Tk()
root.title("MetaMicro - Lancement")
root.configure(bg="#2c3e50")
root.geometry("400x250")

# Polices
title_font = font.Font(family="Arial", size=18, weight="bold")
button_font = font.Font(family="Arial", size=14)

# Titre
label = tk.Label(root, text="Sur quel OS êtes vous ?", font=title_font, bg="#2c3e50", fg="black", pady=20)
label.pack()

# Boutons
btn_train = tk.Button(root, text="Mac", font=button_font, command=lancer_mac,
                      bg="#27ae60", fg="black", activebackground="#2ecc71", padx=10, pady=10)
btn_train.pack(pady=10, fill="x", padx=40)

btn_test = tk.Button(root, text="Linux", font=button_font, command=lancer_linux,
                     bg="#2980b9", fg="black", activebackground="#3498db", padx=10, pady=10)
btn_test.pack(pady=10, fill="x", padx=40)

# Boucle principale
root.mainloop()