import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

##########################################################################################
# CHARGER LES DONNNEES 
# Charger le tableau de predictions
df_predictions = pd.read_csv("predictions_panthere.csv")

# Charger l'audio
audio_file = "audio/panthere_test.wav"
signal, fs = librosa.load(audio_file, sr=None)

##########################################################################################
# RAJOUT DE ON OFF
# Convertir en tableau (liste de tuples)
tableau_predictions = list(zip(df_predictions["Time (s)"], df_predictions ["Label"]))

events = []
previous_event = None  

# Parcourir les prédictions pour détecter les transitions
# passe bas
for i in range(2, len(df_predictions)):  
    prev_prev_label = df_predictions.loc[i-2, "Label"]
    prev_label = df_predictions.loc[i-1, "Label"]  
    current_label = df_predictions.loc[i, "Label"]
    current_time = df_predictions.loc[i, "Time (s)"]

    # ON : Transition "s" → "t"
    if ("s" in [prev_label, prev_prev_label]) and current_label == "t": # Pour eviter de prendre en compte les "a" seuls
        print("prev_prev_label:", prev_prev_label, "prev_label:", prev_label, "current_label:", current_label)
        if not events or events[-1][2] != i - 1:   # Pour eviter d'avoir des ON qui se suivent 
            events.append(("ON", current_time, i))
            previous_event = "ON" 

    # OFF : Transition "a" → "s" 
    if ("a" in [prev_label, prev_prev_label]) and current_label == "s":
        if previous_event == "ON": 
            events.append(("OFF", current_time, i))
            previous_event = "OFF" 

# Convertir en DataFrame
df_events = pd.DataFrame(events, columns=["Event", "Time (s)", "Number"])
print(df_events)

##########################################################################################
# AFFICHAGE

# Créer l'axe temporel pour l'affichage du signal
time = np.linspace(0, len(signal) / fs, num=len(signal))

# Affichage du signal audio
plt.figure(figsize=(12, 4))
librosa.display.waveshow(signal, sr=fs, alpha=0.6, color="black")

# Ajouter les barres pour les événements ON et OFF
for _, row in df_events.iterrows():
    event_type = row["Event"]
    event_time = row["Time (s)"]
    
    if event_type == "ON":
        plt.axvline(x=event_time, color="blue", linestyle="--", label="ON" if "ON" not in plt.gca().get_legend_handles_labels()[1] else "")
    elif event_type == "OFF":
        plt.axvline(x=event_time, color="red", linestyle="--", label="OFF" if "OFF" not in plt.gca().get_legend_handles_labels()[1] else "")

# Ajouter la légende et les labels
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signal Audio avec Indicateurs ON/OFF")
plt.legend()
plt.grid()
plt.show()

# # Sauvegarder les résultats
# df_events.to_csv("events.csv", index=False)
# print(" Fichier 'events.csv' )