import tkinter as tk
from tkinter import ttk
import threading
import pyaudio
import numpy as np
import yaml
import mido

# Charger YAML
yaml_path = "parametre.yaml"
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

main_respiro_param = config["main_respiro"]
rms_max = 0

# Initialisation PyAudio
p = pyaudio.PyAudio()
RATE = main_respiro_param["RATE"]
CHANNELS = main_respiro_param["CHANNELS"]
CHUNK = int((RATE * 0.005))
FORMAT = pyaudio.paInt16

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

##########################################################################################
# FENÊTRE TKINTER 
root = tk.Tk()
root.title("Soufflez fort")
root.geometry("400x200")
root.configure(bg="#2c3e50")

label_title = tk.Label(
    root,
    text="Soufflez le plus fort possible",
    font=("Arial", 16, "bold"),
    bg="#2c3e50",
    fg="#ecf0f1"
)
label_title.pack(pady=10)

rms_bar = ttk.Progressbar(
    root,
    orient="horizontal",
    length=300,
    mode="determinate",
    maximum=1000
)
rms_bar.pack(pady=10)

rms_max_label = tk.Label(
    root,
    text=f"RMS max : {rms_max}",
    font=("Arial", 14),
    bg="#2c3e50",
    fg="#f1c40f"
)
rms_max_label.pack(pady=10)

def save_yaml():
    # S’assurer que rms_max est un float natif
    if isinstance(main_respiro_param["rms_max"], np.generic):
        main_respiro_param["rms_max"] = float(main_respiro_param["rms_max"])
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, sort_keys=False, allow_unicode=True)

def update_gui(rms_value, rms_max_value):
    rms_bar["value"] = rms_value
    rms_max_label.config(text=f"RMS max : {rms_max_value}")

def audio_loop():
    global rms_max
    try:
        print("Go")
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))

            # Mettre à jour l'affichage graphique (dans le thread principal)
            root.after(0, update_gui, rms, rms_max)

            # Mettre à jour rms_max si nécessaire
            if rms > rms_max:
                rms_max = rms
                print(rms_max)
                main_respiro_param["rms_max"] = rms_max
                save_yaml()
    except Exception as e:
        print(f"Erreur audio : {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        root.quit()

# Lancer la logique audio dans un thread séparé
threading.Thread(target=audio_loop, daemon=True).start()

# Lancer Tkinter (dans le thread principal)
root.mainloop()