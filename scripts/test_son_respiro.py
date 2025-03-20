import numpy as np
import matplotlib.pyplot as plt 

t = np.linspace(0, 1, 1000)  
signal = np.sin(2 * np.pi * 5 * t)  

instant_t = 600  
window_size = 50
start = max(0, instant_t - window_size // 2)
end = min(len(signal), instant_t + window_size // 2)
rms = np.sqrt(np.mean(signal[start:end] ** 2))
print(f"Valeur RMS du signal à l'instant {instant_t}: {rms}")

# Affichage du signal
plt.figure(figsize=(10, 5))
plt.plot(t, signal, label="Signal")
plt.axvline(x=t[instant_t], color='r', linestyle='--', label=f"Instant t={instant_t}")
plt.axhline(y=rms, color='b', linestyle='--', label=f"RMS à t={instant_t}")
plt.scatter([t[instant_t]], [rms], color='blue', zorder=3, label=f"RMS={rms:.2f}")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signal et valeur RMS à un instant t")
plt.legend()
plt.grid()
plt.show()