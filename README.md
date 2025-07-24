# Configuration et Installation

## Prérequis

Avant de lancer, assurez-vous d'avoir :

- **Python 3.10+** installé.
- Un **microphone** configuré comme **entrée audio**.
- Un **casque** utilisé comme **sortie audio**.
- Des **ports d’entrée et de sortie différents** (exemple : Micro USB et Casque Jack).

---

## Installation

### Installer les bibliothèques Python requises

```bash
pip install -r requirements.txt
  ```
---

### Installer les dépendances système

#### **Installer `portaudio` (nécessaire pour `pyaudio`)**
- **Linux (Debian/Ubuntu)** :
  ```bash
  sudo apt install portaudio19-dev
  ```
- **macOS** :
  ```bash
  brew install portaudio
  ```
- **Windows** (avec Chocolatey) :
  ```powershell
  choco install portaudio
  ```

#### **Installer `fluidsynth` (nécessaire pour la synthèse MIDI)**
- **Linux (Debian/Ubuntu)** :
  ```bash
  sudo apt install fluidsynth
  ```
- **macOS** :
  ```bash
  brew install fluidsynth
  ```
- **Windows** (avec Chocolatey) :
  ```powershell
  choco install fluidsynth
  ```

#### **Télécharger et ajouter la SoundFont FluidR3_GM.sf2**

Si l'installation de fluidsynth à échoué: 

Téléchargez le fichier `FluidR3_GM.sf2` depuis ce lien :

[FluidR3_GM.sf2](https://member.keymusician.com/Member/FluidR3_GM/index.html)

Puis placez-le suivant ce chemin: **/usr/share/sounds/sf2/FluidR3_GM.sf2**.

---

## Configuration de l'Audio

### **Vérifications Importantes**
- Vérifiez que l’entrée se fait bien via un **microphone** et la sortie via un **casque**.
- Les **ports d’entrée et de sortie doivent être différents**.

### **Changer la source et la sortie audio sous Linux**
Si vous utilisez Linux, vous pouvez configurer les ports avec `pactl` :
```bash
pactl set-default-source <id_microphone>
pactl set-default-sink <id_casque>
```
Obtenir la liste des sources :
```bash
pactl list sources short
```
Obtenir la liste des sorties :
```bash
pactl list sinks short
```

### **Définir l'indice du flux audio**
Dans le code, vous pouvez spécifier l'indice du flux audio en utilisant :
```python
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=<indice_du_micro>,
    frames_per_buffer=CHUNK
)
```

---

## Exécution

Lancez l’application avec :
```bash
python3 main.py
```

---

## Debugging des erreurs courantes

### Problème : Fluidsynth "Failed to set thread to high priority"
```bash
sudo setcap 'cap_sys_nice=eip' $(which fluidsynth)
```
### Problème : ALSA "unable to open slave"
- **Linux (Debian/Ubuntu)** :
  ```bash
  sudo apt install alsa-utilsh
  ```
- **macOS** :
  ```bash
  brew install alsa-utilsh
  ```
- **Windows** :
  ```powershell
  choco install alsa-utilsh
  ```

###  Problème : JACK "connect(2) failed"
- **Linux (Debian/Ubuntu)** :
  ```bash
  sudo apt install jackd2
  ```
- **macOS** :
  ```bash
  brew install jackd2
  ```
- **Windows** :
  ```powershell
  choco install jackd2
  ```
---















































