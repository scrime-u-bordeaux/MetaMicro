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

---

## Exécution

Lancez l’application avec :
```bash
python3 main.py
```

### Lancement de l'application

Une première fenêtre vous demande si vous êtes sur **Linux** ou **Mac**.  
Ensuite, vous pouvez choisir entre deux options :

- **Entraînement**
- **Tester le Meta Micro**

---

### Entraînement

#### 1. Choix des lettres à entraîner

Choisissez parmi :

- a
- i
- u (= le son "ou")
- n
- l
> Le son **t** est automatiquement inclus.
> Il est **recommandé de ne pas sélectionner plus de 3 lettres** à la fois.

#### 2. Enregistrement vocal

Un tutoriel intégré vous guide pour enregistrer votre voix avec les sons choisis. Il est conseillé de faire l'entrainement avec un gain d'entrée élevé.

#### 3. Analyse

Une fenêtre d’analyse s’ouvre automatiquement 

#### 4. Concaténation et script

- Cliquez sur **"Concaténer"**
- Puis sur **"Lancer le script"**

Trois fenêtres s’affichent pour effectuer les calculs.  
> La **2e fenêtre peut être lente à apparaître**.

- Sur Mac, vous pouvez observer des figures détaillant les calculs
- Sur Linux, vous pouvez les récupérer dans le dossier img/

#### 5. Sauvegarde des fichiers

À la fin de chaque étape, il vous sera demandé **où enregistrer les fichiers** :

- Enregistrez-les dans un **dossier d'entraînement**
- Un dossier par défaut est proposé à la racine du projet
- Vous pouvez créer un autre dossier si besoin
> **Ne fermez pas les fenêtres** tant qu’un message indique que les fichiers ont bien été enregistrés.

#### 6. Ajout du fichier `.yaml`

Les paramètres d'entraînement sont sauvegardés dans un fichier `.yaml`.

- Cliquez sur **"Ajouter au dossier"** pour le lier au dossier d'entraînement.

---

### Tester le Meta Micro

Avant tout test, enregistez une premiere fois les paramètres car certains peuvent être différents.

Ensuite, cliquez sur **"Tester le Meta Micro"**
Ensuite vous pouvez choisir entre:

1. Sélectionnez le **dossier d'entraînement**
2. Vous pouvez modifier les **paramètres du fichier `.yaml`**
> Tous les paramètres sont expliqués à l’aide de **boutons "i"** dans l'interface.

---

### Modes de test selon le système

#### Sur macOS :

- **Tester votre souffle maximal**
- **Utiliser l'accordeur de voyelle**
- **Tester sans accordeur**
- Utiliser des logiciels comme **Respiro**

#### Sur Linux :

- **Tester avec ou sans accordeur**
- Possibilité d’utiliser **FluidSynth**

---

## Debugging des erreurs courantes

### **Vérifications Importantes**
- Vérifiez que l’entrée se fait bien via un **microphone** et la sortie via un **casque**.
- Les **ports d’entrée et de sortie doivent être différents**.
- Vérifier que vous aviez bien **ajouté le fichier yaml** au dossier et que vous avez bien **enregistré les modifications**.

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

###  Problème : "ImportError: No module named 'tkinter"
- **Linux (Debian/Ubuntu)** :
  ```bash
  sudo apt install python3-tk
  ```
- **macOS** :
  ```bash
  brew install python-tk
  ```
- **Windows** :