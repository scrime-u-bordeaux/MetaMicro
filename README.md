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

pip install -r requirements.txt


---

### Installer les dépendances système

#### Installer `portaudio` (nécessaire pour `pyaudio`)

- **Linux (Debian/Ubuntu)** :

sudo apt install portaudio19-dev


- **macOS** :

brew install portaudio


- **Windows** (avec Chocolatey) :

choco install portaudio


#### Installer `fluidsynth` (nécessaire pour la synthèse MIDI)

- **Linux (Debian/Ubuntu)** :

sudo apt install fluidsynth


- **macOS** :

brew install fluidsynth


- **Windows** (avec Chocolatey) :

choco install fluidsynth


#### Télécharger et ajouter la SoundFont `FluidR3_GM.sf2`

Si l'installation de fluidsynth a échoué :

1. Téléchargez le fichier depuis ce lien :  
   https://member.keymusician.com/Member/FluidR3_GM/index.html

2. Placez-le dans le dossier suivant :  
   `/usr/share/sounds/sf2/FluidR3_GM.sf2`

---

## Configuration de l'Audio

### Vérifications importantes

- Assurez-vous que :
  - l’entrée se fait via un **microphone**
  - la sortie se fait via un **casque**
- Les **ports d’entrée et de sortie doivent être différents**

### Changer la source et la sortie audio sous Linux

Utilisez `pactl` :

pactl set-default-source <id_microphone>
pactl set-default-sink <id_casque>


Lister les sources disponibles :

pactl list sources short


Lister les sorties disponibles :

pactl list sinks short


---

## Exécution

Lancez l’application avec :

python3 main.py


### Lancement de l'application

Une première fenêtre vous demande si vous êtes sur **Linux** ou **Mac**.  
Ensuite, vous pouvez choisir entre deux options :

- **Entraînement**
- **Tester le Meta Micro**

---

### Entraînement

#### 1. Choix des lettres à entraîner

Choisissez parmi :

a, i, u (= le son "ou"), n, l


Le son **t** est automatiquement inclus.

> ✅ Il est **recommandé de ne pas sélectionner plus de 3 lettres** à la fois.

#### 2. Enregistrement vocal

Un tutoriel intégré vous guide pour enregistrer votre voix avec les sons choisis.

#### 3. Analyse et concaténation

Une fenêtre d’analyse s’ouvre automatiquement :

- Cliquez sur **"Concaténer"**
- Puis sur **"Lancer le script"**

Trois fenêtres s’affichent pour effectuer les calculs.  
> ⚠️ La **2e fenêtre peut être lente à apparaître**. Attendez sans fermer les autres.

#### 4. Sauvegarde des fichiers

À la fin de chaque étape, il vous sera demandé **où enregistrer les fichiers** :

- Enregistrez-les dans un **dossier d'entraînement**
- Un dossier par défaut est proposé à la racine du projet
- Vous pouvez créer un autre dossier si besoin

> ⚠️ **Ne fermez pas les fenêtres** tant qu’un message indique que les fichiers ont bien été enregistrés.

#### 5. Ajout du fichier `.yaml`

Les paramètres d'entraînement sont sauvegardés dans un fichier `.yaml`.

- Cliquez sur **"Ajouter au dossier"** pour le lier au dossier d'entraînement.

---

### Tester le Meta Micro

Après l'entraînement, cliquez sur **"Tester le Meta Micro"** :

1. Sélectionnez le **dossier d'entraînement**
2. Vous pouvez modifier les **paramètres du fichier `.yaml`**

> ℹ️ Tous les paramètres sont expliqués à l’aide de **boutons "i"** dans l'interface.

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

### Fluidsynth : "Failed to set thread to high priority"

sudo setcap 'cap_sys_nice=eip' $(which fluidsynth)


### ALSA : "unable to open slave"

- **Linux** :

sudo apt install alsa-utils


- **macOS** :

brew install alsa-utils


- **Windows** :

choco install alsa-utils


### JACK : "connect(2) failed"

- **Linux** :

sudo apt install jackd2


- **macOS** :

brew install jackd2


- **Windows** :

choco install jackd2