## Prérequis

Avant de lancer, assurez-vous d'avoir :

- **Python 3.10+** installé.
- Un **microphone** configuré comme **entrée audio**.
- Un **casque** utilisé comme **sortie audio** 
- Des **ports d’entrée et de sortie différents** (exemple : Micro USB et Casque Jack).

---

## Installation

### 1 Installer Python et les bibliothèques requises

Installez les dépendances avec la commande :
```bash
pip install pygame pyaudio mido fluidsynth librosa numpy pandas matplotlib joblib xgboost scipy
```
---

## Configuration de l'Audio

**Important** :  
- Vérifiez que l’entrée se fait bien via un micro et la sortie via un casque.
- Les ports d’entrée et de sortie doivent être différents.

### Changer la source et la sortie audio sur Linux :
Si vous utilisez Linux, vous pouvez configurer les ports avec `pactl` :
```bash
pactl set-default-source <id_microphone>
pactl set-default-sink <id_casque>
```
Pour obtenir la liste des sources :
```bash
pactl list sources short
```
Pour obtenir la liste des sorties :
```bash
pactl list sinks short
```

---

## Exécution

Lancez l’application avec :
```bash
python3 main.py
```
---

## Changer l'Instrument MIDI

L’instrument utilisé par FluidSynth peut être modifié dans `main.py` en modifiant cette ligne :
```python
instrument_number = 73  # 73 = Flûtee
fluid.program_select(0, sfid, 0, instrument_number)
```
Voir la liste des instruments ici : [General MIDI](https://fr.wikipedia.org/wiki/General_MIDI)

---

## Changer la mélodie MIDI

La mélodie utilisée par Mido peut être modifiée dans `main.py` en modifiant cette ligne :
```python
# Charger le fichier MIDI
midi_file = mido.MidiFile("midi/amstrong.mid")
```
La liste des mélodies se trouve dans le dossier midi au même endroit ou vous avez ouvert ce README

---
