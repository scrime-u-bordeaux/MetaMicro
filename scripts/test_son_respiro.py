import time
import mido
import mido

port_name = "Gestionnaire IAC Bus 1" 
midi_out = mido.open_output(port_name)

# Jouer une note MIDI (C4 = 60)
midi_out.send(mido.Message('note_on', note=60, velocity=100))
time.sleep(1)  # Maintient la note pendant 1 seconde
midi_out.send(mido.Message('note_off', note=60))

# Modifier la vélocité (dynamique)
for vel in range(50, 127, 10):
    midi_out.send(mido.Message('control_change', control=2, value=vel))
    time.sleep(0.1)

# Fermer la connexion MIDI
midi_out.close()