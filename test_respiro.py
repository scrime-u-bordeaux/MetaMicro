from pynput import keyboard
import rtmidi

# Notes de la gamme de Do majeur sur les touches AZERTY (C D E F G A B C)
key_to_midi = {
    'a': 60,  # C4
    'z': 62,  # D
    'e': 64,  # E
    'r': 65,  # F
    't': 67,  # G
    'y': 69,  # A
}

# MIDI out config
midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()
print(available_ports)

# Choisis ton port virtuel ici
if available_ports:
    # Remplace "loopMIDI" ou "IAC" par le nom de ton port MIDI virtuel
    for i, port in enumerate(available_ports):
        if "loopMIDI" in port or "IAC" in port or "virtual" in port.lower():
            midiout.open_port(i)
            break
    else:
        midiout.open_port(0)
else:
    midiout.open_virtual_port("Virtual Keyboard")

pressed_keys = set()

def on_press(key):
    print("pressed")
    try:
        k = key.char.lower()
        if k in key_to_midi and k not in pressed_keys:
            note = key_to_midi[k]
            midiout.send_message([0x90, note, 100])
            midiout.send_message([0xB0, 2, 100])
            pressed_keys.add(k)
    except AttributeError:
        pass

def on_release(key):
    try:
        k = key.char.lower()
        if k in key_to_midi:
            note = key_to_midi[k]
            midiout.send_message([0x80, note, 0])  # Note OFF
            pressed_keys.discard(k)
    except AttributeError:
        pass
    if key == keyboard.Key.esc:
        return False

print("Joue avec les touches A Z E R T Y. Appuie sur ESC pour quitter.")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()