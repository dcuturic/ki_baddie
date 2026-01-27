import random
import time
from pythonosc.udp_client import SimpleUDPClient

# ============================
# 🔧 GLOBALE PARAMETER
# ============================

OSC_HOST = "127.0.0.1"
OSC_PORT = 39539

# Geschwindigkeit (Sekunden)
SYLLABLE_MIN_DELAY = 0.1   # kleiner = schneller reden
SYLLABLE_MAX_DELAY = 0.3

# Mundöffnung
MOUTH_MIN_OPEN = 0.6        # wie weit mindestens
MOUTH_MAX_OPEN = 1.0        # maximal (1.0 = voll)

# Pausen (natürlicher Effekt)
PAUSE_CHANCE = 0.12         # 12% Chance auf Pause
PAUSE_MIN = 0.2
PAUSE_MAX = 0.5

# Reset-Option
CLEAR_BEFORE_EACH = True    # Mund vor jeder Silbe schließen

# ============================
# OSC Setup
# ============================

client = SimpleUDPClient(OSC_HOST, OSC_PORT)

# Gewichte: A & O dominieren -> offener Mund
VISEMES = [
    ("A", 1.0),
    ("O", 0.9),
    ("E", 0.6),
    ("I", 0.5),
    ("U", 0.4),
]

# ============================
# Funktionen
# ============================

def set_viseme(key: str, value: float):
    client.send_message("/VMC/Ext/Blend/Val", [key, float(value)])
    client.send_message("/VMC/Ext/Blend/Apply", 1)

def clear_mouth():
    for k, _ in VISEMES:
        client.send_message("/VMC/Ext/Blend/Val", [k, 0.0])
    client.send_message("/VMC/Ext/Blend/Apply", 1)

# ============================
# Hauptloop
# ============================

while True:
    # Optionale Pause (wirkt sehr natürlich)
    if random.random() < PAUSE_CHANCE:
        time.sleep(random.uniform(PAUSE_MIN, PAUSE_MAX))

    if CLEAR_BEFORE_EACH:
        clear_mouth()

    # neue "Silbe"
    key, weight = random.choice(VISEMES)

    # Mundöffnung berechnen
    openness = random.uniform(MOUTH_MIN_OPEN, MOUTH_MAX_OPEN) * weight

    set_viseme(key, openness)

    # Sprechtempo
    time.sleep(random.uniform(SYLLABLE_MIN_DELAY, SYLLABLE_MAX_DELAY))
