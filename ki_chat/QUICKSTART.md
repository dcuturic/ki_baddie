# Multi-Character System - Quick Start

## 🎯 Was wurde implementiert?

Das ki_chat System unterstützt jetzt **mehrere unabhängige Characters** mit:
- ✅ Eigene Datenbanken (memory_*.db)
- ✅ Eigene Persönlichkeiten
- ✅ Eigene Erinnerungen & Gedanken
- ✅ Hot-Switching zur Laufzeit
- ✅ Komplett isoliert voneinander

---

## 🚀 Schnellstart

### 1. Server starten
```bash
cd ki_chat
python app.py
```

### 2. Characters auflisten
```bash
curl http://localhost:5001/characters
```

**Response:**
```json
{
  "characters": ["dilara", "alex", "luna", "techbot"],
  "current": "dilara"
}
```

### 3. Character wechseln
```bash
curl -X POST http://localhost:5001/character/switch \
  -H "Content-Type: application/json" \
  -d '{"character": "alex"}'
```

### 4. Chat mit aktuellem Character
```bash
curl -X POST http://localhost:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "user123:hey wie gehts?"}'
```

---

## 📦 Mitgelieferte Characters

### 1. **Dilara** (Horror-Yandere-Streamerin)
- Dark personality
- Pervy guard aktiv
- High thinking rate (70%)

### 2. **Alex** (Casual Streamer)
- Friendly & locker
- Kein pervy guard
- Medium thinking rate (50%)

### 3. **Luna** (Mystische Elfen-Magierin)
- Fantasy-Charakter
- Poetisch & weise
- Low thinking rate (40%)

### 4. **TechBot** (Tech-Support)
- Hilfreich & präzise
- Strukturierte Antworten
- Very low thinking rate (30%)

---

## 🛠️ Neuen Character erstellen

### 1. JSON-Datei anlegen
```bash
ki_chat/characters/meinchar.json
```

### 2. Konfiguration
```json
{
  "name": "MeinChar",
  "db_path": "memory_meinchar.db",
  "model": "deeliar-m4000-perf:latest",
  "system_prompt": "Deine komplette Personality hier...",
  "self_username": "__meinchar__",
  "thinking_rate": 0.60,
  "max_history": 16,
  "max_user_focus": 6,
  "enable_auto_memory": true,
  "enable_pervy_guard": false
}
```

### 3. Laden
```bash
curl -X POST http://localhost:5001/character/switch \
  -d '{"character": "meinchar"}'
```

---

## 🧪 Testen

```bash
cd ki_chat
python test_characters.py
```

---

## 📊 API-Übersicht

| Endpoint | Method | Beschreibung |
|----------|--------|--------------|
| `/characters` | GET | Liste aller Characters |
| `/character/current` | GET | Aktueller Character Info |
| `/character/switch` | POST | Character wechseln |
| `/chat` | POST | Chat mit aktuellem Character |

---

## 💾 Datenbank-Isolation

Jeder Character hat eine **eigene SQLite-Datenbank**:
```
ki_chat/
├── memory_dilara.db    # Dilara's Gedächtnis
├── memory_alex.db      # Alex' Gedächtnis
├── memory_luna.db      # Luna's Gedächtnis
└── memory_techbot.db   # TechBot's Gedächtnis
```

**Wichtig:** Memories, Thoughts und Profiles werden NICHT geteilt!

---

## 🔄 Character-Wechsel im Code

```python
import requests

BASE_URL = "http://localhost:5001"

# Zu Dilara wechseln
requests.post(f"{BASE_URL}/character/switch", 
              json={"character": "dilara"})

# Chat
r = requests.post(f"{BASE_URL}/chat",
                  json={"message": "user:hey"})
print(r.json()["reply"])  # Dilara's Antwort

# Zu Luna wechseln
requests.post(f"{BASE_URL}/character/switch",
              json={"character": "luna"})

# Chat
r = requests.post(f"{BASE_URL}/chat",
                  json={"message": "user:hey"})
print(r.json()["reply"])  # Luna's Antwort
```

---

## 📝 Character-Parameter erklärt

| Parameter | Typ | Funktion |
|-----------|-----|----------|
| `name` | string | Anzeigename |
| `db_path` | string | SQLite DB Pfad |
| `model` | string | Ollama Model |
| `system_prompt` | string | Komplette Persönlichkeit |
| `self_username` | string | Interner Name für Self-Memories |
| `thinking_rate` | float | Denk-Wahrscheinlichkeit (0.0-1.0) |
| `max_history` | int | Chat-History Größe |
| `max_user_focus` | int | User-spezifische Messages |
| `enable_auto_memory` | bool | Auto-Speicherung von Facts |
| `enable_pervy_guard` | bool | Content-Filter |

---

## ⚡ Pro-Tipps

1. **Thinking Rate:**
   - Hoch (0.7): Character denkt viel → mehr Thoughts → komplexere Personality
   - Niedrig (0.3): Character denkt wenig → schneller, weniger DB-Last

2. **System Prompt:**
   - Muss **komplette** Personality enthalten
   - Sollte Sprachstil definieren
   - Muss Emotion-Format erklären (`|| <emotion>`)

3. **DB Migration:**
   - Alte `memory.db` kann als Character-DB verwendet werden
   - Einfach in `dilara.json` angeben: `"db_path": "memory.db"`

4. **Hot-Switching:**
   - Character-Wechsel ohne Server-Neustart
   - DB-Connections werden automatisch geschlossen/geöffnet
   - Kein Datenverlust

---

## 🐛 Troubleshooting

**"Character not found":**
- Prüfe Dateiname (lowercase!)
- Prüfe JSON-Syntax
- Prüfe `characters/` Ordner

**DB-Fehler:**
- DB wird auto-initialisiert beim Switch
- Falls Fehler: alte DB löschen oder umbenennen

**Default Character:**
- System lädt "dilara" beim Start
- Falls nicht vorhanden: ersten verfügbaren Character

---

## 🎓 Weitere Infos

Siehe `CHARACTER_SYSTEM.md` für vollständige Dokumentation.

---

**Have fun! 🚀**
