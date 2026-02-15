# Multi-Character System - Dokumentation

## Übersicht

Das ki_chat System unterstützt jetzt **mehrere Charaktere** mit individuellen:
- **Persönlichkeiten** (System Prompts)
- **Gedächtnissen** (separate SQLite DBs)
- **Einstellungen** (Thinking Rate, Auto-Memory, etc.)

---

## Character-Struktur

Jeder Character wird als JSON-Datei in `ki_chat/characters/` gespeichert.

### Beispiel: `dilara.json`

```json
{
  "name": "Dilara",
  "db_path": "memory_dilara.db",
  "model": "deeliar-m4000-perf:latest",
  "system_prompt": "SYSTEM:\nName: Dilara\n...",
  "self_username": "__dilara__",
  "thinking_rate": 0.70,
  "max_history": 16,
  "max_user_focus": 6,
  "enable_auto_memory": true,
  "enable_pervy_guard": true
}
```

### Parameter

| Parameter | Typ | Beschreibung |
|-----------|-----|-------------|
| `name` | string | Anzeigename des Characters |
| `db_path` | string | Pfad zur character-spezifischen Datenbank |
| `model` | string | Ollama Model Name |
| `system_prompt` | string | Kompletter System Prompt (Persönlichkeit) |
| `self_username` | string | Interner Username für Self-Memories |
| `thinking_rate` | float (0-1) | Wahrscheinlichkeit pro Tick zu "denken" |
| `max_history` | int | Max Chat-History Messages |
| `max_user_focus` | int | Max User-spezifische Messages |
| `enable_auto_memory` | bool | Auto-Memory aktivieren |
| `enable_pervy_guard` | bool | Content-Filter aktivieren |

---

## API Endpoints

### GET `/characters`
Liste aller verfügbaren Characters

**Response:**
```json
{
  "characters": ["dilara", "alex"],
  "current": "dilara"
}
```

### GET `/character/current`
Aktueller Character Info

**Response:**
```json
{
  "name": "Dilara",
  "db_path": "memory_dilara.db",
  "model": "deeliar-m4000-perf:latest",
  "thinking_rate": 0.70,
  ...
}
```

### POST `/character/switch`
Character wechseln

**Request:**
```json
{
  "character": "alex"
}
```

**Response:**
```json
{
  "ok": true,
  "character": "Alex",
  "db_path": "memory_alex.db"
}
```

---

## Neuen Character erstellen

1. **JSON erstellen:**
   ```bash
   ki_chat/characters/mein_char.json
   ```

2. **Konfiguration:**
   ```json
   {
     "name": "MeinChar",
     "db_path": "memory_meinchar.db",
     "model": "deeliar-m4000-perf:latest",
     "system_prompt": "Deine Personality hier...",
     "self_username": "__meinchar__",
     "thinking_rate": 0.50,
     "max_history": 16,
     "max_user_focus": 6,
     "enable_auto_memory": true,
     "enable_pervy_guard": false
   }
   ```

3. **Laden:**
   ```bash
   curl -X POST http://localhost:5001/character/switch \
     -H "Content-Type: application/json" \
     -d '{"character": "mein_char"}'
   ```

---

## Features

### ✅ Separate Datenbanken
Jeder Character hat sein eigenes:
- Chatlog
- Memories (User-Facts)
- Thoughts (innerer Monolog)
- Profiles (Display Names)

### ✅ Hot-Switching
Characters können zur Laufzeit gewechselt werden ohne Server-Neustart.

### ✅ Individuelle Persönlichkeiten
Jeder Character kann komplett unterschiedliche:
- Sprachstile
- Verhaltensweisen
- Regeln
- Emotionen

### ✅ Character-Isolation
Memories und Thoughts werden NICHT geteilt. Jeder Character ist komplett unabhängig.

---

## Beispiel-Usage

```python
import requests

# Liste Characters
r = requests.get("http://localhost:5001/characters")
print(r.json())
# {"characters": ["dilara", "alex"], "current": "dilara"}

# Zu Alex wechseln
r = requests.post("http://localhost:5001/character/switch", 
                  json={"character": "alex"})
print(r.json())
# {"ok": true, "character": "Alex", "db_path": "memory_alex.db"}

# Chat mit Alex
r = requests.post("http://localhost:5001/chat",
                  json={"message": "user123:hey wie gehts?"})
print(r.json())
# {"reply": "Yo alles gut! Was geht bei dir?", "emotion": "fun"}

# Zurück zu Dilara
r = requests.post("http://localhost:5001/character/switch",
                  json={"character": "dilara"})
```

---

## Migration bestehender DBs

Die alte `memory.db` bleibt kompatibel! Du kannst sie als `memory_dilara.db` umbenennen oder in der `dilara.json` den Pfad anpassen:

```json
{
  "db_path": "memory.db"
}
```

---

## Troubleshooting

**Character nicht gefunden:**
- Prüfe ob die JSON-Datei in `ki_chat/characters/` existiert
- Dateiname muss lowercase sein (z.B. `alex.json` für Character "alex")

**DB-Fehler nach Switch:**
- Die DB wird automatisch beim Switch initialisiert
- Alte Connections werden geschlossen

**Default Character:**
- Beim Start wird automatisch "dilara" geladen (wenn vorhanden)
- Falls nicht: erster verfügbarer Character

---

## Beispiel-Characters

Das System kommt mit 2 Beispiel-Characters:

### 1. Dilara (Horror-Yandere-Streamerin)
- Pervy Guard aktiviert
- Thinking Rate: 70%
- Dark Personality

### 2. Alex (Casual Streamer)
- Pervy Guard deaktiviert
- Thinking Rate: 50%
- Friendly Personality

---

**Pro-Tipp:** Du kannst beliebig viele Characters erstellen. Jeder Character ist komplett isoliert und kann zur Laufzeit gewechselt werden! 🚀
