# Global Config System - Dokumentation

## 🎯 Übersicht

Jedes Projekt hat jetzt eine **zentrale `config.json`** für alle wichtigen Parameter. Keine hardcoded Werte mehr!

---

## 📦 Config-Dateien

### 1. **ki_chat/config.json**
```json
{
  "server": {"host": "0.0.0.0", "port": 5001, "debug": true},
  "ollama": {
    "url": "http://localhost:11434/api/chat",
    "default_model": "deeliar-m4000-perf:latest",
    "options": { ... }
  },
  "database": { ... },
  "memory": { ... },
  "chat": { ... },
  "thinking": { ... },
  "characters_dir": "characters",
  "default_character": "dilara"
}
```

**Wichtigste Parameter:**
- `server.port` - Flask Port (default: 5001)
- `ollama.default_model` - LLM Model
- `ollama.options.num_ctx` - Context Size
- `ollama.options.temperature` - Kreativität
- `characters_dir` - Character-Verzeichnis
- `default_character` - Default Character beim Start

---

### 2. **textToSpeech/config.json**
```json
{
  "server": {"host": "0.0.0.0", "port": 5002, "debug": false},
  "gpu": {"index": 0},
  "voice_fx": {
    "enabled": true,
    "global_fx_enabled": true,
    "emotion_fx_enabled": false
  },
  "osc": {"host": "127.0.0.1", "port": 39539, "enabled": true},
  "tts": {
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "language": "de",
    "sample_rate": 24000
  },
  "emotions": {
    "joy": "voices/neutral.wav",
    "angry": "voices/neutral.wav",
    ...
  }
}
```

**Wichtigste Parameter:**
- `server.port` - Flask Port (default: 5002)
- `gpu.index` - GPU für TTS
- `tts.model_name` - XTTS Model
- `tts.language` - Sprache
- `emotions.*` - Voice-Samples pro Emotion
- `osc.enabled` - OSC an/aus

---

### 3. **VroidPoser/config.json**
```json
{
  "server": {"host": "0.0.0.0", "port": 5003, "debug": false},
  "osc": {
    "ip": "127.0.0.1",
    "port": 39539,
    "enabled": true,
    "client_name": "VroidPoser"
  },
  "directories": {
    "poses": "api_used/poses/default",
    "moves": "api_used/moves/default",
    "fight_pose": "api_used/poses/default/fight"
  },
  "animation": {
    "default_speed": 2.9,
    "joy_range": 80
  },
  "head_rotation": {
    "head_look_up_rad": 0.505,
    "neck_look_up_rad": 0.5
  },
  "idle_motion": {
    "update_hz": 30,
    "speed": 0.24,
    "intensity": 5.55
  },
  "blink": {
    "auto_enabled": true,
    "wink_chance": 0.08,
    "duration_min": 0.10,
    "duration_max": 0.20
  }
}
```

**Wichtigste Parameter:**
- `server.port` - Flask Port (default: 5003)
- `osc.enabled` - OSC an/aus
- `animation.default_speed` - Pose-Speed
- `idle_motion.*` - Idle-Animation Tuning
- `blink.*` - Auto-Blink Konfiguration
- `head_rotation.*` - Kamera-Korrektur

---

### 4. **VroidEmotion/config.json**
```json
{
  "server": {"host": "0.0.0.0", "port": 5004, "debug": false},
  "osc": {"host": "127.0.0.1", "port": 39539, "enabled": true},
  "emotion_map": {
    "joy": "Joy",
    "angry": "Angry",
    "sorrow": "Sorrow",
    ...
  },
  "blend_controller": {
    "fade_time": 0.35,
    "fps": 60
  }
}
```

**Wichtigste Parameter:**
- `server.port` - Flask Port (default: 5004)
- `emotion_map.*` - Emotion → Blendshape Mapping
- `blend_controller.fade_time` - Transition-Dauer
- `blend_controller.fps` - Animation FPS

---

### 5. **main_server/config.json**
```json
{
  "server": {"host": "0.0.0.0", "port": 5000, "debug": false},
  "services": {
    "ki_chat": "http://ki_chat:5001",
    "text_to_speech": "http://text_to_speech:5002",
    "vroid_poser": "http://vroid_poser:5003",
    "vroid_emotion": "http://vroid_emotion:5004"
  },
  "stt": {
    "mode": "vosk",
    "vosk_model_path": "models\\vosk-model-de-0.21",
    "vosk_log_level": 0
  },
  "microphone": {
    "device_name": "Headset (WH-1000XM5 Hands-Free AG Audio)",
    "allow_partial_match": true
  }
}
```

**Wichtigste Parameter:**
- `server.port` - Flask Port (default: 5000)
- `services.*` - Service-URLs (für Docker)
- `stt.mode` - STT-Mode (vosk/whisper)
- `stt.vosk_model_path` - VOSK Model Pfad
- `microphone.device_name` - Mikrofon Name

---

## 🔧 Verwendung

### Config laden
Jedes Projekt lädt automatisch beim Start `config.json`:

```python
import json
import os

CONFIG_PATH = "config.json"
CONFIG = {}

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Warning: {CONFIG_PATH} not found, using defaults")
        return {}
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

CONFIG = load_config()
```

### Config-Werte verwenden
```python
# Beispiel: Server-Port
server_cfg = CONFIG.get("server", {})
port = server_cfg.get("port", 5001)  # Fallback auf 5001

# Beispiel: OSC aktiviert?
osc_cfg = CONFIG.get("osc", {})
enabled = osc_cfg.get("enabled", False)
```

### Environment Variables überschreiben Config
```python
# OSC kann auch per ENV überschrieben werden
osc_cfg.ip = os.getenv("OSC_IP", osc_cfg.ip)
osc_cfg.port = int(os.getenv("OSC_PORT", str(osc_cfg.port)))
```

---

## 📝 Beispiel: Config ändern

**Vorher (hardcoded):**
```python
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "deeliar-m4000-perf:latest"
TEMPERATURE = 0.6
```

**Nachher (config.json):**
```json
{
  "ollama": {
    "url": "http://localhost:11434/api/chat",
    "default_model": "deeliar-m4000-perf:latest",
    "options": {
      "temperature": 0.6,
      "num_ctx": 2048
    }
  }
}
```

```python
OLLAMA_URL = CONFIG.get("ollama", {}).get("url", "http://localhost:11434/api/chat")
```

---

## ⚡ Vorteile

### ✅ Zentrale Konfiguration
- Alle Parameter an **einem Ort**
- Keine Suche in Code-Dateien
- Kein Recompile nötig

### ✅ Einfache Anpassung
- JSON ist **human-readable**
- Änderungen ohne Code-Kenntnisse
- Schnelle Experimente

### ✅ Environment-spezifisch
- Unterschiedliche Configs für Dev/Prod
- Docker-Compose kann Configs mounten
- Keine Secrets im Code

### ✅ Dokumentation
- Config-Datei ist selbstdokumentierend
- Defaults sind sichtbar
- Leichter verständlich

---

## 🐛 Troubleshooting

**Config wird nicht geladen:**
- Prüfe ob `config.json` existiert
- Prüfe JSON-Syntax (mit JSONLint)
- Prüfe Encoding (UTF-8)

**Werte werden ignoriert:**
- Prüfe Pfad im Code (`.get("key", default)`)
- Prüfe Fallback-Werte
- Prüfe Environment Variables

**Server startet nicht:**
- Port schon belegt? → `server.port` ändern
- Config-Fehler? → Terminal-Log prüfen

---

## 🎓 Best Practices

1. **Immer Fallbacks:**
   ```python
   port = CONFIG.get("server", {}).get("port", 5001)
   ```

2. **Validierung:**
   ```python
   if not isinstance(port, int) or port < 1 or port > 65535:
       print(f"Invalid port: {port}, using default")
       port = 5001
   ```

3. **Config ändern:**
   - Erst in `config.json` ändern
   - Server neu starten
   - Testen

4. **Versionierung:**
   - `config.json` in Git committen (ohne Secrets!)
   - Secrets in separater `.env` (nicht in Git)
   - Dokumentieren was jeder Wert macht

---

## 📊 Übersicht aller Ports

| Service | Port | Config-Datei |
|---------|------|-------------|
| main_server | 5000 | main_server/config.json |
| ki_chat | 5001 | ki_chat/config.json |
| textToSpeech | 5002 | textToSpeech/config.json |
| VroidPoser | 5003 | VroidPoser/config.json |
| VroidEmotion | 5004 | VroidEmotion/config.json |
| OSC (VSeeFace) | 39539 | alle mit OSC |

---

**Pro-Tipp:** Du kannst jetzt alle wichtigen Parameter zentral pflegen ohne in 1000 Code-Zeilen zu suchen! 🚀
