# Quick Config Guide

## 🚀 Schnellstart

### 1. **Port ändern**
```json
// ki_chat/config.json
{
  "server": {
    "port": 5001  // ← Ändere hier
  }
}
```

### 2. **LLM Model wechseln**
```json
// ki_chat/config.json
{
  "ollama": {
    "default_model": "llama3:latest"  // ← Neues Model
  }
}
```

### 3. **OSC deaktivieren**
```json
// VroidPoser/config.json, VroidEmotion/config.json, textToSpeech/config.json
{
  "osc": {
    "enabled": false  // ← Aus
  }
}
```

### 4. **TTS-Sprache ändern**
```json
// textToSpeech/config.json
{
  "tts": {
    "language": "en"  // ← Deutsch → Englisch
  }
}
```

### 5. **Idle-Animation tunen**
```json
// VroidPoser/config.json
{
  "idle_motion": {
    "speed": 0.24,      // ← Langsamer = kleiner Wert
    "intensity": 5.55   // ← Stärker = größerer Wert
  }
}
```

### 6. **Debug-Modus aktivieren**
```json
// Alle config.json
{
  "server": {
    "debug": true  // ← An (mehr Logs)
  }
}
```

### 7. **Default Character ändern**
```json
// ki_chat/config.json
{
  "default_character": "alex"  // ← Start mit Alex statt Dilara
}
```

### 8. **Mikrofon wechseln**
```json
// main_server/config.json
{
  "microphone": {
    "device_name": "Dein Mikrofon Name"  // ← Neuer Name
  }
}
```

---

## 📋 Häufige Änderungen

### Character erstellen
1. `ki_chat/characters/meinchar.json` erstellen
2. In `ki_chat/config.json`:
   ```json
   {
     "default_character": "meinchar"
   }
   ```

### Thinking Rate ändern
```json
// ki_chat/config.json
{
  "thinking": {
    "interval_seconds": 10  // Schneller denken (war 20)
  }
}
```

### Voice-Sample ändern
```json
// textToSpeech/config.json
{
  "emotions": {
    "joy": "voices/happy.wav",  // ← Neue Datei
    "angry": "voices/mad.wav"
  }
}
```

### Blink-Frequenz ändern
```json
// VroidPoser/config.json
{
  "blink": {
    "normal_wait_min": 1.0,  // ← Schneller blinken
    "normal_wait_max": 3.0
  }
}
```

---

## ⚠️ Wichtig

**Nach Config-Änderung:**
1. Server **neu starten**
2. Logs prüfen
3. Testen

**JSON-Syntax prüfen:**
```bash
# Online: https://jsonlint.com
# Oder Python:
python -m json.tool config.json
```

**Backup erstellen:**
```bash
cp config.json config.json.backup
```

---

## 🎯 Pro-Tipps

1. **Kleine Änderungen:** Nur 1 Parameter auf einmal ändern
2. **Logs lesen:** `debug: true` aktivieren um zu sehen was passiert
3. **Defaults behalten:** Nur ändern was du brauchst
4. **Testen:** Nach jeder Änderung kurz testen

---

**Check CONFIG_SYSTEM.md für vollständige Doku! 📚**
