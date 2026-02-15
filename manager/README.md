# 🎭 KI Girl Manager

Web-basierter Manager für das komplette KI Girl VTuber Projekt.

## Features

✅ **Dashboard** - Übersicht aller Services mit Live-Status  
✅ **Service Management** - Start/Stop/Restart einzelner Services  
✅ **Instanzen** - Verschiedene Konfigurations-Setups  
✅ **Config Editor** - JSON Configs direkt bearbeiten & speichern  
✅ **Logs** - Live-Logs aller Services  
✅ **Process Management** - CPU/RAM Überwachung  
✅ **Toast Notifications** - Sofortiges Feedback bei Aktionen

## Quick Start

### Option 1: PowerShell Script (Empfohlen)
```powershell
cd manager
.\start.ps1
```

### Option 2: Manuell
```powershell
cd manager
pip install -r requirements.txt
python app.py
```

Manager läuft auf: **http://localhost:8000**

## Architektur

```
Manager (Port 8000)
├── Prozess-Verwaltung (subprocess)
├── Live-Status Tracking (Threading)
├── Config Management (JSON)
└── Log-Streaming (In-Memory)

Services werden als Subprozesse gestartet:
├── Ollama (PowerShell Script)
├── KI Chat (Python app.py)
├── Main Server (Python app.py)
├── Text-to-Speech (Python app.py)
├── VRoid Poser (Python app.py)
└── VRoid Emotion (Python app.py)
```

## Instanzen-System

**Instanzen** definieren verschiedene Konfigurations-Setups:

### Vordefinierte Instanzen:
- **default** - Standard mit allen Services
- **dev** - Development (nur Ollama + KI Chat)
- **prod_full** - Production mit Performance-Tuning

### Instanz-Struktur:
```json
{
  "name": "dev",
  "description": "Development Instanz",
  "services": {
    "ki_chat": {"enabled": true, "auto_start": true},
    ...
  },
  "config_overrides": {
    "ki_chat": {
      "thinking": {"thinking_rate": 0.30}
    }
  }
}
```

### Config-Overrides:
Jede Instanz kann Service-Configs überschreiben:
- **Base Config**: `ki_chat/config.json` (Global)
- **Override**: In Instanz definiert
- **Merged**: Manager merged beide beim Start

**Beispiel**:
```json
// Base: ki_chat/config.json
{
  "thinking": {"thinking_rate": 0.70}
}

// Override: instances/dev.json
{
  "config_overrides": {
    "ki_chat": {
      "thinking": {"thinking_rate": 0.30}
    }
  }
}

// Result: thinking_rate = 0.30 (Override gewinnt!)
```

## UI-Features

### 1. Dashboard (/)
- **Service Grid**: 6 Services mit Status (Läuft/Gestoppt)
- **Live-Updates**: Automatisch alle 2 Sekunden
- **Quick Actions**: Alle starten/stoppen
- **System Stats**: Laufende Services, Uptime
- **Instanz-Wechsel**: Dropdown zur Auswahl

### 2. Services (/services)
- **Detaillierte Ansicht**: PID, Uptime, CPU, RAM
- **Start/Stop/Restart**: Pro Service
- **Live-Logs**: Automatisch aktualisiert (alle 5 Sek)
- **Log-Filter**: stdout/stderr farblich getrennt

### 3. Instanzen (/instances)
- **Instanz-Cards**: Übersicht aller Setups
- **Service-Toggles**: Aktivieren/Deaktivieren
- **Auto-Start Badges**: Kennzeichnung
- **CRUD**: Erstellen/Bearbeiten/Löschen

### 4. Configs (/configs)
- **JSON-Editor**: Mit Syntax-Highlighting
- **Live-Validation**: JSON-Prüfung bei Eingabe
- **Alle 6 Configs**: Manager + 5 Services
- **Speichern**: Direktes Schreiben in config.json
- **Service-Restart**: Nach Config-Änderung

## API Endpoints

### Services
```
GET  /api/services/status          - Status aller Services
POST /api/service/<id>/start       - Service starten
POST /api/service/<id>/stop        - Service stoppen
POST /api/service/<id>/restart     - Service neu starten
GET  /api/service/<id>/logs        - Logs abrufen
```

### Instances
```
GET  /api/instances                - Alle Instanzen
GET  /api/instance/<id>            - Spezifische Instanz
POST /api/instance/<id>/start      - Instanz starten
POST /api/instance/<id>/stop       - Instanz stoppen
POST /api/instance/<id>/save       - Instanz speichern
POST /api/instance/<id>/delete     - Instanz löschen
```

### Configs
```
GET  /api/config/manager           - Manager Config
POST /api/config/manager           - Manager Config speichern
GET  /api/config/<service>         - Service Config
POST /api/config/<service>         - Service Config speichern
```

## Prozess-Management

### ProcessManager-Class:
- **Start**: `subprocess.Popen` mit stdout/stderr Pipes
- **Stop**: Graceful `terminate()`, dann `kill()` nach 5 Sek
- **Status**: Via `psutil` (PID, CPU, RAM)
- **Logs**: Threading für stdout/stderr Reading
- **Cleanup**: `atexit` handler stoppt alle beim Exit

### Service-Types:
- **python**: `python app.py` in Service-Dir
- **powershell**: `powershell.exe -File script.ps1`

## Technologies

**Backend:**
- Flask 2.3.3 - Web Framework
- psutil 5.9.5 - Prozess-Monitoring
- subprocess - Prozess-Management
- threading - Log-Streaming

**Frontend:**
- Vanilla JavaScript - Kein Framework
- CSS Grid/Flexbox - Responsive Layout
- Fetch API - AJAX Requests
- Live-Polling - Auto-Updates (2-5 Sek Intervall)

## Ports

- **Manager**: 8000
- **KI Chat**: 5001
- **Main Server**: 5000
- **Text-to-Speech**: 5002
- **VRoid Poser**: 5003
- **VRoid Emotion**: 5004
- **Ollama**: 11434
- **OSC**: 39539

## Workflow-Beispiele

### Neue Instanz erstellen:
1. Dashboard → Instanzen
2. "Neue Instanz" Button
3. Name + Beschreibung eingeben
4. Services auswählen
5. Speichern → Neue JSON-Datei in `instances/`

### Service konfigurieren:
1. Dashboard → Configs
2. Service auswählen (z.B. "KI Chat")
3. JSON bearbeiten
4. "Speichern" → Schreibt in `ki_chat/config.json`
5. "Service neu starten" → Lädt neue Config

### Production deployen:
1. Instanz "prod_full" auswählen
2. "Start" → Alle Services mit Auto-Start
3. Dashboard → Überwachung mit Live-Status
4. Services → Logs überprüfen

## Bekannte Limitationen

- **Windows-Only**: PowerShell-Scripts für Ollama
- **Single-Host**: Keine Remote-Services
- **No Persistence**: Prozesse werden nicht beim Manager-Neustart wiederhergestellt
- **Memory**: Logs auf 1000 Zeilen pro Service limitiert

## Troubleshooting

**Service startet nicht:**
- Logs im Service-Tab überprüfen
- Python/Dependencies installiert?
- Ports bereits belegt?

**Config wird nicht geladen:**
- JSON-Syntax valide?
- Pfade in manager/config.json korrekt?
- Schreibrechte vorhanden?

**Manager läuft nicht:**
```powershell
# Dependencies neu installieren
pip install -r requirements.txt --force-reinstall

# Port 8000 freigeben
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## Next Steps / TODOs

Mögliche Erweiterungen:
- [ ] WebSocket für Live-Updates (statt Polling)
- [ ] Persistent Process Storage (DB)
- [ ] Auto-Restart bei Crash
- [ ] Performance Graphs (CPU/RAM History)
- [ ] Multi-User Authentication
- [ ] Docker Integration
- [ ] Remote Service Support
- [ ] Config-Diff Viewer
- [ ] Backup/Restore System
- [ ] Custom Service Scripts

## Support

Bei Problemen oder Fragen:
1. Logs im Service-Tab checken
2. Browser-Console öffnen (F12)
3. Manager-Console Output prüfen
4. Issue erstellen mit Logs

**Happy Managing!** 🚀
