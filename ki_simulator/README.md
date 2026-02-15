# KI Simulator (2 Bots)

Dieses Tool lässt **2 erreichbare Chat-Bots via HTTP Requests** miteinander sprechen.

Aktuell ist es bewusst auf 2 Bots begrenzt (MVP), aber die Struktur ist für Erweiterung vorbereitet.

## Voraussetzungen

- Beide Bot-APIs laufen und sind erreichbar
- Endpoint akzeptiert `POST /chat` mit Body:
  - `{"message": "username:text"}`
- Antwort enthält Feld `reply`

## Schnellstart

```bash
cd ki_simulator
python simulate_two_bots.py \
  --bot-a-url http://localhost:5001 \
  --bot-b-url http://localhost:5001 \
  --bot-a-name Dilara \
  --bot-b-name Luna \
  --bot-a-role "ruhig, kontrolliert, bindend" \
  --bot-b-role "mystisch, poetisch, neugierig" \
  --topic "Plan für einen düsteren Stream-Abend" \
  --strategy story \
  --turns 12 \
  --starter a \
  --output transcript.json
```

## Mit Config-Datei starten

```bash
python simulate_two_bots.py --config config.example.json
```

Dann einzelne Werte optional per CLI überschreiben.

## Wichtige Parameter

- `--bot-a-url`, `--bot-b-url`: Basis-URL der beiden Bots
- `--endpoint`: Standard-Endpoint (default `/chat`)
- `--bot-a-role`, `--bot-b-role`: Rollenprofil für die Strategie
- `--strategy`: `debate`, `interview`, `story`, `brainstorm`
- `--turns`: Anzahl Gesprächsbeiträge
- `--history-window`: Wie viele letzte Beiträge in den nächsten Prompt einfließen
- `--output`: Schreibt Transkript als JSON
