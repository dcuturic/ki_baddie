"""
Voice Chat Service -- Mikrofon -> STT -> KI Chat -> TTS Pipeline
===============================================================
Hört dauerhaft auf das Mikrofon, erkennt Sprache per Vosk (VAD-basiert),
sendet den erkannten Text an den KI-Chat und lässt die Antwort per TTS vorlesen.

Pipeline:
  [Mikrofon] -> [Vosk STT] -> [ki_chat /chat] -> [textToSpeech /tts] -> [Lautsprecher]

Nutzung:
  1. python app.py
  2. Listening startet automatisch (oder per API /listen/start)
  3. Einfach reden - der Service erkennt automatisch Sprache
"""

import os
import json
import sys
import io
import time
import queue
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import deque

# ===== Bulletproof Windows UTF-8 fix (ä, ö, ü, ß etc.) =====
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import requests as http_requests
from flask import Flask, request, jsonify

# ======================= CONFIG =======================

CONFIG_PATH = "config.json"


def load_config() -> Dict:
    if not os.path.exists(CONFIG_PATH):
        print(f"[Config] {CONFIG_PATH} nicht gefunden, nutze Defaults")
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Config] Fehler: {e}")
        return {}


CONFIG = load_config()

# Server
server_cfg = CONFIG.get("server", {})
SERVER_HOST = server_cfg.get("host", "0.0.0.0")
SERVER_PORT = server_cfg.get("port", 5010)

# Mikrofon
mic_cfg = CONFIG.get("microphone", {})
MIC_DEVICE_NAME = mic_cfg.get("device_name", "")
MIC_ALLOW_PARTIAL = mic_cfg.get("allow_partial_match", True)
MIC_SAMPLE_RATE = mic_cfg.get("sample_rate", 16000)

# STT (Vosk)
stt_cfg = CONFIG.get("stt", {})
VOSK_MODEL_PATH = stt_cfg.get("vosk_model_path", r"..\main_server\models\vosk-model-de-0.21")
VOSK_LOG_LEVEL = stt_cfg.get("vosk_log_level", 0)

# VAD (Voice Activity Detection)
vad_cfg = CONFIG.get("vad", {})
SILENCE_TIMEOUT = vad_cfg.get("silence_timeout", 1.5)
MIN_SPEECH_LENGTH = vad_cfg.get("min_speech_length", 0.5)
ENERGY_THRESHOLD = vad_cfg.get("energy_threshold", 300)

# Services
services_cfg = CONFIG.get("services", {})
KI_CHAT_URL = services_cfg.get("ki_chat", "http://127.0.0.1:5001")
TTS_URL = services_cfg.get("text_to_speech", "http://127.0.0.1:5057")

# Chat
chat_cfg = CONFIG.get("chat", {})
SPEAKER_NAME = chat_cfg.get("speaker_name", "User")
HTTP_TIMEOUT = chat_cfg.get("http_timeout", 120)
AUTO_START = CONFIG.get("auto_start_listening", True)

# ======================= FLASK =======================

app = Flask(__name__)
app.json.ensure_ascii = False

# ======================= GLOBALS =======================

VOSK_MODEL = None
_listener_thread: Optional[threading.Thread] = None
_listener_stop_event = threading.Event()
_is_listening = False
_is_processing = False
_listen_lock = threading.Lock()

# Stats
_stats = {
    "started_at": None,
    "total_recognized": 0,
    "total_sent_to_chat": 0,
    "total_tts_played": 0,
    "total_errors": 0,
    "last_text": None,
    "last_reply": None,
    "last_emotion": None,
    "last_activity": None,
}

# Log buffer
_log_buffer = deque(maxlen=100)


def _log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {msg}"
    print(entry, flush=True)
    _log_buffer.append(entry)


# ======================= AUDIO DEVICE =======================

def _normalize_name(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def find_input_device_index(target_name: str) -> Optional[int]:
    """Findet den sounddevice Device-Index für ein Mikrofon nach Name."""
    try:
        import sounddevice as sd
    except ImportError:
        _log("sounddevice nicht installiert!", "ERROR")
        return None

    if not target_name:
        # Kein Name -> Default Device
        _log("Kein Mikrofon-Name konfiguriert, nutze System-Default")
        return None

    target_norm = _normalize_name(target_name)

    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue

        dev_name = d.get("name", "")
        dev_norm = _normalize_name(dev_name)

        if dev_norm == target_norm:
            _log(f"Mikrofon gefunden (exakt): [{i}] {dev_name}")
            return i

        if MIC_ALLOW_PARTIAL and target_norm in dev_norm:
            _log(f"Mikrofon gefunden (partial): [{i}] {dev_name}")
            return i

    _log(f"Mikrofon nicht gefunden: '{target_name}'", "WARN")
    return None


def list_input_devices() -> List[Dict]:
    """Listet alle Audio-Eingabegeräte auf."""
    try:
        import sounddevice as sd
    except ImportError:
        return []

    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            devices.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "sample_rate": d.get("default_samplerate", 0),
            })
    return devices


# ======================= VOSK MODEL =======================

def load_vosk_model():
    """Lädt das Vosk-Modell einmalig."""
    global VOSK_MODEL

    try:
        from vosk import Model, SetLogLevel
        SetLogLevel(int(VOSK_LOG_LEVEL))
    except ImportError:
        _log("vosk nicht installiert! pip install vosk", "ERROR")
        return False
    except Exception:
        pass

    model_path = os.path.abspath(VOSK_MODEL_PATH)
    if not os.path.isdir(model_path):
        _log(f"Vosk-Modell nicht gefunden: {model_path}", "ERROR")
        return False

    _log(f"Lade Vosk-Modell: {model_path} ...")
    try:
        VOSK_MODEL = Model(model_path)
        _log("Vosk-Modell geladen [OK]")
        return True
    except Exception as e:
        _log(f"Fehler beim Laden: {e}", "ERROR")
        return False


# ======================= CHAT + TTS =======================

def send_to_chat(text: str) -> Optional[Dict]:
    """Sendet erkannten Text an ki_chat /chat."""
    chat_url = f"{KI_CHAT_URL.rstrip('/')}/chat"
    message = f"{SPEAKER_NAME}: {text}" if SPEAKER_NAME else text

    _log(f"-> Chat: {message}")
    try:
        r = http_requests.post(
            chat_url,
            json={"message": message},
            timeout=HTTP_TIMEOUT
        )
        r.raise_for_status()
        data = r.json()
        reply = data.get("reply", "")
        emotion = data.get("emotion", "neutral")
        _log(f"<- Chat: [{emotion}] {reply}")
        _stats["total_sent_to_chat"] += 1
        _stats["last_reply"] = reply
        _stats["last_emotion"] = emotion
        return {"reply": reply, "emotion": emotion}
    except Exception as e:
        _log(f"Chat-Fehler: {e}", "ERROR")
        _stats["total_errors"] += 1
        return None


def send_to_tts(reply: str, emotion: str = "neutral"):
    """Sendet die Chat-Antwort an den TTS Service."""
    tts_url = f"{TTS_URL.rstrip('/')}/tts"

    payload = {
        "text": {"value": reply, "emotion": emotion},
        "play_audio": True,
    }

    _log(f"-> TTS: [{emotion}] {reply[:80]}...")
    try:
        r = http_requests.post(tts_url, json=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        _log("<- TTS: OK")
        _stats["total_tts_played"] += 1
    except Exception as e:
        _log(f"TTS-Fehler: {e}", "ERROR")
        _stats["total_errors"] += 1


def process_speech(text: str):
    """Volle Pipeline: Text -> Chat -> TTS."""
    global _is_processing

    _is_processing = True
    _stats["last_text"] = text
    _stats["last_activity"] = datetime.now().isoformat()

    try:
        # 1. An Chat senden
        result = send_to_chat(text)
        if not result or not result.get("reply"):
            _log("Keine Antwort vom Chat erhalten", "WARN")
            return

        # 2. Antwort per TTS vorlesen
        send_to_tts(result["reply"], result.get("emotion", "neutral"))

    except Exception as e:
        _log(f"Pipeline-Fehler: {e}", "ERROR")
        _stats["total_errors"] += 1
    finally:
        _is_processing = False


# ======================= LISTENER =======================

def _listener_loop(stop_event: threading.Event):
    """
    Hauptschleife: Hört auf das Mikrofon, erkennt Sprache per Vosk,
    und verarbeitet vollständige Sätze.
    """
    global _is_listening

    try:
        import sounddevice as sd
        from vosk import KaldiRecognizer
    except ImportError as e:
        _log(f"Abhängigkeit fehlt: {e}", "ERROR")
        _is_listening = False
        return

    if VOSK_MODEL is None:
        _log("Vosk-Modell nicht geladen - Listener kann nicht starten", "ERROR")
        _is_listening = False
        return

    device = find_input_device_index(MIC_DEVICE_NAME)
    rec = KaldiRecognizer(VOSK_MODEL, MIC_SAMPLE_RATE)
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass  # Ignore minor status messages
        audio_queue.put(bytes(indata))

    _log(f"[MIC] Starte Mikrofon-Listener (Device: {device or 'default'}, Rate: {MIC_SAMPLE_RATE})")
    _is_listening = True
    _stats["started_at"] = datetime.now().isoformat()

    try:
        stream_kwargs = {
            "samplerate": MIC_SAMPLE_RATE,
            "blocksize": 4000,
            "dtype": "int16",
            "channels": 1,
            "callback": audio_callback,
        }
        if device is not None:
            stream_kwargs["device"] = device

        with sd.RawInputStream(**stream_kwargs):
            _log("[MIC] Mikrofon aktiv - sprich jetzt!")

            silence_start = None
            has_speech = False
            speech_start = None

            while not stop_event.is_set():
                try:
                    data = audio_queue.get(timeout=0.3)
                except queue.Empty:
                    continue

                # Wenn gerade TTS abgespielt wird, nicht zuhören (Echo vermeiden)
                if _is_processing:
                    # Recognizer resetten damit kein altes Audio verarbeitet wird
                    rec = KaldiRecognizer(VOSK_MODEL, MIC_SAMPLE_RATE)
                    silence_start = None
                    has_speech = False
                    speech_start = None
                    continue

                # Einfache Energie-Erkennung
                import numpy as np
                audio_array = np.frombuffer(data, dtype=np.int16)
                energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

                if energy > ENERGY_THRESHOLD:
                    if not has_speech:
                        has_speech = True
                        speech_start = time.time()
                        _log("[SPEECH] Sprache erkannt...")
                    silence_start = None
                else:
                    if has_speech and silence_start is None:
                        silence_start = time.time()

                # Audio an Vosk senden
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = (result.get("text") or "").strip()

                    if text and has_speech:
                        speech_duration = time.time() - (speech_start or time.time())
                        if speech_duration >= MIN_SPEECH_LENGTH:
                            _log(f"[STT] Erkannt: \"{text}\"")
                            _stats["total_recognized"] += 1

                            # In eigenem Thread verarbeiten damit Listener weiterlaeuft
                            threading.Thread(
                                target=process_speech,
                                args=(text,),
                                daemon=True
                            ).start()

                        has_speech = False
                        speech_start = None
                        silence_start = None

                # Stille-Timeout: Partial result finalisieren
                if (has_speech and silence_start
                        and (time.time() - silence_start) > SILENCE_TIMEOUT):
                    result = json.loads(rec.FinalResult())
                    text = (result.get("text") or "").strip()

                    if text:
                        speech_duration = time.time() - (speech_start or time.time())
                        if speech_duration >= MIN_SPEECH_LENGTH:
                            _log(f"[STT] Erkannt (Stille): \"{text}\"")
                            _stats["total_recognized"] += 1

                            threading.Thread(
                                target=process_speech,
                                args=(text,),
                                daemon=True
                            ).start()

                    # Reset
                    rec = KaldiRecognizer(VOSK_MODEL, MIC_SAMPLE_RATE)
                    has_speech = False
                    speech_start = None
                    silence_start = None

    except Exception as e:
        _log(f"Listener-Fehler: {e}\n{traceback.format_exc()}", "ERROR")
    finally:
        _is_listening = False
        _log("[MIC] Mikrofon-Listener gestoppt")


def start_listening() -> Dict:
    """Startet den Mikrofon-Listener."""
    global _listener_thread, _listener_stop_event, _is_listening

    with _listen_lock:
        if _is_listening:
            return {"success": False, "error": "Listener läuft bereits"}

        _listener_stop_event = threading.Event()
        _listener_thread = threading.Thread(
            target=_listener_loop,
            args=(_listener_stop_event,),
            daemon=True
        )
        _listener_thread.start()

        # Kurz warten bis der Listener wirklich läuft
        for _ in range(20):
            if _is_listening:
                break
            time.sleep(0.1)

        return {"success": True, "listening": _is_listening}


def stop_listening() -> Dict:
    """Stoppt den Mikrofon-Listener."""
    global _is_listening

    with _listen_lock:
        if not _is_listening:
            return {"success": False, "error": "Listener läuft nicht"}

        _listener_stop_event.set()

        if _listener_thread and _listener_thread.is_alive():
            _listener_thread.join(timeout=5)

        _is_listening = False
        return {"success": True, "listening": False}


# ======================= HTTP API =======================

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "voice_chat",
        "listening": _is_listening,
        "processing": _is_processing,
        "vosk_loaded": VOSK_MODEL is not None,
        "mic_device": MIC_DEVICE_NAME or "(default)",
        "speaker_name": SPEAKER_NAME,
        "services": {
            "ki_chat": KI_CHAT_URL,
            "text_to_speech": TTS_URL,
        },
        "stats": _stats,
    })


@app.post("/listen/start")
def api_listen_start():
    """Startet das Mikrofon-Listening."""
    result = start_listening()
    return jsonify(result), 200 if result["success"] else 400


@app.post("/listen/stop")
def api_listen_stop():
    """Stoppt das Mikrofon-Listening."""
    result = stop_listening()
    return jsonify(result), 200 if result["success"] else 400


@app.get("/listen/status")
def api_listen_status():
    """Aktueller Status des Listeners."""
    return jsonify({
        "listening": _is_listening,
        "processing": _is_processing,
        "stats": _stats,
    })


@app.get("/devices")
def api_devices():
    """Listet alle verfügbaren Audio-Eingabegeräte."""
    devices = list_input_devices()
    return jsonify({
        "devices": devices,
        "current": MIC_DEVICE_NAME or "(default)",
    })


@app.post("/say")
def api_say():
    """Manuell einen Text durch die Pipeline schicken (zum Testen)."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"success": False, "error": "Kein Text angegeben"}), 400

    # In Thread verarbeiten
    threading.Thread(target=process_speech, args=(text,), daemon=True).start()
    return jsonify({"success": True, "text": text, "message": "Wird verarbeitet..."})


@app.post("/tts")
def api_direct_tts():
    """Direkt einen Text per TTS vorlesen (ohne Chat)."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    emotion = data.get("emotion", "neutral")

    if not text:
        return jsonify({"success": False, "error": "Kein Text angegeben"}), 400

    threading.Thread(target=send_to_tts, args=(text, emotion), daemon=True).start()
    return jsonify({"success": True, "text": text, "emotion": emotion})


@app.get("/logs")
def api_logs():
    """Gibt die letzten Log-Einträge zurück."""
    return jsonify({"logs": list(_log_buffer)})


# ======================= START =======================

if __name__ == "__main__":
    print("=" * 60)
    print("  Voice Chat -- Mikrofon -> KI Chat -> TTS")
    print("=" * 60)
    print(f"  Server:      http://localhost:{SERVER_PORT}")
    print(f"  Mikrofon:    {MIC_DEVICE_NAME or '(System-Default)'}")
    print(f"  Vosk-Modell: {os.path.abspath(VOSK_MODEL_PATH)}")
    print(f"  Speaker:     {SPEAKER_NAME}")
    print(f"  KI Chat:     {KI_CHAT_URL}")
    print(f"  TTS:         {TTS_URL}")
    print(f"  Auto-Start:  {AUTO_START}")
    print("=" * 60)

    # Eingabegeräte anzeigen
    devices = list_input_devices()
    if devices:
        print("\n  Audio-Eingabegeräte:")
        for d in devices:
            marker = " <--" if MIC_DEVICE_NAME and _normalize_name(MIC_DEVICE_NAME) in _normalize_name(d["name"]) else ""
            print(f"    [{d['index']}] {d['name']}{marker}")
        print()

    # Vosk laden
    load_vosk_model()

    # Auto-Start Listener
    if AUTO_START and VOSK_MODEL is not None:
        _log("Auto-Start: Listener wird gestartet...")
        start_listening()

    # Flask starten
    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        use_reloader=False
    )
