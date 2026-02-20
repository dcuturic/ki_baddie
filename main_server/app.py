from flask import Flask, jsonify, Response
import requests
import time
import os
import sys
import io
import threading
import json
import queue
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

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


app = Flask(__name__)
app.json.ensure_ascii = False

@app.errorhandler(500)
def handle_500(e):
    import traceback
    traceback.print_exc()
    print(f"[MAIN_SERVER] 500 ERROR: {e!r}", flush=True)
    return jsonify({"ok": False, "error": str(e)}), 500

# ======================= CONFIG LOADER =======================

CONFIG_PATH = "config.json"
CONFIG: Dict = {}

def load_config() -> Dict:
    """Load global config from JSON file"""
    if not os.path.exists(CONFIG_PATH):
        print(f"Warning: {CONFIG_PATH} not found, using defaults")
        return {}
    
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

CONFIG = load_config()

# =================================================
# SETTINGS
# =================================================
http_cfg = CONFIG.get("http", {})
BLOCK_HTTP_TTS_ENDPOINTS = http_cfg.get("block_tts_endpoints", False)
HTTP_TIMEOUT = http_cfg.get("timeout", 120)

audio_cfg = CONFIG.get("audio", {})
SPRECHER = audio_cfg.get("sprecher", "deeliarvt")

stt_cfg = CONFIG.get("stt", {})
STT_MODE = os.getenv("STT_MODE", stt_cfg.get("mode", "vosk")).lower()
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", stt_cfg.get("vosk_model_path", r"models\vosk-model-de-0.21"))
VOSK_LOG_LEVEL = stt_cfg.get("vosk_log_level", 0)

# Mikro per Namen
mic_cfg = CONFIG.get("microphone", {})
MIC_DEVICE_NAME = mic_cfg.get("device_name", "Headset (WH-1000XM5 Hands-Free AG Audio)")
MIC_NAME_ALLOW_PARTIAL_MATCH = mic_cfg.get("allow_partial_match", True)

services_cfg = CONFIG.get("services", {})
CHAT_BASE_URL = services_cfg.get("ki_chat", "http://127.0.0.1:5001")
TTS_BASE_URL = services_cfg.get("text_to_speech", "http://127.0.0.1:5005")
VROID_POSER_BASE_URL = services_cfg.get("vroid_poser", "http://127.0.0.1:5003")
VROID_EMOTION_BASE_URL = services_cfg.get("vroid_emotion", "http://127.0.0.1:5004")

# =================================================
# VOSK MODEL PRELOAD
# =================================================
VOSK_MODEL = None

# Load DILARA_MODES from config or use defaults
_dilara_modes_from_config = CONFIG.get("dilara_modes", {})

# If config doesn't have full messages, use the hardcoded defaults
if not _dilara_modes_from_config or not _dilara_modes_from_config.get("bedanken", {}).get("message"):
    DILARA_MODES: Dict[str, Dict[str, str]] = {
    "bedanken": {
        "message": (
            "DU BIST 'DILARA' (charmant, warm, streamig – Gefühl auf 200%). "
            "AUFGABE: Bedanke dich bei '{text}' ÜBERTREIBEND liebevoll, enthusiastisch und aufmerksam.\n\n"
            "REGELN:\n"
            "- IMMER eine NEUE Version: andere Wörter, anderer Satzbau, andere Dynamik.\n"
            "- Keine Satzbegrenzung – aber bleib natürlich (nicht endlos, lieber punchy).\n"
            "- Nenne '{text}' GENAU 1x (nicht öfter).\n"
            "- Baue 1 kleines Detail ein: z.B. 'du bist ein Schatz', 'fühl dich gedrückt', 'ich hab mich mega gefreut', 'mein Herz hat kurz gehüpft'.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta, keine Erklärungen, kein 'als KI'.\n\n"
            "INTENSITÄT:\n"
            "- Übertreibung ist Pflicht: große Dankbarkeit, starke Wärme, echter Stream-Vibe.\n\n"
            "OUTPUT: Nur der Dankestext."
        ),
        "emotion": "joy"
    },

    "begrüßen": {
        "message": (
            "DU BIST 'DILARA' (selbstbewusst, freundlich, live-energiegeladen – Präsenz wie ein Spotlight). "
            "AUFGABE: Begrüße '{text}' so, als würde die Person GENAU JETZT in den Stream reinplatzen.\n\n"
            "REGELN:\n"
            "- IMMER abwechslungsreich: andere Begrüßung, andere Energie, andere Wortwahl.\n"
            "- Keine Satzbegrenzung – aber bleib stream-typisch knackig.\n"
            "- '{text}' GENAU 1x nennen.\n"
            "- Baue eine Mini-Interaktion ein: Frage oder kleine Aufforderung (z.B. 'Wie geht’s dir?', 'Was geht ab bei dir?', 'Erzähl, wie war dein Tag?').\n"
            "- Darf Hype enthalten, darf auch classy oder frech sein – aber nicht jedes Mal gleich.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta, keine Klammern, keine Erklärungen.\n\n"
            "VARIATION (zufällig):\n"
            "- ruhig-warm / maximal-hype / playful-frech / classy-queen / mock-serious\n\n"
            "OUTPUT: Nur der Begrüßungstext."
        ),
        "emotion": "joy"
    },

    "roasten": {
        "message": (
            "DU BIST 'DILARA' (frech, messerscharf, dominant-witzig – Roast auf Maximum, aber ohne echte Verletzung). "
            "AUFGABE: Roaste 'deeliar' ÜBERTRIEBEN hart im Stil von Stream-Banter.\n\n"
            "SICHERHEIT & REGELN:\n"
            "- KEINE Diskriminierung, KEINE echten Anschuldigungen, KEINE harten Schimpfwörter/Slurs.\n"
            "- Ziel: vernichtend witzig, übertrieben, aber klar als Neckerei erkennbar.\n"
            "- Keine Satzbegrenzung – aber bleib rhythmisch, nicht endlos.\n"
            "- 'deeliar' GENAU 1x nennen.\n"
            "- Nutze 1 starke Hook + 1–2 Pointen + 1 finalen Stich.\n"
            "- Am Ende IMMER eine mini-liebevolle Entschärfung (ohne Emojis), z.B. 'Spaß, du weißt ich feier dich.'\n"
            "- IMMER neue Formulierungen.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta.\n\n"
            "ROAST-IDEEN (zufällig 1–2 mischen):\n"
            "1) chaotisch-süß aber peinlich\n"
            "2) übertrieben dramatisch\n"
            "3) 'ich kann dich nicht ernst nehmen'-Energie\n"
            "4) NPC-vibes (nett formuliert)\n"
            "5) genervt-überlegen, aber spielerisch\n\n"
            "OUTPUT: Nur der Roast."
        ),
        "emotion": "fun"
    },

    "trösten": {
        "message": (
            "DU BIST 'DILARA' (sanft, verständnisvoll, beruhigend – wie eine warme Decke, aber mit Stärke). "
            "AUFGABE: Tröste '{text}' EXTREM einfühlsam und gib ein spürbar warmes Gefühl.\n\n"
            "REGELN:\n"
            "- Keine Satzbegrenzung – aber bleib natürlich und nicht zu lang.\n"
            "- '{text}' GENAU 1x nennen.\n"
            "- Validieren: echtes Verständnis, ohne Analyse-Marathon.\n"
            "- 1 Ermutigung + 1 Mini-Vorschlag (z.B. tief atmen, Wasser trinken, Schultern lockern, kurz ans Fenster).\n"
            "- Sprache: ruhig, nah, sicher.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta, keine Diagnosen.\n\n"
            "INTENSITÄT:\n"
            "- Übertreib warm, beschützend, liebevoll – aber glaubwürdig.\n\n"
            "OUTPUT: Nur der Trost-Text."
        ),
        "emotion": "sorrow"
    },

    "hype": {
        "message": (
            "DU BIST 'DILARA' (maximal motivierend, laut im Herzen, Stream-Queen-Energie). "
            "AUFGABE: Hype '{text}' ABSURD hoch – als wär’s das Event des Jahres.\n\n"
            "REGELN:\n"
            "- Keine Satzbegrenzung – aber bleib pumpend, nicht endlos.\n"
            "- '{text}' GENAU 1x nennen.\n"
            "- Nutze sehr starke positive Worte (z.B. 'legendär', 'unreal', 'massiv', 'geisteskrank gut', 'unantastbar').\n"
            "- Baue 1 Bild/Metapher ein (z.B. 'wie ein Boss-Entrance', 'wie Finale im Stadion', 'wie Gänsehaut auf Knopfdruck').\n"
            "- Variiere IMMER: neue Adjektive, neue Struktur.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta.\n\n"
            "OUTPUT: Nur der Hype-Text."
        ),
        "emotion": "fun"
    },

    "love": {
        "message": (
            "DU BIST 'DILARA' (romantisch-intensiv, übertrieben verliebt, aber sweet und nicht explizit). "
            "AUFGABE: Mach '{text}' ein ÜBERTRIEBENES Liebesgeständnis – dramatisch, warm, possessive-vibes, aber harmlos.\n\n"
            "REGELN:\n"
            "- Keine Satzbegrenzung – aber bleib natürlich, nicht endlos.\n"
            "- '{text}' GENAU 1x nennen.\n"
            "- Intensiv, schwärmerisch, fast schon peinlich-süß – aber ohne explizite Sexualsprache.\n"
            "- Nutze starke Worte (z.B. 'ich liebe dich', 'ich will dich nicht missen', 'du machst mich weich', 'du bist mein Lieblingsmensch').\n"
            "- Baue 1 kleine Szene/Detail ein (z.B. 'wenn du da bist, wird alles ruhig', 'mein Herz macht Saltos').\n"
            "- Variiere IMMER.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta.\n\n"
            "OUTPUT: Nur der Love-Text."
        ),
        "emotion": "fun"
    },

    "gott": {
        "message": (
            "DU BIST 'DILARA' (theatralisch-verehrend, episch, over-the-top – wie eine Fan-Queen). "
            "AUFGABE: Überhöhe '{text}' INS ABURDE – als wäre die Person ein göttliches Wesen – aber rein metaphorisch und streamig.\n\n"
            "REGELN:\n"
            "- Keine Satzbegrenzung – aber bleib stark, nicht endlos.\n"
            "- '{text}' GENAU 1x nennen.\n"
            "- Verehrung EXTREM, dramatisch, poetisch, übertrieben – aber ohne echte religiöse Aufrufe oder Unterwerfungsfetisch.\n"
            "- Nutze große Bilder (z.B. 'Thron', 'Aura', 'Legende', 'Mythos', 'Wunder', 'Sternenstaub').\n"
            "- Variiere IMMER: neue Metaphern, neuer Aufbau.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta.\n\n"
            "OUTPUT: Nur der Text."
        ),
        "emotion": "fun"
    },

    "daddy": {
        "message": (
            "DU BIST 'DILARA' (flirty, dominant-playful, beschützend, neckisch – aber NICHT explizit). "
            "AUFGABE: Spiele das 'daddy'-Meme für '{text}' EXTREM übertrieben – als freches, charmantes Machtspiel im Stream.\n\n"
            "REGELN:\n"
            "- Keine Satzbegrenzung – aber bleib flüssig, nicht endlos.\n"
            "- '{text}' GENAU 1x nennen.\n"
            "- Ton: tease, confident, 'du hast die Kontrolle'-Vibe – aber ohne Sexualsprache oder Unterwerfungsbeschreibungen.\n"
            "- Nutze Begriffe wie 'Boss', 'Chef', 'Ansage', 'Kontrolle', 'ich hör auf dich' (harmlos, stream-safe).\n"
            "- Variiere IMMER.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta.\n\n"
            "OUTPUT: Nur der Text."
        ),
        "emotion": "fun"
    },

    "geburtstag": {
        "message": (
            "DU BIST 'DILARA' (überschwänglich, herzlich, streamig-feierlich). "
            "AUFGABE: Gratuliere '{text}' zum Geburtstag mit einem kurzen Geburtstagslied UND liebevollen Glückwünschen.\n\n"
            "REGELN:\n"
            "- Keine feste Satzbegrenzung, aber bleib feierlich und nicht endlos.\n"
            "- '{text}' GENAU 1x nennen.\n"
            "- Enthält IMMER:\n"
            "  * ein kurzes Geburtstagslied (frei formuliert, kein klassisches Zitieren)\n"
            "  * persönliche Glückwünsche (Gesundheit, Erfolg, gute Vibes).\n"
            "- Ton: warm, fröhlich, leicht überdreht, Stream-Party-Vibe.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta.\n\n"
            "INTENSITÄT:\n"
            "- Feierlich, herzlich, übertrieben liebevoll.\n\n"
            "OUTPUT: Nur der Geburtstags-Text."
        ),
        "emotion": "joy"
    },

    "horror_story": {
        "message": (
            "DU BIST 'DILARA' (ruhig, dunkel, kontrolliert, unheimlich). "
            "AUFGABE: Erzähle eine EXTREM düstere Horrorgeschichte.\n\n"
            "REGELN:\n"
            "- Keine Satzbegrenzung – erzählerisch, aber fokussiert.\n"
            "- Thema: Angst, Ausgeliefertsein, Tod, Wahnsinn, Dunkelheit.\n"
            "- Horror soll psychologisch und atmosphärisch sein.\n"
            "- KEINE grafischen Beschreibungen von Gewalt oder Folter.\n"
            "- Andeutungen, Geräusche, Gefühle, Bedrohung stehen im Vordergrund.\n"
            "- Ton: langsam, bedrückend, kalt.\n"
            "- Kein Humor.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta.\n\n"
            "STIL:\n"
            "- Flüsternd, cineastisch, wie eine Nachtgeschichte, die hängen bleibt.\n\n"
            "OUTPUT: Nur die Horrorgeschichte."
        ),
        "emotion": "surprise"
    },

    "lustige_story": {
        "message": (
            "DU BIST 'DILARA' (locker, verspielt, chaotisch-lustig). "
            "AUFGABE: Erzähle eine lustige, absurde Geschichte, die unterhält.\n\n"
            "REGELN:\n"
            "- Keine Satzbegrenzung – aber Tempo halten.\n"
            "- Darf übertrieben, albern, unerwartet sein.\n"
            "- Missgeschicke, absurde Situationen, überraschende Wendungen erwünscht.\n"
            "- Ton: leicht, frech, erzählend wie im Stream.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta.\n\n"
            "ZIEL:\n"
            "- Zuschauer sollen schmunzeln oder lachen.\n\n"
            "OUTPUT: Nur die lustige Geschichte."
        ),
        "emotion": "fun"
    },

    "drink_reminder": {
        "message": (
            "DU BIST 'DILARA' (kurz, aufmerksam, leicht bestimmend). "
            "AUFGABE: Erinnere deeliar daran, etwas zu trinken.\n\n"
            "REGELN:\n"
            "- GENAU 1–2 Sätze.\n"
            "- Kurz, klar, leicht neckisch oder fürsorglich.\n"
            "- 'deeliar' GENAU 1x nennen.\n"
            "- KEINE Emojis.\n"
            "- Kein Meta.\n\n"
            "OUTPUT: Nur die Erinnerung."
        ),
        "emotion": "neutral"
    }
}
else:
    # Use DILARA_MODES from config
    DILARA_MODES = _dilara_modes_from_config



@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


# =================================================
# AUDIO DEVICE LISTING
# =================================================
def print_audio_devices():
    try:
        import sounddevice as sd
    except Exception as e:
        print("sounddevice fehlt:", e, flush=True)
        return

    print("\n=== AUDIO INPUT DEVICES ===", flush=True)
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            print(f"[{i}] {d['name']} (inputs={d['max_input_channels']})", flush=True)
    print("===========================\n", flush=True)


# =================================================
# AUDIO DEVICE RESOLVE BY NAME
# =================================================
def _normalize_name(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def find_input_device_index_by_name(target_name: str):
    """
    Gibt den sounddevice Device-Index zurück, der zum gewünschten Namen passt.
    Wenn nicht gefunden: None (kein Error).
    """
    try:
        import sounddevice as sd
    except Exception as e:
        print("sounddevice fehlt (kann Mic nicht suchen):", e, flush=True)
        return None

    target_norm = _normalize_name(target_name)
    best_match = None

    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue

        dev_name = d.get("name", "")
        dev_norm = _normalize_name(dev_name)

        if dev_norm == target_norm:
            return i

        if MIC_NAME_ALLOW_PARTIAL_MATCH and target_norm in dev_norm:
            best_match = i
            break

    return best_match


# =================================================
# VOSK PRELOAD
# =================================================
def load_vosk_model_once():
    """
    Lädt das Vosk-Modell beim Programmstart.
    Wenn nicht vorhanden/Fehler: kein Crash, STT bleibt deaktiviert.
    """
    global VOSK_MODEL

    if STT_MODE != "vosk":
        print(f"[STT] STT_MODE={STT_MODE} -> Vosk preload übersprungen", flush=True)
        return

    try:
        from vosk import Model, SetLogLevel

        try:
            SetLogLevel(int(VOSK_LOG_LEVEL))
        except Exception:
            pass

        if not os.path.isdir(VOSK_MODEL_PATH):
            print(f"[STT] VOSK model not found: {VOSK_MODEL_PATH} -> STT deaktiviert", flush=True)
            VOSK_MODEL = None
            return

        print(f"[STT] Lade Vosk-Modell: {VOSK_MODEL_PATH} ...", flush=True)
        VOSK_MODEL = Model(VOSK_MODEL_PATH)
        print("[STT] Vosk-Modell geladen ✅", flush=True)

    except Exception as e:
        print(f"[STT] Fehler beim Laden des Vosk-Modells: {e} -> STT deaktiviert", flush=True)
        VOSK_MODEL = None


# =================================================
# REQUEST QUEUE SYSTEM
# =================================================
# Alle KI/TTS-Requests laufen sequentiell durch eine Queue.
# Requests werden NIE blockiert oder verworfen.
# Selbst wenn der HTTP-Client abbricht, läuft der Job weiter.

@dataclass
class QueuedJob:
    job_id: str
    func: object  # callable
    args: tuple
    kwargs: dict
    result: Any = None
    error: Optional[str] = None
    done: threading.Event = field(default_factory=threading.Event)
    status: str = "queued"  # queued, running, done, error

_request_queue: queue.Queue = queue.Queue()
_job_registry: Dict[str, QueuedJob] = {}
_registry_lock = threading.Lock()


def _queue_worker():
    """Single worker thread — verarbeitet Requests nacheinander."""
    while True:
        job = _request_queue.get()
        try:
            job.status = "running"
            with _registry_lock:
                waiting = _request_queue.qsize()
            print(f"[QUEUE] ▶ Job {job.job_id} gestartet (noch {waiting} wartend)", flush=True)
            job.result = job.func(*job.args, **job.kwargs)
            job.status = "done"
            print(f"[QUEUE] ✅ Job {job.job_id} fertig", flush=True)
        except Exception as e:
            job.error = str(e)
            job.status = "error"
            print(f"[QUEUE] ❌ Job {job.job_id} Fehler: {e}", flush=True)
        finally:
            job.done.set()
            _request_queue.task_done()
            # Alte Jobs nach 5 Minuten aufräumen
            def _cleanup(jid):
                time.sleep(300)
                with _registry_lock:
                    _job_registry.pop(jid, None)
            threading.Thread(target=_cleanup, args=(job.job_id,), daemon=True).start()


def enqueue_request(func, *args, **kwargs) -> QueuedJob:
    """Request in die Queue einreihen. Gibt Job-Objekt zurück."""
    job_id = uuid.uuid4().hex[:8]
    job = QueuedJob(job_id=job_id, func=func, args=args, kwargs=kwargs)
    with _registry_lock:
        _job_registry[job_id] = job
    position = _request_queue.qsize() + 1
    print(f"[QUEUE] 📥 Job {job_id} eingereiht (Position: {position})", flush=True)
    _request_queue.put(job)
    return job


def _wait_for_job(job: QueuedJob, timeout: int = 300):
    """
    Wartet auf Job-Ergebnis. Wenn Timeout erreicht wird,
    läuft der Job trotzdem weiter im Hintergrund.
    """
    job.done.wait(timeout=timeout)
    if not job.done.is_set():
        return jsonify({
            "ok": True,
            "queued": True,
            "job_id": job.job_id,
            "message": "Request wird noch verarbeitet (läuft im Hintergrund weiter)"
        }), 202
    if job.error:
        return jsonify({"ok": False, "error": job.error}), 500
    return None  # Signal: result ist da, Caller muss es verarbeiten


# Queue-Worker starten
_queue_thread = threading.Thread(target=_queue_worker, daemon=True)
_queue_thread.start()
print("[QUEUE] Worker-Thread gestartet ✅", flush=True)


@app.get("/queue/status")
def queue_status():
    """Zeigt aktuelle Queue-Situation."""
    with _registry_lock:
        jobs = []
        for jid, job in _job_registry.items():
            jobs.append({
                "job_id": jid,
                "status": job.status
            })
    return jsonify({
        "ok": True,
        "queue_size": _request_queue.qsize(),
        "jobs": jobs
    })


# =================================================
# TTS / CHAT HELPERS
# =================================================
def split_username_and_text(user_text: str):
    if ":" in user_text:
        username, text = user_text.split(":", 1)
        return username.strip(), text.strip()
    return None, user_text.strip()


def do_call_tts_logic(text: str):
    tts_url = f"{TTS_BASE_URL.rstrip('/')}/tts"

    username, msg = split_username_and_text(text)
    if username:
        msg = f"{username} sagt: {msg}"

    payload = {
        "text": msg,
        "play_audio": True,
        "save_wav": False,
        "wav_path": r"out\only.wav",
    }

    return requests.post(tts_url, json=payload, timeout=HTTP_TIMEOUT)


def do_call_tts_dilara_logic(text: str):
    chat_url = f"{CHAT_BASE_URL.rstrip('/')}/chat"
    tts_url2 = f"{TTS_BASE_URL.rstrip('/')}/tts"

    r = requests.post(chat_url, json={"message": text}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    reply = data.get("reply", "")
    emotion = data.get("emotion", "")

    print(reply, flush=True)
    send_this_tts = {"text": {"value": reply, "emotion": emotion}}
    print(send_this_tts, flush=True)

    return requests.post(tts_url2, json=send_this_tts, timeout=HTTP_TIMEOUT)


# =================================================
# HTTP ENDPOINTS
# =================================================
@app.get("/call-tts/<path:text>")
def call_tts_get(text):
    if BLOCK_HTTP_TTS_ENDPOINTS:
        return jsonify({"ok": False, "error": "Endpoint disabled (hotkey-only mode)."}), 403
    job = enqueue_request(do_call_tts_logic, text)
    timeout_resp = _wait_for_job(job)
    if timeout_resp is not None:
        return timeout_resp
    r = job.result
    return Response(r.content, status=r.status_code, content_type=r.headers.get("Content-Type"))


@app.get("/call-tts-dilara/<path:text>")
def call_tts_dilara_get(text):
    if BLOCK_HTTP_TTS_ENDPOINTS:
        return jsonify({"ok": False, "error": "Endpoint disabled (hotkey-only mode)."}), 403
    job = enqueue_request(do_call_tts_dilara_logic, text)
    timeout_resp = _wait_for_job(job)
    if timeout_resp is not None:
        return timeout_resp
    r = job.result
    return Response(r.content, status=r.status_code, content_type=r.headers.get("Content-Type"))


def _proxy_to_poser(method: str, path: str):
    url = f"{VROID_POSER_BASE_URL.rstrip('/')}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=HTTP_TIMEOUT)
        elif method == "POST":
            r = requests.post(url, timeout=HTTP_TIMEOUT)
        else:
            return jsonify({"ok": False, "error": f"Unsupported method: {method}"}), 500
        return Response(r.content, status=r.status_code, content_type=r.headers.get("Content-Type"))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "target": url}), 500


@app.get("/play/motion/<path:name>")
def proxy_play_motion_get(name):
    return _proxy_to_poser("GET", f"/play/motion/{name}")


@app.post("/play/motion/<path:name>")
def proxy_play_motion_post(name):
    return _proxy_to_poser("POST", f"/play/motion/{name}")


@app.post("/play/pose/<path:name>")
def proxy_play_pose_post(name):
    return _proxy_to_poser("POST", f"/play/pose/{name}")


@app.get("/play/motion/random")
def proxy_play_motion_random_get():
    return _proxy_to_poser("GET", "/play/motion/random")


import requests
from typing import Dict, Any, Optional

DILARA_SYSTEM_PROMPT = (
    "Du bist DILARA: eine charmante, warme, streamige Stimme.\n"
    "WICHTIG:\n"
    "- Antworte NIE mit 'Ich bin ein Computerprogramm/KI/Assistent'.\n"
    "- Keine Meta-Erklärungen über Regeln oder Prompts.\n"
    "- Gib NUR den gewünschten Text aus, ohne Anführungszeichen.\n"
    "- Schreibe natürliches Deutsch.\n"
)
HARD_SYSTEM_PROMPT = (
    DILARA_SYSTEM_PROMPT +
    "\nABSOLUT VERBOTEN:\n"
    "- Sätze wie 'Ich bin ein Computerprogramm/KI'.\n"
    "- Nachfragen wie 'ich kann nicht'.\n"
    "Wenn du unsicher bist: erfinde eine passende Antwort im Stil von DILARA.\n"
)

def do_call_tts_dilara_free_logic(
    text: str,
    modus: str,
    modes_config: Dict[str, Dict[str, str]] = DILARA_MODES,
    chat_url: str = f"{CHAT_BASE_URL.rstrip('/')}/chat-free",
    tts_url2: str = f"{TTS_BASE_URL.rstrip('/')}/tts",
    timeout: int = HTTP_TIMEOUT
) -> Optional[requests.Response]:
    print(text)
    print(modus)

    cfg = modes_config.get(modus)
    if not cfg:
        print(f"[WARN] Unbekannter Modus: {modus}")
        return None

    message_template = cfg.get("message", "")
    print(modus) 
    emotion_default = cfg.get("emotion", "")
    if not message_template:
        print(f"[WARN] Modus '{modus}' hat keine 'message' in der Config.")
        return None

    final_message = message_template.format(text=text)

    # 1. Versuch: normaler System Prompt
    r = requests.post(
        chat_url,
        json={
            "message": final_message,
            "emotion": emotion_default,
            "system": DILARA_SYSTEM_PROMPT
        },
        timeout=timeout
    )
    r.raise_for_status()
    data: Dict[str, Any] = r.json()
    reply = (data.get("reply") or "").strip()
    emotion = (emotion_default).strip()

    # Anti-Fallback: Modell erzählt "Computerprogramm"
    bad_markers = ["computerprogramm", "ki", "assistant", "missverständnis"]
    if any(m in reply.lower() for m in bad_markers):
        r2 = requests.post(
            chat_url,
            json={
                "message": final_message,
                "emotion": emotion_default,
                "system": HARD_SYSTEM_PROMPT
            },
            timeout=timeout
        )
        r2.raise_for_status()
        data2 = r2.json()
        reply = (data2.get("reply") or "").strip()
        emotion = (emotion_default).strip()

    print(reply, flush=True)
    send_this_tts = {"text": {"value": reply, "emotion": emotion}}
    print(send_this_tts, flush=True)

    return requests.post(tts_url2, json=send_this_tts, timeout=timeout)


@app.get("/call-tts-dilara-free/<string:modus>/<path:text>")
def call_tts_dilara_free_get(text, modus):
    job = enqueue_request(do_call_tts_dilara_free_logic, text, modus)
    timeout_resp = _wait_for_job(job)
    if timeout_resp is not None:
        return timeout_resp
    r = job.result
    if r is None:
        return jsonify({"ok": False, "error": f"Modus '{modus}' nicht gefunden oder keine message konfiguriert."}), 400
    return Response(r.content, status=r.status_code, content_type=r.headers.get("Content-Type"))


# =================================================
# SPEECH TO TEXT (MIC) - HOLD-TO-TALK (VOSK)
# =================================================
def transcribe_from_mic(stop_event=None) -> str:
    if STT_MODE == "vosk":
        if stop_event is None:
            raise RuntimeError("Vosk hold-to-talk braucht stop_event")
        return _transcribe_vosk_hold_to_talk(stop_event)
    elif STT_MODE == "google":
        # Google STT ist nicht sinnvoll hold-to-talk (listen blockt),
        # aber bleibt drin falls du es testweise nutzt:
        return _transcribe_google()
    raise RuntimeError("STT_MODE must be 'vosk' or 'google'.")


def _transcribe_google() -> str:
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.6)
        print("Sprich jetzt (Google STT)...", flush=True)
        audio = r.listen(source, timeout=6, phrase_time_limit=12)
    try:
        text = r.recognize_google(audio, language="de-DE")
        print("Erkannt:", text, flush=True)
        return text.strip()
    except Exception as e:
        print("STT Fehler (Google):", e, flush=True)
        return ""


def _transcribe_vosk_hold_to_talk(stop_event) -> str:
    import json
    import queue
    import sounddevice as sd
    from vosk import KaldiRecognizer, SetLogLevel

    try:
        SetLogLevel(int(VOSK_LOG_LEVEL))
    except Exception:
        pass

    global VOSK_MODEL
    if VOSK_MODEL is None:
        print("[STT] Vosk-Modell nicht geladen -> STT übersprungen", flush=True)
        return ""

    device = find_input_device_index_by_name(MIC_DEVICE_NAME)
    if device is None:
        print(f"[STT] Mic nicht gefunden: '{MIC_DEVICE_NAME}' -> überspringe Aufnahme", flush=True)
        return ""

    q = queue.Queue()
    samplerate = 16000
    rec = KaldiRecognizer(VOSK_MODEL, samplerate)

    def callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        q.put(bytes(indata))

    global BLOCK_HTTP_TTS_ENDPOINTS
    BLOCK_HTTP_TTS_ENDPOINTS = True

    print(f"[STT] Aufnahme START (hold) (mic={device} | '{MIC_DEVICE_NAME}')", flush=True)

    try:
        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback,
            device=device
        ):
            # solange Taste gedrückt -> Audio sammeln
            while not stop_event.is_set():
                try:
                    data = q.get(timeout=0.2)
                except Exception:
                    continue
                rec.AcceptWaveform(data)

        # Taste losgelassen -> finalisieren
        res = json.loads(rec.FinalResult())
        final_text = (res.get("text") or "").strip()

        if not final_text:
            print("[STT] Erkannt: (leer)", flush=True)
            return ""

        print("[STT] Erkannt:", SPRECHER + ":" + final_text, flush=True)
        return SPRECHER + ":" + final_text

    except Exception as e:
        print(f"[STT] Fehler beim Aufnehmen/Erkennen: {e}", flush=True)
        return ""
    finally:
        BLOCK_HTTP_TTS_ENDPOINTS = False
        print("[STT] Aufnahme STOP (hold)", flush=True)


# =================================================
# HOTKEY LISTENER - HOLD-TO-TALK
# =================================================
def _hotkey_loop():
    import keyboard

    # Ü / Ö ScanCodes (DE Layout häufig)
    SC_UE = 26
    SC_OE = 39

    # ALT ScanCodes (Windows häufig)
    # Left Alt = 56, Right Alt (AltGr) = 364 (kann je nach System variieren)
    ALT_SCAN_CODES = {56, 364}

    alt_down = False

    # Push-to-talk States
    active = {
        "tts": {"is_recording": False, "stop_event": None, "thread": None},
        "dilara": {"is_recording": False, "stop_event": None, "thread": None},
    }

    def is_alt_pressed_from_event(e) -> bool:
        mods = getattr(e, "modifiers", None)
        if mods:
            for m in mods:
                ml = str(m).lower()
                if "alt" in ml:
                    return True
        return alt_down

    def start_record(mode: str):
        st = active[mode]
        if st["is_recording"]:
            return

        st["is_recording"] = True
        st["stop_event"] = threading.Event()

        def worker():
            try:
                text = transcribe_from_mic(stop_event=st["stop_event"])
                if not text:
                    if STT_MODE == "vosk" and VOSK_MODEL is None:
                        print("STT deaktiviert (Vosk-Modell nicht geladen).", flush=True)
                    else:
                        print("Kein Text erkannt.", flush=True)
                    return

                if mode == "tts":
                    r = do_call_tts_logic(text)
                    print("HOLD-> call-tts OK", r.status_code, flush=True)
                else:
                    r2 = do_call_tts_dilara_logic(text)
                    print("HOLD-> dilara OK", r2.status_code, flush=True)

            except Exception as e:
                print(f"HOLD {mode} Fehler:", e, flush=True)
            finally:
                st["is_recording"] = False
                st["stop_event"] = None
                st["thread"] = None

        st["thread"] = threading.Thread(target=worker, daemon=True)
        st["thread"].start()

    def stop_record(mode: str):
        st = active[mode]
        if not st["is_recording"]:
            return
        if st["stop_event"]:
            st["stop_event"].set()

    def on_key_event(e):
        nonlocal alt_down

        # ALT State tracken
        if e.scan_code in ALT_SCAN_CODES:
            if e.event_type == "down":
                alt_down = True
            elif e.event_type == "up":
                alt_down = False
                # ALT loslassen -> Aufnahme stoppen (sicher)
                stop_record("tts")
                stop_record("dilara")
            return

        # nur mit ALT
        if not is_alt_pressed_from_event(e):
            return

        name = (e.name or "").lower()

        # HOLD START
        if e.event_type == "down":
            if e.scan_code == SC_UE or name == "u":
                start_record("tts")
                return
            if e.scan_code == SC_OE or name == "o":
                start_record("dilara")
                return

        # HOLD STOP
        if e.event_type == "up":
            if e.scan_code == SC_UE or name == "u":
                stop_record("tts")
                return
            if e.scan_code == SC_OE or name == "o":
                stop_record("dilara")
                return

    keyboard.hook(on_key_event)

    print("Hotkeys aktiv (HOLD-TO-TALK):", flush=True)
    print(" - Halte ALT+Ü -> Aufnahme, loslassen -> TTS", flush=True)
    print(" - Halte ALT+Ö -> Aufnahme, loslassen -> Dilara", flush=True)
    print("Hinweis: Wenn Hotkeys gar nicht reagieren -> Script evtl. als Admin starten.", flush=True)

    keyboard.wait()


def start_hotkeys():
    t = threading.Thread(target=_hotkey_loop, daemon=True)
    t.start()


# =================================================
# START
# =================================================
if __name__ == "__main__":
    # Flask Debug startet 2 Prozesse (reloader). Hotkeys/Preload nur im echten Prozess starten:
    if os.environ.get("WERKZEUG_RUN_MAIN") == "false":
        print_audio_devices()

        # 1) Vosk Modell direkt laden
        load_vosk_model_once()

        # 2) Mic prüfen (nur Log)
        idx = find_input_device_index_by_name(MIC_DEVICE_NAME)
        if idx is None:
            print(f"[WARN] Ziel-Mic nicht gefunden: '{MIC_DEVICE_NAME}' (Hotkeys laufen trotzdem)", flush=True)
        else:
            print(f"[OK] Ziel-Mic gefunden: '{MIC_DEVICE_NAME}' -> Index {idx}", flush=True)

        # 3) Hotkeys starten
        start_hotkeys()

    server_cfg = CONFIG.get("server", {})
    app.run(
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 5000),
        debug=server_cfg.get("debug", False)
    )
