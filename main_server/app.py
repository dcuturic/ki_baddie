from flask import Flask, jsonify, request, Response
import requests
import time
import os
import threading

app = Flask(__name__)

# =================================================
# SETTINGS
# =================================================
# Wenn True: blockiert HTTP Requests an /call-tts/<...> und /call-tts-dilara/<...>
# Hotkeys funktionieren trotzdem (rufen Funktionen direkt auf).
BLOCK_HTTP_TTS_ENDPOINTS = False

SPRECHER = "deeliar"

# Hotkeys (DE Layout)
HOTKEY_CALL_TTS = "alt+ü"
HOTKEY_CALL_DILARA = "alt+ö"

# Fallback Hotkeys, falls ü/ö nicht erkannt wird
HOTKEY_CALL_TTS_FALLBACK = "alt+u"
HOTKEY_CALL_DILARA_FALLBACK = "alt+o"

# Speech-to-text: "vosk" (offline) oder "google" (online)
STT_MODE = os.getenv("STT_MODE", "vosk").lower()

# Vosk German Model Path
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", r"models\vosk-model-small-de-0.15")

# >>> WICHTIG: Mikro auswählen <<<
# None = Windows Default Microphone
# Sonst: Index aus print_audio_devices() nehmen
MIC_DEVICE_INDEX = 17  # z.B. 3

# Optional: Vosk Logs leiser machen
# "0" = keine Logs, "1" = normal
VOSK_LOG_LEVEL = 0

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
# TTS / CHAT HELPERS
# =================================================

def split_username_and_text(user_text: str):
    if ":" in user_text:
        username, text = user_text.split(":", 1)
        return username.strip(), text.strip()
    else:
        return None, user_text.strip()

def _block_if_configured():
    if not BLOCK_HTTP_TTS_ENDPOINTS:
        return None
    return jsonify({"ok": False, "error": "Endpoint disabled (hotkey-only mode)."}), 403

def do_call_tts_logic(text: str):
    """
    gleiche Logik wie /call-tts/<path:text>, aber als Funktion
    """
    tts_url = "http://127.0.0.1:5003/tts"

    username, msg = split_username_and_text(text)
    if username:
        msg = f"{username} sagt: {msg}"

    payload = {
        "text": msg,
        "play_audio": True,
        "save_wav": False,
        "wav_path": r"out\only.wav",
    }

    r = requests.post(tts_url, json=payload, timeout=30)
    return r

def do_call_tts_dilara_logic(text: str):
    """
    gleiche Logik wie /call-tts-dilara/<path:text>, aber als Funktion
    """
    chat_url = "http://127.0.0.1:5001/chat"
    tts_url2 = "http://127.0.0.1:5003/tts"

    send_this = {"message": text}

    r = requests.post(chat_url, json=send_this, timeout=30)
    r.raise_for_status()
    reply = r.json()["reply"]
    emotion = r.json()["emotion"]

    print(reply, flush=True)

    send_this_tts = {"text": {"value": reply, "emotion": emotion}}
    print(send_this_tts, flush=True)

    r2 = requests.post(tts_url2, json=send_this_tts, timeout=30)
    return r2


# =================================================
# HTTP ENDPOINTS
# =================================================

@app.get("/call-tts/<path:text>")
def call_tts_get(text):
    blocked = _block_if_configured()
    if blocked:
        return blocked
    try:
        r = do_call_tts_logic(text)
        return Response(r.content, status=r.status_code, content_type=r.headers.get("Content-Type"))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/call-tts-dilara/<path:text>")
def call_tts_dilara_get(text):
    blocked = _block_if_configured()
    if blocked:
        return blocked
    try:
        r2 = do_call_tts_dilara_logic(text)
        return Response(r2.content, status=r2.status_code, content_type=r2.headers.get("Content-Type"))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# =================================================
# SPEECH TO TEXT (MIC)
# =================================================

def transcribe_from_mic() -> str:
    if STT_MODE == "vosk":
        return _transcribe_vosk()
    elif STT_MODE == "google":
        return _transcribe_google()
    else:
        raise RuntimeError("STT_MODE must be 'vosk' or 'google'.")

def _transcribe_google() -> str:
    # (nicht empfohlen bei deinem Setup, weil PyAudio unter Py3.14 stresst)
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

def _transcribe_vosk() -> str:
    import json
    import queue
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer, SetLogLevel

    # Logs runterdrehen
    try:
        SetLogLevel(int(VOSK_LOG_LEVEL))
    except Exception:
        pass

    if not os.path.isdir(VOSK_MODEL_PATH):
        raise RuntimeError(
            f"VOSK model not found: {VOSK_MODEL_PATH}\n"
            f"Download German model and set VOSK_MODEL_PATH."
        )

    q = queue.Queue()
    samplerate = 16000

    # Modell NUR einmal laden (Performance + weniger Spam)
    # -> wir cachen es in einer Funktion-Static Variable
    if not hasattr(_transcribe_vosk, "_model"):
        _transcribe_vosk._model = Model(VOSK_MODEL_PATH)

    rec = KaldiRecognizer(_transcribe_vosk._model, samplerate)

    def callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        q.put(bytes(indata))
    BLOCK_HTTP_TTS_ENDPOINTS = True
    device = MIC_DEVICE_INDEX  # None = default
    print(f"Sprich jetzt (Vosk offline STT)... (mic={device})", flush=True)

    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
        device=device
    ):
        start = time.time()
        final_text = ""

        # Du hast 12 Sekunden Zeit zu sprechen
        while time.time() - start < 12:
            data = q.get()
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                final_text = (res.get("text") or "").strip()
                break

        if not final_text:
            res = json.loads(rec.FinalResult())
            final_text = (res.get("text") or "").strip()

    print("Erkannt:",SPRECHER+":"+final_text, flush=True)
    BLOCK_HTTP_TTS_ENDPOINTS = False
    return SPRECHER+":"+final_text


# =================================================
# HOTKEY LISTENER
# =================================================

def _hotkey_loop():
    import keyboard

    def fire_tts():
        try:
            text = transcribe_from_mic()
            if not text:
                print("Kein Text erkannt.", flush=True)
                return
            r = do_call_tts_logic(text)
            print("ALT+Ü -> call-tts OK", r.status_code, flush=True)
        except Exception as e:
            print("ALT+Ü Fehler:", e, flush=True)

    def fire_dilara():
        try:
            text = transcribe_from_mic()
            if not text:
                print("Kein Text erkannt.", flush=True)
                return
            r2 = do_call_tts_dilara_logic(text)
            print("ALT+Ö -> dilara OK", r2.status_code, flush=True)
        except Exception as e:
            print("ALT+Ö Fehler:", e, flush=True)

    keyboard.add_hotkey(HOTKEY_CALL_TTS, fire_tts)
    keyboard.add_hotkey(HOTKEY_CALL_DILARA, fire_dilara)

    # fallback
    keyboard.add_hotkey(HOTKEY_CALL_TTS_FALLBACK, fire_tts)
    keyboard.add_hotkey(HOTKEY_CALL_DILARA_FALLBACK, fire_dilara)

    print("Hotkeys aktiv:", flush=True)
    print(f" - {HOTKEY_CALL_TTS} (fallback {HOTKEY_CALL_TTS_FALLBACK}) -> TTS", flush=True)
    print(f" - {HOTKEY_CALL_DILARA} (fallback {HOTKEY_CALL_DILARA_FALLBACK}) -> Dilara", flush=True)

    keyboard.wait()

def start_hotkeys():
    t = threading.Thread(target=_hotkey_loop, daemon=True)
    t.start()


# =================================================
# START
# =================================================

if __name__ == "__main__":
    # Flask Debug startet 2 Prozesse (reloader). Hotkeys nur im echten Prozess starten:
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print_audio_devices()
        start_hotkeys()

    app.run(host="0.0.0.0", port=5050, debug=True)
