import io
import os
import wave
import asyncio
import threading
import random
import time
from datetime import datetime
import requests
import numpy as np
import sounddevice as sd
import miniaudio
import edge_tts

from flask import Flask, request, jsonify, Response
from pythonosc.udp_client import SimpleUDPClient

# =====================================================
# FLASK
# =====================================================

app = Flask(__name__)

# =====================================================
# OSC / VSEEFACE
# =====================================================

OSC_HOST = "127.0.0.1"
OSC_PORT = 39539

osc_client = SimpleUDPClient(OSC_HOST, OSC_PORT)

# =====================================================
# FAKE LIPSYNC – GLOBAL CONFIG
# =====================================================

FAKE_LIPSYNC_ENABLED = False          # wird automatisch gesetzt
FAKE_LIPSYNC_STOP_EVENT = threading.Event()

# realistische Sprachgeschwindigkeit (6–10 Silben/s)
SYLLABLE_MIN_DELAY = 0.07
SYLLABLE_MAX_DELAY = 0.16

# Mundöffnung
MOUTH_MIN_OPEN = 0.55
MOUTH_MAX_OPEN = 1.0

# Pausen
PAUSE_CHANCE = 0.18
PAUSE_MIN = 0.25
PAUSE_MAX = 0.6

CLEAR_BEFORE_EACH = True

VISEMES = [
    ("A", 1.0),
    ("O", 0.9),
    ("E", 0.6),
    ("I", 0.5),
    ("U", 0.4),
]

# =====================================================
# LIPSYNC HELPERS
# =====================================================

def set_viseme(key: str, value: float):
    osc_client.send_message("/VMC/Ext/Blend/Val", [key, float(value)])
    osc_client.send_message("/VMC/Ext/Blend/Apply", 1)

def clear_mouth():
    for k, _ in VISEMES:
        osc_client.send_message("/VMC/Ext/Blend/Val", [k, 0.0])
    osc_client.send_message("/VMC/Ext/Blend/Apply", 1)

# =====================================================
# LIPSYNC THREAD
# =====================================================

def lipsync_loop():
    global FAKE_LIPSYNC_ENABLED

    while not FAKE_LIPSYNC_STOP_EVENT.is_set():

        if not FAKE_LIPSYNC_ENABLED:
            clear_mouth()
            time.sleep(0.1)
            continue

        if random.random() < PAUSE_CHANCE:
            clear_mouth()
            time.sleep(random.uniform(PAUSE_MIN, PAUSE_MAX))
            continue

        if CLEAR_BEFORE_EACH:
            clear_mouth()

        key, weight = random.choice(VISEMES)
        openness = random.uniform(MOUTH_MIN_OPEN, MOUTH_MAX_OPEN) * weight

        set_viseme(key, openness)

        time.sleep(random.uniform(SYLLABLE_MIN_DELAY, SYLLABLE_MAX_DELAY))


lipsync_thread = threading.Thread(
    target=lipsync_loop,
    daemon=True
)
lipsync_thread.start()

# =====================================================
# AUDIO OUTPUT DEVICE
# =====================================================

VOICEMOD_OUTPUT_NAME_SUBSTRING = "cable input"

def find_output_device_id(name_substring: str) -> int:
    name_substring = name_substring.lower()
    devices = sd.query_devices()

    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            if name_substring in dev["name"].lower():
                return idx

    raise RuntimeError("Audio output device not found")

CABLE_DEVICE_ID = find_output_device_id(VOICEMOD_OUTPUT_NAME_SUBSTRING)
AUDIO_LOCK = threading.Lock()

# =====================================================
# EDGE TTS CONFIG
# =====================================================
# de-DE-AmalaNeural
# de-DE-ConradNeural
# de-DE-KatjaNeural
# de-DE-KillianNeural
VOICE = "de-DE-AmalaNeural"
RATE = "+15%"
#PITCH = "-25Hz"
PITCH = "+20Hz"

# =====================================================
# TTS HELPERS
# =====================================================

async def synthesize_mp3_bytes(text: str) -> bytes:
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=RATE,
        pitch=PITCH,
    )
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()

def mp3_to_pcm(mp3_bytes: bytes):
    decoded = miniaudio.decode(mp3_bytes)
    samples = np.frombuffer(decoded.samples, dtype=np.int16)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels)
    return samples, decoded.sample_rate, decoded.nchannels

def pcm_to_wav_bytes(pcm16, sr, ch):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()

# =====================================================
# AUDIO PLAYBACK (AUTO LIPSYNC)
# =====================================================

def play_audio_blocking(pcm16, sr,emotion):
    global FAKE_LIPSYNC_ENABLED

    with AUDIO_LOCK:
        FAKE_LIPSYNC_ENABLED = True

        send_this_emotion ={"emotion":emotion}
        tts_url_emotion = "http://127.0.0.1:5004/emotion"
        r = requests.post(tts_url_emotion, json=send_this_emotion, timeout=30)

        sd.stop()
        sd.play(pcm16, sr, device=CABLE_DEVICE_ID)
        sd.wait()
        FAKE_LIPSYNC_ENABLED = False

        send_this_emotion ={"emotion":"natural"}
        tts_url_emotion = "http://127.0.0.1:5004/emotion"
        r = requests.post(tts_url_emotion, json=send_this_emotion, timeout=30)

        clear_mouth()

# =====================================================
# SPEAK TASK
# =====================================================

def speak_task(text, save_wav, play_audio, wav_path,emotion):
    try:
        mp3 = asyncio.run(synthesize_mp3_bytes(text))
        pcm16, sr, ch = mp3_to_pcm(mp3)

        if save_wav:
            if not wav_path:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_path = f"tts_{ts}.wav"
            with open(wav_path, "wb") as f:
                f.write(pcm_to_wav_bytes(pcm16, sr, ch))

        if play_audio:
            play_audio_blocking(pcm16, sr,emotion)

    except Exception as e:
        print("TTS error:", e)

# =====================================================
# ROUTES
# =====================================================

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "voice": VOICE,
        "rate": RATE,
        "pitch": PITCH,
        "lipsync_running": lipsync_thread.is_alive(),
        "output_device": sd.query_devices(CABLE_DEVICE_ID)["name"]
    })


@app.post("/tts")
def tts():
    """
    JSON body example:
    {
      "text": "Hallo Meister!",
      "save_wav": true,
      "play_audio": true,
      "wav_path": "out/test.wav",
      "return_wav": false
    }
    """
    payload = request.get_json(silent=True) or {}
    print(payload,flush=True)
    text = (payload.get("text") or "")
    emotion = (payload.get("emotion") or "natural")
    if("value" in text):

        if("emotion" in text):
            emotion = text["emotion"] 
        text = text["value"]

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    # optional overrides for voice params (global)
    global VOICE, RATE, PITCH
    if payload.get("voice"):
        VOICE = str(payload["voice"])
    if payload.get("rate"):
        RATE = str(payload["rate"])
    if payload.get("pitch"):
        PITCH = str(payload["pitch"])

    save_wav = bool(payload.get("save_wav", False))
    play_audio = bool(payload.get("play_audio", True))
    wav_path = payload.get("wav_path")  # optional
    return_wav = bool(payload.get("return_wav", False))

    # Wenn du WAV direkt als Response willst:
    if return_wav:
        try:
            mp3_bytes = asyncio.run(synthesize_mp3_bytes(text))
            pcm16, sr, ch = mp3_bytes_to_pcm16_numpy(mp3_bytes)
            wav_bytes = pcm16_to_wav_bytes(pcm16, sr, ch)

            # optional zusätzlich speichern/abspielen
            if save_wav:
                if not wav_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wav_path = f"tts_{ts}.wav"
                save_wav_file(pcm16, sr, ch, wav_path)

            if play_audio:
                threading.Thread(target=play_audio_blocking, args=(pcm16, sr,emotion), daemon=True).start()

            return Response(wav_bytes, mimetype="audio/wav")

        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    # Standard: async task (nicht blockieren)
    threading.Thread(
        target=speak_task,
        args=(text, save_wav, play_audio, wav_path,emotion),
        daemon=True
    ).start()

    return jsonify({
        "ok": True,
        "status": "queued",
        "save_wav": save_wav,
        "play_audio": play_audio,
        "wav_path": wav_path,
        "return_wav": return_wav
    })


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
