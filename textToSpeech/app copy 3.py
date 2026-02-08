import io
import os
import wave
import threading
import random
import time
from datetime import datetime

import requests
import numpy as np
import sounddevice as sd
from flask import Flask, request, jsonify, Response
from pythonosc.udp_client import SimpleUDPClient

# =====================================================
# GPU CONFIG (MANUAL SELECTION)
# =====================================================
# None  = CPU
# 0     = cuda:0
# 1     = cuda:1
GPU_INDEX = 0  # <<< HIER wählen: 0 oder 1 oder None

# =====================================================
# TORCH + PATCHES (XTTS compatibility)
# =====================================================
import torch

# --- FIX PyTorch 2.6+ weights_only default (required for Coqui XTTS checkpoints) ---
_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = torch_load_compat

# --- FIX torchaudio/torchcodec: WAV via soundfile laden statt torchcodec ---
import soundfile as sf
try:
    import torchaudio

    def sf_load(path):
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 1:
            audio = audio[None, :]      # (1, T)
        else:
            audio = audio.T             # (C, T)
        return torch.from_numpy(audio), sr

    torchaudio.load = sf_load
except Exception:
    pass

from TTS.api import TTS

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
FAKE_LIPSYNC_ENABLED = False
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

lipsync_thread = threading.Thread(target=lipsync_loop, daemon=True)
lipsync_thread.start()

# =====================================================
# AUDIO OUTPUT DEVICE
# =====================================================
VOICEMOD_OUTPUT_NAME_SUBSTRING = "cable input"

def find_output_device_id(name_substring: str) -> int:
    name_substring = name_substring.lower()
    devices = sd.query_devices()

    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) > 0:
            if name_substring in dev["name"].lower():
                return idx

    raise RuntimeError(f"Audio output device not found for substring: {name_substring!r}")

CABLE_DEVICE_ID = find_output_device_id(VOICEMOD_OUTPUT_NAME_SUBSTRING)
AUDIO_LOCK = threading.Lock()

# =====================================================
# XTTS CONFIG
# =====================================================
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "de"

# Emotion -> Referenz-WAV (Speaker Reference)
EMOTION_WAVS = {
    "neutral": "voices/neutral.wav",
    # "happy":   "voices/happy.wav",
    # "sad":     "voices/sad.wav",
    # "angry":   "voices/angry.wav",
}
DEFAULT_EMOTION = "neutral"

# XTTS v2 typischerweise 24000 Hz, mono
XTTS_SR = 24000

# =====================================================
# HELPERS
# =====================================================
def ensure_voice_files():
    missing = [p for p in EMOTION_WAVS.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing voice reference wav(s). Create these files:\n" + "\n".join(missing)
        )

def choose_torch_device() -> str:
    if GPU_INDEX is None:
        print("[XTTS] GPU disabled -> using CPU")
        return "cpu"

    if not torch.cuda.is_available():
        print("[XTTS] CUDA not available -> fallback to CPU")
        return "cpu"

    count = torch.cuda.device_count()
    if GPU_INDEX < 0 or GPU_INDEX >= count:
        print(f"[XTTS] Invalid GPU index {GPU_INDEX} (found {count}) -> CPU")
        return "cpu"

    name = torch.cuda.get_device_name(GPU_INDEX)
    props = torch.cuda.get_device_properties(GPU_INDEX)
    vram = props.total_memory / (1024 ** 3)

    print(f"[XTTS] Using GPU {GPU_INDEX}: {name} ({vram:.1f} GB VRAM)")
    return f"cuda:{GPU_INDEX}"

def float32_to_int16(wav_f32: np.ndarray) -> np.ndarray:
    wav_f32 = np.asarray(wav_f32, dtype=np.float32)
    wav_f32 = np.clip(wav_f32, -1.0, 1.0)
    return (wav_f32 * 32767.0).astype(np.int16)

def pcm16_to_wav_bytes(pcm16: np.ndarray, sr: int, ch: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()

def save_wav_file(pcm16: np.ndarray, sr: int, ch: int, path: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as f:
        f.write(pcm16_to_wav_bytes(pcm16, sr, ch))

# =====================================================
# LOAD MODEL ON STARTUP
# =====================================================
ensure_voice_files()
TTS_DEVICE = choose_torch_device()

print(f"[XTTS] Loading model {TTS_MODEL_NAME} on {TTS_DEVICE} ...")
tts_model = TTS(model_name=TTS_MODEL_NAME).to(TTS_DEVICE)

# Warmup (macht ersten Play schneller)
try:
    _ = tts_model.tts(text="Warmup.", speaker_wav=EMOTION_WAVS[DEFAULT_EMOTION], language=LANGUAGE)
except Exception:
    pass

# =====================================================
# XTTS SYNTH
# =====================================================
def synthesize_wav_f32(text: str, emotion: str):
    emotion = (emotion or DEFAULT_EMOTION).lower().strip()
    ref = EMOTION_WAVS.get(emotion, EMOTION_WAVS[DEFAULT_EMOTION])

    wav = tts_model.tts(
        text=text,
        speaker_wav=ref,
        language=LANGUAGE,
    )

    wav_f32 = np.asarray(wav, dtype=np.float32)
    wav_f32 = np.clip(wav_f32, -1.0, 1.0)
    return wav_f32, XTTS_SR, 1, emotion, ref

# =====================================================
# AUDIO PLAYBACK (AUTO LIPSYNC + EMOTION POST)
# =====================================================
def play_audio_blocking(wav_f32: np.ndarray, sr: int, emotion: str):
    global FAKE_LIPSYNC_ENABLED

    with AUDIO_LOCK:
        FAKE_LIPSYNC_ENABLED = True

        # Emotion an dein anderes System schicken (wie vorher)
        try:
            requests.post("http://127.0.0.1:5004/emotion", json={"emotion": emotion}, timeout=30)
        except Exception:
            pass

        sd.stop()
        sd.play(wav_f32, sr, device=CABLE_DEVICE_ID)
        sd.wait()

        FAKE_LIPSYNC_ENABLED = False

        # zurück auf natural (wie vorher)
        try:
            requests.post("http://127.0.0.1:5004/emotion", json={"emotion": "natural"}, timeout=30)
        except Exception:
            pass

        clear_mouth()

# =====================================================
# SPEAK TASK
# =====================================================
def speak_task(text: str, save_wav: bool, play_audio: bool, wav_path: str | None, emotion: str):
    try:
        wav_f32, sr, ch, used_emotion, ref = synthesize_wav_f32(text, emotion)

        if save_wav:
            if not wav_path:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_path = f"tts_{ts}.wav"
            pcm16 = float32_to_int16(wav_f32)
            save_wav_file(pcm16, sr, ch, wav_path)

        if play_audio:
            play_audio_blocking(wav_f32, sr, used_emotion)

    except Exception as e:
        print("TTS error:", repr(e), flush=True)

# =====================================================
# ROUTES
# =====================================================
@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "model": TTS_MODEL_NAME,
        "language": LANGUAGE,
        "device": TTS_DEVICE,
        "gpu_index": GPU_INDEX,
        "lipsync_running": lipsync_thread.is_alive(),
        "output_device": sd.query_devices(CABLE_DEVICE_ID)["name"],
        "emotions": list(EMOTION_WAVS.keys()),
        "default_emotion": DEFAULT_EMOTION,
    })

@app.post("/tts")
def tts():
    """
    JSON body example:
    {
      "text": "Hallo Meister!",
      "emotion": "neutral",
      "save_wav": true,
      "play_audio": true,
      "wav_path": "out/test.wav",
      "return_wav": false
    }

    Backward compatible: Falls text ein Objekt ist wie:
      { "value": "...", "emotion": "neutral" }
    """
    payload = request.get_json(silent=True) or {}
    print(payload, flush=True)

    text = payload.get("text") or ""
    emotion = payload.get("emotion") or "natural"

    # backward compatible handling (dein alter Sonderfall)
    if isinstance(text, dict) and "value" in text:
        if "emotion" in text:
            emotion = text.get("emotion") or emotion
        text = text.get("value") or ""

    text = str(text).strip()
    emotion = str(emotion).strip().lower()

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    save_wav = bool(payload.get("save_wav", False))
    play_audio = bool(payload.get("play_audio", True))
    wav_path = payload.get("wav_path")
    return_wav = bool(payload.get("return_wav", False))

    # WAV direkt zurückgeben (blocking synth, optional zusätzlich abspielen/speichern)
    if return_wav:
        try:
            wav_f32, sr, ch, used_emotion, ref = synthesize_wav_f32(text, emotion)
            pcm16 = float32_to_int16(wav_f32)
            wav_bytes = pcm16_to_wav_bytes(pcm16, sr, ch)

            if save_wav:
                if not wav_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wav_path = f"tts_{ts}.wav"
                save_wav_file(pcm16, sr, ch, wav_path)

            if play_audio:
                threading.Thread(
                    target=play_audio_blocking,
                    args=(wav_f32, sr, used_emotion),
                    daemon=True
                ).start()

            return Response(wav_bytes, mimetype="audio/wav")
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    # Standard: async task (nicht blockieren)
    threading.Thread(
        target=speak_task,
        args=(text, save_wav, play_audio, wav_path, emotion),
        daemon=True
    ).start()

    return jsonify({
        "ok": True,
        "status": "queued",
        "save_wav": save_wav,
        "play_audio": play_audio,
        "wav_path": wav_path,
        "return_wav": return_wav,
        "emotion": emotion,
    })

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
