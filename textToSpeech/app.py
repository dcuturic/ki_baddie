import io
import os
import wave
import asyncio
import threading
from datetime import datetime

from flask import Flask, request, jsonify, Response
import edge_tts

import numpy as np
import sounddevice as sd
import miniaudio

app = Flask(__name__)

# ---------------------------
# OUTPUT DEVICE CONFIG
# ---------------------------
VOICEMOD_OUTPUT_NAME_SUBSTRING = "cable input"  # z.B. "cable input"

def find_output_device_id(name_substring: str) -> int:
    name_substring = (name_substring or "").lower().strip()
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) > 0:
            if name_substring in (dev.get("name") or "").lower():
                return idx

    print("Available output devices:")
    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) > 0:
            print(f"  [{idx}] {dev.get('name')} (outputs={dev.get('max_output_channels')})")

    raise RuntimeError(f"Output device not found containing: '{name_substring}'")

CABLE_DEVICE_ID = find_output_device_id(VOICEMOD_OUTPUT_NAME_SUBSTRING)
print(f"[AUDIO] Using output device id={CABLE_DEVICE_ID}: {sd.query_devices(CABLE_DEVICE_ID)['name']}")

AUDIO_LOCK = threading.Lock()

# ---------------------------
# EDGE TTS CONFIG
# ---------------------------
VOICE = "de-DE-KatjaNeural"
RATE = "+10%"
PITCH = "+15Hz"


# ---------------------------
# TTS + AUDIO HELPERS
# ---------------------------
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


def mp3_bytes_to_pcm16_numpy(mp3_bytes: bytes):
    decoded = miniaudio.decode(mp3_bytes)  # PCM16
    samples = np.frombuffer(decoded.samples, dtype=np.int16)
    if decoded.nchannels > 1:
        samples = samples.reshape(-1, decoded.nchannels)
    return samples, decoded.sample_rate, decoded.nchannels


def pcm16_to_wav_bytes(pcm16: np.ndarray, sample_rate: int, channels: int) -> bytes:
    pcm16 = np.asarray(pcm16, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def save_wav_file(pcm16: np.ndarray, sample_rate: int, channels: int, wav_path: str) -> str:
    wav_bytes = pcm16_to_wav_bytes(pcm16, sample_rate, channels)
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    with open(wav_path, "wb") as f:
        f.write(wav_bytes)
    return wav_path


def play_audio_blocking(pcm16: np.ndarray, sample_rate: int):
    with AUDIO_LOCK:
        sd.stop()
        sd.play(pcm16, sample_rate, device=CABLE_DEVICE_ID)
        sd.wait()


def speak_task(text: str, save_wav: bool, play_audio: bool, wav_path: str | None):
    try:
        mp3_bytes = asyncio.run(synthesize_mp3_bytes(text))
        pcm16, sr, ch = mp3_bytes_to_pcm16_numpy(mp3_bytes)

        saved_path = None
        if save_wav:
            if not wav_path:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_path = f"tts_{ts}.wav"
            saved_path = save_wav_file(pcm16, sr, ch, wav_path)

        if play_audio:
            play_audio_blocking(pcm16, sr)

        if saved_path:
            print(f"[WAV] saved: {saved_path}")

    except Exception as e:
        print("TTS/Playback error:", repr(e))


# ---------------------------
# ROUTES
# ---------------------------
@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "voice": VOICE,
        "rate": RATE,
        "pitch": PITCH,
        "output_device_id": CABLE_DEVICE_ID,
        "output_device_name": sd.query_devices(CABLE_DEVICE_ID)["name"],
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

    if("value" in text):
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
                threading.Thread(target=play_audio_blocking, args=(pcm16, sr), daemon=True).start()

            return Response(wav_bytes, mimetype="audio/wav")

        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    # Standard: async task (nicht blockieren)
    threading.Thread(
        target=speak_task,
        args=(text, save_wav, play_audio, wav_path),
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
