import os
import numpy as np
import sounddevice as sd

# OPTIONAL: Quadro komplett ausblenden (dann keine Warnungen)
# Wenn du die GPU-Auswahl behalten willst: kommentiere die nächste Zeile aus.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # in deiner Python-Liste ist 0 = RTX 5060 Ti

import torch

# --- FIX PyTorch 2.6+ weights_only default (required for Coqui XTTS checkpoints) ---
_torch_load = torch.load
def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = torch_load_compat

# --- FIX torchaudio/torchcodec: WAV via soundfile laden statt torchcodec ---
# XTTS nutzt intern torchaudio.load(...) für speaker_wav. Wir patchen das.
import soundfile as sf
try:
    import torchaudio

    def sf_load(path):
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        # torchaudio.load gibt (Tensor[C,T], sr) zurück
        if audio.ndim == 1:
            audio = audio[None, :]      # (1, T)
        else:
            audio = audio.T             # (C, T)
        return torch.from_numpy(audio), sr

    torchaudio.load = sf_load
except Exception:
    # Falls torchaudio gar nicht da ist, ist ok – XTTS bringt's meist als dep mit.
    pass

from TTS.api import TTS

# =========================
# CONFIG
# =========================
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "de"

# Output device (optional). Wenn du einfach Standard-Output willst: OUTPUT_NAME_SUBSTRING = None
OUTPUT_NAME_SUBSTRING = "cable input"   # z.B. VB-Cable/Voicemod; setze None für default

EMOTION_WAVS = {
    "neutral": "voices/neutral.wav",
    # "happy":   "voices/happy.wav",
    # "sad":     "voices/sad.wav",
    # "angry":   "voices/angry.wav",
}
DEFAULT_EMOTION = "neutral"

TEXT = "Hallo Meister. Das ist ein lokaler Test der deutschen Frauenstimme mit Emotionen. Ich bin total traurig darüber, was habe ich dir getan. Ich will dich töten uf aufschlitzen du penner!"

# =========================
# HELPERS
# =========================
def find_output_device_id(name_substring: str) -> int:
    name_substring = name_substring.lower()
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) > 0 and name_substring in dev["name"].lower():
            return idx
    raise RuntimeError(f"Audio output device not found for substring: {name_substring!r}")

def ensure_voice_files():
    missing = [p for p in EMOTION_WAVS.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing voice reference wav(s). Create these files:\n" + "\n".join(missing)
        )

def choose_torch_device() -> str:
    """
    Returns a device string for TTS.to(...):
      - "cpu"
      - "cuda:0"
      - "cuda:1"
    """
    if not torch.cuda.is_available():
        print("CUDA not available -> using CPU")
        return "cpu"

    n = torch.cuda.device_count()
    print("\n=== GPU SELECTION ===")
    print(f"CUDA available. Found {n} GPU(s):")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  [{i}] {name}  ({vram_gb:.1f} GB VRAM)")

    while True:
        sel = input("Choose GPU index (ENTER = 0, 'c' = CPU): ").strip().lower()
        if sel == "":
            return "cuda:0"
        if sel == "c":
            return "cpu"
        if sel.isdigit():
            idx = int(sel)
            if 0 <= idx < n:
                return f"cuda:{idx}"
        print("Invalid input. Try again.")

def synthesize(tts: TTS, text: str, emotion: str):
    emotion = (emotion or DEFAULT_EMOTION).lower().strip()
    ref = EMOTION_WAVS.get(emotion, EMOTION_WAVS[DEFAULT_EMOTION])

    wav = tts.tts(
        text=text,
        speaker_wav=ref,
        language=LANGUAGE,
    )

    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)

    # XTTS v2 ist typischerweise 24000 Hz, mono
    sr = 24000
    return wav, sr, emotion, ref

def play_audio(wav_f32: np.ndarray, sr: int, device_id: int | None):
    sd.stop()
    sd.play(wav_f32, sr, device=device_id)
    sd.wait()

# =========================
# MAIN
# =========================
def main():
    ensure_voice_files()

    device_id = None
    if OUTPUT_NAME_SUBSTRING:
        device_id = find_output_device_id(OUTPUT_NAME_SUBSTRING)

    DEVICE = choose_torch_device()

    print("\nLoading XTTS v2...")
    tts = TTS(model_name=TTS_MODEL_NAME).to(DEVICE)

    # Warmup (macht ersten Play schneller)
    try:
        _ = synthesize(tts, "Warmup.", "neutral")
    except Exception:
        pass

    print("\n=== LOCAL TTS TEST ===")
    print(f"Model: {TTS_MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Language: {LANGUAGE}")
    if device_id is None:
        print("Audio Device: DEFAULT")
    else:
        print("Audio Device:", sd.query_devices(device_id)["name"])
    print("\nEmotion options:", ", ".join(EMOTION_WAVS.keys()))
    print("Commands: [ENTER]=play | e=change emotion | t=change text | q=quit\n")

    emotion = DEFAULT_EMOTION
    text = TEXT

    while True:
        cmd = input(">").strip().lower()

        if cmd == "q":
            break

        if cmd == "e":
            emotion_in = input(f"Emotion ({'/'.join(EMOTION_WAVS.keys())}) [{emotion}]: ").strip().lower()
            if emotion_in:
                emotion = emotion_in
            continue

        if cmd == "t":
            new_text = input("New text: ").strip()
            if new_text:
                text = new_text
            continue

        print(f"\n[TTS] emotion={emotion} | text={text!r}")
        wav_f32, sr, used_emotion, ref = synthesize(tts, text, emotion)
        print(f"[OK] ref={ref} | len={len(wav_f32)/sr:.2f}s\n")
        play_audio(wav_f32, sr, device_id)

    print("Bye.")

if __name__ == "__main__":
    main()
