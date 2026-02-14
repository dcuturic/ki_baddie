import io
import os
import json
import wave
import threading
import time
import subprocess
import random
from datetime import datetime
from dataclasses import dataclass
import re

import requests
import numpy as np
import sounddevice as sd
from flask import Flask, request, jsonify, Response
from pythonosc.udp_client import SimpleUDPClient

GPU_INDEX = 0

VOICE_FX_ENABLED = True
GLOBAL_FX_ENABLED = True
EMOTION_FX_ENABLED = False

END_PAD_MS = 320
END_FADE_MS = 45

OSC_HOST = "127.0.0.1"
OSC_PORT = 39539

VOICEMOD_OUTPUT_NAME_SUBSTRING = "cable input"

TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "de"
XTTS_SR = 24000

EMOTION_WAVS = {
    "joy": "voices/neutral.wav",
    "angry": "voices/neutral.wav",
    "sorrow": "voices/neutral.wav",
    "fun": "voices/neutral.wav",
    "neutral": "voices/neutral.wav",
    "surprise": "voices/neutral.wav",
}
DEFAULT_EMOTION = "neutral"

FX_DIR = "voice_fx"

import torch

_torch_load = torch.load


def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = torch_load_compat

import soundfile as sf

try:
    import torchaudio

    def sf_load(path):
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 1:
            audio = audio[None, :]
        else:
            audio = audio.T
        return torch.from_numpy(audio), sr

    torchaudio.load = sf_load
except Exception:
    pass

from TTS.api import TTS
import librosa
import scipy.signal

try:
    import pyloudnorm as pyln
    HAS_LOUDNORM = True
except Exception:
    HAS_LOUDNORM = False

try:
    import pyrubberband as pyrb
    HAS_RUBBERBAND_PY = True
except Exception:
    HAS_RUBBERBAND_PY = False


def rubberband_cli_available() -> bool:
    try:
        r = subprocess.run(["rubberband", "--version"], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


HAS_RUBBERBAND_CLI = rubberband_cli_available()
HAS_RUBBERBAND = bool(HAS_RUBBERBAND_PY and HAS_RUBBERBAND_CLI)


def set_repro_seed(seed: int):
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _normalize_eq(eq):
    out = []
    if not isinstance(eq, list):
        return out
    for band in eq:
        if not (isinstance(band, (list, tuple)) and len(band) == 3):
            continue
        freq = _safe_float(band[0], None)
        gain = _safe_float(band[1], None)
        Q = _safe_float(band[2], None)
        if freq is None or gain is None or Q is None:
            continue
        out.append([freq, gain, Q])
    return out


def merge_fx(base: dict, override: dict) -> dict:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def normalize_fx_profile(p: dict) -> dict:
    if not isinstance(p, dict):
        return {}

    out = dict(p)

    if "time_stretch" in out and out["time_stretch"] is not None:
        ts = _safe_float(out["time_stretch"], 1.0)
        if ts is None or ts <= 0.0:
            ts = 1.0
        out["time_stretch"] = ts

    for k in ("pitch_semitones", "time_stretch", "lowcut_hz", "highcut_hz", "normalize_lufs"):
        if k in out and out[k] is not None:
            out[k] = _safe_float(out[k], out[k])

    if "eq" in out:
        out["eq"] = _normalize_eq(out["eq"])

    if "compressor" in out and isinstance(out["compressor"], dict):
        c = dict(out["compressor"])
        for kk in ("threshold_db", "ratio", "attack_ms", "release_ms", "makeup_db"):
            if kk in c and c[kk] is not None:
                c[kk] = _safe_float(c[kk], c[kk])
        if "enabled" in c:
            c["enabled"] = bool(c["enabled"])
        out["compressor"] = c

    if "limiter" in out and isinstance(out["limiter"], dict):
        l = dict(out["limiter"])
        if "ceiling_db" in l and l["ceiling_db"] is not None:
            l["ceiling_db"] = _safe_float(l["ceiling_db"], l["ceiling_db"])
        if "enabled" in l:
            l["enabled"] = bool(l["enabled"])
        out["limiter"] = l

    return out


@dataclass
class FXStore:
    root_dir: str
    scan_interval_sec: float = 0.25

    _cache_global: dict = None
    _cache_emotions: dict = None
    _mtimes: dict = None
    _last_scan: float = 0.0

    def __post_init__(self):
        self.root_dir = self.root_dir or "voice_fx"
        self.global_path = os.path.join(self.root_dir, "global.json")
        self.emotions_dir = os.path.join(self.root_dir, "emotions")
        self._cache_global = {}
        self._cache_emotions = {}
        self._mtimes = {}

    def _file_mtime(self, path: str) -> float:
        try:
            return os.path.getmtime(path)
        except Exception:
            return 0.0

    def _needs_rescan(self) -> bool:
        now = time.time()
        if (now - self._last_scan) < self.scan_interval_sec:
            return False
        self._last_scan = now
        return True

    def reload_if_changed(self) -> bool:
        if not self._needs_rescan():
            return False

        changed = False

        gp = self.global_path
        gm = self._file_mtime(gp)
        if self._mtimes.get(gp) != gm:
            if os.path.exists(gp):
                try:
                    self._cache_global = normalize_fx_profile(_read_json(gp))
                except Exception:
                    self._cache_global = {}
            else:
                self._cache_global = {}
            self._mtimes[gp] = gm
            changed = True

        if os.path.isdir(self.emotions_dir):
            for fn in os.listdir(self.emotions_dir):
                if not fn.lower().endswith(".json"):
                    continue
                key = os.path.splitext(fn)[0].lower().strip()
                path = os.path.join(self.emotions_dir, fn)
                mt = self._file_mtime(path)
                if self._mtimes.get(path) != mt:
                    try:
                        self._cache_emotions[key] = normalize_fx_profile(_read_json(path))
                    except Exception:
                        self._cache_emotions[key] = {}
                    self._mtimes[path] = mt
                    changed = True

        return changed

    def get_global(self) -> dict:
        self.reload_if_changed()
        return dict(self._cache_global or {})

    def get_emotion(self, emotion: str) -> dict:
        self.reload_if_changed()
        e = (emotion or "").strip().lower()
        return dict((self._cache_emotions or {}).get(e, {}))

    def loaded_emotions(self) -> list[str]:
        self.reload_if_changed()
        return sorted(list((self._cache_emotions or {}).keys()))


app = Flask(__name__)
osc_client = SimpleUDPClient(OSC_HOST, OSC_PORT)

CLEAR_BEFORE_EACH = True
VISEMES = [("A", 1.0), ("O", 0.9), ("E", 0.6), ("I", 0.5), ("U", 0.4)]


def set_viseme(key: str, value: float):
    osc_client.send_message("/VMC/Ext/Blend/Val", [key, float(value)])
    osc_client.send_message("/VMC/Ext/Blend/Apply", 1)


def clear_mouth():
    for k, _ in VISEMES:
        osc_client.send_message("/VMC/Ext/Blend/Val", [k, 0.0])
    osc_client.send_message("/VMC/Ext/Blend/Apply", 1)


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

fx_store = FXStore(root_dir=FX_DIR)


def ensure_voice_files():
    missing = [p for p in EMOTION_WAVS.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing voice reference wav(s). Create these files:\n" + "\n".join(missing))


def ensure_fx_files():
    os.makedirs(FX_DIR, exist_ok=True)
    os.makedirs(os.path.join(FX_DIR, "emotions"), exist_ok=True)

    global_path = os.path.join(FX_DIR, "global.json")
    neutral_path = os.path.join(FX_DIR, "emotions", "neutral.json")

    if not os.path.exists(global_path):
        default_global = {
            "pitch_semitones": 0.0,
            "time_stretch": 1.0,
            "lowcut_hz": 120.0,
            "highcut_hz": 15500.0,
            "eq": [
                [120.0, -3.0, 1.0],
                [220.0, -1.5, 1.0],
                [420.0, -2.0, 1.0],
                [3200.0, 1.2, 1.0],
                [11000.0, 0.8, 0.7],
            ],
            "compressor": {
                "enabled": True,
                "threshold_db": -20.5,
                "ratio": 2.4,
                "attack_ms": 7.0,
                "release_ms": 110.0,
                "makeup_db": 1.0,
            },
            "normalize_lufs": -16.0,
            "limiter": {"enabled": True, "ceiling_db": -1.5},
        }
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(default_global, f, ensure_ascii=False, indent=2)

    if not os.path.exists(neutral_path):
        default_neutral = {
            "pitch_semitones": 0.0,
            "time_stretch": 1.0,
            "lowcut_hz": 120.0,
            "eq": [
                [120.0, -2.5, 1.0],
                [220.0, -1.2, 1.0],
                [420.0, -1.8, 1.0],
                [3200.0, 1.0, 1.0],
                [11000.0, 0.7, 0.7],
            ],
            "compressor": {
                "enabled": True,
                "threshold_db": -20.5,
                "ratio": 2.4,
                "attack_ms": 7.0,
                "release_ms": 115.0,
                "makeup_db": 1.0,
            },
            "normalize_lufs": -16.0,
            "limiter": {"enabled": True, "ceiling_db": -1.5},
        }
        with open(neutral_path, "w", encoding="utf-8") as f:
            json.dump(default_neutral, f, ensure_ascii=False, indent=2)

    fx_store.reload_if_changed()


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


def compute_envelope(wav_f32: np.ndarray, sr: int, frame_ms: int = 20, hop_ms: int = 10):
    if wav_f32.ndim > 1:
        wav = wav_f32.mean(axis=1)
    else:
        wav = wav_f32
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if frame <= 0 or hop <= 0 or len(wav) < frame:
        return np.array([], dtype=np.float32), hop_ms / 1000.0
    env = []
    for i in range(0, len(wav) - frame, hop):
        chunk = wav[i:i + frame]
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        env.append(rms)
    return np.asarray(env, dtype=np.float32), hop_ms / 1000.0


def smooth_attack_release(x: np.ndarray, attack: float, release: float, dt: float):
    if len(x) == 0:
        return x
    y = np.zeros_like(x)
    a_a = np.exp(-dt / max(attack, 1e-6))
    a_r = np.exp(-dt / max(release, 1e-6))
    y[0] = x[0]
    for i in range(1, len(x)):
        if x[i] > y[i - 1]:
            y[i] = a_a * y[i - 1] + (1 - a_a) * x[i]
        else:
            y[i] = a_r * y[i - 1] + (1 - a_r) * x[i]
    return y


def lipsync_from_audio(wav_f32: np.ndarray, sr: int, stop_event: threading.Event):
    env, dt = compute_envelope(wav_f32, sr, frame_ms=20, hop_ms=10)
    if env.size == 0:
        clear_mouth()
        return

    p10 = float(np.percentile(env, 10))
    p95 = float(np.percentile(env, 95))
    env = (env - p10) / max((p95 - p10), 1e-6)
    env = np.clip(env, 0.0, 1.0)

    gate = 0.06
    env = np.where(env < gate, 0.0, (env - gate) / (1.0 - gate))
    env = smooth_attack_release(env, attack=0.02, release=0.08, dt=dt)

    for v in env:
        if stop_event.is_set():
            break

        openness = float(np.clip(v, 0.0, 1.0))

        if CLEAR_BEFORE_EACH:
            clear_mouth()

        set_viseme("A", openness)
        if openness > 0.15:
            set_viseme("O", openness * 0.35)
            set_viseme("E", openness * 0.20)

        time.sleep(dt)

    clear_mouth()


def db_to_lin(db: float) -> float:
    return float(10 ** (db / 20.0))


def butter_filter(wav: np.ndarray, sr: int, lowcut: float | None, highcut: float | None) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    nyq = 0.5 * sr

    if lowcut and lowcut > 0:
        b, a = scipy.signal.butter(2, lowcut / nyq, btype="highpass")
        x = scipy.signal.filtfilt(b, a, x).astype(np.float32)

    if highcut and highcut < nyq:
        b, a = scipy.signal.butter(2, highcut / nyq, btype="lowpass")
        x = scipy.signal.filtfilt(b, a, x).astype(np.float32)

    return x


def peaking_eq(wav: np.ndarray, sr: int, freq: float, gain_db: float, Q: float) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * (freq / sr)
    alpha = np.sin(w0) / (2 * max(Q, 1e-6))

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

    y = scipy.signal.filtfilt(b, a, x.astype(np.float64)).astype(np.float32)
    return y


def simple_compressor(wav: np.ndarray, sr: int, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, makeup_db: float) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    eps = 1e-12

    attack = np.exp(-1.0 / (sr * max(attack_ms, 1e-6) / 1000.0))
    release = np.exp(-1.0 / (sr * max(release_ms, 1e-6) / 1000.0))

    env = np.zeros_like(x)
    for i in range(1, len(x)):
        rect = abs(x[i])
        if rect > env[i - 1]:
            env[i] = attack * env[i - 1] + (1 - attack) * rect
        else:
            env[i] = release * env[i - 1] + (1 - release) * rect

    env_db = 20.0 * np.log10(env + eps)

    over = env_db - threshold_db
    gain_db = np.where(over > 0, -over * (1.0 - 1.0 / max(ratio, 1e-6)), 0.0)
    gain = (10 ** (gain_db / 20.0)).astype(np.float32)

    y = x * gain
    y *= db_to_lin(makeup_db)
    return y


def limiter(wav: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    ceiling = db_to_lin(ceiling_db)
    peak = float(np.max(np.abs(x)) + 1e-12)
    if peak > ceiling:
        x = x * (ceiling / peak)
    return x


def loudness_normalize(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    if not HAS_LOUDNORM:
        return np.asarray(wav, dtype=np.float32)
    x = np.asarray(wav, dtype=np.float32)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x)
    return pyln.normalize.loudness(x, loudness, target_lufs).astype(np.float32)


def apply_pitch_and_time(wav: np.ndarray, sr: int, pitch_semitones: float, time_stretch: float) -> tuple[np.ndarray, str]:
    x = np.asarray(wav, dtype=np.float32)

    if time_stretch is None or time_stretch <= 0:
        time_stretch = 1.0

    backend = "librosa"

    if abs(time_stretch - 1.0) > 1e-3:
        if HAS_RUBBERBAND:
            try:
                x = pyrb.time_stretch(x, sr, time_stretch)
                backend = "rubberband"
            except Exception:
                x = librosa.effects.time_stretch(x, rate=time_stretch)
        else:
            x = librosa.effects.time_stretch(x, rate=time_stretch)

    if pitch_semitones and abs(pitch_semitones) > 1e-3:
        if HAS_RUBBERBAND:
            try:
                x = pyrb.pitch_shift(
                    x,
                    sr,
                    pitch_semitones,
                    rbargs=["--formant", "--pitch-hq", "--transients", "mixed"]
                )
                backend = "rubberband"
            except Exception:
                x = librosa.effects.pitch_shift(x, sr=sr, n_steps=pitch_semitones)
        else:
            x = librosa.effects.pitch_shift(x, sr=sr, n_steps=pitch_semitones)

    return x.astype(np.float32), backend


def apply_voice_fx(wav_f32: np.ndarray, sr: int, fx: dict) -> tuple[np.ndarray, str]:
    x = np.asarray(wav_f32, dtype=np.float32)

    pitch = float(fx.get("pitch_semitones", 0.0) or 0.0)
    stretch = float(fx.get("time_stretch", 1.0) or 1.0)

    x, pitch_backend = apply_pitch_and_time(x, sr, pitch, stretch)

    lowcut = fx.get("lowcut_hz", None)
    highcut = fx.get("highcut_hz", None)
    if lowcut is not None or highcut is not None:
        x = butter_filter(
            x,
            sr,
            float(lowcut) if lowcut is not None else None,
            float(highcut) if highcut is not None else None,
        )

    for band in (fx.get("eq") or []):
        try:
            freq, gain_db, Q = band
            x = peaking_eq(x, sr, float(freq), float(gain_db), float(Q))
        except Exception:
            pass

    comp = fx.get("compressor") or {}
    if comp.get("enabled", True):
        x = simple_compressor(
            x,
            sr,
            threshold_db=float(comp.get("threshold_db", -19.0)),
            ratio=float(comp.get("ratio", 2.4)),
            attack_ms=float(comp.get("attack_ms", 7.0)),
            release_ms=float(comp.get("release_ms", 110.0)),
            makeup_db=float(comp.get("makeup_db", 1.0)),
        )

    target_lufs = fx.get("normalize_lufs", None)
    if target_lufs is not None:
        try:
            x = loudness_normalize(x, sr, float(target_lufs))
        except Exception:
            pass

    lim = fx.get("limiter") or {}
    if lim.get("enabled", True):
        x = limiter(x, float(lim.get("ceiling_db", -1.5)))

    return np.clip(x, -1.0, 1.0).astype(np.float32), pitch_backend


def add_end_padding_and_fade(wav_f32: np.ndarray, sr: int, pad_ms: int, fade_ms: int) -> np.ndarray:
    x = np.asarray(wav_f32, dtype=np.float32).copy()

    if fade_ms and fade_ms > 0:
        fade_n = int(sr * fade_ms / 1000.0)
        if fade_n > 1 and x.size >= fade_n:
            ramp = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
            x[-fade_n:] *= ramp

    if pad_ms and pad_ms > 0:
        pad_n = int(sr * pad_ms / 1000.0)
        if pad_n > 0:
            noise = (np.random.randn(pad_n).astype(np.float32) * 0.00035)
            x = np.concatenate([x, noise], axis=0)

    return x


def sanitize_audio(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if not np.isfinite(wav).all():
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(wav, -1.0, 1.0)


def resolve_emotion_key(emotion: str) -> str:
    e = (emotion or "").strip().lower()
    if e in ():
        e = DEFAULT_EMOTION
    if e in EMOTION_WAVS:
        return e
    return DEFAULT_EMOTION


_END_PUNCT = (".", "!", "?", "…", ":", ";")

def coerce_text_to_string(text_obj) -> str:
    if isinstance(text_obj, dict):
        text_obj = text_obj.get("value", "")

    if isinstance(text_obj, (list, tuple)):
        parts = []
        for x in text_obj:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            parts.append(s)

        out = ""
        for p in parts:
            if not out:
                out = p
                continue
            if out.rstrip().endswith(_END_PUNCT):
                out += " " + p
            else:
                out += ". " + p
        text_obj = out

    s = str(text_obj or "").strip()
    s = re.sub(r"\s+", " ", s)
    if s and not s.endswith(_END_PUNCT):
        s += "."
    return s


ensure_voice_files()
ensure_fx_files()
TTS_DEVICE = choose_torch_device()

print(f"[XTTS] Loading model {TTS_MODEL_NAME} on {TTS_DEVICE} ...")
tts_model = TTS(model_name=TTS_MODEL_NAME).to(TTS_DEVICE)

try:
    _ = tts_model.tts(text="Warmup.", speaker_wav=EMOTION_WAVS[DEFAULT_EMOTION], language=LANGUAGE)
except Exception:
    pass
try:
    tts_model.model = torch.compile(tts_model.model, mode="max-autotune")
except Exception as e:
    print("torch.compile not available/failed:", e)
print("[FX] HAS_RUBBERBAND_PY:", HAS_RUBBERBAND_PY, "HAS_RUBBERBAND_CLI:", HAS_RUBBERBAND_CLI, "USING_RUBBERBAND:", HAS_RUBBERBAND)


def synthesize_wav_f32(text: str, emotion: str, tuning: dict | None = None):
    emotion_key = resolve_emotion_key(emotion)
    ref = EMOTION_WAVS.get(emotion_key, EMOTION_WAVS[DEFAULT_EMOTION])

    wav = tts_model.tts(text=text, speaker_wav=ref, language=LANGUAGE)
    wav_f32 = np.asarray(wav, dtype=np.float32)
    wav_f32 = np.clip(wav_f32, -1.0, 1.0)

    pitch_backend_used = "none"

    if VOICE_FX_ENABLED:
        fx = {}
        if GLOBAL_FX_ENABLED:
            fx = merge_fx(fx, fx_store.get_global())
        if EMOTION_FX_ENABLED:
            fx = merge_fx(fx, fx_store.get_emotion(emotion_key))
        if tuning and isinstance(tuning, dict):
            fx = merge_fx(fx, tuning)

        if fx:
            wav_f32, pitch_backend_used = apply_voice_fx(wav_f32, XTTS_SR, fx)

    wav_f32 = add_end_padding_and_fade(wav_f32, XTTS_SR, END_PAD_MS, END_FADE_MS)
    wav_f32 = sanitize_audio(wav_f32)

    return wav_f32, XTTS_SR, 1, emotion_key, ref, pitch_backend_used


liste_dict = [
    {
        "emotion":"default",
        "desc":"standard sachen",
        "elements" : [{
            "datei":"pose_20260210225531227",
            "desc":"mach"
        },{
            "datei":"pose_20260210225631127",
            "desc":"mach"
        },{
            "datei":"pose_20260210225643907",
            "desc":"mach"
        } ]
    }
]
import random
import math

def play_audio_blocking(wav_f32: np.ndarray, sr: int, emotion: str):
    with AUDIO_LOCK:
        idx_emotion = math.floor(random.random() * len(liste_dict))
        emotion_pose_dict = liste_dict[idx_emotion]

        elements = emotion_pose_dict["elements"]
        idx_pose = math.floor(random.random() * len(elements))
        emotion_pose = elements[idx_pose]

        emotion_pose_file = emotion_pose["datei"]
        emotion_pose_name = emotion_pose_dict["emotion"]


        try:
            requests.post("http://127.0.0.1:5004/emotion", json={"emotion": emotion}, timeout=30)
        except Exception:
            pass
        

        try:
            requests.post("http://127.0.0.1:5000/play/pose/"+emotion_pose_name+"/"+emotion_pose_file+"", timeout=30)
        except Exception:
            pass

        local_stop = threading.Event()
        lip_thread = threading.Thread(target=lipsync_from_audio, args=(wav_f32, sr, local_stop), daemon=True)
        lip_thread.start()

        

        sd.stop()
        sd.play(wav_f32, sr, device=CABLE_DEVICE_ID)
        sd.wait()

        local_stop.set()
        clear_mouth()

        try:
            requests.post("http://127.0.0.1:5004/emotion", json={"emotion": "neutral"}, timeout=30)
        except Exception:
            pass
        

        try:
            requests.post("http://127.0.0.1:5000/play/pose/default/idle", timeout=30)
        except Exception:
            pass


def speak_task(text: str, save_wav: bool, play_audio: bool, wav_path: str | None, emotion: str, tuning: dict | None):
    try:
        wav_f32, sr, ch, used_emotion, ref, backend = synthesize_wav_f32(text, emotion, tuning=tuning)

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


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "model": TTS_MODEL_NAME,
        "language": LANGUAGE,
        "device": TTS_DEVICE,
        "gpu_index": GPU_INDEX,
        "output_device": sd.query_devices(CABLE_DEVICE_ID)["name"],
        "emotions_available_wavs": list(EMOTION_WAVS.keys()),
        "default_emotion": DEFAULT_EMOTION,
        "has_rubberband_py": HAS_RUBBERBAND_PY,
        "has_rubberband_cli": HAS_RUBBERBAND_CLI,
        "using_rubberband": HAS_RUBBERBAND,
        "has_loudnorm": HAS_LOUDNORM,
        "voice_fx_enabled": VOICE_FX_ENABLED,
        "global_fx_enabled": GLOBAL_FX_ENABLED,
        "emotion_fx_enabled": EMOTION_FX_ENABLED,
        "fx_dir": FX_DIR,
        "fx_global_json": os.path.join(FX_DIR, "global.json"),
        "fx_emotions_dir": os.path.join(FX_DIR, "emotions"),
        "fx_loaded_emotions": fx_store.loaded_emotions(),
        "end_pad_ms": END_PAD_MS,
        "end_fade_ms": END_FADE_MS,
    })


@app.get("/fx_state")
def fx_state():
    fx_store.reload_if_changed()
    return jsonify({
        "ok": True,
        "voice_fx_enabled": VOICE_FX_ENABLED,
        "global_fx_enabled": GLOBAL_FX_ENABLED,
        "emotion_fx_enabled": EMOTION_FX_ENABLED,
        "fx_dir": FX_DIR,
        "global_path": fx_store.global_path,
        "emotions_dir": fx_store.emotions_dir,
        "loaded_emotions": fx_store.loaded_emotions(),
        "global_keys": sorted(list((fx_store._cache_global or {}).keys())),
        "end_pad_ms": END_PAD_MS,
        "end_fade_ms": END_FADE_MS,
    })


@app.post("/fx_toggle")
def fx_toggle():
    global VOICE_FX_ENABLED, GLOBAL_FX_ENABLED, EMOTION_FX_ENABLED
    payload = request.get_json(silent=True) or {}
    if "enabled" in payload:
        VOICE_FX_ENABLED = bool(payload["enabled"])
    if "global_enabled" in payload:
        GLOBAL_FX_ENABLED = bool(payload["global_enabled"])
    if "emotion_enabled" in payload:
        EMOTION_FX_ENABLED = bool(payload["emotion_enabled"])
    return jsonify({
        "ok": True,
        "voice_fx_enabled": VOICE_FX_ENABLED,
        "global_fx_enabled": GLOBAL_FX_ENABLED,
        "emotion_fx_enabled": EMOTION_FX_ENABLED,
    })


@app.post("/tts")
def tts():
    payload = request.get_json(silent=True) or {}
    print(payload, flush=True)

    seed = payload.get("seed", None)
    if seed is not None:
        try:
            seed = int(seed)
            set_repro_seed(seed)
        except Exception:
            seed = None

    text_raw = payload.get("text") or ""
    emotion = payload.get("emotion") or payload.get("emotion") or"neutral"

    if isinstance(text_raw, dict) and "value" in text_raw:
        if isinstance(text_raw, dict) and "emotion" in text_raw:
            emotion = text_raw.get("emotion") or emotion

    tuning = payload.get("tuning")
    if tuning is not None and not isinstance(tuning, dict):
        tuning = None

    text = coerce_text_to_string(text_raw)
    emotion = str(emotion).strip().lower()

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'"}), 400

    save_wav = bool(payload.get("save_wav", False))
    play_audio = bool(payload.get("play_audio", True))
    wav_path = payload.get("wav_path")
    return_wav = bool(payload.get("return_wav", False))

    if return_wav:
        try:
            wav_f32, sr, ch, used_emotion, ref, backend = synthesize_wav_f32(text, emotion, tuning=tuning)
            pcm16 = float32_to_int16(wav_f32)
            wav_bytes = pcm16_to_wav_bytes(pcm16, sr, ch)

            if save_wav:
                if not wav_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wav_path = f"tts_{ts}.wav"
                save_wav_file(pcm16, sr, ch, wav_path)

            if play_audio:
                threading.Thread(target=play_audio_blocking, args=(wav_f32, sr, used_emotion), daemon=True).start()

            resp = Response(wav_bytes, mimetype="audio/wav")
            resp.headers["X-Pitch-Backend"] = backend
            if seed is not None:
                resp.headers["X-Seed"] = str(seed)
            return resp
        except Exception as e:
            return jsonify({"ok": False, "error": repr(e)}), 500

    threading.Thread(
        target=speak_task,
        args=(text, save_wav, play_audio, wav_path, emotion, tuning),
        daemon=True
    ).start()

    return jsonify({
        "ok": True,
        "status": "queued",
        "save_wav": save_wav,
        "play_audio": play_audio,
        "wav_path": wav_path,
        "return_wav": return_wav,
        "emotion_in": emotion,
        "emotion_used": resolve_emotion_key(emotion),
        "has_rubberband_py": HAS_RUBBERBAND_PY,
        "has_rubberband_cli": HAS_RUBBERBAND_CLI,
        "using_rubberband": HAS_RUBBERBAND,
        "has_loudnorm": HAS_LOUDNORM,
        "seed": seed,
        "fx_dir": FX_DIR,
        "voice_fx_enabled": VOICE_FX_ENABLED,
        "global_fx_enabled": GLOBAL_FX_ENABLED,
        "emotion_fx_enabled": EMOTION_FX_ENABLED,
        "end_pad_ms": END_PAD_MS,
        "end_fade_ms": END_FADE_MS,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
