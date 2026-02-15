from dataclasses import dataclass
from typing import Dict, Optional
import threading
import time
import math
import json
import os

from flask import Flask, request, jsonify
from pythonosc.udp_client import SimpleUDPClient

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

# ----------------------------
# VSeeFace / VMC OSC Settings
# ----------------------------
osc_cfg = CONFIG.get("osc", {})
VSEE_HOST = osc_cfg.get("host", "127.0.0.1")
VSEE_PORT = osc_cfg.get("port", 39539)
OSC_ENABLED = osc_cfg.get("enabled", True)

# ----------------------------
# Mapping (deine Commands -> Keys)
# ----------------------------
EMOTION_MAP: Dict[str, str] = CONFIG.get("emotion_map", {
    "joy": "Joy",
    "angry": "Angry",
    "sorrow": "Sorrow",
    "fun": "Fun",
    "neutral": "Neutral",
    "surprise": "Surprised",
    "surprised": "Surprised",
})

# ---------------------------- surprise, angry, sorrow, fun, neutral,joy
# WICHTIG: Reset-Liste erweitern
# Damit wirklich nix "hängen bleibt"
# (Keys, die es nicht gibt, werden meistens ignoriert)
# ----------------------------
RESET_KEYS = sorted(set(CONFIG.get("reset_keys", [
    "Joy", "Angry", "Anger", "Sorrow", "Fun", "Neutral",
    "Surprised", "Surprise", "Relaxed", "Sad", "Happy",
    "A", "I", "U", "E", "O",
    "Blink", "Blink_L", "Blink_R",
    "Smile", "Smirk", "Frown",
])))

# zusätzlich auch die aus EMOTION_MAP
RESET_KEYS = sorted(set(RESET_KEYS + list(EMOTION_MAP.values())))

# OSC Paths
osc_paths_cfg = CONFIG.get("osc_paths", {})
OSC_VAL = osc_paths_cfg.get("blend_val", "/VMC/Ext/Blend/Val")
OSC_APPLY = osc_paths_cfg.get("blend_apply", "/VMC/Ext/Blend/Apply")


@dataclass
class CurrentEmotion:
    key: str
    value: float


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def ease_in_out(t: float) -> float:
    # smoothstep (0..1)
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


class SmoothBlendController:
    """
    Smooth blending:
    - hält current_values pro key
    - target_values pro key
    - background thread interpoliert current -> target über fade_time
    """

    def __init__(self, host: str, port: int, keys: list[str]):
        self.client = SimpleUDPClient(host, port)
        self.lock = threading.Lock()

        self.keys = keys[:]  # alle keys, die wir managen/resetten
        self.current_values: Dict[str, float] = {k: 0.0 for k in self.keys}
        self.target_values: Dict[str, float] = {k: 0.0 for k in self.keys}

        self.current: Optional[CurrentEmotion] = None

        self.fade_time = 0.35   # default: schöner Übergang (Sekunden)
        self.fps = 60

        blend_cfg = CONFIG.get("blend_controller", {})
        self.fade_time = blend_cfg.get("fade_time", self.fade_time)
        self.fps = blend_cfg.get("fps", self.fps)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _send_val(self, key: str, value: float):
        self.client.send_message(OSC_VAL, [key, float(value)])

    def _apply(self):
        self.client.send_message(OSC_APPLY, 1)

    def set_fade_time(self, seconds: float):
        with self.lock:
            self.fade_time = max(0.0, float(seconds))

    def set_exclusive_target(self, key: str, value: float):
        """
        Exclusive target:
        - setzt Ziele für ALLE keys auf 0
        - setzt nur key auf value
        """
        value = clamp01(value)
        with self.lock:
            for k in self.keys:
                self.target_values[k] = 0.0
            if key not in self.target_values:
                # falls du einen Key schickst, der nicht in RESET_KEYS ist:
                self.keys.append(key)
                self.current_values[key] = 0.0
                self.target_values[key] = 0.0
            self.target_values[key] = value
            self.current = CurrentEmotion(key=key, value=value)

    def reset_all(self):
        with self.lock:
            for k in self.keys:
                self.target_values[k] = 0.0
            self.current = None

    def snapshot(self):
        with self.lock:
            return {
                "fade_time": self.fade_time,
                "current": (self.current.__dict__ if self.current else None),
                "targets": dict(self.target_values),
            }

    def _loop(self):
        dt = 1.0 / float(self.fps)
        while self._running:
            start = time.time()

            # copy state
            with self.lock:
                fade = self.fade_time
                keys = list(self.keys)
                cur = dict(self.current_values)
                tgt = dict(self.target_values)

            # no smoothing -> snap
            if fade <= 0.0:
                changed = []
                for k in keys:
                    nv = tgt.get(k, 0.0)
                    if abs(cur.get(k, 0.0) - nv) > 0.0005:
                        changed.append((k, nv))
                        cur[k] = nv

                if changed:
                    # send only changed
                    for k, v in changed:
                        self._send_val(k, v)
                    self._apply()

                with self.lock:
                    self.current_values.update(cur)

            else:
                # smoothing step
                step = dt / fade  # Anteil pro Tick
                changed = []
                for k in keys:
                    cv = cur.get(k, 0.0)
                    tv = tgt.get(k, 0.0)
                    if abs(cv - tv) <= 0.0005:
                        continue

                    # lerp mit ease-in-out pro step (kleines smoothing)
                    # wir nähern uns pro Tick ein Stück an
                    # (stabiler als "t from start", weil target jederzeit wechseln kann)
                    a = clamp01(step)
                    # ease the step a bit
                    a = ease_in_out(a)
                    nv = cv + (tv - cv) * a

                    changed.append((k, nv))
                    cur[k] = nv

                if changed:
                    for k, v in changed:
                        self._send_val(k, v)
                    self._apply()

                with self.lock:
                    self.current_values.update(cur)

            # sleep to maintain fps
            elapsed = time.time() - start
            to_sleep = max(0.0, dt - elapsed)
            time.sleep(to_sleep)


# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)
blend = SmoothBlendController(VSEE_HOST, VSEE_PORT, RESET_KEYS) if OSC_ENABLED else None


@app.get("/health")
def health():
    state = blend.snapshot() if blend else {"osc_enabled": False}
    return jsonify(ok=True, host=VSEE_HOST, port=VSEE_PORT, osc_enabled=OSC_ENABLED, state=state)


@app.post("/emotion")
def set_emotion():
    """
    POST /emotion?emotion=angry&intensity=1&fade=0.35
    JSON geht auch: {"emotion":"angry","intensity":1,"fade":0.35}
    """
    data = request.get_json(silent=True) or {}
    print(data)
    emotion = (data.get("emotion") or request.args.get("emotion") or "").strip().lower()
    if not emotion:
        return jsonify(ok=False, error="Missing 'emotion'"), 400

    intensity = data.get("intensity", request.args.get("intensity", 1.0))
    fade = data.get("fade", request.args.get("fade", "0.1"))

    try:
        intensity = float(intensity)
    except Exception:
        return jsonify(ok=False, error="Invalid 'intensity'"), 400

    if fade is not None:
        try:
            if blend:
                blend.set_fade_time(float(fade))
        except Exception:
            return jsonify(ok=False, error="Invalid 'fade'"), 400

    key = EMOTION_MAP.get(emotion, emotion)  # erlaubt direkte Keys
    print(key, intensity)
    if blend:
        blend.set_exclusive_target(key, intensity)
    print("emotion",emotion)
    return jsonify(ok=True, emotion=emotion, key=key, intensity=clamp01(intensity), fade=blend.fade_time if blend else 0)


@app.post("/reset")
def reset():
    if blend:
        blend.reset_all()
    return jsonify(ok=True, reset=True)


@app.post("/fade")
def set_fade():
    """
    POST /fade?seconds=0.4
    """
    data = request.get_json(silent=True) or {}
    seconds = data.get("seconds", request.args.get("seconds", None))
    if seconds is None:
        return jsonify(ok=False, error="Missing 'seconds'"), 400
    try:
        if blend:
            blend.set_fade_time(float(seconds))
    except Exception:
        return jsonify(ok=False, error="Invalid 'seconds'"), 400
    return jsonify(ok=True, fade=blend.fade_time if blend else 0)


if __name__ == "__main__":
    server_cfg = CONFIG.get("server", {})
    app.run(
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 5004),
        debug=server_cfg.get("debug", False),
        threaded=True
    )
