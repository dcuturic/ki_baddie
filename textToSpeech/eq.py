import json
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import requests
import wave
import io

FX_DIR = "voice_fx"
GLOBAL_PATH = os.path.join(FX_DIR, "global.json")
EMOTIONS_DIR = os.path.join(FX_DIR, "emotions")

TTS_URL = "http://127.0.0.1:5003/tts"
DEFAULT_TEST_TEXT = "Hallo Meister. Das ist ein kurzer Stimmtest."

DEFAULT_EQ = [
    [120.0, -3.0, 1.0],
    [220.0, -1.6, 1.0],
    [420.0, -2.5, 1.0],
    [3200.0, 2.2, 1.0],
    [11000.0, 1.0, 0.7],
]

def read_json(path, fallback=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback if fallback is not None else {}

def write_json_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def list_emotions():
    os.makedirs(EMOTIONS_DIR, exist_ok=True)
    out = []
    for fn in os.listdir(EMOTIONS_DIR):
        if fn.lower().endswith(".json"):
            out.append(os.path.splitext(fn)[0].lower())
    out.sort()
    return out

def wav_bytes_to_float32(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        n = wf.getnframes()
        pcm = wf.readframes(n)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch)
    return audio, sr

def stable_seed_from(text: str, emotion: str) -> int:
    s = (emotion or "") + "|" + (text or "")
    h = 2166136261
    for c in s:
        h ^= ord(c)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice FX Editor (JSON live)")
        self.geometry("1000x740")

        os.makedirs(FX_DIR, exist_ok=True)
        os.makedirs(EMOTIONS_DIR, exist_ok=True)

        if not os.path.exists(GLOBAL_PATH):
            write_json_atomic(GLOBAL_PATH, {
                "pitch_semitones": 1.2,
                "time_stretch": 1.0,
                "lowcut_hz": 120.0,
                "highcut_hz": 15500.0,
                "eq": DEFAULT_EQ,
                "compressor": {"enabled": True, "threshold_db": -20.5, "ratio": 2.6, "attack_ms": 7.0, "release_ms": 110.0, "makeup_db": 2.0},
                "normalize_lufs": -16.0,
                "limiter": {"enabled": True, "ceiling_db": -1.0}
            })

        self.mode = tk.StringVar(value="emotion")
        self.emotion = tk.StringVar(value="neutral")
        self.test_text = tk.StringVar(value=DEFAULT_TEST_TEXT)

        self.seed_mode = tk.StringVar(value="manual")  # "manual" | "auto"
        self.seed_value = tk.IntVar(value=1337)

        self.vars = {}
        self._build_ui()
        self._refresh_emotions()
        self._load_selected()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=12, pady=10)

        ttk.Label(top, text="Edit Mode:").pack(side="left")
        ttk.Radiobutton(top, text="Emotion", variable=self.mode, value="emotion", command=self._load_selected).pack(side="left", padx=6)
        ttk.Radiobutton(top, text="Global", variable=self.mode, value="global", command=self._load_selected).pack(side="left", padx=6)

        ttk.Label(top, text="Emotion:").pack(side="left", padx=(18, 6))
        self.emotion_combo = ttk.Combobox(top, textvariable=self.emotion, state="readonly", width=18)
        self.emotion_combo.pack(side="left")
        self.emotion_combo.bind("<<ComboboxSelected>>", lambda e: self._load_selected())

        ttk.Button(top, text="Reload disk", command=self._load_selected).pack(side="right")
        ttk.Button(top, text="Save", command=self._save_selected).pack(side="right", padx=8)

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=False, padx=(12, 0))

        self._add_num(left, "pitch_semitones", "Pitch (semitones)", -6.0, 6.0, 0.1)
        self._add_num(left, "time_stretch", "Time stretch (rate)", 0.6, 1.6, 0.01)
        self._add_num(left, "lowcut_hz", "Lowcut (Hz)", 0.0, 500.0, 1.0)
        self._add_num(left, "highcut_hz", "Highcut (Hz)", 1000.0, 20000.0, 10.0)
        self._add_num(left, "normalize_lufs", "Normalize (LUFS)", -30.0, -10.0, 0.1)

        comp = ttk.LabelFrame(left, text="Compressor")
        comp.pack(fill="x", pady=10)
        self.vars["compressor.enabled"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(comp, text="Enabled", variable=self.vars["compressor.enabled"]).pack(anchor="w", padx=8, pady=4)
        self._add_num(comp, "compressor.threshold_db", "Threshold (dB)", -40.0, 0.0, 0.1)
        self._add_num(comp, "compressor.ratio", "Ratio", 1.0, 10.0, 0.1)
        self._add_num(comp, "compressor.attack_ms", "Attack (ms)", 0.1, 60.0, 0.1)
        self._add_num(comp, "compressor.release_ms", "Release (ms)", 10.0, 500.0, 1.0)
        self._add_num(comp, "compressor.makeup_db", "Makeup (dB)", 0.0, 12.0, 0.1)

        lim = ttk.LabelFrame(left, text="Limiter")
        lim.pack(fill="x", pady=10)
        self.vars["limiter.enabled"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(lim, text="Enabled", variable=self.vars["limiter.enabled"]).pack(anchor="w", padx=8, pady=4)
        self._add_num(lim, "limiter.ceiling_db", "Ceiling (dB)", -6.0, 0.0, 0.1)

        eqf = ttk.LabelFrame(left, text="EQ bands (JSON list of [freq, gain_db, Q])")
        eqf.pack(fill="both", expand=True, pady=10)
        self.eq_text = tk.Text(eqf, height=10, wrap="none")
        self.eq_text.pack(fill="both", expand=True, padx=8, pady=8)

        testf = ttk.LabelFrame(right, text="Live test")
        testf.pack(fill="both", expand=True)

        ttk.Label(testf, text="TTS URL:").pack(anchor="w", padx=8, pady=(8, 0))
        self.tts_url_entry = ttk.Entry(testf, width=44)
        self.tts_url_entry.insert(0, TTS_URL)
        self.tts_url_entry.pack(anchor="w", padx=8, pady=6)

        ttk.Label(testf, text="Test text:").pack(anchor="w", padx=8, pady=(8, 0))
        self.test_entry = ttk.Entry(testf, textvariable=self.test_text, width=44)
        self.test_entry.pack(anchor="w", padx=8, pady=6)

        seedf = ttk.LabelFrame(testf, text="Seed")
        seedf.pack(fill="x", padx=8, pady=(12, 8))

        row1 = ttk.Frame(seedf)
        row1.pack(fill="x", pady=6)

        ttk.Radiobutton(row1, text="Manual", variable=self.seed_mode, value="manual", command=self._on_seed_mode).pack(side="left")
        ttk.Radiobutton(row1, text="Auto (from text+emotion)", variable=self.seed_mode, value="auto", command=self._on_seed_mode).pack(side="left", padx=10)

        row2 = ttk.Frame(seedf)
        row2.pack(fill="x", pady=6)

        ttk.Label(row2, text="Seed value:", width=12).pack(side="left")
        self.seed_scale = ttk.Scale(row2, from_=0, to=2**31 - 1, orient="horizontal", command=self._seed_scale_changed)
        self.seed_scale.pack(side="left", fill="x", expand=True, padx=8)

        self.seed_entry = ttk.Entry(row2, width=14)
        self.seed_entry.pack(side="right")
        self.seed_entry.insert(0, str(self.seed_value.get()))

        row3 = ttk.Frame(seedf)
        row3.pack(fill="x", pady=(0, 6))
        ttk.Button(row3, text="Set Auto Seed now", command=self._set_auto_seed_now).pack(side="left")
        self.seed_preview = tk.StringVar(value="(seed preview)")
        ttk.Label(row3, textvariable=self.seed_preview).pack(side="left", padx=10)

        ttk.Button(testf, text="Save + Test", command=self._save_and_test).pack(anchor="w", padx=8, pady=(12, 6))
        ttk.Button(testf, text="Test only (no save)", command=self._test_only).pack(anchor="w", padx=8, pady=6)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=12, pady=(0, 10))

        self._sync_seed_widgets()

    def _on_seed_mode(self):
        self._sync_seed_widgets()
        self._update_seed_preview()

    def _seed_scale_changed(self, _val):
        if self.seed_mode.get() != "manual":
            return
        v = int(float(self.seed_scale.get()))
        self.seed_value.set(v)
        self.seed_entry.delete(0, tk.END)
        self.seed_entry.insert(0, str(v))
        self._update_seed_preview()

    def _sync_seed_widgets(self):
        try:
            cur = int(self.seed_entry.get().strip() or str(self.seed_value.get()))
        except Exception:
            cur = self.seed_value.get()
        self.seed_value.set(cur)
        self.seed_scale.set(cur)

        state = "normal" if self.seed_mode.get() == "manual" else "disabled"
        self.seed_entry.configure(state=state)
        try:
            self.seed_scale.configure(state=state)
        except Exception:
            pass

        def entry_commit(_evt=None):
            if self.seed_mode.get() != "manual":
                return
            try:
                v = int(float(self.seed_entry.get().strip()))
                if v < 0:
                    v = 0
                if v > 2**31 - 1:
                    v = 2**31 - 1
                self.seed_value.set(v)
                self.seed_scale.set(v)
                self.seed_entry.delete(0, tk.END)
                self.seed_entry.insert(0, str(v))
                self._update_seed_preview()
            except Exception:
                pass

        self.seed_entry.bind("<Return>", entry_commit)
        self.seed_entry.bind("<FocusOut>", entry_commit)

    def _set_auto_seed_now(self):
        s = stable_seed_from(self.test_text.get().strip(), self.emotion.get().strip().lower())
        self.seed_value.set(s)
        self.seed_entry.configure(state="normal")
        self.seed_entry.delete(0, tk.END)
        self.seed_entry.insert(0, str(s))
        self.seed_entry.configure(state="disabled" if self.seed_mode.get() == "auto" else "normal")
        self.seed_scale.set(s)
        self.seed_preview.set(f"Auto seed = {s}")

    def _update_seed_preview(self):
        if self.seed_mode.get() == "auto":
            s = stable_seed_from(self.test_text.get().strip(), self.emotion.get().strip().lower())
            self.seed_preview.set(f"Auto seed = {s}")
        else:
            self.seed_preview.set(f"Manual seed = {self.seed_value.get()}")

    def _add_num(self, parent, key, label, mn, mx, step):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=22).pack(side="left")
        v = tk.DoubleVar(value=0.0)
        self.vars[key] = v
        s = ttk.Scale(row, from_=mn, to=mx, orient="horizontal", variable=v)
        s.pack(side="left", fill="x", expand=True, padx=8)
        e = ttk.Entry(row, width=10)
        e.pack(side="right")
        e.insert(0, "0.0")

        def sync_entry(*_):
            e.delete(0, tk.END)
            e.insert(0, f"{v.get():.4g}")
        v.trace_add("write", sync_entry)

        def entry_commit(_evt=None):
            try:
                v.set(float(e.get()))
            except Exception:
                pass
        e.bind("<Return>", entry_commit)
        e.bind("<FocusOut>", entry_commit)

    def _refresh_emotions(self):
        emos = list_emotions()
        if "neutral" not in emos:
            write_json_atomic(os.path.join(EMOTIONS_DIR, "neutral.json"), {
                "pitch_semitones": 1.2,
                "time_stretch": 1.0,
                "lowcut_hz": 120.0,
                "eq": DEFAULT_EQ,
                "compressor": {"enabled": True, "threshold_db": -20.5, "ratio": 2.6, "attack_ms": 7.0, "release_ms": 115.0, "makeup_db": 2.0},
                "normalize_lufs": -16.0
            })
            emos = list_emotions()
        self.emotion_combo["values"] = emos
        if self.emotion.get() not in emos and emos:
            self.emotion.set(emos[0])

    def _selected_path(self):
        if self.mode.get() == "global":
            return GLOBAL_PATH
        return os.path.join(EMOTIONS_DIR, f"{self.emotion.get().lower().strip()}.json")

    def _load_selected(self):
        self._refresh_emotions()
        path = self._selected_path()
        data = read_json(path, fallback={})

        def get(path_key, default=None):
            cur = data
            for part in path_key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur

        for k, var in self.vars.items():
            if isinstance(var, tk.BooleanVar):
                var.set(bool(get(k, True)))
            else:
                val = get(k, None)
                if val is None:
                    val = 1.0 if k.endswith("time_stretch") else 0.0
                try:
                    var.set(float(val))
                except Exception:
                    pass

        eq = get("eq", [])
        try:
            eq_str = json.dumps(eq, ensure_ascii=False, indent=2)
        except Exception:
            eq_str = "[]"
        self.eq_text.delete("1.0", tk.END)
        self.eq_text.insert("1.0", eq_str)

        self._update_seed_preview()
        self.status.set(f"Loaded: {path}")

    def _collect_current(self):
        out = {}

        def set_nested(path_key, value):
            cur = out
            parts = path_key.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value

        for k, var in self.vars.items():
            if isinstance(var, tk.BooleanVar):
                set_nested(k, bool(var.get()))
            else:
                set_nested(k, float(var.get()))

        try:
            eq = json.loads(self.eq_text.get("1.0", tk.END).strip() or "[]")
            out["eq"] = eq
        except Exception:
            out["eq"] = []

        return out

    def _save_selected(self):
        path = self._selected_path()
        data = self._collect_current()
        try:
            write_json_atomic(path, data)
            self.status.set(f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save failed", repr(e))

    def _save_and_test(self):
        self._save_selected()
        self._test_only()

    def _get_seed_for_request(self, text: str, emotion: str):
        if self.seed_mode.get() == "auto":
            return stable_seed_from(text, emotion)
        return int(self.seed_value.get())

    def _test_only(self):
        url = self.tts_url_entry.get().strip() or TTS_URL
        emotion = self.emotion.get().strip().lower()
        text = self.test_text.get().strip() or DEFAULT_TEST_TEXT
        seed = self._get_seed_for_request(text, emotion)

        self._update_seed_preview()
        self.status.set(f"Testing... (seed={seed})")

        def run():
            try:
                r = requests.post(url, json={
                    "text": text,
                    "emotion": emotion,
                    "return_wav": True,
                    "play_audio": True,
                    "save_wav": False,
                    "seed": seed
                }, timeout=120)
                r.raise_for_status()
                _audio, _sr = wav_bytes_to_float32(r.content)
                self.status.set(f"Test done. (seed={seed})")
            except Exception as e:
                self.status.set("Test failed.")
                messagebox.showerror("Test failed", repr(e))

        threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    App().mainloop()
