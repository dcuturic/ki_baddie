"""
Web Avatar — Browser-basierter VRM Avatar Renderer
===================================================
Ersetzt VSeeFace komplett: Empfängt Bone/Expression-Daten via OSC (VMC-Protokoll),
rendert das VRM-Modell im Browser mit three-vrm.

Architektur:
  [VroidPoser]  ──OSC──┐
  [VroidEmotion]──OSC──┤──> [web_avatar :39539] ──WebSocket──> [Browser: three-vrm]
  [TTS Lipsync] ──OSC──┘         │
                          (optional: OSC forward → VSeeFace)

Nutzung:
  1. VRM-Datei in web_avatar/models/dilara.vrm ablegen
  2. python app.py
  3. Browser öffnen: http://localhost:5006
  4. VSeeFace stoppen (web_avatar übernimmt Port 39539)
"""

import os
import json
import time
import threading
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file, Response
from flask_socketio import SocketIO, emit

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

server_cfg = CONFIG.get("server", {})
SERVER_HOST = server_cfg.get("host", "0.0.0.0")
SERVER_PORT = server_cfg.get("port", 5006)

osc_cfg = CONFIG.get("osc", {})
OSC_ENABLED = osc_cfg.get("enabled", True)
OSC_LISTEN_HOST = osc_cfg.get("listen_host", "0.0.0.0")
OSC_LISTEN_PORT = osc_cfg.get("listen_port", 39539)
OSC_FORWARD_ENABLED = osc_cfg.get("forward_enabled", False)
OSC_FORWARD_HOST = osc_cfg.get("forward_host", "127.0.0.1")
OSC_FORWARD_PORT = osc_cfg.get("forward_port", 39540)

vrm_cfg = CONFIG.get("vrm", {})
VRM_MODEL_PATH = vrm_cfg.get("model_path", "models/dilara.vrm")

_ext = os.path.splitext(VRM_MODEL_PATH)[1].lower()
if _ext == ".glb":
    MODEL_FORMAT = "glb"
elif _ext == ".fbx":
    MODEL_FORMAT = "fbx"
else:
    MODEL_FORMAT = "vrm"

cam_cfg = CONFIG.get("camera", {})
CAM_POSITION = cam_cfg.get("position", [0, 1.35, 0.9])
CAM_TARGET = cam_cfg.get("target", [0, 1.25, 0])
CAM_FOV = cam_cfg.get("fov", 28)

# ======================= FLASK + SOCKET.IO =======================

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["SECRET_KEY"] = "web-avatar-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ======================= STATE BUFFERS =======================
# Bone data: accumulated per frame, broadcast at 30fps
_bone_buffer: Dict[str, Dict] = {}
_bone_lock = threading.Lock()

# Blend shape data: accumulated until /VMC/Ext/Blend/Apply
_blend_buffer: Dict[str, float] = {}
_blend_lock = threading.Lock()
_blend_dirty = False

# Current camera state — sent to every newly connected browser
_camera_state: Dict = {}   # {"type": "preset", "preset": "front"} or {"type": "position", ...}
_camera_lock = threading.Lock()

# ======================= OSC FORWARDING =======================

_osc_forward_client = None

if OSC_FORWARD_ENABLED:
    try:
        from pythonosc.udp_client import SimpleUDPClient
        _osc_forward_client = SimpleUDPClient(OSC_FORWARD_HOST, OSC_FORWARD_PORT)
        print(f"[OSC] Forwarding aktiviert → {OSC_FORWARD_HOST}:{OSC_FORWARD_PORT}")
    except Exception as e:
        print(f"[OSC] Forward-Client Fehler: {e}")


def _forward_osc(addr, args):
    """Forward OSC message to VSeeFace (optional)."""
    if _osc_forward_client:
        try:
            _osc_forward_client.send_message(addr, list(args))
        except Exception:
            pass


# ======================= OSC HANDLERS =======================

def _handle_bone(addr, *args):
    """
    /VMC/Ext/Bone/Pos <name> <px> <py> <pz> <qx> <qy> <qz> <qw>
    Empfängt Bone-Transforms im VMC-Format (Unity left-handed).
    """
    if len(args) < 8:
        return

    name = str(args[0])
    px, py, pz = float(args[1]), float(args[2]), float(args[3])
    qx, qy, qz, qw = float(args[4]), float(args[5]), float(args[6]), float(args[7])

    with _bone_lock:
        _bone_buffer[name] = {
            "p": [px, py, pz],
            "q": [qx, qy, qz, qw]
        }

    _forward_osc(addr, args)


def _handle_blend_val(addr, *args):
    """
    /VMC/Ext/Blend/Val <name> <value>
    Sammelt Blend Shape Werte bis /VMC/Ext/Blend/Apply kommt.
    """
    global _blend_dirty
    if len(args) < 2:
        return

    name = str(args[0])
    value = float(args[1])

    with _blend_lock:
        _blend_buffer[name] = value
        _blend_dirty = True

    _forward_osc(addr, args)


def _handle_blend_apply(addr, *args):
    """
    /VMC/Ext/Blend/Apply
    Sendet alle gesammelten Blend Shape Werte an den Browser.
    """
    global _blend_dirty

    with _blend_lock:
        if _blend_dirty and _blend_buffer:
            data = dict(_blend_buffer)
            _blend_dirty = False
        else:
            data = None

    if data:
        socketio.emit("blend", data)

    _forward_osc(addr, args if args else [1])


# ======================= OSC SERVER =======================

def _start_osc_server():
    """Startet den OSC-Empfänger (VMC-Protokoll)."""
    try:
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer
    except ImportError:
        print("[OSC] python-osc nicht installiert! pip install python-osc")
        return

    dispatcher = Dispatcher()
    dispatcher.map("/VMC/Ext/Bone/Pos", _handle_bone)
    dispatcher.map("/VMC/Ext/Blend/Val", _handle_blend_val)
    dispatcher.map("/VMC/Ext/Blend/Apply", _handle_blend_apply)

    try:
        server = ThreadingOSCUDPServer((OSC_LISTEN_HOST, OSC_LISTEN_PORT), dispatcher)
        print(f"[OSC] Empfänger aktiv auf {OSC_LISTEN_HOST}:{OSC_LISTEN_PORT}")
        print(f"[OSC] (Ersetzt VSeeFace — gleicher Port wie VSeeFace default)")
        server.serve_forever()
    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            print(f"[OSC] FEHLER: Port {OSC_LISTEN_PORT} bereits belegt!")
            print(f"[OSC] → VSeeFace läuft noch? Bitte zuerst beenden.")
            print(f"[OSC] → Oder in config.json 'osc.listen_port' ändern.")
        else:
            print(f"[OSC] Server Fehler: {e}")


# ======================= BROADCAST THREAD =======================

def _broadcast_bones_loop():
    """Sendet Bone-Daten an alle Browser-Clients mit ~30fps."""
    while True:
        time.sleep(1.0 / 30.0)

        with _bone_lock:
            if _bone_buffer:
                data = dict(_bone_buffer)
            else:
                data = None

        if data:
            socketio.emit("bones", data)


# ======================= HTTP API =======================

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "web_avatar",
        "osc_enabled": OSC_ENABLED,
        "osc_port": OSC_LISTEN_PORT if OSC_ENABLED else None,
        "osc_forward": OSC_FORWARD_ENABLED,
        "vrm_model": VRM_MODEL_PATH,
        "vrm_exists": os.path.exists(VRM_MODEL_PATH),
    })


@app.post("/api/bones")
def api_bones():
    """Empfängt Bone-Daten per HTTP (Alternative zu OSC)."""
    data = request.get_json() or {}
    bones = data.get("bones", data)
    with _bone_lock:
        _bone_buffer.update(bones)
    socketio.emit("bones", bones)
    return jsonify({"ok": True})


@app.post("/api/expression")
def api_expression():
    """Empfängt Expression-Daten per HTTP."""
    data = request.get_json() or {}
    socketio.emit("expression", data)
    return jsonify({"ok": True})


@app.post("/api/blend")
def api_blend():
    """Empfängt Blend Shape Werte per HTTP."""
    data = request.get_json() or {}
    socketio.emit("blend", data)
    return jsonify({"ok": True})


@app.post("/api/lipsync")
def api_lipsync():
    """Empfängt Lip-Sync Daten per HTTP."""
    data = request.get_json() or {}
    socketio.emit("lipsync", data)
    return jsonify({"ok": True})


# ======================= VRM MODEL SERVING =======================

@app.get("/vrm/model")
def serve_vrm():
    """Liefert die VRM/GLB-Datei an den Browser."""
    if not os.path.exists(VRM_MODEL_PATH):
        return jsonify({
            "ok": False,
            "error": f"Model-Datei nicht gefunden: {VRM_MODEL_PATH}",
            "hint": "Lege deine .vrm, .glb oder .fbx Datei in web_avatar/models/ ab"
        }), 404
    ext = os.path.splitext(VRM_MODEL_PATH)[1].lower()
    if ext == ".glb":
        mime = "model/gltf-binary"
        dl_name = "model.glb"
    elif ext == ".fbx":
        mime = "application/octet-stream"
        dl_name = "model.fbx"
    else:
        mime = "application/octet-stream"
        dl_name = "model.vrm"
    return send_file(
        os.path.abspath(VRM_MODEL_PATH),
        mimetype=mime,
        as_attachment=False,
        download_name=dl_name
    )


@app.get("/vrm/config")
def vrm_config():
    """Liefert Konfiguration für den Viewer."""
    # Live aus config.json lesen damit Änderungen sofort wirken
    live_cfg = load_config()
    live_cam = live_cfg.get("camera", {})
    return jsonify({
        "camera": {
            "position": live_cam.get("position", CAM_POSITION),
            "target": live_cam.get("target", CAM_TARGET),
            "fov": live_cam.get("fov", CAM_FOV),
        },
        "vrm_url": "/vrm/model",
        "vrm_exists": os.path.exists(VRM_MODEL_PATH),
        "model_format": MODEL_FORMAT,
        "model_path": VRM_MODEL_PATH,
    })


@app.route("/api/camera/save", methods=["POST"])
def api_camera_save():
    """Speichert die aktuelle Kamera-Position in die config.json."""
    try:
        data = request.get_json(force=True)
        pos = data.get("position")
        target = data.get("target")
        fov = data.get("fov")

        if not pos or not target:
            return jsonify({"success": False, "error": "position und target erforderlich"}), 400

        # Config laden, camera updaten, speichern
        cfg = load_config()
        cfg.setdefault("camera", {})
        cfg["camera"]["position"] = [round(v, 4) for v in pos]
        cfg["camera"]["target"] = [round(v, 4) for v in target]
        if fov is not None:
            cfg["camera"]["fov"] = round(fov, 1)

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        print(f"[Camera] Gespeichert: pos={cfg['camera']['position']}, target={cfg['camera']['target']}, fov={cfg['camera'].get('fov')}")
        return jsonify({"success": True, "camera": cfg["camera"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/camera/view", methods=["POST"])
def api_camera_view():
    """Kamera zu einem Preset springen lassen (per WebSocket an Browser)."""
    try:
        data = request.get_json(force=True)
        preset = data.get("preset") or data.get("view") or data.get("id")

        if not preset:
            # Liste aller verfügbaren Presets zurückgeben
            presets = [
                "front", "front_bust", "face", "face_close",
                "front_hip", "front_thigh", "front_legs", "front_feet",
                "back", "back_bust", "back_butt", "back_butt_close",
                "back_thigh", "back_legs",
                "side_right", "side_right_close", "side_left", "side_left_close",
                "top", "bottom",
            ]
            return jsonify({"success": False, "error": "preset/view/id required", "available_presets": presets}), 400

        # State speichern, damit neue Clients denselben View bekommen
        with _camera_lock:
            _camera_state.clear()
            _camera_state.update({"type": "preset", "preset": preset})

        socketio.emit("camera_view", {"preset": preset})
        print(f"[Camera] View-Wechsel: {preset}")
        return jsonify({"success": True, "preset": preset})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/camera/position", methods=["POST"])
def api_camera_position():
    """Kamera direkt zu einer Position springen lassen (per WebSocket an Browser)."""
    try:
        data = request.get_json(force=True)
        pos = data.get("position")
        target = data.get("target")
        fov = data.get("fov")

        if not pos or not target:
            return jsonify({"success": False, "error": "position und target erforderlich"}), 400

        payload = {"position": pos, "target": target}
        if fov is not None:
            payload["fov"] = fov

        # State speichern, damit neue Clients dieselbe Position bekommen
        with _camera_lock:
            _camera_state.clear()
            _camera_state.update({"type": "position", **payload})

        socketio.emit("camera_move", payload)
        print(f"[Camera] Position-Wechsel: pos={pos}, target={target}")
        return jsonify({"success": True, **payload})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ======================= MAIN PAGE =======================

@app.get("/")
def index():
    return send_file(os.path.join("static", "index.html"))


# ======================= SOCKET.IO EVENTS =======================

@socketio.on("connect")
def on_connect():
    print("[WS] Browser-Client verbunden", flush=True)
    # Sende aktuellen Bone-State an neuen Client
    with _bone_lock:
        if _bone_buffer:
            emit("bones", dict(_bone_buffer))
    with _blend_lock:
        if _blend_buffer:
            emit("blend", dict(_blend_buffer))
    # Sende aktuellen Kamera-State an neuen Client
    with _camera_lock:
        if _camera_state:
            ctype = _camera_state.get("type")
            if ctype == "preset":
                emit("camera_view", {"preset": _camera_state["preset"]})
            elif ctype == "position":
                payload = {k: v for k, v in _camera_state.items() if k != "type"}
                emit("camera_move", payload)


@socketio.on("disconnect")
def on_disconnect():
    print("[WS] Browser-Client getrennt", flush=True)


# ======================= MODELS DIR =======================

os.makedirs("models", exist_ok=True)


# ======================= START =======================

if __name__ == "__main__":
    print("=" * 60)
    print("  Web Avatar — Browser-basierter VRM Renderer")
    print("=" * 60)
    print(f"  Server:     http://localhost:{SERVER_PORT}")
    print(f"  Model-Datei: {VRM_MODEL_PATH} ({MODEL_FORMAT.upper()}) ({'OK' if os.path.exists(VRM_MODEL_PATH) else 'FEHLT!'})")
    print(f"  OSC:        {'Port ' + str(OSC_LISTEN_PORT) if OSC_ENABLED else 'deaktiviert'}")
    if OSC_FORWARD_ENABLED:
        print(f"  OSC Fwd:    {OSC_FORWARD_HOST}:{OSC_FORWARD_PORT}")
    print("=" * 60)

    if not os.path.exists(VRM_MODEL_PATH):
        print(f"\n  [!] Model-Datei fehlt: {VRM_MODEL_PATH}")
        print(f"  [!] Lege deine .vrm, .glb oder .fbx Datei dort ab und starte neu.\n")

    # Start OSC listener
    if OSC_ENABLED:
        threading.Thread(target=_start_osc_server, daemon=True).start()

    # Start bone broadcast loop
    threading.Thread(target=_broadcast_bones_loop, daemon=True).start()

    # Start Flask + Socket.IO
    socketio.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
        use_reloader=False
    )
