import os
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from flask import Flask, request, jsonify

from osc4py3.as_eventloop import osc_startup, osc_terminate, osc_udp_client, osc_send, osc_process
from osc4py3 import oscbuildparse

app = Flask(__name__)

# =========================
# Config / State
# =========================

POSES_DIR = "Poses"
MOVES_DIR = "Moves"

JOY_RANGE = 80  # same concept as your joyRange
DEFAULT_SPEED = 2.5

@dataclass
class OSCConfig:
    enabled: bool = False
    ip: str = "127.0.0.1"
    port: int = 39539
    client_name: str = "VroidPoser"

@dataclass
class RuntimeState:
    # Travel and destination (hips position)
    travel: List[float] = None
    destination: List[float] = None
    offset: List[float] = None

    # Pose data (final values after last play)
    moved: Dict[str, List[float]] = None  # bone -> [q, w, rot] in "angle space" like your saved files

    # playback control
    lock: threading.Lock = None
    stop_flag: bool = False
    is_playing: bool = False

osc_cfg = OSCConfig()
state = RuntimeState(
    travel=[0.0, 0.0, 0.0],
    destination=[0.0, 0.0, 0.0],
    offset=[0.0, 0.0, 0.0],
    moved={},
    lock=threading.Lock(),
    stop_flag=False,
    is_playing=False
)

# =========================
# OSC helpers (matching your mapping)
# =========================

def ensure_osc_started():
    """Start/Restart OSC client based on osc_cfg."""
    if not osc_cfg.enabled:
        return
    # We terminate and re-start to be safe when changing target
    try:
        osc_terminate()
    except Exception:
        pass
    osc_startup()
    osc_udp_client(osc_cfg.ip, osc_cfg.port, osc_cfg.client_name)

def sendosc(bone: str, x: float, y: float, z: float):
    if not osc_cfg.enabled:
        return

    if bone == "Hips":
        msg = oscbuildparse.OSCMessage(
            "/VMC/Ext/Bone/Pos",
            None,
            [
                bone,
                float(state.destination[0] + state.offset[0]),
                float(state.destination[1] + state.offset[1]),
                float(state.destination[2] + state.offset[2]),
                float(x),
                float(y),
                float(z),
                float(1)
            ]
        )
    else:
        msg = oscbuildparse.OSCMessage(
            "/VMC/Ext/Bone/Pos",
            None,
            [bone, 0.0, 0.0, 0.0, float(x), float(y), float(z), float(1)]
        )

    osc_send(msg, osc_cfg.client_name)
    osc_process()

# bone groups for axis mapping (copied logic)
L_LEG = {"LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes"}
R_LEG = {"RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"}
BODY = {"RightEye", "LeftEye", "Head", "Neck", "Chest", "Spine", "Hips"}

def sieve(joint: str, q: float, w: float, rot: float):
    if joint in L_LEG:
        sendosc(joint, w, rot, q)
    elif joint in R_LEG:
        sendosc(joint, -1 * w, rot, q)
    elif joint in BODY and "Eye" not in joint:
        sendosc(joint, -1 * w, -1 * rot, -1 * q)
    else:
        sendosc(joint, rot, q, w)

# =========================
# Pose / Motion file parsing + playback
# =========================

def list_poses() -> List[str]:
    if not os.path.isdir(POSES_DIR):
        return []
    out = []
    for f in os.listdir(POSES_DIR):
        if f.lower().endswith(".txt"):
            out.append(os.path.splitext(f)[0])
    out.sort()
    return out

def list_motions() -> List[str]:
    if not os.path.isdir(MOVES_DIR):
        return []
    out = []
    for root, dirs, files in os.walk(MOVES_DIR):
        # only direct subfolders
        if root == MOVES_DIR:
            out.extend(dirs)
            break
    out.sort()
    return out

def parse_pose_file(path: str) -> Tuple[float, List[float], Dict[str, List[float]]]:
    """
    Returns: (speed, destination_travel, bone_targets)
      - bone_targets: bone -> [q, w, rot_raw] where q,w are angle-ish values from file, rot is raw from file.
    """
    speed = DEFAULT_SPEED
    dest = state.travel.copy()
    bone_targets: Dict[str, List[float]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if parts[0] == "speed" and len(parts) >= 2:
                try:
                    speed = float(parts[1])
                except ValueError:
                    speed = DEFAULT_SPEED
            elif parts[0] == "travel" and len(parts) >= 4:
                # NOTE: in your file it stores absolute travel
                try:
                    dest = [float(parts[1]), float(parts[2]), float(parts[3])]
                except ValueError:
                    dest = state.travel.copy()
            else:
                # bone,q,w,rot
                if len(parts) >= 4:
                    bone = parts[0]
                    try:
                        q = float(parts[1])
                        w = float(parts[2])
                        rot = float(parts[3])
                        bone_targets[bone] = [q, w, rot]
                    except ValueError:
                        pass

    return speed, dest, bone_targets

def play_pose_file(path: str):
    """
    Smoothly transitions from current state.moved to pose targets, like your posepicker loop.
    Uses x 0..1 steps with step size speed/100.
    """
    speed, new_dest, targets = parse_pose_file(path)

    # Initialize moved bones if first time
    with state.lock:
        if not state.moved:
            # start from zeros for bones we see
            for b in targets.keys():
                state.moved[b] = [0.0, 0.0, 0.0]

    step = max(0.001, speed / 100.0)

    x = 0.0
    while x <= 1.000001:
        with state.lock:
            if state.stop_flag:
                return
            # interpolate destination/travel
            state.destination = [
                state.travel[0] + (new_dest[0] - state.travel[0]) * x,
                state.travel[1] + (new_dest[1] - state.travel[1]) * x,
                state.travel[2] + (new_dest[2] - state.travel[2]) * x,
            ]

        # interpolate each bone
        for bone, end in targets.items():
            with state.lock:
                start = state.moved.get(bone, [0.0, 0.0, 0.0])
                if state.stop_flag:
                    return

            q0, w0, r0 = start
            q1, w1, r1 = end

            hor = q0 + (q1 - q0) * x
            ver = w0 + (w1 - w0) * x
            rot = r0 + (r1 - r0) * x
            rot = (1.0 * rot) ** 3  # same cubic

            sieve(bone, hor, ver, rot)

        x += step
        # small sleep so it doesn't hog CPU; OSC still updates quickly
        time.sleep(0.002)

    # commit final state
    with state.lock:
        for bone, end in targets.items():
            state.moved[bone] = end[:]  # store raw end (not cubed), like your moved dict stores
        state.travel = new_dest[:]     # travel becomes destination

def play_motion(name: str):
    anim_path = os.path.join(MOVES_DIR, name, "animate.txt")
    if not os.path.exists(anim_path):
        raise FileNotFoundError(anim_path)

    with open(anim_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # play each pose in order
    for pose_name in lines:
        with state.lock:
            if state.stop_flag:
                return
        pose_path = os.path.join(MOVES_DIR, name, f"{pose_name}.txt")
        if not os.path.exists(pose_path):
            # fallback: sometimes animate references a pose that only exists in Poses/
            alt = os.path.join(POSES_DIR, f"{pose_name}.txt")
            if os.path.exists(alt):
                pose_path = alt
            else:
                continue
        play_pose_file(pose_path)

# =========================
# Threaded playback control
# =========================

def _run_play(fn):
    with state.lock:
        state.stop_flag = False
        state.is_playing = True
    try:
        fn()
    finally:
        with state.lock:
            state.is_playing = False
            state.stop_flag = False

def start_background(fn):
    t = threading.Thread(target=_run_play, args=(fn,), daemon=True)
    t.start()

# =========================
# API
# =========================

@app.get("/health")
def health():
    with state.lock:
        return jsonify({
            "ok": True,
            "osc_enabled": osc_cfg.enabled,
            "osc_target": {"ip": osc_cfg.ip, "port": osc_cfg.port},
            "is_playing": state.is_playing,
            "travel": state.travel,
            "destination": state.destination,
            "offset": state.offset
        })

@app.get("/poses")
def poses():
    return jsonify({"poses": list_poses()})

@app.get("/motions")
def motions():
    return jsonify({"motions": list_motions()})

@app.post("/osc/config")
def osc_config():
    data = request.get_json(force=True, silent=True) or {}
    ip = data.get("ip", osc_cfg.ip)
    port = data.get("port", osc_cfg.port)
    enabled = bool(data.get("enabled", osc_cfg.enabled))

    osc_cfg.ip = str(ip)
    osc_cfg.port = int(port)
    osc_cfg.enabled = enabled

    if osc_cfg.enabled:
        ensure_osc_started()
    else:
        try:
            osc_terminate()
        except Exception:
            pass

    return jsonify({"ok": True, "osc": {"enabled": osc_cfg.enabled, "ip": osc_cfg.ip, "port": osc_cfg.port}})

@app.post("/offset")
def set_offset():
    data = request.get_json(force=True, silent=True) or {}
    off = data.get("offset", None)
    if not isinstance(off, list) or len(off) != 3:
        return jsonify({"ok": False, "error": "offset must be [x,y,z]"}), 400
    with state.lock:
        state.offset = [float(off[0]), float(off[1]), float(off[2])]
    return jsonify({"ok": True, "offset": state.offset})

@app.post("/play/pose/<name>")
def play_pose(name):
    path = os.path.join(POSES_DIR, f"{name}.txt")
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": f"Pose not found: {name}"}), 404

    with state.lock:
        if state.is_playing:
            return jsonify({"ok": False, "error": "Already playing. Call /stop first."}), 409

    start_background(lambda: play_pose_file(path))
    return jsonify({"ok": True, "playing": {"type": "pose", "name": name}})

@app.post("/play/motion/<name>")
def play_motion_route(name):
    folder = os.path.join(MOVES_DIR, name)
    if not os.path.isdir(folder):
        return jsonify({"ok": False, "error": f"Motion not found: {name}"}), 404

    with state.lock:
        if state.is_playing:
            return jsonify({"ok": False, "error": "Already playing. Call /stop first."}), 409

    start_background(lambda: play_motion(name))
    return jsonify({"ok": True, "playing": {"type": "motion", "name": name}})

@app.post("/stop")
def stop():
    with state.lock:
        state.stop_flag = True
    return jsonify({"ok": True, "stopping": True})

@app.post("/reset")
def reset():
    """
    Simple reset: sets travel/destination to 0 and sends zero rot/pos for known bones.
    (We only know bones we've seen via moved dict.)
    """
    with state.lock:
        state.stop_flag = True

    # give playback thread a moment to stop
    time.sleep(0.05)

    with state.lock:
        state.travel = [0.0, 0.0, 0.0]
        state.destination = [0.0, 0.0, 0.0]
        bones = list(state.moved.keys())

    for b in bones:
        sendosc(b, 0.0, 0.0, 0.0)

    with state.lock:
        for b in bones:
            state.moved[b] = [0.0, 0.0, 0.0]
        state.stop_flag = False
        state.is_playing = False

    return jsonify({"ok": True, "reset": True})

if __name__ == "__main__":
    # Optional: auto enable OSC via env
    # export OSC_IP=127.0.0.1 OSC_PORT=39539 OSC_ENABLED=1
    osc_cfg.ip = "127.0.0.1"
    osc_cfg.port = int("39539")
    osc_cfg.enabled = "0"
    if osc_cfg.enabled:
        ensure_osc_started()

    app.run(host="0.0.0.0", port=5000, debug=True)
