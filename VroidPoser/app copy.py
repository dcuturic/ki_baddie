import os
import time
import threading
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from flask import Flask, request, jsonify

from osc4py3.as_eventloop import osc_startup, osc_terminate, osc_udp_client, osc_send, osc_process
from osc4py3 import oscbuildparse

app = Flask(__name__)

POSES_DIR = "Poses"
MOVES_DIR = "Moves"

JOY_RANGE = 80
DEFAULT_SPEED = 2.5

@dataclass
class OSCConfig:
    enabled: bool = False
    ip: str = "127.0.0.1"
    port: int = 39539
    client_name: str = "VroidPoser"

@dataclass
class RuntimeState:
    travel: List[float] = None
    destination: List[float] = None
    offset: List[float] = None
    moved: Dict[str, List[float]] = None
    lock: threading.Lock = None
    stop_flag: bool = False
    is_playing: bool = False
    face_lock: threading.Lock = None
    auto_blink_enabled: bool = True
    auto_blink_thread_running: bool = False

osc_cfg = OSCConfig()
state = RuntimeState(
    travel=[0.0, 0.0, 0.0],
    destination=[0.0, 0.0, 0.0],
    offset=[0.0, 0.0, 0.0],
    moved={},
    lock=threading.Lock(),
    stop_flag=False,
    is_playing=False,
    face_lock=threading.Lock(),
    auto_blink_enabled=True,
    auto_blink_thread_running=False
)

def ensure_osc_started():
    if not osc_cfg.enabled:
        return
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

def send_blendshape(name: str, value: float):
    if not osc_cfg.enabled:
        return
    v = float(max(0.0, min(1.0, value)))
    msg = oscbuildparse.OSCMessage("/VMC/Ext/Blend/Val", None, [str(name), v])
    osc_send(msg, osc_cfg.client_name)
    osc_process()

def apply_blendshapes():
    if not osc_cfg.enabled:
        return
    msg = oscbuildparse.OSCMessage("/VMC/Ext/Blend/Apply", None, [])
    osc_send(msg, osc_cfg.client_name)
    osc_process()

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

def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def _blink_curve(t: float) -> float:
    close_end = 0.28
    hold_end = 0.40
    if t <= close_end:
        u = t / close_end
        return _smoothstep(u) ** 0.75
    if t <= hold_end:
        return 1.0
    u = (t - hold_end) / (1.0 - hold_end)
    return 1.0 - (_smoothstep(u) ** 1.25)

def _do_blink(duration_s: float, max_close: float, mode: str):
    steps = max(10, int(duration_s / 0.01))
    for i in range(steps + 1):
        with state.lock:
            if state.stop_flag:
                return
        t = i / steps
        v = _blink_curve(t) * max_close
        if mode == "both":
            send_blendshape("Blink", v)
        elif mode == "left":
            send_blendshape("Blink_L", v)
        elif mode == "right":
            send_blendshape("Blink_R", v)
        if i % 3 == 0:
            apply_blendshapes()
        time.sleep(duration_s / steps)
    if mode == "both":
        send_blendshape("Blink", 0.0)
    elif mode == "left":
        send_blendshape("Blink_L", 0.0)
    elif mode == "right":
        send_blendshape("Blink_R", 0.0)
    apply_blendshapes()

def _auto_blink_loop():
    with state.face_lock:
        if state.auto_blink_thread_running:
            return
        state.auto_blink_thread_running = True
    try:
        while True:
            with state.lock:
                enabled = state.auto_blink_enabled
            if not enabled:
                time.sleep(0.25)
                continue
            if random.random() < 0.15:
                wait_s = random.uniform(7.0, 12.0)
            else:
                wait_s = random.uniform(2.2, 6.2)
            wait_s += random.uniform(-0.25, 0.25)
            time.sleep(max(0.4, wait_s))
            if not osc_cfg.enabled:
                continue
            is_wink = random.random() < 0.08
            duration = random.uniform(0.10, 0.20)
            strength = random.uniform(0.80, 1.00)
            if is_wink:
                side = "left" if random.random() < 0.5 else "right"
                _do_blink(duration_s=duration * random.uniform(1.05, 1.35), max_close=min(1.0, strength * 1.05), mode=side)
            else:
                _do_blink(duration_s=duration, max_close=strength, mode="both")
    finally:
        with state.face_lock:
            state.auto_blink_thread_running = False

def start_auto_blink():
    t = threading.Thread(target=_auto_blink_loop, daemon=True)
    t.start()

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
        if root == MOVES_DIR:
            out.extend(dirs)
            break
    out.sort()
    return out

def parse_pose_file(path: str) -> Tuple[float, List[float], Dict[str, List[float]]]:
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
                try:
                    dest = [float(parts[1]), float(parts[2]), float(parts[3])]
                except ValueError:
                    dest = state.travel.copy()
            else:
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
    speed, new_dest, targets = parse_pose_file(path)
    with state.lock:
        if not state.moved:
            for b in targets.keys():
                state.moved[b] = [0.0, 0.0, 0.0]
    step = max(0.001, speed / 100.0)
    x = 0.0
    while x <= 1.000001:
        with state.lock:
            if state.stop_flag:
                return
            state.destination = [
                state.travel[0] + (new_dest[0] - state.travel[0]) * x,
                state.travel[1] + (new_dest[1] - state.travel[1]) * x,
                state.travel[2] + (new_dest[2] - state.travel[2]) * x,
            ]
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
            rot = (1.0 * rot) ** 3
            sieve(bone, hor, ver, rot)
        x += step
        time.sleep(0.002)
    with state.lock:
        for bone, end in targets.items():
            state.moved[bone] = end[:]
        state.travel = new_dest[:]

def play_motion(name: str):
    anim_path = os.path.join(MOVES_DIR, name, "animate.txt")
    if not os.path.exists(anim_path):
        raise FileNotFoundError(anim_path)
    with open(anim_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for pose_name in lines:
        with state.lock:
            if state.stop_flag:
                return
        pose_path = os.path.join(MOVES_DIR, name, f"{pose_name}.txt")
        if not os.path.exists(pose_path):
            alt = os.path.join(POSES_DIR, f"{pose_name}.txt")
            if os.path.exists(alt):
                pose_path = alt
            else:
                continue
        play_pose_file(pose_path)

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
            "offset": state.offset,
            "auto_blink_enabled": state.auto_blink_enabled
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

@app.post("/face/auto_blink")
def face_auto_blink():
    data = request.get_json(force=True, silent=True) or {}
    enabled = bool(data.get("enabled", True))
    with state.lock:
        state.auto_blink_enabled = enabled
    return jsonify({"ok": True, "auto_blink_enabled": state.auto_blink_enabled})

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
    with state.lock:
        state.stop_flag = True
    time.sleep(0.05)
    with state.lock:
        state.travel = [0.0, 0.0, 0.0]
        state.destination = [0.0, 0.0, 0.0]
        bones = list(state.moved.keys())
    for b in bones:
        sendosc(b, 0.0, 0.0, 0.0)
    send_blendshape("Blink", 0.0)
    send_blendshape("Blink_L", 0.0)
    send_blendshape("Blink_R", 0.0)
    apply_blendshapes()
    with state.lock:
        for b in bones:
            state.moved[b] = [0.0, 0.0, 0.0]
        state.stop_flag = False
        state.is_playing = False
    return jsonify({"ok": True, "reset": True})

if __name__ == "__main__":
    osc_cfg.ip = os.getenv("OSC_IP", "127.0.0.1")
    osc_cfg.port = int(os.getenv("OSC_PORT", "39539"))
    osc_cfg.enabled = os.getenv("OSC_ENABLED", "1") == "1"
    if osc_cfg.enabled:
        ensure_osc_started()
    start_auto_blink()
    app.run(host="0.0.0.0", port=5000, debug=True)
