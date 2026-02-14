import os
import time
import threading
import random
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

from flask import Flask, request, jsonify
from osc4py3.as_eventloop import osc_startup, osc_terminate, osc_udp_client, osc_send, osc_process
from osc4py3 import oscbuildparse

app = Flask(__name__)

POSES_DIR = "api_used/poses"
MOVES_DIR = "api_used/moves"
FIGHT_POSE = "api_used/poses/default/fight"
RES_DIR = "Resources"

JOY_RANGE = 80
DEFAULT_SPEED = 2.5

POSE_EXTS = (".txt", ".json")

POSE_JSON_CFG_PATH = os.path.join(RES_DIR, "pose_json_config.json")

IDLE_UPDATE_HZ = 30
IDLE_TARGET_INTERVAL = (1.0, 2.0)
IDLE_SPEED = 0.24
IDLE_INTENSITY = 5.55

def _ensure_resources():
    os.makedirs(RES_DIR, exist_ok=True)

def _load_pose_json_cfg() -> dict:
    _ensure_resources()
    defaults = {
        "vrm_to_vmc": True,
        "use_hips_position": True,
        "swap_lr": False,
        "thumb_fix": True,
    }
    if not os.path.exists(POSE_JSON_CFG_PATH):
        return defaults
    try:
        with open(POSE_JSON_CFG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return defaults
        for k in list(defaults.keys()):
            if k in data:
                defaults[k] = bool(data[k])
        return defaults
    except Exception:
        return defaults

def _save_pose_json_cfg(cfg: dict) -> None:
    _ensure_resources()
    out = {
        "vrm_to_vmc": bool(cfg.get("vrm_to_vmc", True)),
        "use_hips_position": bool(cfg.get("use_hips_position", True)),
        "swap_lr": bool(cfg.get("swap_lr", False)),
        "thumb_fix": bool(cfg.get("thumb_fix", True)),
    }
    with open(POSE_JSON_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

POSE_JSON_CFG = _load_pose_json_cfg()

L_LEG = {"LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes"}
R_LEG = {"RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"}
BODY = {"RightEye", "LeftEye", "Head", "Neck", "Chest", "Spine", "Hips"}

VSEEFACE_BONES = [
    "RightHand","RightLowerArm","RightUpperArm","RightShoulder",
    "LeftShoulder","LeftUpperArm","LeftLowerArm","LeftHand",
    "RightThumbMetacarpal","RightThumbIntermediate","RightThumbDistal",
    "RightIndexProximal","RightIndexIntermediate","RightIndexDistal",
    "RightMiddleProximal","RightMiddleIntermediate","RightMiddleDistal",
    "RightRingProximal","RightRingIntermediate","RightRingDistal",
    "RightLittleProximal","RightLittleIntermediate","RightLittleDistal",
    "LeftThumbMetacarpal","LeftThumbIntermediate","LeftThumbDistal",
    "LeftIndexProximal","LeftIndexIntermediate","LeftIndexDistal",
    "LeftMiddleProximal","LeftMiddleIntermediate","LeftMiddleDistal",
    "LeftRingProximal","LeftRingIntermediate","LeftRingDistal",
    "LeftLittleProximal","LeftLittleIntermediate","LeftLittleDistal",
    "RightUpperLeg","RightLowerLeg","RightFoot","RightToes",
    "LeftUpperLeg","LeftLowerLeg","LeftFoot","LeftToes",
    "RightEye","LeftEye","Head","Neck","Chest","Spine","Hips"
]

POSEJSON_TO_VSEEFACE = {
    "hips": "Hips",
    "spine": "Spine",
    "chest": "Chest",
    "upperChest": "Chest",
    "neck": "Neck",
    "head": "Head",
    "leftEye": "LeftEye",
    "rightEye": "RightEye",
    "leftShoulder": "LeftShoulder",
    "leftUpperArm": "LeftUpperArm",
    "leftLowerArm": "LeftLowerArm",
    "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder",
    "rightUpperArm": "RightUpperArm",
    "rightLowerArm": "RightLowerArm",
    "rightHand": "RightHand",
    "leftUpperLeg": "LeftUpperLeg",
    "leftLowerLeg": "LeftLowerLeg",
    "leftFoot": "LeftFoot",
    "leftToes": "LeftToes",
    "rightUpperLeg": "RightUpperLeg",
    "rightLowerLeg": "RightLowerLeg",
    "rightFoot": "RightFoot",
    "rightToes": "RightToes",
    "leftThumbMetacarpal": "LeftThumbMetacarpal",
    "leftThumbProximal": "LeftThumbIntermediate",
    "leftThumbDistal": "LeftThumbDistal",
    "rightThumbMetacarpal": "RightThumbMetacarpal",
    "rightThumbProximal": "RightThumbIntermediate",
    "rightThumbDistal": "RightThumbDistal",
    "leftIndexProximal": "LeftIndexProximal",
    "leftIndexIntermediate": "LeftIndexIntermediate",
    "leftIndexDistal": "LeftIndexDistal",
    "leftMiddleProximal": "LeftMiddleProximal",
    "leftMiddleIntermediate": "LeftMiddleIntermediate",
    "leftMiddleDistal": "LeftMiddleDistal",
    "leftRingProximal": "LeftRingProximal",
    "leftRingIntermediate": "LeftRingIntermediate",
    "leftRingDistal": "LeftRingDistal",
    "leftLittleProximal": "LeftLittleProximal",
    "leftLittleIntermediate": "LeftLittleIntermediate",
    "leftLittleDistal": "LeftLittleDistal",
    "rightIndexProximal": "RightIndexProximal",
    "rightIndexIntermediate": "RightIndexIntermediate",
    "rightIndexDistal": "RightIndexDistal",
    "rightMiddleProximal": "RightMiddleProximal",
    "rightMiddleIntermediate": "RightMiddleIntermediate",
    "rightMiddleDistal": "RightMiddleDistal",
    "rightRingProximal": "RightRingProximal",
    "rightRingIntermediate": "RightRingIntermediate",
    "rightRingDistal": "RightRingDistal",
    "rightLittleProximal": "RightLittleProximal",
    "rightLittleIntermediate": "RightLittleIntermediate",
    "rightLittleDistal": "RightLittleDistal",
}

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
    idle_motion_enabled: bool = True
    idle_motion_thread_running: bool = False

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
    auto_blink_thread_running=False,
    idle_motion_enabled=True,
    idle_motion_thread_running=False
)

current_json_quat: Dict[str, List[float]] = {b: [0.0, 0.0, 0.0, 1.0] for b in VSEEFACE_BONES}

def quat_norm(q):
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x/n, y/n, z/n, w/n]

def quat_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

def quat_slerp(a, b, t):
    a = quat_norm(a)
    b = quat_norm(b)
    dot = quat_dot(a, b)
    if dot < 0.0:
        b = [-b[0], -b[1], -b[2], -b[3]]
        dot = -dot
    if dot > 0.9995:
        out = [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]
        return quat_norm(out)
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    sin0 = math.sin(theta0)
    theta = theta0 * t
    sinT = math.sin(theta)
    s0 = math.cos(theta) - dot * (sinT / sin0)
    s1 = sinT / sin0
    return [
        a[0]*s0 + b[0]*s1,
        a[1]*s0 + b[1]*s1,
        a[2]*s0 + b[2]*s1,
        a[3]*s0 + b[3]*s1,
    ]

def euler_xyz_to_quat(rx: float, ry: float, rz: float) -> List[float]:
    cx = math.cos(rx * 0.5); sx = math.sin(rx * 0.5)
    cy = math.cos(ry * 0.5); sy = math.sin(ry * 0.5)
    cz = math.cos(rz * 0.5); sz = math.sin(rz * 0.5)
    qw = cx*cy*cz - sx*sy*sz
    qx = sx*cy*cz + cx*sy*sz
    qy = cx*sy*cz - sx*cy*sz
    qz = cx*cy*sz + sx*sy*cz
    return quat_norm([qx, qy, qz, qw])

def _swap_lr_bone(name: str) -> str:
    if not POSE_JSON_CFG.get("swap_lr", False):
        return name
    if name.startswith("Left"):
        return "Right" + name[4:]
    if name.startswith("Right"):
        return "Left" + name[5:]
    return name

def _vrm_to_vmc_quat(q):
    qx, qy, qz, qw = q
    qz = -qz
    qw = -qw
    return quat_norm([qx, qy, qz, qw])

def _vrm_to_vmc_pos(pos):
    x, y, z = pos
    return [x, y, -z]

def ensure_osc_started():
    if not osc_cfg.enabled:
        return
    try:
        osc_terminate()
    except Exception:
        pass
    osc_startup()
    osc_udp_client(osc_cfg.ip, osc_cfg.port, osc_cfg.client_name)

def sendosc_quat(bone: str, px: float, py: float, pz: float, qx: float, qy: float, qz: float, qw: float):
    if not osc_cfg.enabled:
        return
    msg = oscbuildparse.OSCMessage(
        "/VMC/Ext/Bone/Pos",
        None,
        [bone, float(px), float(py), float(pz), float(qx), float(qy), float(qz), float(qw)]
    )
    osc_send(msg, osc_cfg.client_name)
    osc_process()

def sendosc_triplet(bone: str, x: float, y: float, z: float):
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
                float(x), float(y), float(z), float(1)
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

def sieve(joint: str, q: float, w: float, rot: float):
    if joint in L_LEG:
        sendosc_triplet(joint, w, rot, q)
    elif joint in R_LEG:
        sendosc_triplet(joint, -1 * w, rot, q)
    elif joint in BODY and "Eye" not in joint:
        sendosc_triplet(joint, -1 * w, -1 * rot, -1 * q)
    else:
        sendosc_triplet(joint, rot, q, w)

IDLE_BONES = ["Spine", "Chest", "Neck", "Head"]
IDLE_SMOOTH = 0.02
idle_current: Dict[str, List[float]] = {b: [0.0, 0.0, 0.0, 1.0] for b in IDLE_BONES}

def _send_idle_quat(bone: str, target_quat: List[float]):
    cur = idle_current.get(bone, [0.0, 0.0, 0.0, 1.0])
    sm = quat_slerp(cur, target_quat, IDLE_SMOOTH)
    idle_current[bone] = sm
    if bone == "Hips":
        sendosc_quat(
            bone,
            float(state.destination[0] + state.offset[0]),
            float(state.destination[1] + state.offset[1]),
            float(state.destination[2] + state.offset[2]),
            sm[0], sm[1], sm[2], sm[3]
        )
    else:
        sendosc_quat(bone, 0.0, 0.0, 0.0, sm[0], sm[1], sm[2], sm[3])

def _send_idle_euler(bone: str, ex: float, ey: float, ez: float):
    q = euler_xyz_to_quat(ex, ey, ez)
    _send_idle_quat(bone, q)

def sieve_idle_quat(joint: str, q: float, w: float, rot: float):
    if joint in L_LEG:
        _send_idle_euler(joint, w, rot, q)
    elif joint in R_LEG:
        _send_idle_euler(joint, -1 * w, rot, q)
    elif joint in BODY and "Eye" not in joint:
        _send_idle_euler(joint, -1 * w, -1 * rot, -1 * q)
    else:
        _send_idle_euler(joint, rot, q, w)

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

IDLE_LIMITS = {
    "Head":  (0.070, 0.070, 0.095),
    "Neck":  (0.045, 0.045, 0.065),
    "Chest": (0.030, 0.030, 0.050),
    "Spine": (0.020, 0.020, 0.035),
}

def _clamp(x, a, b):
    return a if x < a else (b if x > b else x)

def _lerp(a, b, t):
    return a + (b - a) * t

def _smooth(t):
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return quat_norm([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ])
def _idle_motion_loop():
    with state.face_lock:
        if state.idle_motion_thread_running:
            return
        state.idle_motion_thread_running = True

    target_dt = 1.0 / float(max(1, int(IDLE_UPDATE_HZ)))
    t_prev = time.time()

    breath_speed = random.uniform(10.06, 0.12) * float(IDLE_SPEED)
    breath_phase = random.uniform(10.0, math.pi * 2.0)

    micro_speed = random.uniform(10.10, 0.25) * float(IDLE_SPEED)
    micro_phase = random.uniform(10.0, math.pi * 2.0)

    shift_from = {"Head": (0.0, 0.0, 0.0), "Neck": (0.0, 0.0, 0.0), "Chest": (0.0, 0.0, 0.0), "Spine": (0.0, 0.0, 0.0)}
    shift_to   = {"Head": (0.0, 0.0, 0.0), "Neck": (0.0, 0.0, 0.0), "Chest": (0.0, 0.0, 0.0), "Spine": (0.0, 0.0, 0.0)}

    shift_start = time.time()
    shift_dur = random.uniform(0.8, 1.6) / max(0.05, float(IDLE_SPEED))
    next_shift_at = time.time() + random.uniform(float(IDLE_TARGET_INTERVAL[0]), float(IDLE_TARGET_INTERVAL[1]))

    def new_posture_target():
        scale = float(IDLE_INTENSITY)

        chest_q = random.uniform(-0.014, 0.014) * scale
        chest_w = random.uniform(-0.012, 0.012) * scale
        chest_r = random.uniform(-0.020, 0.020) * scale

        spine_q = random.uniform(-0.010, 0.010) * scale
        spine_w = random.uniform(-0.008, 0.008) * scale
        spine_r = random.uniform(-0.014, 0.014) * scale

        head_q = _clamp(random.uniform(-0.020, 0.020) * scale, -0.030, 0.030)
        head_w = _clamp(random.uniform(-0.018, 0.018) * scale, -0.028, 0.028)
        head_r = _clamp(random.uniform(-0.028, 0.028) * scale, -0.040, 0.040)

        neck_q = head_q * 0.55
        neck_w = head_w * 0.55
        neck_r = head_r * 0.55

        return {
            "Chest": (chest_q, chest_w, chest_r),
            "Spine": (spine_q, spine_w, spine_r),
            "Neck":  (neck_q,  neck_w,  neck_r),
            "Head":  (head_q,  head_w,  head_r),
        }

    shift_to.update(new_posture_target())

    tau = 0.55 / max(0.05, float(IDLE_SPEED))
    smoothed = dict(shift_to)

    try:
        while True:
            with state.lock:
                enabled = state.idle_motion_enabled
                playing = state.is_playing
                stopping = state.stop_flag

            if stopping or (not enabled) or playing:
                time.sleep(0.1)
                t_prev = time.time()
                continue

            if not osc_cfg.enabled:
                time.sleep(0.1)
                t_prev = time.time()
                continue

            now = time.time()
            dt = now - t_prev
            t_prev = now

            breath_phase += breath_speed * dt
            micro_phase += micro_speed * dt

            breath = math.sin(breath_phase)
            breath2 = math.sin(breath_phase * 0.5 + 1.2)
            breath_mix = (breath * 0.75 + breath2 * 0.25)

            micro = (
                math.sin(micro_phase) * 0.6
                + math.sin(micro_phase * 2.3 + 0.9) * 0.25
                + math.sin(micro_phase * 3.8 + 2.1) * 0.15
            )

            if now >= next_shift_at:
                next_shift_at = now + random.uniform(float(IDLE_TARGET_INTERVAL[0]), float(IDLE_TARGET_INTERVAL[1]))
                shift_start = now
                shift_dur = random.uniform(0.8, 1.6) / max(0.05, float(IDLE_SPEED))
                shift_from = dict(shift_to)
                shift_to = dict(shift_to)
                shift_to.update(new_posture_target())

                breath_speed = _clamp(breath_speed + random.uniform(-0.01, 0.01) * float(IDLE_SPEED), 0.03 * float(IDLE_SPEED), 0.18 * float(IDLE_SPEED))
                micro_speed  = _clamp(micro_speed  + random.uniform(-0.02, 0.02) * float(IDLE_SPEED), 0.05 * float(IDLE_SPEED), 0.35 * float(IDLE_SPEED))

            u = (now - shift_start) / max(0.001, shift_dur)
            u = _smooth(u)

            posture_target = {}
            for k in shift_to.keys():
                a = shift_from[k]
                b = shift_to[k]
                posture_target[k] = (
                    _lerp(a[0], b[0], u),
                    _lerp(a[1], b[1], u),
                    _lerp(a[2], b[2], u),
                )

            alpha = 1.0 - math.exp(-dt / max(1e-6, tau))
            for k in posture_target.keys():
                cur = smoothed[k]
                tgt = posture_target[k]
                smoothed[k] = (
                    _lerp(cur[0], tgt[0], alpha),
                    _lerp(cur[1], tgt[1], alpha),
                    _lerp(cur[2], tgt[2], alpha),
                )

            chest_q, chest_w, chest_r = smoothed["Chest"]
            spine_q, spine_w, spine_r = smoothed["Spine"]
            neck_q,  neck_w,  neck_r  = smoothed["Neck"]
            head_q,  head_w,  head_r  = smoothed["Head"]

            bm = breath_mix * float(IDLE_INTENSITY)
            mc = micro * float(IDLE_INTENSITY)

            chest_e = (chest_q + bm * 0.004, chest_w, chest_r + bm * 0.003)
            spine_e = (spine_q + bm * 0.003, spine_w, spine_r + bm * 0.002)
            neck_e  = (neck_q + mc * 0.0018, neck_w + mc * 0.0015, neck_r + mc * 0.0015)
            head_e  = (head_q + mc * 0.0025, head_w + mc * 0.0020, head_r + mc * 0.0020)

            for bone, (ex, ey, ez) in [("Spine", spine_e), ("Chest", chest_e), ("Neck", neck_e), ("Head", head_e)]:
                lim = IDLE_LIMITS[bone]
                ex = _clamp(ex, -lim[0], lim[0])
                ey = _clamp(ey, -lim[1], lim[1])
                ez = _clamp(ez, -lim[2], lim[2])

                base = current_json_quat.get(bone, [0.0, 0.0, 0.0, 1.0])
                delta = euler_xyz_to_quat(ex, ey, ez)
                target = quat_mul(base, delta)

                cur = idle_current.get(bone, [0.0, 0.0, 0.0, 1.0])
                sm = quat_slerp(cur, target, IDLE_SMOOTH)
                idle_current[bone] = sm

                sendosc_quat(bone, 0.0, 0.0, 0.0, sm[0], sm[1], sm[2], sm[3])

            spent = time.time() - now
            sleep_for = target_dt - spent
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        with state.face_lock:
            state.idle_motion_thread_running = False

def start_idle_motion():
    t = threading.Thread(target=_idle_motion_loop, daemon=True)
    t.start()

def list_poses() -> List[str]:
    if not os.path.isdir(POSES_DIR):
        return []
    bases = set()
    for f in os.listdir(POSES_DIR):
        lf = f.lower()
        if lf.endswith(".txt") or lf.endswith(".json"):
            bases.add(os.path.splitext(f)[0])
    return sorted(bases)

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

def parse_pose_json(path: str) -> Tuple[float, List[float], Dict[str, List[float]]]:
    speed = DEFAULT_SPEED
    dest = state.travel.copy()
    targets_quat: Dict[str, List[float]] = {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pose = data.get("pose", {})
    if not isinstance(pose, dict):
        return speed, dest, targets_quat

    if POSE_JSON_CFG.get("use_hips_position", True):
        hips = pose.get("hips", None)
        if isinstance(hips, dict):
            pos = hips.get("position", None)
            if isinstance(pos, list) and len(pos) == 3:
                try:
                    p = [float(pos[0]), float(pos[1]), float(pos[2])]
                    if POSE_JSON_CFG.get("vrm_to_vmc", True):
                        p = _vrm_to_vmc_pos(p)
                    dest = p
                except Exception:
                    pass

    for json_bone_name, payload in pose.items():
        vseeface_bone = POSEJSON_TO_VSEEFACE.get(json_bone_name)
        if not vseeface_bone:
            continue

        vseeface_bone = _swap_lr_bone(vseeface_bone)
        if vseeface_bone not in VSEEFACE_BONES:
            continue

        rot = (payload or {}).get("rotation", [0, 0, 0, 1])
        if not isinstance(rot, list) or len(rot) != 4:
            continue

        try:
            q = [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])]
        except Exception:
            continue

        if POSE_JSON_CFG.get("vrm_to_vmc", True):
            q = _vrm_to_vmc_quat(q)

        targets_quat[vseeface_bone] = q

    if POSE_JSON_CFG.get("thumb_fix", True):
        for side in ("Left", "Right"):
            meta = side + "ThumbMetacarpal"
            prox = side + "ThumbIntermediate"
            if meta not in targets_quat and prox in targets_quat and meta in VSEEFACE_BONES:
                targets_quat[meta] = targets_quat[prox]

    return speed, dest, targets_quat

def parse_pose_txt(path: str) -> Tuple[float, List[float], Dict[str, List[float]]]:
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

def parse_pose_file(path: str):
    lp = path.lower()
    if lp.endswith(".json"):
        return ("json",) + parse_pose_json(path)
    return ("txt",) + parse_pose_txt(path)

def play_pose_file(path: str):
    kind, speed, new_dest, targets = parse_pose_file(path)

    step = max(0.001, float(speed) / 100.0)
    x = 0.0
    print(kind)
    print(path)
    if kind == "json":
        targets_quat: Dict[str, List[float]] = targets
        bones = list(targets_quat.keys())

        while x <= 1.000001:
            with state.lock:
                if state.stop_flag:
                    return
                state.destination = [
                    state.travel[0] + (new_dest[0] - state.travel[0]) * x,
                    state.travel[1] + (new_dest[1] - state.travel[1]) * x,
                    state.travel[2] + (new_dest[2] - state.travel[2]) * x,
                ]
                dest_now = state.destination[:]
                off_now = state.offset[:]

            for bone in bones:
                q0 = current_json_quat.get(bone, [0.0, 0.0, 0.0, 1.0])
                q1 = targets_quat[bone]
                qt = quat_slerp(q0, q1, x)

                if bone == "Hips":
                    sendosc_quat(
                        bone,
                        float(dest_now[0] + off_now[0]),
                        float(dest_now[1] + off_now[1]),
                        float(dest_now[2] + off_now[2]),
                        qt[0], qt[1], qt[2], qt[3]
                    )
                else:
                    sendosc_quat(bone, 0.0, 0.0, 0.0, qt[0], qt[1], qt[2], qt[3])

            x += step
            time.sleep(0.01)

        for bone in bones:
            current_json_quat[bone] = targets_quat[bone]
        with state.lock:
            state.travel = new_dest[:]
        return

    targets_triplet: Dict[str, List[float]] = targets
    with state.lock:
        if not state.moved:
            for b in targets_triplet.keys():
                state.moved[b] = [0.0, 0.0, 0.0]

    while x <= 1.000001:
        with state.lock:
            if state.stop_flag:
                return
            state.destination = [
                state.travel[0] + (new_dest[0] - state.travel[0]) * x,
                state.travel[1] + (new_dest[1] - state.travel[1]) * x,
                state.travel[2] + (new_dest[2] - state.travel[2]) * x,
            ]
        for bone, end in targets_triplet.items():
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
        time.sleep(0.01)

    with state.lock:
        for bone, end in targets_triplet.items():
            state.moved[bone] = end[:]
        state.travel = new_dest[:]

def _find_pose_path_in_motion(folder: str, pose_name: str) -> str:
    candidates = []
    if pose_name.lower().endswith(".txt") or pose_name.lower().endswith(".json"):
        candidates.append(os.path.join(folder, pose_name))
        candidates.append(os.path.join(POSES_DIR, pose_name))
    else:
        candidates.append(os.path.join(folder, f"{pose_name}.txt"))
        candidates.append(os.path.join(folder, f"{pose_name}.json"))
        candidates.append(os.path.join(POSES_DIR, f"{pose_name}.txt"))
        candidates.append(os.path.join(POSES_DIR, f"{pose_name}.json"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

def play_motion(name: str):
    anim_path = os.path.join(MOVES_DIR, name, "animate.txt")
    if not os.path.exists(anim_path):
        raise FileNotFoundError(anim_path)
    folder = os.path.join(MOVES_DIR, name)
    with open(anim_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for pose_name in lines:
        with state.lock:
            if state.stop_flag:
                return
        pose_path = _find_pose_path_in_motion(folder, pose_name)
        if not pose_path:
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
            "auto_blink_enabled": state.auto_blink_enabled,
            "idle_motion_enabled": state.idle_motion_enabled,
            "pose_json": {
                "config_path": POSE_JSON_CFG_PATH,
                "vrm_to_vmc": POSE_JSON_CFG.get("vrm_to_vmc", True),
                "use_hips_position": POSE_JSON_CFG.get("use_hips_position", True),
                "swap_lr": POSE_JSON_CFG.get("swap_lr", False),
                "thumb_fix": POSE_JSON_CFG.get("thumb_fix", True),
            }
        })

@app.get("/pose_json/config")
def get_pose_json_config():
    return jsonify({"ok": True, "pose_json_config": POSE_JSON_CFG})

@app.post("/pose_json/config")
def set_pose_json_config():
    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "json body must be object"}), 400
    POSE_JSON_CFG.update({
        "vrm_to_vmc": bool(data.get("vrm_to_vmc", POSE_JSON_CFG.get("vrm_to_vmc", True))),
        "use_hips_position": bool(data.get("use_hips_position", POSE_JSON_CFG.get("use_hips_position", True))),
        "swap_lr": bool(data.get("swap_lr", POSE_JSON_CFG.get("swap_lr", False))),
        "thumb_fix": bool(data.get("thumb_fix", POSE_JSON_CFG.get("thumb_fix", True))),
    })
    _save_pose_json_cfg(POSE_JSON_CFG)
    return jsonify({"ok": True, "pose_json_config": POSE_JSON_CFG})

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

@app.post("/idle_motion")
def idle_motion():
    data = request.get_json(force=True, silent=True) or {}
    enabled = bool(data.get("enabled", True))
    with state.lock:
        state.idle_motion_enabled = enabled
    if enabled:
        start_idle_motion()
    return jsonify({"ok": True, "idle_motion_enabled": state.idle_motion_enabled})

@app.post("/play/pose/<path:name>")
def play_pose(name):
    path_txt = os.path.join(POSES_DIR, f"{name}.txt")
    path_json = os.path.join(POSES_DIR, f"{name}.json")
    path = path_txt if os.path.exists(path_txt) else path_json
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": f"Pose not found: {name} (.txt/.json)"}), 404
    with state.lock:
        if state.is_playing:
            return jsonify({"ok": False, "error": "Already playing. Call /stop first."}), 409
    start_background(lambda: play_pose_file(path))
    return jsonify({"ok": True, "playing": {"type": "pose", "name": name, "file": os.path.basename(path)}})

path = os.path.join(POSES_DIR, f"default/idle.json")
start_background(lambda: play_pose_file(path))

@app.post("/play/motion/<path:name>")
def play_motion_route(name):
    folder = os.path.join(MOVES_DIR, name)
    if not os.path.isdir(folder):
        return jsonify({"ok": False, "error": f"Motion not found: {name}"}), 404
    with state.lock:
        if state.is_playing:
            return jsonify({"ok": False, "error": "Already playing. Call /stop first."}), 409
    start_background(lambda: play_motion(name))
    return jsonify({"ok": True, "playing": {"type": "motion", "name": name}})

def play_motion_without_filepath(path,lines: list):
    for pose_name in lines:
        with state.lock:
            if state.stop_flag:
                return
        play_pose_file(path+"/"+pose_name)

@app.get("/play/motion/random")
def play_motion_route_random():
    all_files = [
        f for f in os.listdir(FIGHT_POSE)
        if os.path.isfile(os.path.join(FIGHT_POSE, f))
    ]

    if len(all_files) < 3:
        return jsonify({
            "ok": False,
            "error": "Not enough motion files available"
        }), 400

    random_files = random.sample(all_files, 6)
    random_files.append("../idle.json")
    with state.lock:
        if state.is_playing:
            return jsonify({
                "ok": False,
                "error": "Already playing. Call /stop first."
            }), 409
    print(FIGHT_POSE)
    print(random_files)
    start_background(lambda: play_motion_without_filepath(FIGHT_POSE,random_files))

    return jsonify({
        "ok": True,
        "playing": {
            "type": "motion",
            "name": random_files
        }
    })


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
        bones_txt = list(state.moved.keys())

    for b in bones_txt:
        sendosc_triplet(b, 0.0, 0.0, 0.0)

    for b in VSEEFACE_BONES:
        current_json_quat[b] = [0.0, 0.0, 0.0, 1.0]
        sendosc_quat(b, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    for b in IDLE_BONES:
        idle_current[b] = [0.0, 0.0, 0.0, 1.0]

    send_blendshape("Blink", 0.0)
    send_blendshape("Blink_L", 0.0)
    send_blendshape("Blink_R", 0.0)
    apply_blendshapes()

    with state.lock:
        for b in bones_txt:
            state.moved[b] = [0.0, 0.0, 0.0]
        state.stop_flag = False
        state.is_playing = False

    return jsonify({"ok": True, "reset": True})

if __name__ == "__main__":
    os.makedirs(POSES_DIR, exist_ok=True)
    os.makedirs(MOVES_DIR, exist_ok=True)
    _ensure_resources()
    if not os.path.exists(POSE_JSON_CFG_PATH):
        _save_pose_json_cfg(POSE_JSON_CFG)

    osc_cfg.ip = os.getenv("OSC_IP", "127.0.0.1")
    osc_cfg.port = int(os.getenv("OSC_PORT", "39539"))
    osc_cfg.enabled = os.getenv("OSC_ENABLED", "1") == "1"
    if osc_cfg.enabled:
        ensure_osc_started()

    start_auto_blink()
    start_idle_motion()

    app.run(host="0.0.0.0", port=5000, debug=False)
