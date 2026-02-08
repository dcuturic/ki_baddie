import json
import math
import os
import sys
from typing import Dict, Tuple, List

# =======================
# CONFIG (Basics)
# =======================
SPEED = 1.2
OUTPUT_DEGREES = False   # normalerweise False (radians)
TRAVEL = (0, 0, 0)

# Wie viele Varianten erzeugen?
EULER_MODES = ["xyz", "zyx"]             # für Nicht-Finger
FINGER_AXES = ["x", "y", "z"]            # Curl-Achse Tests
FINGER_FLIPS = ["none", "right", "left", "both"]  # Vorzeichen invertieren pro Seite

# Finger-Strategien:
# - "twist": extrahiert Twist um Achse (stabil)
# - "euler_zonly": nimmt Euler und schreibt nur z (kann bei manchen rigs besser matchen)
FINGER_STRATEGIES = ["twist", "euler_zonly"]

# Optional: Finger-Werte clampen (verhindert Extremwerte)
CLAMP_FINGER = True
CLAMP_MIN = -1.6   # ~ -92°
CLAMP_MAX =  1.6   # ~  92°

# =======================
# Helpers
# =======================
def fmt(x: float) -> str:
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s

def quat_norm(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    return qx/n, qy/n, qz/n, qw/n

def quat_to_euler_xyz(qx, qy, qz, qw) -> Tuple[float, float, float]:
    qx, qy, qz, qw = quat_norm(qx, qy, qz, qw)

    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw*qy - qz*qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi/2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def quat_to_euler_zyx(qx, qy, qz, qw) -> Tuple[float, float, float]:
    qx, qy, qz, qw = quat_norm(qx, qy, qz, qw)

    yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

    sinp = 2*(qw*qy - qz*qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))

    return roll, pitch, yaw

def twist_angle_around_axis(qx, qy, qz, qw, axis: str) -> float:
    qx, qy, qz, qw = quat_norm(qx, qy, qz, qw)

    if axis == "x":
        ax, ay, az = 1.0, 0.0, 0.0
    elif axis == "y":
        ax, ay, az = 0.0, 1.0, 0.0
    else:
        ax, ay, az = 0.0, 0.0, 1.0

    # project vector part onto axis => twist quaternion
    dot = qx*ax + qy*ay + qz*az
    px, py, pz = ax*dot, ay*dot, az*dot
    tx, ty, tz, tw = quat_norm(px, py, pz, qw)

    vmag = math.sqrt(tx*tx + ty*ty + tz*tz)
    ang = 2.0 * math.atan2(vmag, tw)

    # signed
    if (tx*ax + ty*ay + tz*az) < 0:
        ang = -ang

    # wrap to [-pi, pi]
    while ang > math.pi:
        ang -= 2*math.pi
    while ang < -math.pi:
        ang += 2*math.pi

    return ang

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def get_bone_quat(pose: Dict, bone: str) -> Tuple[float, float, float, float]:
    b = pose.get(bone)
    if not b:
        return 0.0, 0.0, 0.0, 1.0
    rot = b.get("rotation", [0, 0, 0, 1])
    if not isinstance(rot, list) or len(rot) != 4:
        return 0.0, 0.0, 0.0, 1.0
    return float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])

def is_finger(key: str) -> bool:
    return any(s in key for s in ["Thumb", "Index", "Middle", "Ring", "Little"])

def apply_flip(key: str, ang: float, flip_mode: str) -> float:
    if flip_mode == "none":
        return ang
    if key.startswith("Right") and flip_mode in ("right", "both"):
        return -ang
    if key.startswith("Left") and flip_mode in ("left", "both"):
        return -ang
    return ang

def euler_from_quat(qx, qy, qz, qw, euler_mode: str) -> Tuple[float, float, float]:
    if euler_mode == "zyx":
        ex, ey, ez = quat_to_euler_zyx(qx, qy, qz, qw)
    else:
        ex, ey, ez = quat_to_euler_xyz(qx, qy, qz, qw)

    if OUTPUT_DEGREES:
        ex, ey, ez = math.degrees(ex), math.degrees(ey), math.degrees(ez)
    return ex, ey, ez

# =======================
# Mapping & Order
# =======================
VSEEFACE_KEYS: List[str] = [
    "RightHand","RightLowerArm","RightUpperArm","RightShoulder",
    "LeftShoulder","LeftUpperArm","LeftLowerArm","LeftHand",

    "RightThumbIntermediate","RightThumbDistal",
    "RightIndexProximal","RightIndexIntermediate","RightIndexDistal",
    "RightMiddleProximal","RightMiddleIntermediate","RightMiddleDistal",
    "RightRingProximal","RightRingIntermediate","RightRingDistal",
    "RightLittleProximal","RightLittleIntermediate","RightLittleDistal",

    "LeftThumbIntermediate","LeftThumbDistal",
    "LeftIndexProximal","LeftIndexIntermediate","LeftIndexDistal",
    "LeftMiddleProximal","LeftMiddleIntermediate","LeftMiddleDistal",
    "LeftRingProximal","LeftRingIntermediate","LeftRingDistal",
    "LeftLittleProximal","LeftLittleIntermediate","LeftLittleDistal",

    "RightUpperLeg","RightLowerLeg","RightFoot","RightToes",
    "LeftUpperLeg","LeftLowerLeg","LeftFoot","LeftToes",

    "RightEye","LeftEye",
    "Head","Neck","Chest","Spine","Hips",
]

MAP_TO_VRM = {
    "Hips": "hips",
    "Spine": "spine",
    "Chest": "chest",
    "Neck": "neck",
    "Head": "head",
    "LeftEye": "leftEye",
    "RightEye": "rightEye",

    "LeftShoulder": "leftShoulder",
    "LeftUpperArm": "leftUpperArm",
    "LeftLowerArm": "leftLowerArm",
    "LeftHand": "leftHand",

    "RightShoulder": "rightShoulder",
    "RightUpperArm": "rightUpperArm",
    "RightLowerArm": "rightLowerArm",
    "RightHand": "rightHand",

    "LeftUpperLeg": "leftUpperLeg",
    "LeftLowerLeg": "leftLowerLeg",
    "LeftFoot": "leftFoot",
    "LeftToes": "leftToes",

    "RightUpperLeg": "rightUpperLeg",
    "RightLowerLeg": "rightLowerLeg",
    "RightFoot": "rightFoot",
    "RightToes": "rightToes",

    # Fingers: VSeeFace Intermediate -> VRM Proximal
    "LeftThumbIntermediate": "leftThumbProximal",
    "LeftThumbDistal": "leftThumbDistal",
    "LeftIndexProximal": "leftIndexProximal",
    "LeftIndexIntermediate": "leftIndexIntermediate",
    "LeftIndexDistal": "leftIndexDistal",
    "LeftMiddleProximal": "leftMiddleProximal",
    "LeftMiddleIntermediate": "leftMiddleIntermediate",
    "LeftMiddleDistal": "leftMiddleDistal",
    "LeftRingProximal": "leftRingProximal",
    "LeftRingIntermediate": "leftRingIntermediate",
    "LeftRingDistal": "leftRingDistal",
    "LeftLittleProximal": "leftLittleProximal",
    "LeftLittleIntermediate": "leftLittleIntermediate",
    "LeftLittleDistal": "leftLittleDistal",

    "RightThumbIntermediate": "rightThumbProximal",
    "RightThumbDistal": "rightThumbDistal",
    "RightIndexProximal": "rightIndexProximal",
    "RightIndexIntermediate": "rightIndexIntermediate",
    "RightIndexDistal": "rightIndexDistal",
    "RightMiddleProximal": "rightMiddleProximal",
    "RightMiddleIntermediate": "rightMiddleIntermediate",
    "RightMiddleDistal": "rightMiddleDistal",
    "RightRingProximal": "rightRingProximal",
    "RightRingIntermediate": "rightRingIntermediate",
    "RightRingDistal": "rightRingDistal",
    "RightLittleProximal": "rightLittleProximal",
    "RightLittleIntermediate": "rightLittleIntermediate",
    "RightLittleDistal": "rightLittleDistal",
}

# =======================
# Writer
# =======================
def write_variant(pose: Dict, out_path: str, euler_mode: str, finger_axis: str, finger_flip: str, finger_strategy: str):
    lines = []
    lines.append(f"speed,{SPEED}")
    lines.append(f"travel,{TRAVEL[0]},{TRAVEL[1]},{TRAVEL[2]}")

    for key in VSEEFACE_KEYS:
        vrm_name = MAP_TO_VRM.get(key)
        qx, qy, qz, qw = get_bone_quat(pose, vrm_name) if vrm_name else (0,0,0,1)

        if is_finger(key):
            if finger_strategy == "twist":
                ang = twist_angle_around_axis(qx, qy, qz, qw, finger_axis)
            else:
                # Euler Z-only strategy: take Euler then use only one component as "curl"
                ex, ey, ez = euler_from_quat(qx, qy, qz, qw, euler_mode)
                # pick the component matching the axis
                if finger_axis == "x":
                    ang = ex
                elif finger_axis == "y":
                    ang = ey
                else:
                    ang = ez

            if OUTPUT_DEGREES:
                ang = math.degrees(ang)

            ang = apply_flip(key, ang, finger_flip)

            if CLAMP_FINGER:
                ang = clamp(ang, CLAMP_MIN, CLAMP_MAX)

            # VSeeFace demo-style: only third column used for finger curl
            lines.append(f"{key},0,0,{fmt(ang)}")

        else:
            ex, ey, ez = euler_from_quat(qx, qy, qz, qw, euler_mode)
            lines.append(f"{key},{fmt(ex)},{fmt(ey)},{fmt(ez)}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# =======================
# Main
# =======================
def main():
    if len(sys.argv) < 2:
        print("Usage: python make_variants.py input_pose.json [output_dir]")
        print("  input_pose.json must contain 'pose' with bone rotations as quaternions.")
        sys.exit(1)

    in_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "vseeface_variants"

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "muscles" in data and "pose" not in data:
        raise SystemExit(
            "ERROR: Das ist die 'muscles' HumanPose-Datei. Daraus kann man ohne Unity/VRM-Rig keine Bone-Rotationen erzeugen.\n"
            "Bitte nutze als Input das VRM Pose JSON mit 'pose': { hips: {rotation:[x,y,z,w]}, ... }."
        )

    pose = data.get("pose", {})
    if not isinstance(pose, dict) or len(pose) == 0:
        raise SystemExit("ERROR: JSON hat kein gültiges 'pose' Objekt.")

    os.makedirs(out_dir, exist_ok=True)

    created = 0
    for euler_mode in EULER_MODES:
        for finger_strategy in FINGER_STRATEGIES:
            for axis in FINGER_AXES:
                for flip in FINGER_FLIPS:
                    name = f"pose__euler-{euler_mode}__finger-{finger_strategy}__axis-{axis}__flip-{flip}.txt"
                    out_path = os.path.join(out_dir, name)
                    write_variant(pose, out_path, euler_mode, axis, flip, finger_strategy)
                    created += 1

    print(f"OK: created {created} files in: {out_dir}")
    print("Teste in VSeeFace die Varianten und sag mir den Dateinamen, der am besten aussieht.")
    print("Wenn du mir den besten Dateinamen gibst + was noch leicht falsch ist (z.B. Daumen zu weit offen), fine-tunen wir sofort.")

if __name__ == "__main__":
    main()
