import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional


# ----------------------------
# Quaternion helpers
# ----------------------------
def quat_norm(q: List[float]) -> List[float]:
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x/n, y/n, z/n, w/n]

def euler_xyz_to_quat(rx: float, ry: float, rz: float) -> List[float]:
    cx = math.cos(rx * 0.5); sx = math.sin(rx * 0.5)
    cy = math.cos(ry * 0.5); sy = math.sin(ry * 0.5)
    cz = math.cos(rz * 0.5); sz = math.sin(rz * 0.5)
    qw = cx*cy*cz - sx*sy*sz
    qx = sx*cy*cz + cx*sy*sz
    qy = cx*sy*cz - sx*cy*sz
    qz = cx*cy*sz + sx*sy*cz
    return quat_norm([qx, qy, qz, qw])


# ----------------------------
# VRM Pose template
# ----------------------------
VRM_POSE_BONES = [
    "hips","spine","chest","upperChest","neck","head","leftEye","rightEye",
    "leftUpperLeg","leftLowerLeg","leftFoot","leftToes",
    "rightUpperLeg","rightLowerLeg","rightFoot","rightToes",
    "leftShoulder","leftUpperArm","leftLowerArm","leftHand",
    "rightShoulder","rightUpperArm","rightLowerArm","rightHand",
    "leftThumbMetacarpal","leftThumbProximal","leftThumbDistal",
    "leftIndexProximal","leftIndexIntermediate","leftIndexDistal",
    "leftMiddleProximal","leftMiddleIntermediate","leftMiddleDistal",
    "leftRingProximal","leftRingIntermediate","leftRingDistal",
    "leftLittleProximal","leftLittleIntermediate","leftLittleDistal",
    "rightThumbMetacarpal","rightThumbProximal","rightThumbDistal",
    "rightIndexProximal","rightIndexIntermediate","rightIndexDistal",
    "rightMiddleProximal","rightMiddleIntermediate","rightMiddleDistal",
    "rightRingProximal","rightRingIntermediate","rightRingDistal",
    "rightLittleProximal","rightLittleIntermediate","rightLittleDistal",
]

def make_empty_vrm_pose_json() -> Dict[str, Any]:
    pose = {}
    for b in VRM_POSE_BONES:
        pose[b] = {"position": [0,0,0], "rotation": [0,0,0,1]}
    return {
        "vrmMetaVersion": "0",
        "pose": pose,
        "expressions": {
            "happy":0,"angry":0,"sad":0,"relaxed":0,"surprised":0,
            "aa":0,"ih":0,"ou":0,"ee":0,"oh":0,
            "blink":0,"blinkLeft":0,"blinkRight":0
        },
        "guiSliders": {
            "lefts":{"leftThumb":0,"leftIndexFinger":0,"leftMiddleFinger":0,"leftRingFinger":0,"leftLittleFinger":0},
            "rights":{"rightThumb":0,"rightIndexFinger":0,"rightMiddleFinger":0,"rightRingFinger":0,"rightLittleFinger":0},
            "fingerSettings": {},
            "fingerJointSettings": {},
            "gages":{"pitch":0,"yaw":0}
        },
        "gages":{"pitch":0,"yaw":0}
    }


# ----------------------------
# Mapping file
# ----------------------------
def load_mapping(path: Optional[str]) -> Dict[str, Any]:
    """
    Mapping JSON:
    {
      "apply_root_position": false,
      "apply_root_rotation": false,

      "root_y_mode": "zero" | "keep" | "offset",
      "root_y_offset": 0.0,

      "muscle_to_bone_euler": {
        "spine": {"indices":[0,1,2], "scale":[0.25,0.25,0.25], "offset":[0,0,0]}
      }
    }
    """
    defaults = {
        "apply_root_position": False,   # <- wichtig!
        "apply_root_rotation": False,   # <- wichtig!
        "root_y_mode": "zero",          # zero = Y wird 0 gesetzt (kein fliegen)
        "root_y_offset": 0.0,
        "muscle_to_bone_euler": {}
    }
    if not path:
        return defaults
    p = Path(path)
    if not p.exists():
        return defaults
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return defaults
        for k in ("apply_root_position","apply_root_rotation"):
            if k in data:
                defaults[k] = bool(data[k])
        if "root_y_mode" in data and str(data["root_y_mode"]) in ("zero","keep","offset"):
            defaults["root_y_mode"] = str(data["root_y_mode"])
        if "root_y_offset" in data:
            defaults["root_y_offset"] = float(data["root_y_offset"])
        mbe = data.get("muscle_to_bone_euler")
        if isinstance(mbe, dict):
            defaults["muscle_to_bone_euler"] = mbe
        return defaults
    except Exception:
        return defaults


# ----------------------------
# Conversion
# ----------------------------
def get_vec3_from_xyz_dict(d: Any) -> Optional[List[float]]:
    if not isinstance(d, dict):
        return None
    try:
        return [float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("z", 0.0))]
    except Exception:
        return None

def get_quat_from_xyzw_dict(d: Any) -> Optional[List[float]]:
    if not isinstance(d, dict):
        return None
    try:
        q = [float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("z", 0.0)), float(d.get("w", 1.0))]
        return quat_norm(q)
    except Exception:
        return None

def convert(humanpose: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    out = make_empty_vrm_pose_json()
    pose = out["pose"]

    muscles = humanpose.get("muscles", [])
    if not isinstance(muscles, list):
        muscles = []

    # --- ROOT position (OFF by default)
    if mapping.get("apply_root_position", False):
        bp = get_vec3_from_xyz_dict(humanpose.get("bodyPosition"))
        if bp is not None:
            # kill "flying"
            mode = mapping.get("root_y_mode", "zero")
            if mode == "zero":
                bp[1] = 0.0
            elif mode == "offset":
                bp[1] = float(mapping.get("root_y_offset", 0.0))
            # keep = leave as is

            pose["hips"]["position"] = bp

    # --- ROOT rotation (OFF by default)
    if mapping.get("apply_root_rotation", False):
        br = get_quat_from_xyzw_dict(humanpose.get("bodyRotation"))
        if br is not None:
            pose["hips"]["rotation"] = br  # BUT: usually you keep this OFF to prevent spinning

    # --- muscles -> bone rotations via mapping
    m2b = mapping.get("muscle_to_bone_euler", {})
    if isinstance(m2b, dict) and muscles:
        for bone_name, cfg in m2b.items():
            if bone_name not in pose:
                continue
            if not isinstance(cfg, dict):
                continue

            idx = cfg.get("indices", None)
            sca = cfg.get("scale", [1.0, 1.0, 1.0])
            off = cfg.get("offset", [0.0, 0.0, 0.0])

            if not (isinstance(idx, list) and len(idx) == 3):
                continue

            try:
                ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])
                sx, sy, sz = float(sca[0]), float(sca[1]), float(sca[2])
                ox, oy, oz = float(off[0]), float(off[1]), float(off[2])
            except Exception:
                continue

            def m(i: int) -> float:
                if 0 <= i < len(muscles):
                    try:
                        return float(muscles[i])
                    except Exception:
                        return 0.0
                return 0.0

            ex = m(ix) * sx + ox
            ey = m(iy) * sy + oy
            ez = m(iz) * sz + oz

            q = euler_xyz_to_quat(ex, ey, ez)
            pose[bone_name]["rotation"] = q

    # Most important safety default:
    # hips.rotation is identity unless you explicitly mapped muscles into it
    # hips.position is 0 unless apply_root_position is true

    return out


def main():
    ap = argparse.ArgumentParser(description="HumanPose/MusclePose -> VRM Pose (root-safe)")
    ap.add_argument("input", help="input humanpose json")
    ap.add_argument("output", help="output vrm pose json")
    ap.add_argument("--map", default=None, help="mapping json")
    args = ap.parse_args()

    hp = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(hp, dict):
        raise SystemExit("Input JSON must be an object")

    mapping = load_mapping(args.map)
    vrm = convert(hp, mapping)

    Path(args.output).write_text(json.dumps(vrm, indent=2, ensure_ascii=False), encoding="utf-8")
    print("OK:", args.output)


if __name__ == "__main__":
    main()
