import json, struct
from pathlib import Path

# --- GLB helpers ------------------------------------------------------------

def read_glb(path: str):
    data = Path(path).read_bytes()
    if data[:4] != b'glTF':
        raise ValueError("Not a valid GLB (missing 'glTF' magic).")

    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"Unsupported GLB version: {version} (expected 2).")

    offset = 12
    json_chunk = None
    bin_chunk = b""

    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset: offset + chunk_len]
        offset += chunk_len

        if chunk_type == 0x4E4F534A:  # JSON
            json_chunk = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == 0x004E4942:  # BIN
            bin_chunk = chunk_data

    if json_chunk is None:
        raise ValueError("GLB has no JSON chunk.")
    return json_chunk, bin_chunk


def get_accessor_floats(gltf, bin_chunk: bytes, accessor_index: int):
    accessor = gltf["accessors"][accessor_index]
    bv = gltf["bufferViews"][accessor["bufferView"]]

    component_type = accessor["componentType"]
    if component_type != 5126:
        raise ValueError(f"Only FLOAT(5126) supported here, got {component_type}")

    type_str = accessor["type"]
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[type_str]
    count = accessor["count"]

    byte_offset = bv.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    byte_length = bv["byteLength"]

    # Slice the bufferView range, then offset into it
    view_data = bin_chunk[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + byte_length]
    raw = view_data[accessor.get("byteOffset", 0): accessor.get("byteOffset", 0) + 4 * comps * count]

    floats = struct.unpack("<" + "f" * (comps * count), raw)
    # Return as list of lists for vectors, or list for scalars
    if comps == 1:
        return list(floats)
    return [list(floats[i*comps:(i+1)*comps]) for i in range(count)]


# --- Conversion: VRMC_vrm_animation -> Pose JSON ----------------------------

# Bone names you use in your JSON
POSE_BONES = [
    "hips","spine","chest","upperChest","neck","head",
    "leftEye","rightEye",
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

def extract_pose_json(glb_path: str):
    gltf, bin_chunk = read_glb(glb_path)

    # VRMC_vrm_animation humanoid mapping: boneName -> {node: index}
    ext = (gltf.get("extensions") or {}).get("VRMC_vrm_animation")
    if not ext:
        raise ValueError("No VRMC_vrm_animation extension found.")
    human_bones = (((ext.get("humanoid") or {}).get("humanBones")) or {})

    # Build nodeIndex -> boneName map
    node_to_bone = {}
    for bone_name, info in human_bones.items():
        if "node" in info:
            node_to_bone[info["node"]] = bone_name

    # Prepare defaults from node transforms
    nodes = gltf.get("nodes", [])
    pose = {}
    for bone in POSE_BONES:
        pose[bone] = {
            "position": [0,0,0],
            "rotation": [0,0,0,1]
        }

    def set_from_node(node_idx: int, bone: str):
        n = nodes[node_idx]
        if "translation" in n:
            pose[bone]["position"] = list(n["translation"])
        if "rotation" in n:
            pose[bone]["rotation"] = list(n["rotation"])

    for node_idx, bone in node_to_bone.items():
        if bone in pose:
            set_from_node(node_idx, bone)

    # Apply animation values (if present)
    anims = gltf.get("animations") or []
    if anims:
        anim = anims[0]
        samplers = anim.get("samplers") or []
        channels = anim.get("channels") or []

        for ch in channels:
            target = ch.get("target") or {}
            node_idx = target.get("node")
            path = target.get("path")  # "rotation" or "translation"
            if node_idx is None or path not in ("rotation", "translation"):
                continue

            bone = node_to_bone.get(node_idx)
            if not bone or bone not in pose:
                continue

            sampler = samplers[ch["sampler"]]
            out_acc = sampler["output"]

            out_vals = get_accessor_floats(gltf, bin_chunk, out_acc)

            # Your file has count=1 -> take first key
            if not out_vals:
                continue

            if path == "rotation":
                pose[bone]["rotation"] = list(out_vals[0])
            elif path == "translation":
                # translation accessor is VEC3 -> out_vals[0] list of 3
                pose[bone]["position"] = list(out_vals[0])

    # Fill your full structure (expressions/guiSliders/gages) with defaults
    result = {
        "vrmMetaVersion": "0",
        "pose": pose,
        "expressions": {
            "happy": 0, "angry": 0, "sad": 0, "relaxed": 0,
            "surprised": None,
            "aa": 0, "ih": 0, "ou": 0, "ee": 0, "oh": 0,
            "blink": 0, "blinkLeft": 0, "blinkRight": 0
        },
        "guiSliders": {
            "lefts": {"leftThumb":0,"leftIndexFinger":0,"leftMiddleFinger":0,"leftRingFinger":0,"leftLittleFinger":0},
            "rights": {"rightThumb":0,"rightIndexFinger":0,"rightMiddleFinger":0,"rightRingFinger":0,"rightLittleFinger":0},
            "fingerSettings": {
                "leftThumb":{"x":0,"y":0,"z":0},"leftIndex":{"x":0,"y":0,"z":0},"leftMiddle":{"x":0,"y":0,"z":0},
                "leftRing":{"x":0,"y":0,"z":0},"leftLittle":{"x":0,"y":0,"z":0},
                "rightThumb":{"x":0,"y":0,"z":0},"rightIndex":{"x":0,"y":0,"z":0},"rightMiddle":{"x":0,"y":0,"z":0},
                "rightRing":{"x":0,"y":0,"z":0},"rightLittle":{"x":0,"y":0,"z":0},
            },
            "fingerJointSettings": {
                "leftThumb":{"proximal":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
                "leftIndex":{"proximal":{"x":0,"y":0,"z":0},"intermediate":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
                "leftMiddle":{"proximal":{"x":0,"y":0,"z":0},"intermediate":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
                "leftRing":{"proximal":{"x":0,"y":0,"z":0},"intermediate":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
                "leftLittle":{"proximal":{"x":0,"y":0,"z":0},"intermediate":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
                "rightThumb":{"proximal":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
                "rightIndex":{"proximal":{"x":0,"y":0,"z":0},"intermediate":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
                "rightMiddle":{"proximal":{"x":0,"y":0,"z":0},"intermediate":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
                "rightRing":{"proximal":{"x":0,"y":0,"z":0},"intermediate":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
                "rightLittle":{"proximal":{"x":0,"y":0,"z":0},"intermediate":{"x":0,"y":0,"z":0},"distal":{"x":0,"y":0,"z":0}},
            },
            "gages": {"pitch": 0, "yaw": 0}
        },
        "gages": {"pitch": 0, "yaw": 0}
    }

    return result


if __name__ == "__main__":
    p = Path(r"C:\Users\Deeliar\Downloads\animations\pose_20260210180046260.vrma")  # <- anpassen
    b = p.read_bytes()[:32]
    print("first 32 bytes:", b)
    print("as text:", b.decode("utf-8", errors="replace"))
    print("hex:", b.hex(" "))
    
    glb_path = r"C:\Users\Deeliar\Downloads\animations\pose_20260210180046260.vrma"   # <- deine Datei


    out_path = "pose2.json"

    pose_json = extract_pose_json(glb_path)
    Path(out_path).write_text(json.dumps(pose_json, indent=2), encoding="utf-8")
    print("Wrote:", out_path)
