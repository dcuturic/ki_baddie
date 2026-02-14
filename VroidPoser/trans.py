import json
import struct
from pathlib import Path


# -----------------------------------------------------------------------------
# GLB helpers
# -----------------------------------------------------------------------------

GLB_MAGIC = b"glTF"
CHUNK_TYPE_JSON = 0x4E4F534A  # JSON
CHUNK_TYPE_BIN  = 0x004E4942  # BIN


def read_glb(path: str):
    data = Path(path).read_bytes()

    if data[:4] != GLB_MAGIC:
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

        if chunk_type == CHUNK_TYPE_JSON:
            txt = chunk_data.decode("utf-8", errors="strict").rstrip("\x00").strip()
            json_chunk = json.loads(txt)
        elif chunk_type == CHUNK_TYPE_BIN:
            bin_chunk = chunk_data

    if json_chunk is None:
        raise ValueError("GLB has no JSON chunk.")

    return json_chunk, bin_chunk


# -----------------------------------------------------------------------------
# Accessor reader (FLOAT only) with proper byteStride support
# -----------------------------------------------------------------------------

def get_accessor_floats(gltf: dict, bin_chunk: bytes, accessor_index: int):
    accessor = gltf["accessors"][accessor_index]
    bv = gltf["bufferViews"][accessor["bufferView"]]

    component_type = accessor["componentType"]
    if component_type != 5126:
        raise ValueError(f"Only FLOAT(5126) supported here, got {component_type}")

    type_str = accessor["type"]
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[type_str]
    count = int(accessor["count"])

    bv_offset = int(bv.get("byteOffset", 0))
    acc_offset = int(accessor.get("byteOffset", 0))
    base = bv_offset + acc_offset

    stride = int(bv.get("byteStride", 0))
    elem_size = 4 * comps

    if stride == 0:
        stride = elem_size

    if stride < elem_size:
        raise ValueError(f"Invalid byteStride={stride} for elem_size={elem_size}")

    out = []
    for i in range(count):
        off = base + i * stride
        raw = bin_chunk[off: off + elem_size]
        if len(raw) != elem_size:
            raise ValueError(
                f"Truncated accessor read: need {elem_size} bytes at {off}, got {len(raw)}"
            )

        vals = struct.unpack("<" + "f" * comps, raw)
        if comps == 1:
            out.append(float(vals[0]))
        else:
            out.append([float(v) for v in vals])

    return out


# -----------------------------------------------------------------------------
# Conversion: VRMC_vrm_animation -> Pose JSON
# -----------------------------------------------------------------------------

POSE_BONES = [
    "hips", "spine", "chest", "upperChest", "neck", "head",
    "leftEye", "rightEye",
    "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
    "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
    "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
    "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
    "leftThumbMetacarpal", "leftThumbProximal", "leftThumbDistal",
    "leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal",
    "leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal",
    "leftRingProximal", "leftRingIntermediate", "leftRingDistal",
    "leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal",
    "rightThumbMetacarpal", "rightThumbProximal", "rightThumbDistal",
    "rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal",
    "rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal",
    "rightRingProximal", "rightRingIntermediate", "rightRingDistal",
    "rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal",
]


def extract_pose_json(glb_path: str) -> dict:
    gltf, bin_chunk = read_glb(glb_path)

    ext = (gltf.get("extensions") or {}).get("VRMC_vrm_animation")
    if not ext:
        raise ValueError("No VRMC_vrm_animation extension found.")

    human_bones = (((ext.get("humanoid") or {}).get("humanBones")) or {})
    nodes = gltf.get("nodes", [])

    node_to_bone = {}
    for bone_name, info in human_bones.items():
        if isinstance(info, dict) and "node" in info:
            node_to_bone[int(info["node"])] = bone_name

    pose = {
        bone: {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}
        for bone in POSE_BONES
    }

    for node_idx, bone in node_to_bone.items():
        if bone not in pose:
            continue
        if node_idx < 0 or node_idx >= len(nodes):
            continue
        n = nodes[node_idx] or {}
        if isinstance(n.get("translation"), list) and len(n["translation"]) == 3:
            pose[bone]["position"] = [float(x) for x in n["translation"]]
        if isinstance(n.get("rotation"), list) and len(n["rotation"]) == 4:
            pose[bone]["rotation"] = [float(x) for x in n["rotation"]]

    anims = gltf.get("animations") or []
    if anims:
        anim = anims[0]
        samplers = anim.get("samplers") or []
        channels = anim.get("channels") or []

        for ch in channels:
            target = (ch or {}).get("target") or {}
            node_idx = target.get("node")
            path = target.get("path")
            if node_idx is None or path not in ("rotation", "translation"):
                continue

            bone = node_to_bone.get(int(node_idx))
            if not bone or bone not in pose:
                continue

            sampler_idx = ch.get("sampler")
            if sampler_idx is None or sampler_idx < 0 or sampler_idx >= len(samplers):
                continue
            sampler = samplers[sampler_idx] or {}
            out_acc = sampler.get("output")
            if out_acc is None:
                continue

            out_vals = get_accessor_floats(gltf, bin_chunk, int(out_acc))
            if not out_vals:
                continue

            if path == "rotation":
                v = out_vals[0]
                if isinstance(v, list) and len(v) == 4:
                    pose[bone]["rotation"] = [float(x) for x in v]
            elif path == "translation":
                v = out_vals[0]
                if isinstance(v, list) and len(v) == 3:
                    pose[bone]["position"] = [float(x) for x in v]

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
            "lefts": {
                "leftThumb": 0, "leftIndexFinger": 0, "leftMiddleFinger": 0,
                "leftRingFinger": 0, "leftLittleFinger": 0
            },
            "rights": {
                "rightThumb": 0, "rightIndexFinger": 0, "rightMiddleFinger": 0,
                "rightRingFinger": 0, "rightLittleFinger": 0
            },
            "fingerSettings": {
                "leftThumb": {"x": 0, "y": 0, "z": 0},
                "leftIndex": {"x": 0, "y": 0, "z": 0},
                "leftMiddle": {"x": 0, "y": 0, "z": 0},
                "leftRing": {"x": 0, "y": 0, "z": 0},
                "leftLittle": {"x": 0, "y": 0, "z": 0},
                "rightThumb": {"x": 0, "y": 0, "z": 0},
                "rightIndex": {"x": 0, "y": 0, "z": 0},
                "rightMiddle": {"x": 0, "y": 0, "z": 0},
                "rightRing": {"x": 0, "y": 0, "z": 0},
                "rightLittle": {"x": 0, "y": 0, "z": 0},
            },
            "fingerJointSettings": {
                "leftThumb": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
                "leftIndex": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "intermediate": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
                "leftMiddle": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "intermediate": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
                "leftRing": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "intermediate": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
                "leftLittle": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "intermediate": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
                "rightThumb": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
                "rightIndex": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "intermediate": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
                "rightMiddle": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "intermediate": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
                "rightRing": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "intermediate": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
                "rightLittle": {
                    "proximal": {"x": 0, "y": 0, "z": 0},
                    "intermediate": {"x": 0, "y": 0, "z": 0},
                    "distal": {"x": 0, "y": 0, "z": 0}
                },
            },
            "gages": {"pitch": 0, "yaw": 0}
        },
        "gages": {"pitch": 0, "yaw": 0}
    }

    return result


# -----------------------------------------------------------------------------
# Folder conversion
# -----------------------------------------------------------------------------

SUPPORTED_EXTS = {".vrma", ".glb"}  # falls du nur vrma willst: {".vrma"}


def convert_folder(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist or is not a folder: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0

    for file_path in input_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        rel = file_path.relative_to(input_dir)
        out_file = (output_dir / rel).with_suffix(".json")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            pose_json = extract_pose_json(str(file_path))
            out_file.write_text(json.dumps(pose_json, indent=2), encoding="utf-8")
            ok += 1
            print(f"[OK]   {rel} -> {out_file.relative_to(output_dir)}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {rel}  ({type(e).__name__}: {e})")

    return ok, fail


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Beispiel-Pfade anpassen:
    input_folder = Path(r"C:\Users\Deeliar\Downloads\animations")
    output_folder = Path(r"C:\Users\Deeliar\Downloads\animations_output")

    ok, fail = convert_folder(input_folder, output_folder)
    print(f"\nDone. Converted: {ok}, Failed: {fail}")
