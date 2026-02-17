"""
Auto-Rig: Adds a humanoid skeleton to a static GLB model using Blender.
============================================================================
Usage (PowerShell):
  & "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" -b --factory-startup -P auto_rig.py -- input.glb [output.glb]

If output is omitted, overwrites input with "_rigged" suffix.

What it does:
  1. Imports the static GLB
  2. Analyzes the mesh to find body proportions
  3. Creates a humanoid armature with standard bone names
  4. Adds facial bones (jaw, eyelids, lips) for lipsync
  5. Parents mesh to armature with Automatic Weights
  6. Exports as new GLB

Bone naming convention matches web_avatar's buildSmartBoneMap():
  JNT_C_Head, JNT_L_UpperArm, JNT_R_Forearm, etc.
"""

import os
import sys
import math
import traceback

import bpy
import bmesh
from mathutils import Vector, Matrix


# ===================== CONFIG =====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Parse args after '--' separator."""
    argv = sys.argv
    if "--" not in argv:
        # Default: look for any .glb in models/
        models_dir = os.path.join(SCRIPT_DIR, "models")
        glbs = [f for f in os.listdir(models_dir) if f.lower().endswith(".glb")] if os.path.isdir(models_dir) else []
        if not glbs:
            print("[ERROR] No GLB file specified and none found in models/")
            sys.exit(1)
        inp = os.path.join(models_dir, glbs[0])
        base, ext = os.path.splitext(inp)
        out = base + "_rigged" + ext
        return inp, out

    args = argv[argv.index("--") + 1:]
    if len(args) < 1:
        print("Usage: blender -b -P auto_rig.py -- input.glb [output.glb]")
        sys.exit(1)

    inp = os.path.abspath(args[0])
    if len(args) >= 2:
        out = os.path.abspath(args[1])
    else:
        base, ext = os.path.splitext(inp)
        out = base + "_rigged" + ext
    return inp, out


# ===================== SCENE MANAGEMENT =====================

def reset_scene():
    """Clear everything."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=True)
    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                       bpy.data.images, bpy.data.armatures, bpy.data.cameras,
                       bpy.data.lights, bpy.data.actions, bpy.data.node_groups]:
        for block in list(block_type):
            block_type.remove(block)
    for col in list(bpy.context.scene.collection.children):
        bpy.context.scene.collection.children.unlink(col)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def import_glb(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    result = bpy.ops.import_scene.gltf(filepath=path)
    if result != {"FINISHED"}:
        raise RuntimeError(f"GLB import failed: {result}")
    print(f"[OK] Imported: {path}")


def get_all_meshes():
    """Get all mesh objects in the scene."""
    return [o for o in bpy.data.objects if o.type == "MESH"]


# ===================== MESH ANALYSIS =====================

def analyze_mesh_bounds(meshes):
    """
    Analyze all meshes to find the bounding box and approximate body landmarks.
    Returns a dict with key points for skeleton placement.
    """
    # Combine all mesh bounds
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

    all_verts = []
    for obj in meshes:
        mesh = obj.data
        mat = obj.matrix_world
        for v in mesh.vertices:
            co = mat @ v.co
            all_verts.append(co)
            min_co.x = min(min_co.x, co.x)
            min_co.y = min(min_co.y, co.y)
            min_co.z = min(min_co.z, co.z)
            max_co.x = max(max_co.x, co.x)
            max_co.y = max(max_co.y, co.y)
            max_co.z = max(max_co.z, co.z)

    height = max_co.z - min_co.z
    width = max_co.x - min_co.x
    depth = max_co.y - min_co.y
    center_x = (min_co.x + max_co.x) / 2
    center_y = (min_co.y + max_co.y) / 2
    bottom_z = min_co.z
    top_z = max_co.z

    print(f"[Mesh] Height: {height:.3f}, Width: {width:.3f}, Depth: {depth:.3f}")
    print(f"[Mesh] Bottom: {bottom_z:.3f}, Top: {top_z:.3f}")

    # Estimate body proportions (humanoid ratios)
    # Standard human proportions based on total height:
    feet_z = bottom_z
    ankle_z = feet_z + height * 0.05
    knee_z = feet_z + height * 0.28
    hip_z = feet_z + height * 0.50
    spine_z = feet_z + height * 0.56
    chest_z = feet_z + height * 0.65
    upper_chest_z = feet_z + height * 0.72
    neck_z = feet_z + height * 0.80
    head_z = feet_z + height * 0.85
    head_top_z = top_z

    # Shoulder width estimate (slightly less than mesh width)
    shoulder_w = width * 0.38
    hip_w = width * 0.15
    arm_len = height * 0.17  # upper arm length
    forearm_len = height * 0.14
    hand_len = height * 0.06

    # Front/back (Y) — use center
    body_y = center_y

    return {
        'height': height,
        'width': width,
        'depth': depth,
        'center_x': center_x,
        'center_y': body_y,
        'bottom_z': bottom_z,
        'top_z': top_z,
        # Body landmarks
        'feet_z': feet_z,
        'ankle_z': ankle_z,
        'knee_z': knee_z,
        'hip_z': hip_z,
        'spine_z': spine_z,
        'chest_z': chest_z,
        'upper_chest_z': upper_chest_z,
        'neck_z': neck_z,
        'head_z': head_z,
        'head_top_z': head_top_z,
        'shoulder_w': shoulder_w,
        'hip_w': hip_w,
        'arm_len': arm_len,
        'forearm_len': forearm_len,
        'hand_len': hand_len,
    }


# ===================== ARMATURE CREATION =====================

def create_humanoid_armature(landmarks):
    """
    Create a humanoid armature based on mesh landmarks.
    Uses naming convention compatible with web_avatar's buildSmartBoneMap().
    """
    cx = landmarks['center_x']
    cy = landmarks['center_y']
    sw = landmarks['shoulder_w']
    hw = landmarks['hip_w']
    al = landmarks['arm_len']
    fl = landmarks['forearm_len']
    hl = landmarks['hand_len']

    # Create armature
    arm_data = bpy.data.armatures.new("Armature")
    arm_obj = bpy.data.objects.new("Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')

    bones = {}

    def add_bone(name, head, tail, parent_name=None, connect=False):
        b = arm_data.edit_bones.new(name)
        b.head = Vector(head)
        b.tail = Vector(tail)
        if parent_name and parent_name in bones:
            b.parent = bones[parent_name]
            b.use_connect = connect
        bones[name] = b
        return b

    # ===== CORE BODY =====
    hip_z = landmarks['hip_z']
    spine_z = landmarks['spine_z']
    chest_z = landmarks['chest_z']
    uchest_z = landmarks['upper_chest_z']
    neck_z = landmarks['neck_z']
    head_z = landmarks['head_z']
    head_top_z = landmarks['head_top_z']

    # Root / COG
    add_bone("JNT_C_Cog", (cx, cy, hip_z), (cx, cy, hip_z + 0.05))

    # Pelvis
    add_bone("JNT_C_Pelvis", (cx, cy, hip_z), (cx, cy, hip_z - 0.03), "JNT_C_Cog")

    # Spine chain
    add_bone("JNT_C_Spine_01", (cx, cy, spine_z), (cx, cy, chest_z), "JNT_C_Cog")
    add_bone("JNT_C_Spine_02", (cx, cy, chest_z), (cx, cy, uchest_z), "JNT_C_Spine_01", True)
    add_bone("JNT_C_Spine_03", (cx, cy, uchest_z), (cx, cy, neck_z), "JNT_C_Spine_02", True)

    # Neck + Head
    add_bone("JNT_C_Neck", (cx, cy, neck_z), (cx, cy, head_z), "JNT_C_Spine_03", True)
    add_bone("JNT_C_Head", (cx, cy, head_z), (cx, cy, head_top_z), "JNT_C_Neck", True)

    # ===== LEGS =====
    ankle_z = landmarks['ankle_z']
    knee_z = landmarks['knee_z']
    feet_z = landmarks['feet_z']

    for side, sign in [("L", 1), ("R", -1)]:
        hx = cx + sign * hw

        # Upper leg
        add_bone(f"JNT_{side}_UpperLeg", (hx, cy, hip_z), (hx, cy, knee_z), "JNT_C_Cog")
        # Lower leg
        add_bone(f"JNT_{side}_Calf", (hx, cy, knee_z), (hx, cy, ankle_z), f"JNT_{side}_UpperLeg", True)
        # Foot
        foot_front_y = cy + landmarks['depth'] * 0.15
        add_bone(f"JNT_{side}_Foot", (hx, cy, ankle_z), (hx, foot_front_y, feet_z), f"JNT_{side}_Calf", True)
        # Toe
        add_bone(f"JNT_{side}_Toe", (hx, foot_front_y, feet_z), (hx, foot_front_y + 0.04, feet_z), f"JNT_{side}_Foot", True)

    # ===== ARMS =====
    for side, sign in [("L", 1), ("R", -1)]:
        sx = cx + sign * sw * 0.6  # clavicle start
        shoulder_x = cx + sign * sw  # shoulder joint

        # Clavicle
        add_bone(f"JNT_{side}_Clavicle", (sx, cy, uchest_z), (shoulder_x, cy, uchest_z), "JNT_C_Spine_03")

        # Upper arm (T-pose: extends horizontally)
        ua_end_x = shoulder_x + sign * al
        add_bone(f"JNT_{side}_UpperArm", (shoulder_x, cy, uchest_z), (ua_end_x, cy, uchest_z), f"JNT_{side}_Clavicle", True)

        # Forearm
        fa_end_x = ua_end_x + sign * fl
        add_bone(f"JNT_{side}_Forearm", (ua_end_x, cy, uchest_z), (fa_end_x, cy, uchest_z), f"JNT_{side}_UpperArm", True)

        # Hand
        h_end_x = fa_end_x + sign * hl
        add_bone(f"JNT_{side}_Hand", (fa_end_x, cy, uchest_z), (h_end_x, cy, uchest_z), f"JNT_{side}_Forearm", True)

        # Fingers (simplified: 3 bones each for Index, Middle, Ring, Pinky, Thumb)
        finger_len = hl * 0.3
        finger_seg = finger_len / 3
        finger_spread = hl * 0.15

        finger_names = ["Thumb", "Index", "Middle", "Pointer", "Pinky"]
        finger_offsets_y = [-0.6, -0.3, 0.0, 0.3, 0.55]  # spread in Y

        for fi, (fname, fy_off) in enumerate(zip(finger_names, finger_offsets_y)):
            fx = fa_end_x + sign * hl  # start at hand end
            fz = uchest_z
            fy = cy + fy_off * finger_spread

            if fname == "Thumb":
                # Thumb angles differently
                for seg in range(1, 4):
                    seg_start_x = fx + sign * finger_seg * (seg - 1) * 0.7
                    seg_end_x = fx + sign * finger_seg * seg * 0.7
                    parent = f"JNT_{side}_Hand" if seg == 1 else f"JNT_{side}_{fname}_{seg-1}"
                    add_bone(f"JNT_{side}_{fname}_{seg}",
                             (seg_start_x, fy - sign * 0.01, fz - 0.005 * seg),
                             (seg_end_x, fy - sign * 0.02, fz - 0.005 * (seg + 1)),
                             parent, seg > 1)
            else:
                for seg in range(1, 4):
                    seg_start_x = fx + sign * finger_seg * (seg - 1)
                    seg_end_x = fx + sign * finger_seg * seg
                    parent = f"JNT_{side}_Hand" if seg == 1 else f"JNT_{side}_{fname}_{seg-1}"
                    add_bone(f"JNT_{side}_{fname}_{seg}",
                             (seg_start_x, fy, fz),
                             (seg_end_x, fy, fz),
                             parent, seg > 1)

    # ===== EYES =====
    eye_z = head_z + (head_top_z - head_z) * 0.45
    eye_y = cy + landmarks['depth'] * 0.2
    eye_sep = landmarks['width'] * 0.07

    add_bone("JNT_L_Eye", (cx + eye_sep, eye_y, eye_z), (cx + eye_sep, eye_y + 0.015, eye_z), "JNT_C_Head")
    add_bone("JNT_R_Eye", (cx - eye_sep, eye_y, eye_z), (cx - eye_sep, eye_y + 0.015, eye_z), "JNT_C_Head")

    # ===== FACIAL BONES (for lipsync & expressions) =====
    mouth_z = head_z + (head_top_z - head_z) * 0.15
    mouth_y = cy + landmarks['depth'] * 0.22
    jaw_z = head_z + (head_top_z - head_z) * 0.05

    # Jaw
    add_bone("JNT_C_Jaw", (cx, mouth_y - 0.01, jaw_z + 0.015), (cx, mouth_y, jaw_z), "JNT_C_Head")

    # Eyelids
    for side, sign in [("L", 1), ("R", -1)]:
        ex = cx + sign * eye_sep
        add_bone(f"JNT_{side}_eyelid_upper", (ex, eye_y, eye_z + 0.005), (ex, eye_y + 0.008, eye_z + 0.005), "JNT_C_Head")
        add_bone(f"JNT_{side}_eyelid_lower", (ex, eye_y, eye_z - 0.005), (ex, eye_y + 0.008, eye_z - 0.005), "JNT_C_Head")

    # Eyebrows
    for side, sign in [("L", 1), ("R", -1)]:
        brow_z = eye_z + (head_top_z - head_z) * 0.12
        brow_x_inner = cx + sign * eye_sep * 0.4
        brow_x_mid = cx + sign * eye_sep
        brow_x_outer = cx + sign * eye_sep * 1.5
        add_bone(f"JNT_{side}_eyebrow_inner", (brow_x_inner, eye_y, brow_z), (brow_x_inner, eye_y + 0.008, brow_z), "JNT_C_Head")
        add_bone(f"JNT_{side}_eyebrow_mid", (brow_x_mid, eye_y, brow_z), (brow_x_mid, eye_y + 0.008, brow_z), "JNT_C_Head")
        add_bone(f"JNT_{side}_eyebrow_outer", (brow_x_outer, eye_y, brow_z), (brow_x_outer, eye_y + 0.008, brow_z), "JNT_C_Head")

    # Lips
    lip_spread = landmarks['width'] * 0.05
    add_bone("JNT_lipsUpper", (cx, mouth_y, mouth_z + 0.003), (cx, mouth_y + 0.006, mouth_z + 0.003), "JNT_C_Head")
    add_bone("JNT_lips_lower", (cx, mouth_y, mouth_z - 0.003), (cx, mouth_y + 0.006, mouth_z - 0.003), "JNT_C_Jaw")

    for side, sign in [("L", 1), ("R", -1)]:
        lx = cx + sign * lip_spread
        add_bone(f"JNT_lips_{side}_corner", (lx, mouth_y, mouth_z), (lx, mouth_y + 0.005, mouth_z), "JNT_C_Head")
        add_bone(f"JNT_lips_{side}_upInner", (cx + sign * lip_spread * 0.4, mouth_y, mouth_z + 0.002),
                 (cx + sign * lip_spread * 0.4, mouth_y + 0.005, mouth_z + 0.002), "JNT_C_Head")
        add_bone(f"JNT_lips_{side}_lowInner", (cx + sign * lip_spread * 0.4, mouth_y, mouth_z - 0.002),
                 (cx + sign * lip_spread * 0.4, mouth_y + 0.005, mouth_z - 0.002), "JNT_C_Jaw")
        add_bone(f"JNT_lips_{side}_upOuter", (lx, mouth_y, mouth_z + 0.002),
                 (lx, mouth_y + 0.005, mouth_z + 0.002), "JNT_C_Head")
        add_bone(f"JNT_lips_{side}_lowOuter", (lx, mouth_y, mouth_z - 0.002),
                 (lx, mouth_y + 0.005, mouth_z - 0.002), "JNT_C_Jaw")

    # Tongue
    tongue_y = mouth_y - 0.005
    add_bone("JNT_Tongue_01", (cx, tongue_y, mouth_z - 0.002), (cx, tongue_y + 0.01, mouth_z - 0.002), "JNT_C_Jaw")
    add_bone("JNT_Tongue_02", (cx, tongue_y + 0.01, mouth_z - 0.002), (cx, tongue_y + 0.02, mouth_z - 0.002), "JNT_Tongue_01", True)

    bpy.ops.object.mode_set(mode='OBJECT')

    total = len(arm_data.bones)
    print(f"[OK] Armature created with {total} bones")

    return arm_obj


# ===================== MESH-TO-ARMATURE BINDING =====================

def join_meshes(meshes):
    """Join all mesh objects into one (needed for automatic weights)."""
    if len(meshes) <= 1:
        return meshes[0] if meshes else None

    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def clean_mesh_for_weights(mesh_obj):
    """Clean mesh to avoid automatic weights failures."""
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Remove doubles
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    # Recalculate normals
    bpy.ops.mesh.normals_make_consistent(inside=False)
    # Delete loose verts
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[OK] Mesh cleaned: {len(mesh_obj.data.vertices)} verts")


def parent_with_automatic_weights(armature, mesh_obj):
    """
    Parent mesh to armature with Automatic Weights (heat map).
    Falls back to Bone Envelope if automatic weights fail.
    """
    # Clear any existing parent
    mesh_obj.parent = None
    mesh_obj.matrix_parent_inverse = Matrix.Identity(4)

    # Remove existing armature modifiers
    for mod in list(mesh_obj.modifiers):
        if mod.type == 'ARMATURE':
            mesh_obj.modifiers.remove(mod)

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Try automatic weights first
    try:
        result = bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        if result == {"FINISHED"}:
            print("[OK] Automatic Weights applied successfully!")
            _verify_binding(armature, mesh_obj)
            return True
    except Exception as e:
        print(f"[WARN] Automatic Weights failed: {e}")

    # Fallback: Bone Envelope
    try:
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        result = bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
        if result == {"FINISHED"}:
            print("[OK] Bone Envelope weights applied (fallback)")
            _verify_binding(armature, mesh_obj)
            return True
    except Exception as e:
        print(f"[WARN] Bone Envelope also failed: {e}")

    # Last resort: just parent with empty groups
    try:
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.parent_set(type='ARMATURE_NAME')
        print("[OK] Parented with empty vertex groups (manual weight painting needed)")
        _verify_binding(armature, mesh_obj)
        return True
    except Exception as e:
        print(f"[ERROR] All parenting methods failed: {e}")
        return False


def _verify_binding(armature, mesh_obj):
    """Verify and fix the mesh-armature binding."""
    # Ensure parent is set
    if mesh_obj.parent != armature:
        mesh_obj.parent = armature
        mesh_obj.parent_type = 'OBJECT'
        print("[FIX] Parent relationship set manually")

    # Ensure armature modifier exists
    has_arm_mod = False
    for mod in mesh_obj.modifiers:
        if mod.type == 'ARMATURE':
            if mod.object != armature:
                mod.object = armature
                print("[FIX] Armature modifier target corrected")
            has_arm_mod = True
            break

    if not has_arm_mod:
        mod = mesh_obj.modifiers.new("Armature", 'ARMATURE')
        mod.object = armature
        print("[FIX] Armature modifier added manually")

    # Count vertex groups
    vg_count = len(mesh_obj.vertex_groups)

    # Quick check: do any vertices actually have weights?
    has_weights = False
    for v in mesh_obj.data.vertices[:200]:  # check first 200
        if len(v.groups) > 0:
            has_weights = True
            break

    print(f"[VERIFY] {vg_count} vertex groups, has_weights={has_weights}")

    if not has_weights and vg_count > 0:
        print("[WARN] Automatic weights failed — using distance-based assignment")
        assign_weights_by_distance(armature, mesh_obj)


def assign_weights_by_distance(armature, mesh_obj):
    """
    Assign vertex weights based on distance to nearest bone.
    Uses smooth falloff so vertices blend between nearby bones.
    """
    import numpy as np

    # Get bone positions in world space
    bone_data = []  # [(bone_name, head_world, tail_world)]
    arm_mat = armature.matrix_world

    for bone in armature.data.bones:
        head_w = arm_mat @ bone.head_local
        tail_w = arm_mat @ bone.tail_local
        bone_data.append((bone.name, head_w, tail_w))

    mesh_mat = mesh_obj.matrix_world
    verts = mesh_obj.data.vertices

    # Clear existing groups
    mesh_obj.vertex_groups.clear()

    # Create vertex groups for each bone
    vg_map = {}
    for bname, _, _ in bone_data:
        vg = mesh_obj.vertex_groups.new(name=bname)
        vg_map[bname] = vg

    # For each vertex, find closest bone and assign weight
    NUM_BONES_INFLUENCE = 4  # max bones per vertex
    total_assigned = 0

    for v in verts:
        v_world = mesh_mat @ v.co
        distances = []

        for bname, head, tail in bone_data:
            # Distance from vertex to bone segment (head-tail)
            d = point_to_segment_dist(v_world, head, tail)
            distances.append((d, bname))

        # Sort by distance, take closest N
        distances.sort(key=lambda x: x[0])
        closest = distances[:NUM_BONES_INFLUENCE]

        # Skip if too far from all bones
        if closest[0][0] > 2.0:
            # Assign to nearest bone anyway
            vg_map[closest[0][1]].add([v.index], 1.0, 'REPLACE')
            total_assigned += 1
            continue

        # Compute smooth weights using inverse distance
        min_dist = max(closest[0][0], 0.0001)
        weights = []
        for d, bname in closest:
            # Smooth falloff: closer = higher weight
            w = 1.0 / max(d, 0.0001)
            weights.append((bname, w))

        # Normalize
        total_w = sum(w for _, w in weights)
        if total_w > 0:
            for bname, w in weights:
                normalized = w / total_w
                if normalized > 0.01:  # skip tiny weights
                    vg_map[bname].add([v.index], normalized, 'REPLACE')

        total_assigned += 1

    print(f"[OK] Distance-based weights assigned to {total_assigned} vertices")


def point_to_segment_dist(point, seg_a, seg_b):
    """Distance from a point to a line segment."""
    ab = seg_b - seg_a
    ap = point - seg_a
    ab_len_sq = ab.dot(ab)

    if ab_len_sq < 0.00001:
        return (point - seg_a).length

    t = max(0, min(1, ap.dot(ab) / ab_len_sq))
    closest = seg_a + ab * t
    return (point - closest).length


# ===================== EXPORT =====================

def export_glb(output_path):
    """Export scene as GLB with proper skin data."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Select all for export
    bpy.ops.object.select_all(action='SELECT')

    # Debug: print scene structure before export
    print("[Export] Scene objects:")
    for obj in bpy.data.objects:
        parent_name = obj.parent.name if obj.parent else "None"
        mods = [m.type for m in obj.modifiers] if hasattr(obj, 'modifiers') else []
        print(f"  {obj.name} ({obj.type}) parent={parent_name} mods={mods}")

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=False,
        export_apply=False,
        export_animations=False,
        export_skins=True,
        export_morph=True,
        export_lights=False,
        export_cameras=False,
        export_yup=True,
    )
    print(f"[OK] Exported: {output_path}")


# ===================== MAIN =====================

def main():
    input_path, output_path = parse_args()
    print(f"\n{'='*60}")
    print(f"  Auto-Rig: Adding skeleton to static GLB")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # 1. Reset & Import
    reset_scene()
    import_glb(input_path)

    # 2. Get meshes
    meshes = get_all_meshes()
    if not meshes:
        print("[ERROR] No meshes found in GLB!")
        sys.exit(1)
    print(f"[OK] Found {len(meshes)} mesh objects")

    # 3. Check if already rigged
    existing_arm = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if existing_arm:
        bone_count = sum(len(a.data.bones) for a in existing_arm)
        if bone_count > 5:
            print(f"[INFO] Model already has {bone_count} bones — skipping auto-rig")
            print(f"[INFO] Exporting as-is to: {output_path}")
            export_glb(output_path)
            return

    # 4. CRITICAL: Preserve parent transforms BEFORE removing empties!
    #    The GLB importer creates parent empties with Y-up → Z-up rotation.
    #    We must bake that rotation into mesh vertices.
    bpy.context.view_layer.update()  # Ensure transforms are current

    # First: unparent meshes while preserving their world transform
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH' and obj.parent:
            # Save the full world matrix (includes parent chain rotation)
            world_mat = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world_mat
            print(f"  [Unparent] {obj.name}: world matrix preserved")

    bpy.context.view_layer.update()

    # Now apply transforms to bake world matrix into vertex data
    meshes = get_all_meshes()
    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
        bpy.context.view_layer.objects.active = m
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print("[OK] Transforms applied (parent rotations baked in)")

    # Now safe to remove empties and non-mesh objects
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.type not in ('MESH', 'ARMATURE'):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    if removed:
        print(f"[OK] Removed {removed} non-mesh objects")

    # Re-get meshes
    meshes = get_all_meshes()

    # 5. Analyze mesh — check which axis is actually "up"
    landmarks_raw = analyze_mesh_bounds(meshes)
    raw_height_z = landmarks_raw['height']  # Z range
    raw_width_x = landmarks_raw['width']    # X range
    raw_depth_y = landmarks_raw['depth']    # Y range

    print(f"[Axes] X(width)={raw_width_x:.1f}, Y(depth)={raw_depth_y:.1f}, Z(height)={raw_height_z:.1f}")

    # Detect if model needs rotation (Y might be up instead of Z)
    # In Blender, Z should be up. If Y is significantly taller, rotate.
    if raw_depth_y > raw_height_z * 1.5:
        print(f"[FIX] Model appears Y-up (Y={raw_depth_y:.1f} > Z={raw_height_z:.1f})")
        print(f"      Rotating -90° around X to make Z-up")
        bpy.ops.object.select_all(action='DESELECT')
        for m in meshes:
            m.select_set(True)
            bpy.context.view_layer.objects.active = m
        bpy.ops.transform.rotate(value=-math.pi/2, orient_axis='X')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # Re-analyze
        meshes = get_all_meshes()
        landmarks_raw = analyze_mesh_bounds(meshes)
        raw_height_z = landmarks_raw['height']
        print(f"[OK] After rotation: Z(height)={raw_height_z:.1f}")

    raw_height = raw_height_z

    # 6. Scale model to ~1.6m if needed
    TARGET_HEIGHT = 1.6
    if raw_height > 10 or raw_height < 0.1:
        scale_factor = TARGET_HEIGHT / raw_height
        print(f"[INFO] Model height {raw_height:.1f} -> scaling by {scale_factor:.6f} to {TARGET_HEIGHT}m")

        bpy.ops.object.select_all(action='DESELECT')
        for m in meshes:
            m.select_set(True)
            bpy.context.view_layer.objects.active = m

        # Scale from bottom center
        cx = landmarks_raw['center_x']
        cy = landmarks_raw['center_y']
        bz = landmarks_raw['bottom_z']

        # Move to origin, scale, move back (then apply)
        for m in meshes:
            m.location = (0, 0, 0)
            m.scale = (scale_factor, scale_factor, scale_factor)

        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Center at origin - feet on Z=0
        meshes = get_all_meshes()
        bounds2 = analyze_mesh_bounds(meshes)
        offset_x = -bounds2['center_x']
        offset_y = -bounds2['center_y']
        offset_z = -bounds2['bottom_z']

        for m in meshes:
            m.location = (offset_x, offset_y, offset_z)
        bpy.ops.object.select_all(action='DESELECT')
        for m in meshes:
            m.select_set(True)
            bpy.context.view_layer.objects.active = m
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        print(f"[OK] Model scaled and centered at origin, feet on Z=0")

    # 8. Re-analyze after scaling
    meshes = get_all_meshes()
    landmarks = analyze_mesh_bounds(meshes)
    print(f"[OK] Body analysis complete — height={landmarks['height']:.3f}")

    # 9. Join meshes
    mesh_obj = join_meshes(meshes)
    mesh_obj.name = "Body"  # Clean name for exporter
    mesh_obj.data.name = "Body"  # Also rename the mesh DATA block
    print(f"[OK] Meshes joined: {mesh_obj.name}, data: {mesh_obj.data.name}")

    # 10. Clean mesh
    clean_mesh_for_weights(mesh_obj)

    # 11. Create armature
    armature = create_humanoid_armature(landmarks)

    # 12. Apply transforms on mesh before binding
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # 13. Parent with automatic weights
    success = parent_with_automatic_weights(armature, mesh_obj)
    if not success:
        print("[WARN] Automatic weights failed entirely, using distance-based fallback")
        # Set parent manually
        mesh_obj.parent = armature
        mesh_obj.parent_type = 'OBJECT'
        mod = mesh_obj.modifiers.new("Armature", 'ARMATURE')
        mod.object = armature

    # 14. Verify binding and fix with distance weights if needed
    _verify_binding(armature, mesh_obj)

    # 15. Final state check
    has_any_weights = False
    for v in mesh_obj.data.vertices[:100]:
        if len(v.groups) > 0:
            has_any_weights = True
            break
    print(f"\n[Pre-Export] Scene state:")
    print(f"  Mesh: {mesh_obj.name} (data: {mesh_obj.data.name})")
    print(f"  Parent: {mesh_obj.parent.name if mesh_obj.parent else 'NONE'}")
    print(f"  Modifiers: {[(m.name, m.type, m.object.name if m.object else 'None') for m in mesh_obj.modifiers]}")
    print(f"  Vertex groups: {len(mesh_obj.vertex_groups)}")
    print(f"  Has vertex weights: {has_any_weights}")
    print(f"  Total vertices: {len(mesh_obj.data.vertices)}")
    print(f"  Armature bones: {len(armature.data.bones)}")

    if not has_any_weights:
        print("[CRITICAL] Still no weights after fallback! Force-assigning all vertices to root bone.")
        root_name = "JNT_C_Cog"
        if root_name not in [vg.name for vg in mesh_obj.vertex_groups]:
            mesh_obj.vertex_groups.new(name=root_name)
        vg = mesh_obj.vertex_groups[root_name]
        all_verts = [v.index for v in mesh_obj.data.vertices]
        vg.add(all_verts, 1.0, 'REPLACE')
        print(f"  Assigned {len(all_verts)} vertices to {root_name}")

    # 16. Export
    export_glb(output_path)

    # 17. Verify output has bones
    import struct, json as json_mod
    with open(output_path, 'rb') as f:
        f.read(4)  # magic
        f.read(4)  # version
        f.read(4)  # total length
        chunk_len = struct.unpack('<I', f.read(4))[0]
        f.read(4)  # chunk type
        gltf_json = json_mod.loads(f.read(chunk_len).decode('utf-8'))

    skin_count = len(gltf_json.get('skins', []))
    joint_count = sum(len(s.get('joints', [])) for s in gltf_json.get('skins', []))
    print(f"[VERIFY] Output has {skin_count} skins, {joint_count} joints/bones")

    if joint_count == 0:
        print("[ERROR] Export has NO bones! Something went wrong with skinning.")
    else:
        print(f"\n{'='*60}")
        print(f"  DONE! Rigged model saved to:")
        print(f"  {output_path}")
        print(f"  Bones: {joint_count}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
