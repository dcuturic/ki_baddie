"""
Blend → VRM Converter Script (läuft INNERHALB von Blender headless)
====================================================================
Wird von app.py via subprocess aufgerufen:
  blender --background --factory-startup --python converter.py -- input.blend output.vrm job_id

Features:
  - VRM Addon auto-install & force-register
  - Humanoid Bone-Mapping (automatisch + bekannte Namens-Patterns)
  - Shape Keys → VRM Expressions (Gesichtsemotionen)
  - Model normalisieren (zentrieren, skalieren)
  - VRM Meta-Daten setzen
"""

import os
import sys
import json
import importlib
import traceback

import bpy


# ======================= CONTEXT HELPERS =======================

def _ensure_object_mode():
    """Sicherstellen dass wir im OBJECT-Modus sind (headless-safe)."""
    bpy.context.view_layer.update()
    try:
        if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass


def _select_all(select=True):
    """Alle Objekte (de)selektieren ohne Operator (headless-safe)."""
    for obj in bpy.context.view_layer.objects:
        try:
            obj.select_set(select)
        except Exception:
            pass


def _select_only(obj):
    """Nur ein Objekt selektieren, alle anderen deselektieren."""
    _select_all(False)
    try:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    except Exception:
        pass


def _get_context_override():
    """Context-Override fuer Operator-Aufrufe im Headless-Modus erstellen."""
    ctx = {}
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        ctx = {
                            'window': window,
                            'screen': window.screen,
                            'area': area,
                            'region': region,
                        }
                        return ctx
    # Fallback: erstes Window/Screen
    if bpy.context.window_manager.windows:
        window = bpy.context.window_manager.windows[0]
        ctx = {'window': window, 'screen': window.screen}
        if window.screen.areas:
            ctx['area'] = window.screen.areas[0]
            for region in window.screen.areas[0].regions:
                if region.type == 'WINDOW':
                    ctx['region'] = region
                    break
    return ctx


def _apply_transforms(obj, location=False, rotation=True, scale=True):
    """Transforms anwenden ohne Operator (headless-safe)."""
    import mathutils
    if scale and obj.scale != mathutils.Vector((1, 1, 1)):
        if obj.type == 'MESH' and obj.data:
            sx, sy, sz = obj.scale
            for v in obj.data.vertices:
                v.co.x *= sx
                v.co.y *= sy
                v.co.z *= sz
            obj.scale = (1, 1, 1)
        elif obj.type == 'ARMATURE' and obj.data:
            # Fuer Armatures Scale auf Children propagieren
            pass

    if rotation:
        # Rotation in Mesh einbacken ist komplex, wir versuchen den Operator mit Override
        pass

    if location and obj.location != mathutils.Vector((0, 0, 0)):
        if obj.type == 'MESH' and obj.data:
            for v in obj.data.vertices:
                v.co += obj.location
            obj.location = (0, 0, 0)


# ======================= ARGS =======================

def parse_args():
    """Parse arguments after '--' separator."""
    argv = sys.argv
    if "--" not in argv:
        print("[FAIL] Keine Argumente übergeben. Usage: blender -b --factory-startup -P converter.py -- input.blend output.vrm [job_id]")
        sys.exit(1)
    args = argv[argv.index("--") + 1:]
    if len(args) < 2:
        print("[FAIL] Mindestens input und output Pfad nötig")
        sys.exit(1)
    options = {}
    if len(args) > 3:
        try:
            options = json.loads(args[3])
        except (json.JSONDecodeError, Exception) as e:
            print(f"[WARN] Options JSON ungueltig: {e}")
    return {
        "input": args[0],
        "output": args[1],
        "job_id": args[2] if len(args) > 2 else "unknown",
        "options": options,
    }


def progress(pct: int):
    print(f"[PROGRESS:{pct}]", flush=True)


def status(msg: str):
    print(f"[STATUS:{msg}]", flush=True)


# ======================= VRM ADDON =======================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VRM_ADDON_ZIP = os.path.join(SCRIPT_DIR, "vrm_addon.zip")


def _blender_addons_dir():
    p = bpy.utils.user_resource("SCRIPTS", path="addons")
    if p:
        os.makedirs(p, exist_ok=True)
    return p


def ensure_vrm_addon():
    """VRM Addon installieren & aktivieren."""
    import addon_utils

    # Check if already available
    for mod in addon_utils.modules():
        name = getattr(mod, "__name__", "")
        if "VRM_Addon_for_Blender" in name:
            enabled, _loaded = addon_utils.check(name)
            if not enabled:
                bpy.ops.preferences.addon_enable(module=name)
                bpy.ops.wm.save_userpref()
            print(f"[OK] VRM Add-on aktiv: {name}")
            return name

    # Install from zip
    if not os.path.exists(VRM_ADDON_ZIP):
        # Try parent directory (shared with glb_to_vrm)
        alt_zip = os.path.join(os.path.dirname(SCRIPT_DIR), "glb_to_vrm", "vrm_addon.zip")
        if os.path.exists(alt_zip):
            import shutil
            shutil.copy2(alt_zip, VRM_ADDON_ZIP)
            print(f"[INFO] vrm_addon.zip von glb_to_vrm kopiert")
        else:
            raise RuntimeError(
                f"vrm_addon.zip nicht gefunden!\n"
                f"Bitte VRM_Addon_for_Blender ZIP in {SCRIPT_DIR} ablegen.\n"
                f"Download: https://github.com/saturday06/VRM-Addon-for-Blender/releases"
            )

    print("[INFO] Installiere VRM Add-on aus ZIP...")
    bpy.ops.preferences.addon_install(filepath=VRM_ADDON_ZIP)

    addons_dir = _blender_addons_dir()
    if not addons_dir:
        raise RuntimeError("Blender Addons-Verzeichnis nicht gefunden")

    # Find installed folder
    candidates = [e for e in os.listdir(addons_dir)
                  if os.path.isdir(os.path.join(addons_dir, e)) and "VRM_Addon_for_Blender" in e]
    if not candidates:
        raise RuntimeError("VRM Add-on Ordner nach Installation nicht gefunden")

    candidates.sort(key=len, reverse=True)
    folder = candidates[0]

    # Fix hyphens in folder name
    fixed = folder.replace("-", "_")
    if fixed != folder:
        src = os.path.join(addons_dir, folder)
        dst = os.path.join(addons_dir, fixed)
        if not os.path.exists(dst):
            os.rename(src, dst)
        folder = fixed

    bpy.ops.preferences.addon_enable(module=folder)
    bpy.ops.wm.save_userpref()
    print(f"[OK] VRM Add-on installiert & aktiviert: {folder}")
    return folder


def force_register_vrm(module_name: str):
    """Force re-register um sicherzustellen dass alle Operatoren verfügbar sind."""
    addons_dir = _blender_addons_dir()
    if not addons_dir:
        return

    parent = os.path.dirname(addons_dir)
    if parent not in sys.path:
        sys.path.append(parent)

    try:
        mod = importlib.import_module(module_name)
        mod = importlib.reload(mod)

        if hasattr(mod, "unregister"):
            try:
                mod.unregister()
            except Exception:
                pass

        if hasattr(mod, "register"):
            try:
                mod.register()
                print(f"[OK] Force-Register erfolgreich: {module_name}")
            except Exception as e:
                if "translations" in str(e).lower():
                    try:
                        bpy.app.translations.unregister(module_name)
                    except Exception:
                        pass
                    mod.register()
                    print(f"[OK] Register nach Translation-Fix erfolgreich")
                else:
                    raise
    except Exception:
        print(f"[WARN] Force-Register fehlgeschlagen:")
        traceback.print_exc()


# ======================= BONE MAPPING =======================

def _camel_to_snake(name):
    """Convert camelCase to snake_case: leftUpperArm -> left_upper_arm."""
    import re
    return re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name).lower()


def _normalize_for_match(name):
    """Normalize bone/pattern name: strip suffixes/prefixes, unify separators to underscore."""
    n = name.lower().strip()
    # Strip common bone suffixes
    for suffix in ["_jnt", ".jnt", "_bone", ".bone", "_bn", ".bn", "_joint", ".joint",
                    "_def", ".def", "_sk", "_end", ".end", "_null", ".null"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
            break
    # Strip common prefixes
    for prefix in ["def-", "def_", "org-", "org_", "mch-", "mch_", "jnt_", "jnt-",
                    "bn_", "bone_", "bip_", "bip01_", "bip01 ",
                    "j_", "sk_", "b_", "mixamorig:", "mixamorig_",
                    "valvebip01_", "rig_"]:
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    # Unify ALL separators to underscore
    n = n.replace(".", "_").replace("-", "_").replace(" ", "_")
    while "__" in n:
        n = n.replace("__", "_")
    return n.strip("_")


def _compact(name):
    """Remove ALL separators for ultra-fuzzy comparison."""
    return name.replace("_", "").replace(".", "").replace("-", "").replace(" ", "")


# Standard VRM humanoid bone names → common Blender bone name patterns
# After normalization, dots become underscores, _JNT gets stripped, etc.
VRM_BONE_PATTERNS = {
    "hips": ["hips", "pelvis", "hip", "cog", "root", "center"],
    "spine": ["spine", "spine1", "spine_01", "spine_1", "spine.001", "spine.01"],
    "chest": ["chest", "spine2", "spine_02", "spine_2", "spine.002", "spine.02",
              "upper_body", "upperbody"],
    "upperChest": ["upper_chest", "spine3", "spine_03", "spine_3", "spine.003", "spine.03",
                   "upper_body2", "upper_body_2"],
    "neck": ["neck", "neck_01", "neck_1"],
    "head": ["head"],

    # --- Shoulders ---
    "leftShoulder": ["shoulder.l", "shoulder_l", "leftshoulder", "left_shoulder",
                     "l_shoulder", "clavicle.l", "clavicle_l", "l_clavicle"],
    "rightShoulder": ["shoulder.r", "shoulder_r", "rightshoulder", "right_shoulder",
                      "r_shoulder", "clavicle.r", "clavicle_r", "r_clavicle"],

    # --- Arms (IMPORTANT: more specific patterns first to avoid mismatches) ---
    "leftUpperArm": ["upper_arm.l", "upper_arm_l", "upperarm_l", "arm_upper_l",
                     "leftupperarm", "left_upper_arm", "l_upperarm", "l_upper_arm",
                     "arm.l", "arm_l", "l_arm"],
    "leftLowerArm": ["forearm.l", "forearm_l", "lower_arm.l", "lower_arm_l", "lowerarm_l",
                     "arm_lower_l", "leftlowerarm", "left_lower_arm", "l_forearm",
                     "l_lower_arm", "l_lowerarm", "elbow_l", "elbow.l"],
    "leftHand": ["hand.l", "hand_l", "lefthand", "left_hand", "l_hand",
                 "wrist.l", "wrist_l", "l_wrist"],

    "rightUpperArm": ["upper_arm.r", "upper_arm_r", "upperarm_r", "arm_upper_r",
                      "rightupperarm", "right_upper_arm", "r_upperarm", "r_upper_arm",
                      "arm.r", "arm_r", "r_arm"],
    "rightLowerArm": ["forearm.r", "forearm_r", "lower_arm.r", "lower_arm_r", "lowerarm_r",
                      "arm_lower_r", "rightlowerarm", "right_lower_arm", "r_forearm",
                      "r_lower_arm", "r_lowerarm", "elbow_r", "elbow.r"],
    "rightHand": ["hand.r", "hand_r", "righthand", "right_hand", "r_hand",
                  "wrist.r", "wrist_r", "r_wrist"],

    # --- Legs ---
    "leftUpperLeg": ["thigh.l", "thigh_l", "upper_leg.l", "upper_leg_l", "upperleg_l",
                     "leg_upper_l", "leftupperleg", "left_upper_leg", "l_upperleg",
                     "l_upper_leg", "l_thigh", "leg.l", "leg_l"],
    "leftLowerLeg": ["shin.l", "shin_l", "lower_leg.l", "lower_leg_l", "lowerleg_l",
                     "leg_lower_l", "leftlowerleg", "left_lower_leg", "l_shin",
                     "l_lower_leg", "l_lowerleg", "calf.l", "calf_l", "l_calf",
                     "knee.l", "knee_l"],
    "leftFoot": ["foot.l", "foot_l", "leftfoot", "left_foot", "l_foot",
                 "ankle.l", "ankle_l"],
    "leftToes": ["toe.l", "toe_l", "toes.l", "toes_l", "lefttoes", "left_toes",
                 "l_toes", "l_toe", "ball.l", "ball_l"],

    "rightUpperLeg": ["thigh.r", "thigh_r", "upper_leg.r", "upper_leg_r", "upperleg_r",
                      "leg_upper_r", "rightupperleg", "right_upper_leg", "r_upperleg",
                      "r_upper_leg", "r_thigh", "leg.r", "leg_r"],
    "rightLowerLeg": ["shin.r", "shin_r", "lower_leg.r", "lower_leg_r", "lowerleg_r",
                      "leg_lower_r", "rightlowerleg", "right_lower_leg", "r_shin",
                      "r_lower_leg", "r_lowerleg", "calf.r", "calf_r", "r_calf",
                      "knee.r", "knee_r"],
    "rightFoot": ["foot.r", "foot_r", "rightfoot", "right_foot", "r_foot",
                  "ankle.r", "ankle_r"],
    "rightToes": ["toe.r", "toe_r", "toes.r", "toes_r", "righttoes", "right_toes",
                  "r_toes", "r_toe", "ball.r", "ball_r"],

    # --- Eyes/Jaw ---
    "leftEye": ["eye.l", "eye_l", "lefteye", "left_eye", "l_eye"],
    "rightEye": ["eye.r", "eye_r", "righteye", "right_eye", "r_eye"],
    "jaw": ["jaw", "jaw_01"],

    # --- Finger bones (left) ---
    "leftThumbMetacarpal": ["thumb.01.l", "thumb_01_l", "thumb_1_l", "l_thumb1", "thumb1.l",
                            "thumb_metacarpal_l", "l_thumb_01"],
    "leftThumbProximal": ["thumb.02.l", "thumb_02_l", "thumb_2_l", "l_thumb2", "thumb2.l",
                          "thumb_proximal_l", "l_thumb_02"],
    "leftThumbDistal": ["thumb.03.l", "thumb_03_l", "thumb_3_l", "l_thumb3", "thumb3.l",
                        "thumb_distal_l", "l_thumb_03"],
    "leftIndexProximal": ["f_index.01.l", "index.01.l", "index_01_l", "index_1_l",
                          "l_index1", "index1.l", "index_proximal_l", "l_index_01"],
    "leftIndexIntermediate": ["f_index.02.l", "index.02.l", "index_02_l", "index_2_l",
                              "l_index2", "index2.l", "index_intermediate_l", "l_index_02"],
    "leftIndexDistal": ["f_index.03.l", "index.03.l", "index_03_l", "index_3_l",
                        "l_index3", "index3.l", "index_distal_l", "l_index_03"],
    "leftMiddleProximal": ["f_middle.01.l", "middle.01.l", "middle_01_l", "middle_1_l",
                           "l_middle1", "middle1.l", "middle_proximal_l", "l_middle_01"],
    "leftMiddleIntermediate": ["f_middle.02.l", "middle.02.l", "middle_02_l", "middle_2_l",
                               "l_middle2", "middle2.l", "middle_intermediate_l", "l_middle_02"],
    "leftMiddleDistal": ["f_middle.03.l", "middle.03.l", "middle_03_l", "middle_3_l",
                         "l_middle3", "middle3.l", "middle_distal_l", "l_middle_03"],
    "leftRingProximal": ["f_ring.01.l", "ring.01.l", "ring_01_l", "ring_1_l",
                         "l_ring1", "ring1.l", "ring_proximal_l", "l_ring_01"],
    "leftRingIntermediate": ["f_ring.02.l", "ring.02.l", "ring_02_l", "ring_2_l",
                             "l_ring2", "ring2.l", "ring_intermediate_l", "l_ring_02"],
    "leftRingDistal": ["f_ring.03.l", "ring.03.l", "ring_03_l", "ring_3_l",
                       "l_ring3", "ring3.l", "ring_distal_l", "l_ring_03"],
    "leftLittleProximal": ["f_pinky.01.l", "pinky.01.l", "pinky_01_l", "little.01.l",
                           "little_01_l", "l_pinky1", "pinky1.l", "l_pinky_01"],
    "leftLittleIntermediate": ["f_pinky.02.l", "pinky.02.l", "pinky_02_l", "little.02.l",
                               "little_02_l", "l_pinky2", "pinky2.l", "l_pinky_02"],
    "leftLittleDistal": ["f_pinky.03.l", "pinky.03.l", "pinky_03_l", "little.03.l",
                         "little_03_l", "l_pinky3", "pinky3.l", "l_pinky_03"],

    # --- Finger bones (right) ---
    "rightThumbMetacarpal": ["thumb.01.r", "thumb_01_r", "thumb_1_r", "r_thumb1", "thumb1.r",
                             "thumb_metacarpal_r", "r_thumb_01"],
    "rightThumbProximal": ["thumb.02.r", "thumb_02_r", "thumb_2_r", "r_thumb2", "thumb2.r",
                           "thumb_proximal_r", "r_thumb_02"],
    "rightThumbDistal": ["thumb.03.r", "thumb_03_r", "thumb_3_r", "r_thumb3", "thumb3.r",
                         "thumb_distal_r", "r_thumb_03"],
    "rightIndexProximal": ["f_index.01.r", "index.01.r", "index_01_r", "index_1_r",
                           "r_index1", "index1.r", "index_proximal_r", "r_index_01"],
    "rightIndexIntermediate": ["f_index.02.r", "index.02.r", "index_02_r", "index_2_r",
                               "r_index2", "index2.r", "index_intermediate_r", "r_index_02"],
    "rightIndexDistal": ["f_index.03.r", "index.03.r", "index_03_r", "index_3_r",
                         "r_index3", "index3.r", "index_distal_r", "r_index_03"],
    "rightMiddleProximal": ["f_middle.01.r", "middle.01.r", "middle_01_r", "middle_1_r",
                            "r_middle1", "middle1.r", "middle_proximal_r", "r_middle_01"],
    "rightMiddleIntermediate": ["f_middle.02.r", "middle.02.r", "middle_02_r", "middle_2_r",
                                "r_middle2", "middle2.r", "middle_intermediate_r", "r_middle_02"],
    "rightMiddleDistal": ["f_middle.03.r", "middle.03.r", "middle_03_r", "middle_3_r",
                          "r_middle3", "middle3.r", "middle_distal_r", "r_middle_03"],
    "rightRingProximal": ["f_ring.01.r", "ring.01.r", "ring_01_r", "ring_1_r",
                          "r_ring1", "ring1.r", "ring_proximal_r", "r_ring_01"],
    "rightRingIntermediate": ["f_ring.02.r", "ring.02.r", "ring_02_r", "ring_2_r",
                              "r_ring2", "ring2.r", "ring_intermediate_r", "r_ring_02"],
    "rightRingDistal": ["f_ring.03.r", "ring.03.r", "ring_03_r", "ring_3_r",
                        "r_ring3", "ring3.r", "ring_distal_r", "r_ring_03"],
    "rightLittleProximal": ["f_pinky.01.r", "pinky.01.r", "pinky_01_r", "little.01.r",
                            "little_01_r", "r_pinky1", "pinky1.r", "r_pinky_01"],
    "rightLittleIntermediate": ["f_pinky.02.r", "pinky.02.r", "pinky_02_r", "little.02.r",
                                "little_02_r", "r_pinky2", "pinky2.r", "r_pinky_02"],
    "rightLittleDistal": ["f_pinky.03.r", "pinky.03.r", "pinky_03_r", "little.03.r",
                          "little_03_r", "r_pinky3", "pinky3.r", "r_pinky_03"],
}

# Required VRM bones (minimum for valid humanoid)
REQUIRED_BONES = ["hips", "spine", "head", "leftUpperArm", "leftLowerArm", "leftHand",
                  "rightUpperArm", "rightLowerArm", "rightHand", "leftUpperLeg", "leftLowerLeg",
                  "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]


def find_bone_match(bone_name: str, patterns: list) -> int:
    """
    Check if bone_name matches any of the patterns using normalized comparison.
    Returns match quality score: 0=no match, 3=exact normalized, 2=exact compact, 1=substring.
    Normalization unifies separators (./-/space → _) and strips suffixes (_JNT etc).
    """
    norm_bone = _normalize_for_match(bone_name)
    compact_bone = _compact(norm_bone)

    best_score = 0
    for pat in patterns:
        norm_pat = _normalize_for_match(pat)
        compact_pat = _compact(norm_pat)

        # Exact normalized match (highest quality)
        if norm_bone == norm_pat:
            return 3
        # Exact compact match (all separators removed)
        if compact_bone == compact_pat:
            best_score = max(best_score, 2)
            continue
        # Substring match (normalized form, min length 4 to reduce false positives)
        if len(norm_pat) >= 4 and norm_pat in norm_bone:
            best_score = max(best_score, 1)
        elif len(norm_bone) >= 4 and norm_bone in norm_pat:
            best_score = max(best_score, 1)
        # Substring match (compact form, min length 5)
        elif len(compact_pat) >= 5 and compact_pat in compact_bone:
            best_score = max(best_score, 1)
        elif len(compact_bone) >= 5 and compact_bone in compact_pat:
            best_score = max(best_score, 1)

    return best_score


def auto_map_bones(armature) -> dict:
    """Automatisch Blender-Bones zu VRM Humanoid Bones mappen.
    Uses multi-pass matching: exact matches first, then fuzzy,
    followed by hierarchy-based fallback for missing required bones."""
    mapping = {}
    all_bones = [b.name for b in armature.data.bones]
    bones_by_name = {b.name: b for b in armature.data.bones}
    used_bones = set()

    # Multi-pass: higher quality matches first to prevent conflicts
    for min_score in [3, 2, 1]:
        for vrm_name, patterns in VRM_BONE_PATTERNS.items():
            if vrm_name in mapping:
                continue
            best_match = None
            best_score = 0
            for bone_name in all_bones:
                if bone_name in used_bones:
                    continue
                score = find_bone_match(bone_name, patterns)
                if score >= min_score and score > best_score:
                    best_match = bone_name
                    best_score = score
            if best_match:
                mapping[vrm_name] = best_match
                used_bones.add(best_match)

    # === Hierarchy-based fallback for missing required bones ===
    print("[INFO] Pruefe fehlende Pflicht-Bones mit Hierarchie-Analyse...")

    # --- SPINE fallback ---
    # If spine is missing but hips+chest are mapped, check the chain between them.
    # If only one bone between hips and neck (or head), it should be spine (required), not chest (optional).
    if "spine" not in mapping and "hips" in mapping:
        hips_bone = bones_by_name.get(mapping["hips"])
        if hips_bone:
            # Get children of hips bone
            hips_children = [c for c in hips_bone.children if c.name not in used_bones or
                             (c.name in used_bones and mapping.get("chest") == c.name)]
            # Find the bone that leads toward neck/head
            neck_bone_name = mapping.get("neck") or mapping.get("head")
            if neck_bone_name and "chest" in mapping:
                chest_name = mapping["chest"]
                chest_bone = bones_by_name.get(chest_name)
                # If chest bone is direct child of hips, there's no spine between them
                # -> Reassign: chest becomes spine (since spine is required, chest is optional)
                if chest_bone and chest_bone.parent and chest_bone.parent.name == mapping["hips"]:
                    # Check if there's another bone between chest and neck that could be chest
                    new_chest = None
                    for child in (chest_bone.children or []):
                        if child.name == neck_bone_name or child.name == mapping.get("neck"):
                            break
                        # Check if this child leads to neck
                        if child.name not in used_bones:
                            # Walk down to see if it connects to neck
                            walker = child
                            while walker:
                                if walker.name == neck_bone_name:
                                    new_chest = child
                                    break
                                walker_children = list(walker.children)
                                walker = walker_children[0] if walker_children else None
                    
                    print(f"[INFO] Spine-Fallback: Verschiebe chest -> spine ({chest_name})")
                    mapping["spine"] = chest_name
                    if new_chest:
                        mapping["chest"] = new_chest.name
                        used_bones.add(new_chest.name)
                        print(f"[INFO] Neuer chest: {new_chest.name}")
                    else:
                        del mapping["chest"]
                        print(f"[INFO] Kein separater chest-Bone gefunden")
            elif not mapping.get("chest"):
                # No chest mapped either - look for any child of hips leading to neck/head
                if hips_bone.children:
                    for child in hips_bone.children:
                        if child.name not in used_bones:
                            # Check if going up from this child leads toward head
                            walker = child
                            depth = 0
                            while walker and depth < 5:
                                if walker.name == neck_bone_name:
                                    mapping["spine"] = child.name
                                    used_bones.add(child.name)
                                    print(f"[INFO] Spine-Fallback (Kette): {child.name}")
                                    break
                                walker_children = list(walker.children)
                                walker = walker_children[0] if walker_children else None
                                depth += 1
                            if "spine" in mapping:
                                break

    # --- UPPER ARM fallback ---
    # In game rigs: clavicle=shoulder, shoulder_bone=upper_arm, elbow=lower_arm, wrist=hand
    # If leftUpperArm is missing but leftShoulder and leftLowerArm are mapped,
    # look for a bone between shoulder and lowerArm in the hierarchy.
    for side, side_label in [("left", "l"), ("right", "r")]:
        upper_key = f"{side}UpperArm"
        shoulder_key = f"{side}Shoulder"
        lower_key = f"{side}LowerArm"
        if upper_key not in mapping and shoulder_key in mapping:
            shoulder_bone = bones_by_name.get(mapping[shoulder_key])
            lower_bone_name = mapping.get(lower_key)
            if shoulder_bone:
                # Look at children of shoulder bone
                for child in shoulder_bone.children:
                    if child.name in used_bones and child.name != lower_bone_name:
                        continue
                    # If this child IS the lower arm, or leads to lower arm, it's the upper arm
                    if lower_bone_name:
                        # Check if lower_arm is child or grandchild of this bone
                        if child.name == lower_bone_name:
                            # Shoulder's direct child is the lower arm - no upper arm bone exists
                            # Use the shoulder as upper arm and un-map shoulder
                            # Actually: just assign child to upper arm since it was lower arm
                            # This doesn't help. Skip.
                            continue
                        is_ancestor = False
                        walker = bones_by_name.get(lower_bone_name)
                        while walker and walker.parent:
                            if walker.parent.name == child.name:
                                is_ancestor = True
                                break
                            walker = walker.parent
                        if is_ancestor and child.name not in used_bones:
                            mapping[upper_key] = child.name
                            used_bones.add(child.name)
                            print(f"[INFO] {upper_key}-Fallback (Hierarchie): {child.name}")
                            break
                    elif child.name not in used_bones:
                        # No lower arm mapped, just take first unmapped child
                        mapping[upper_key] = child.name
                        used_bones.add(child.name)
                        print(f"[INFO] {upper_key}-Fallback (Kind von Shoulder): {child.name}")
                        break

    return mapping


def apply_bone_mapping(armature, mapping: dict):
    """
    VRM Humanoid Bone-Mapping auf das Armature anwenden.
    Nutzt die VRM Add-on Datenstruktur.
    Entscheidet anhand spec_version ob VRM 0.x oder 1.0 Pfad.
    """
    if not hasattr(armature.data, "vrm_addon_extension"):
        print("[WARN] VRM Add-on Extension nicht auf Armature gefunden")
        return False

    vrm_ext = armature.data.vrm_addon_extension
    is_vrm1 = getattr(vrm_ext, "spec_version", "0.0") == "1.0"

    if is_vrm1:
        # VRM 1.0: Properties sind snake_case (left_upper_arm, nicht leftUpperArm)
        human_bones = vrm_ext.vrm1.humanoid.human_bones
        mapped = 0
        for vrm_name, blender_bone in mapping.items():
            attr_name = _camel_to_snake(vrm_name)
            if hasattr(human_bones, attr_name):
                bone_prop = getattr(human_bones, attr_name)
                if hasattr(bone_prop, "node") and hasattr(bone_prop.node, "bone_name"):
                    bone_prop.node.bone_name = blender_bone
                    mapped += 1
                    print(f"  [VRM1] {vrm_name} ({attr_name}) -> {blender_bone}")
            else:
                print(f"  [VRM1] {vrm_name} ({attr_name}) -> NICHT GEFUNDEN in human_bones")

        print(f"[INFO] VRM1 Bone-Mapping: {mapped}/{len(mapping)} Bones gemappt")
        return mapped > 0

    else:
        # VRM 0.x: Collection von HumanBone-Eintraegen
        human_bones_list = vrm_ext.vrm0.humanoid.human_bones
        # Bestehende Eintraege loeschen
        while len(human_bones_list) > 0:
            human_bones_list.remove(0)
        mapped = 0
        for vrm_name, blender_bone in mapping.items():
            try:
                hb = human_bones_list.add()
                hb.bone = vrm_name
                hb.node.bone_name = blender_bone
                mapped += 1
            except Exception as e:
                print(f"[WARN] Bone {vrm_name} -> {blender_bone}: {e}")

        print(f"[INFO] VRM0 Bone-Mapping: {mapped}/{len(mapping)} Bones gemappt")
        return mapped > 0


# ======================= EXPRESSION MAPPING =======================

# Shape key name patterns → VRM expressions
EXPRESSION_PATTERNS = {
    "happy": ["happy", "smile", "joy", "lachen", "freude", "froh", "grinsen"],
    "angry": ["angry", "anger", "wut", "ärger", "böse", "zorn", "mad"],
    "sad": ["sad", "sadness", "trauer", "traurig", "cry", "weinen"],
    "relaxed": ["relaxed", "relax", "entspannt", "calm", "ruhig"],
    "surprised": ["surprised", "surprise", "überrasch", "shock", "schock", "wow"],
    "aa": ["aa", "a", "mouth_a", "vrc.v_aa", "mth_a", "あ"],
    "ih": ["ih", "i", "mouth_i", "vrc.v_ih", "mth_i", "い"],
    "ou": ["ou", "u", "mouth_u", "vrc.v_ou", "mth_u", "う"],
    "ee": ["ee", "e", "mouth_e", "vrc.v_ee", "mth_e", "え"],
    "oh": ["oh", "o", "mouth_o", "vrc.v_oh", "mth_o", "お"],
    "blink": ["blink", "blinzel", "close_eyes", "eye_close", "augen_zu"],
    "blinkLeft": ["blink_l", "blink.l", "wink_l", "wink.l", "left_blink", "blink_left"],
    "blinkRight": ["blink_r", "blink.r", "wink_r", "wink.r", "right_blink", "blink_right"],
    "lookUp": ["look_up", "lookup", "eye_up"],
    "lookDown": ["look_down", "lookdown", "eye_down"],
    "lookLeft": ["look_left", "lookleft", "eye_left"],
    "lookRight": ["look_right", "lookright", "eye_right"],
    "neutral": ["neutral", "default", "basis", "normal"],
}


def collect_shape_keys(armature) -> dict:
    """Alle Shape Keys von allen Meshes finden die zum Armature gehören."""
    shape_keys = {}  # mesh_name → [shape_key_names]

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        # Check if parented to armature
        if obj.parent != armature and obj.find_armature() != armature:
            continue
        if obj.data.shape_keys and obj.data.shape_keys.key_blocks:
            keys = [kb.name for kb in obj.data.shape_keys.key_blocks if kb.name != "Basis"]
            if keys:
                shape_keys[obj.name] = keys

    return shape_keys


def auto_map_expressions(shape_keys: dict) -> dict:
    """Shape Keys automatisch zu VRM Expressions mappen."""
    all_keys = []
    for mesh_name, keys in shape_keys.items():
        for key in keys:
            all_keys.append((mesh_name, key))

    mapping = {}  # vrm_expression → [(mesh_name, shape_key_name)]

    for expr_name, patterns in EXPRESSION_PATTERNS.items():
        for mesh_name, key_name in all_keys:
            clean = key_name.lower().strip()
            for pat in patterns:
                if clean == pat.lower() or pat.lower() in clean:
                    if expr_name not in mapping:
                        mapping[expr_name] = []
                    mapping[expr_name].append((mesh_name, key_name))
                    break

    return mapping


def apply_expression_mapping(armature, expression_mapping: dict):
    """VRM Expression Bindings setzen. Nutzt spec_version fuer VRM 0.x/1.0."""
    if not hasattr(armature.data, "vrm_addon_extension"):
        return

    vrm_ext = armature.data.vrm_addon_extension
    is_vrm1 = getattr(vrm_ext, "spec_version", "0.0") == "1.0"

    if is_vrm1:
        # VRM 1.0: Preset-Expressions als snake_case Properties
        expressions = vrm_ext.vrm1.expressions
        applied = 0

        for expr_name, bindings in expression_mapping.items():
            # VRM 1.0 Addon nutzt snake_case: blinkLeft -> blink_left
            snake_name = _camel_to_snake(expr_name)
            preset_attr = None

            if hasattr(expressions, "preset"):
                preset = expressions.preset
                # Versuche snake_case zuerst, dann original
                if hasattr(preset, snake_name):
                    preset_attr = getattr(preset, snake_name)
                elif hasattr(preset, expr_name):
                    preset_attr = getattr(preset, expr_name)

            if preset_attr is None:
                continue

            for mesh_name, shape_key_name in bindings:
                mesh_obj = bpy.data.objects.get(mesh_name)
                if not mesh_obj:
                    continue
                try:
                    bind = preset_attr.morph_target_binds.add()
                    bind.node.mesh_object_name = mesh_name
                    bind.index = shape_key_name
                    bind.weight = 1.0
                    applied += 1
                except Exception as e:
                    print(f"[WARN] Expression-Binding fehlgeschlagen ({expr_name}/{shape_key_name}): {e}")

        print(f"[INFO] VRM1 Expression-Bindings: {applied} angewendet")

    else:
        # VRM 0.x: BlendShapeGroups
        blend_shape_master = vrm_ext.vrm0.blend_shape_master
        applied = 0

        vrm0_map = {
            "happy": "joy", "angry": "angry", "sad": "sorrow",
            "relaxed": "fun", "surprised": "surprised",
            "aa": "a", "ih": "i", "ou": "u", "ee": "e", "oh": "o",
            "blink": "blink", "blinkLeft": "blink_l", "blinkRight": "blink_r",
            "lookUp": "lookup", "lookDown": "lookdown",
            "lookLeft": "lookleft", "lookRight": "lookright",
            "neutral": "neutral",
        }

        for expr_name, bindings in expression_mapping.items():
            vrm0_name = vrm0_map.get(expr_name, expr_name)

            group = None
            for g in blend_shape_master.blend_shape_groups:
                if g.preset_name == vrm0_name or g.name.lower() == vrm0_name.lower():
                    group = g
                    break

            if group is None:
                group = blend_shape_master.blend_shape_groups.add()
                group.name = vrm0_name
                group.preset_name = vrm0_name

            for mesh_name, shape_key_name in bindings:
                try:
                    bind = group.binds.add()
                    bind.mesh.mesh_object_name = mesh_name
                    bind.index = shape_key_name
                    bind.weight = 1.0
                    applied += 1
                except Exception as e:
                    print(f"[WARN] VRM0 Binding fehlgeschlagen: {e}")

        print(f"[INFO] VRM0 Expression-Bindings: {applied} angewendet")


# ======================= MODEL SETUP =======================

def find_armature():
    """Hauptarmature finden (das mit den meisten Bones)."""
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE" and o.data and len(o.data.bones) > 0]
    if not armatures:
        return None
    armatures.sort(key=lambda a: len(a.data.bones), reverse=True)
    return armatures[0]


def normalize_model(armature):
    """Model zentrieren: Fuesse auf Y=0, horizontal zentriert (headless-safe)."""
    import mathutils

    _ensure_object_mode()

    # Transform apply via Operator mit Context-Override
    _select_all(True)
    bpy.context.view_layer.objects.active = armature
    ctx = _get_context_override()
    try:
        if ctx:
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        else:
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        print("[INFO] Transforms applied (rotation + scale)")
    except Exception as e:
        print(f"[WARN] transform_apply: {e} — versuche manuell")
        # Manuell Scale anwenden
        for obj in bpy.context.view_layer.objects:
            if obj.type == 'MESH' and obj.scale != mathutils.Vector((1, 1, 1)):
                sx, sy, sz = obj.scale
                if obj.data:
                    for v in obj.data.vertices:
                        v.co.x *= sx
                        v.co.y *= sy
                        v.co.z *= sz
                obj.scale = (1, 1, 1)

    all_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not all_meshes:
        all_meshes = [armature]

    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    for obj in all_meshes:
        for corner in obj.bound_box:
            wc = obj.matrix_world @ mathutils.Vector(corner)
            min_x = min(min_x, wc.x); max_x = max(max_x, wc.x)
            min_y = min(min_y, wc.y); max_y = max(max_y, wc.y)
            min_z = min(min_z, wc.z); max_z = max(max_z, wc.z)

    offset = mathutils.Vector((-((min_x + max_x) / 2.0), -min_y, -((min_z + max_z) / 2.0)))
    print(f"[INFO] Bounds: X[{min_x:.3f},{max_x:.3f}] Y[{min_y:.3f},{max_y:.3f}] Z[{min_z:.3f},{max_z:.3f}]")
    print(f"[INFO] Offset: ({offset.x:.3f}, {offset.y:.3f}, {offset.z:.3f})")

    height = max_y - min_y
    print(f"[INFO] Model-Hoehe: {height:.3f} m")

    # Offset direkt auf root-Objekte anwenden
    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.location += offset

    # Location apply
    _select_all(True)
    bpy.context.view_layer.objects.active = armature
    try:
        if ctx:
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
        else:
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
        print("[INFO] Location applied")
    except Exception as e:
        print(f"[WARN] location apply: {e} — Offset bleibt als Location")

    _select_only(armature)


def ensure_rest_pose(armature):
    """Sicherstellen dass Armature in Rest-Pose ist."""
    _ensure_object_mode()
    _select_only(armature)

    # Clear all pose bone transforms
    if armature.pose:
        for pb in armature.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
            pb.scale = (1, 1, 1)


def setup_vrm_version(armature, vrm_version="0.x"):
    """VRM spec_version auf dem Armature setzen (bestimmt Export-Format)."""
    if not hasattr(armature.data, "vrm_addon_extension"):
        print("[WARN] VRM Extension nicht auf Armature gefunden")
        return

    vrm_ext = armature.data.vrm_addon_extension
    if vrm_version == "1.0":
        vrm_ext.spec_version = "1.0"
        print("[INFO] VRM spec_version = 1.0")
    else:
        vrm_ext.spec_version = "0.0"
        print("[INFO] VRM spec_version = 0.0 (VRM 0.x / VSeeFace kompatibel)")


def set_vrm_meta(armature, filename: str, vrm_version="0.x"):
    """VRM Meta-Daten setzen — ALLE Pflichtfelder fuer valides VRM."""
    if not hasattr(armature.data, "vrm_addon_extension"):
        return

    vrm_ext = armature.data.vrm_addon_extension
    name = os.path.splitext(filename)[0]

    if vrm_version == "1.0":
        meta = vrm_ext.vrm1.meta
        # Name
        if hasattr(meta, "vrm_name"):
            meta.vrm_name = name
        # Version
        if hasattr(meta, "version"):
            meta.version = "1.0"
        # Authors (mindestens einer noetig)
        if hasattr(meta, "authors"):
            while len(meta.authors) > 0:
                meta.authors.remove(0)
            author = meta.authors.add()
            author.value = "blend_to_vrm"
        # Licensing (Pflichtfelder fuer VRM 1.0)
        if hasattr(meta, "avatar_permission"):
            meta.avatar_permission = "everyone"
        if hasattr(meta, "allow_excessively_violent_usage"):
            meta.allow_excessively_violent_usage = False
        if hasattr(meta, "allow_excessively_sexual_usage"):
            meta.allow_excessively_sexual_usage = False
        if hasattr(meta, "commercial_usage"):
            meta.commercial_usage = "personalNonProfit"
        if hasattr(meta, "allow_political_or_religious_usage"):
            meta.allow_political_or_religious_usage = False
        if hasattr(meta, "allow_antisocial_or_hate_usage"):
            meta.allow_antisocial_or_hate_usage = False
        if hasattr(meta, "credit_notation"):
            meta.credit_notation = "unnecessary"
        if hasattr(meta, "allow_redistribution"):
            meta.allow_redistribution = True
        if hasattr(meta, "modification"):
            meta.modification = "allowModificationRedistribution"
        if hasattr(meta, "other_license_url"):
            meta.other_license_url = ""
        print(f"[INFO] VRM 1.0 Meta gesetzt: {name}")

    else:
        # VRM 0.x — ALLE Pflichtfelder setzen (sonst ungueltige VRM!)
        meta = vrm_ext.vrm0.meta
        if hasattr(meta, "title"):
            meta.title = name
        if hasattr(meta, "version"):
            meta.version = "1.0"
        if hasattr(meta, "author"):
            meta.author = "blend_to_vrm"
        if hasattr(meta, "contact_information"):
            meta.contact_information = ""
        if hasattr(meta, "reference"):
            meta.reference = ""
        # Lizenz-Felder (PFLICHT fuer gueltige VRM 0.x!)
        if hasattr(meta, "allowed_user_name"):
            meta.allowed_user_name = "Everyone"
        if hasattr(meta, "violent_ussage_name"):
            meta.violent_ussage_name = "Disallow"
        if hasattr(meta, "sexual_ussage_name"):
            meta.sexual_ussage_name = "Disallow"
        if hasattr(meta, "commercial_ussage_name"):
            meta.commercial_ussage_name = "Disallow"
        if hasattr(meta, "license_name"):
            meta.license_name = "Redistribution_Prohibited"
        if hasattr(meta, "other_permission_url"):
            meta.other_permission_url = ""
        if hasattr(meta, "other_license_url"):
            meta.other_license_url = ""
        print(f"[INFO] VRM 0.x Meta gesetzt: {name} (alle Lizenzfelder konfiguriert)")


def clean_scene_for_export():
    """Unnoetige Objekte entfernen (Kameras, Lichter, etc.)."""
    _ensure_object_mode()

    to_remove = []
    for obj in bpy.data.objects:
        if obj.type in ("CAMERA", "LIGHT", "SPEAKER"):
            to_remove.append(obj)
        elif obj.type == "EMPTY":
            # Keep empties that are parents of meshes/armatures
            if any(c.type in ("MESH", "ARMATURE") for c in obj.children):
                continue
            to_remove.append(obj)

    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    removed = len(to_remove)
    if removed:
        print(f"[INFO] {removed} unnoetige Objekte entfernt (Kameras, Lichter, etc.)")

    # Sicherstellen: nur EIN Armature bleibt (VRM braucht genau eins)
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if len(armatures) > 1:
        # Das mit den meisten Bones behalten
        armatures.sort(key=lambda a: len(a.data.bones) if a.data else 0, reverse=True)
        keep = armatures[0]
        for arm in armatures[1:]:
            # Meshes umparenten
            for child in list(arm.children):
                if child.type == 'MESH':
                    child.parent = keep
            bpy.data.objects.remove(arm, do_unlink=True)
        print(f"[INFO] {len(armatures)-1} zusaetzliche Armatures entfernt, behalten: {keep.name}")


def ensure_mesh_parenting(armature):
    """Sicherstellen dass alle Meshes zum Armature geparented sind."""
    orphan_meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.parent is None]
    for mesh in orphan_meshes:
        mesh.parent = armature
        if not any(m.type == "ARMATURE" for m in mesh.modifiers):
            mod = mesh.modifiers.new(name="Armature", type="ARMATURE")
            mod.object = armature
            print(f"[INFO] Mesh '{mesh.name}' zum Armature geparented + Armature-Modifier")

    # Negative Scale fixen (VRM Addon blockt Export bei negativem Scale)
    for obj in bpy.data.objects:
        if obj.type in ('MESH', 'ARMATURE'):
            sx, sy, sz = obj.scale
            if sx < 0 or sy < 0 or sz < 0:
                obj.scale = (abs(sx), abs(sy), abs(sz))
                print(f"[INFO] Negativer Scale bei '{obj.name}' korrigiert -> ({abs(sx):.3f}, {abs(sy):.3f}, {abs(sz):.3f})")


def ensure_mesh_materials():
    """Sicherstellen dass alle Meshes mindestens ein Material haben.
    VRM Export schlaegt fehl wenn Meshes kein Material zugewiesen haben."""
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if obj.data.materials and len(obj.data.materials) > 0:
            # Check for None/empty material slots
            has_valid = False
            for i, mat in enumerate(obj.data.materials):
                if mat is not None:
                    has_valid = True
                else:
                    # Fill empty slot with a default material
                    default_mat = bpy.data.materials.get("VRM_Default")
                    if not default_mat:
                        default_mat = bpy.data.materials.new(name="VRM_Default")
                        default_mat.use_nodes = True
                    obj.data.materials[i] = default_mat
                    print(f"[INFO] Mesh '{obj.name}' leerer Material-Slot {i} mit Default gefuellt")
            if has_valid:
                continue
        # No materials at all
        default_mat = bpy.data.materials.get("VRM_Default")
        if not default_mat:
            default_mat = bpy.data.materials.new(name="VRM_Default")
            default_mat.use_nodes = True
            # Set a basic diffuse color
            if default_mat.node_tree:
                bsdf = default_mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        obj.data.materials.append(default_mat)
        print(f"[INFO] Mesh '{obj.name}' hatte kein Material - VRM_Default zugewiesen")


def validate_vrm_model(armature):
    """VRM-Validierung manuell ausfuehren und Fehler ausgeben."""
    vrm_ext = armature.data.vrm_addon_extension
    is_vrm1 = getattr(vrm_ext, "spec_version", "0.0") == "1.0"

    print(f"\n{'='*50}")
    print(f"  VRM VALIDIERUNG (spec_version={getattr(vrm_ext, 'spec_version', '?')})")
    print(f"{'='*50}")

    # 1. Check Armature count
    arm_count = sum(1 for o in bpy.data.objects if o.type == 'ARMATURE')
    mesh_count = sum(1 for o in bpy.data.objects if o.type == 'MESH')
    print(f"[CHECK] Armatures: {arm_count} (braucht genau 1)")
    print(f"[CHECK] Meshes: {mesh_count} (braucht mindestens 1)")

    if mesh_count == 0:
        print("[ERROR] KEIN MESH GEFUNDEN - Export wird fehlschlagen!")
    if arm_count > 1:
        print("[ERROR] MEHRERE ARMATURES - Export wird fehlschlagen!")

    # 2. Check bone mapping
    if is_vrm1:
        human_bones = vrm_ext.vrm1.humanoid.human_bones
        print("\n[CHECK] VRM 1.0 Human Bone Mapping:")
        for req_bone in REQUIRED_BONES:
            attr_name = _camel_to_snake(req_bone)
            if hasattr(human_bones, attr_name):
                bp = getattr(human_bones, attr_name)
                bone_name = ""
                if hasattr(bp, 'node') and hasattr(bp.node, 'bone_name'):
                    bone_name = bp.node.bone_name
                status_str = f"-> {bone_name}" if bone_name else "-> NICHT ZUGEWIESEN!"
                level = "OK" if bone_name else "FEHLT"
                print(f"  [{level}] {req_bone} ({attr_name}) {status_str}")
            else:
                print(f"  [FEHLT] {req_bone} ({attr_name}) -> Property existiert nicht")
    else:
        human_bones_list = vrm_ext.vrm0.humanoid.human_bones
        print(f"\n[CHECK] VRM 0.x Human Bones ({len(human_bones_list)} Eintraege):")
        mapped_vrm_bones = set()
        for hb in human_bones_list:
            bone_name = hb.node.bone_name if hasattr(hb.node, 'bone_name') else '?'
            vrm_name = hb.bone if hasattr(hb, 'bone') else '?'
            actual_exists = bone_name in [b.name for b in armature.data.bones] if bone_name else False
            status_str = "OK" if (bone_name and actual_exists) else "FEHLT/UNGUELTIG"
            print(f"  [{status_str}] {vrm_name} -> {bone_name}")
            if vrm_name:
                mapped_vrm_bones.add(vrm_name)

        for req in REQUIRED_BONES:
            if req not in mapped_vrm_bones:
                print(f"  [FEHLT!] Pflicht-Bone '{req}' nicht gemappt!")

    # 3. Check meta
    if is_vrm1:
        meta = vrm_ext.vrm1.meta
        print(f"\n[CHECK] VRM 1.0 Meta:")
        print(f"  vrm_name: {getattr(meta, 'vrm_name', '?')}")
        print(f"  authors: {len(getattr(meta, 'authors', []))}")
    else:
        meta = vrm_ext.vrm0.meta
        print(f"\n[CHECK] VRM 0.x Meta:")
        print(f"  title: {getattr(meta, 'title', '?')}")
        print(f"  author: {getattr(meta, 'author', '?')}")
        print(f"  allowed_user_name: {getattr(meta, 'allowed_user_name', '?')}")
        print(f"  license_name: {getattr(meta, 'license_name', '?')}")

    # 4. Check negative scale
    for obj in bpy.data.objects:
        if obj.type in ('MESH', 'ARMATURE'):
            sx, sy, sz = obj.scale
            if sx < 0 or sy < 0 or sz < 0:
                print(f"[ERROR] Objekt '{obj.name}' hat negativen Scale: ({sx}, {sy}, {sz})")

    # 5. Try addon's own validator
    try:
        result = bpy.ops.vrm.model_validate(
            'EXEC_DEFAULT',
            show_successful_message=False,
            armature_object_name=armature.name,
        )
        print(f"\n[CHECK] VRM Addon Validator: {result}")
        if result != {'FINISHED'}:
            print("[WARN] Addon-Validator hat Fehler erkannt - Export wird wahrscheinlich fehlschlagen")
    except Exception as e:
        print(f"[WARN] Kann Addon-Validator nicht ausfuehren: {e}")

    # 6. Read addon's stored validation messages
    try:
        # VRM addon stores messages in armature extension
        if not is_vrm1:
            hb_list = vrm_ext.vrm0.humanoid.human_bones
            for hb in hb_list:
                # Check node reference
                if hasattr(hb.node, 'bone_name'):
                    bn = hb.node.bone_name
                    if bn and bn not in [b.name for b in armature.data.bones]:
                        print(f"[ERROR] Bone '{bn}' fuer {hb.bone} existiert nicht im Armature!")

        # Check materials
        print(f"\n[CHECK] Material-Pruefung:")
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                if not obj.data.materials or len(obj.data.materials) == 0:
                    print(f"[ERROR] Mesh '{obj.name}' hat KEIN Material!")
                else:
                    for i, mat in enumerate(obj.data.materials):
                        if mat is None:
                            print(f"[ERROR] Mesh '{obj.name}' Material-Slot {i} ist leer!")
                        else:
                            print(f"[CHECK] Mesh '{obj.name}' Material {i}: {mat.name}")

        # Check mesh parenting
        print(f"\n[CHECK] Mesh-Parenting:")
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                parent = obj.parent
                has_arm_mod = any(m.type == 'ARMATURE' for m in obj.modifiers)
                print(f"[CHECK] {obj.name}: parent={parent.name if parent else 'NONE'}, armature_mod={has_arm_mod}")

        # Check for duplicate bone names in VRM mapping
        if not is_vrm1:
            bone_names_used = []
            for hb in vrm_ext.vrm0.humanoid.human_bones:
                bn = hb.node.bone_name if hasattr(hb.node, 'bone_name') else ''
                if bn in bone_names_used:
                    print(f"[ERROR] Bone '{bn}' wird MEHRFACH verwendet!")
                bone_names_used.append(bn)

        # Try to capture export errors by intercepting addon error reports
        print(f"\n[CHECK] Scene-Objekte:")
        for obj in bpy.data.objects:
            print(f"[CHECK]   {obj.name} type={obj.type} parent={obj.parent.name if obj.parent else 'NONE'}")

    except Exception as e:
        print(f"[WARN] Erweiterte Validierung fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

    # 7. Try to read addon's internal validation messages
    try:
        # Find the VRM addon module and use its internal validation
        vrm_module = None
        for mod_name, mod in sys.modules.items():
            if 'vrm' in mod_name.lower() and hasattr(mod, '__path__'):
                print(f"[DEBUG] VRM Modul: {mod_name} -> {getattr(mod, '__path__', '?')}")
                vrm_module = mod

        # Try direct validation through addon internals
        for mod_name in sorted(sys.modules.keys()):
            if 'vrm' in mod_name.lower() and 'valid' in mod_name.lower():
                print(f"[DEBUG] VRM Validation Modul: {mod_name}")
            if 'vrm' in mod_name.lower() and 'export' in mod_name.lower():
                print(f"[DEBUG] VRM Export Modul: {mod_name}")

        # Try to find and call validation directly
        for mod_name, mod in sys.modules.items():
            if 'validation' in mod_name.lower() and 'vrm' in mod_name.lower():
                if hasattr(mod, 'WM_OT_vrm_validator') or hasattr(mod, 'VrmValidationError'):
                    print(f"[DEBUG] VRM Validator-Klasse gefunden in {mod_name}")
                # Look for any callable validation function
                for attr_name in dir(mod):
                    if 'valid' in attr_name.lower() or 'error' in attr_name.lower():
                        print(f"[DEBUG]   {mod_name}.{attr_name}")

    except Exception as e:
        print(f"[DEBUG] VRM Modul-Analyse fehlgeschlagen: {e}")

    print(f"{'='*50}\n")


# ======================= MAIN CONVERSION =======================

def convert(input_path: str, output_path: str, job_id: str, options: dict = None):
    """Hauptkonvertierungs-Pipeline."""
    if options is None:
        options = {}

    vrm_version = options.get("vrm_version", "1.0")
    do_normalize = options.get("normalize", True)
    do_expressions = options.get("expression_mapping", "auto") != "none"
    print(f"[INFO] Optionen: VRM={vrm_version}, Normalize={do_normalize}, Expressions={do_expressions}")

    status("VRM Add-on wird geladen...")
    progress(10)
    vrm_module = ensure_vrm_addon()
    force_register_vrm(vrm_module)
    progress(15)

    # Verify VRM export operator exists
    vrm_ops = {}
    for cat_name in ["export_scene"]:
        cat = getattr(bpy.ops, cat_name, None)
        if cat:
            names = [n for n in dir(cat) if "vrm" in n.lower()]
            if names:
                vrm_ops[cat_name] = names
    print(f"[INFO] VRM Operatoren: {vrm_ops}")

    if "export_scene" not in vrm_ops or "vrm" not in vrm_ops["export_scene"]:
        raise RuntimeError("VRM Export-Operator nicht verfügbar! VRM Add-on korrekt installiert?")

    # Load blend file
    status("Blend-Datei wird geladen...")
    progress(20)
    print(f"[INFO] Lade: {input_path}")
    bpy.ops.wm.open_mainfile(filepath=input_path)
    progress(30)

    # WICHTIG: Context nach open_mainfile neu initialisieren
    bpy.context.view_layer.update()
    _ensure_object_mode()

    # Sicherstellen dass ein 3D-Viewport existiert (fuer Operator-Context)
    # Bei manchen .blend Dateien gibt es keinen VIEW_3D Area
    found_3d = False
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                found_3d = True
                break
    if not found_3d:
        # Ersten Area zu VIEW_3D umwandeln
        for window in bpy.context.window_manager.windows:
            if window.screen.areas:
                window.screen.areas[0].type = 'VIEW_3D'
                print("[INFO] Area zu VIEW_3D umgewandelt fuer Operator-Context")
                break

    # List all objects
    print(f"[INFO] Szene-Objekte: {len(bpy.data.objects)}")
    for obj in bpy.data.objects:
        print(f"  - {obj.name} (Typ: {obj.type})")

    # Clean scene
    status("Szene wird aufgeräumt...")
    clean_scene_for_export()
    progress(35)

    # Find armature
    armature = find_armature()
    if not armature:
        raise RuntimeError(
            "Kein Armature/Rig in der Blend-Datei gefunden!\n"
            "Das Model muss ein Armature mit Bones haben."
        )
    print(f"[INFO] Armature: {armature.name} ({len(armature.data.bones)} Bones)")

    # List bones
    status("Bones werden analysiert...")
    progress(40)
    print("[INFO] Alle Bones im Armature:")
    for bone in armature.data.bones:
        parent_name = bone.parent.name if bone.parent else "ROOT"
        norm = _normalize_for_match(bone.name)
        print(f"[INFO]   - {bone.name} (parent: {parent_name}) [normalized: {norm}]")

    # VRM Version auf Armature setzen (MUSS vor Bone/Expression-Mapping passieren)
    status("VRM-Version wird konfiguriert...")
    setup_vrm_version(armature, vrm_version)

    # Ensure mesh parenting
    ensure_mesh_parenting(armature)

    # Normalize model
    if do_normalize:
        status("Model wird normalisiert...")
        progress(45)
        normalize_model(armature)
    else:
        print("[INFO] Normalisierung uebersprungen (Option deaktiviert)")
        progress(45)

    # Rest pose
    ensure_rest_pose(armature)
    progress(50)

    # Auto bone mapping
    status("Bone-Mapping wird erstellt...")
    progress(55)
    bone_mapping = auto_map_bones(armature)

    print(f"[INFO] Bone-Mapping ({len(bone_mapping)} gefunden):")
    for vrm_name, blender_bone in sorted(bone_mapping.items()):
        print(f"[INFO]   {vrm_name} → {blender_bone}")

    # Check required bones
    missing = [b for b in REQUIRED_BONES if b not in bone_mapping]
    if missing:
        print(f"[WARN] Fehlende Pflicht-Bones: {missing}")
        print("[WARN] VRM Export könnte fehlschlagen. Das VRM Add-on versucht automatisches Mapping.")

    # Apply bone mapping
    status("Bone-Mapping wird angewendet...")
    progress(60)
    apply_bone_mapping(armature, bone_mapping)

    # Shape keys / expressions
    status("Shape Keys werden analysiert...")
    progress(65)
    shape_keys = collect_shape_keys(armature)
    total_keys = sum(len(v) for v in shape_keys.values())
    print(f"[INFO] Shape Keys gefunden: {total_keys} auf {len(shape_keys)} Meshes")

    for mesh_name, keys in shape_keys.items():
        print(f"  {mesh_name}: {', '.join(keys[:10])}" + (" ..." if len(keys) > 10 else ""))

    if total_keys > 0 and do_expressions:
        status("Expressions werden gemappt...")
        progress(70)
        expr_mapping = auto_map_expressions(shape_keys)
        print(f"[INFO] Expression-Mapping ({len(expr_mapping)} Expressions):")
        for expr_name, bindings in sorted(expr_mapping.items()):
            binding_str = ", ".join(f"{m}:{k}" for m, k in bindings)
            print(f"  {expr_name} -> {binding_str}")

        apply_expression_mapping(armature, expr_mapping)
        progress(75)
    else:
        if not do_expressions:
            print("[INFO] Expression-Mapping uebersprungen (Option deaktiviert)")
        else:
            print("[INFO] Keine Shape Keys gefunden - Expression-Mapping uebersprungen")
        progress(75)

    # Set VRM meta
    status("VRM Meta-Daten werden gesetzt...")
    progress(80)
    set_vrm_meta(armature, os.path.basename(input_path), vrm_version)

    # Ensure all meshes have materials (VRM requires this)
    status("Materialien werden geprueft...")
    ensure_mesh_materials()

    # Select armature for export
    _ensure_object_mode()
    _select_only(armature)

    # ===== VRM Validierung vor Export =====
    status("VRM wird validiert...")
    progress(82)
    validate_vrm_model(armature)

    # Export VRM
    status("VRM wird exportiert...")
    progress(85)
    print(f"[INFO] Exportiere nach: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export-Parameter: armature_object_name + ignore_warning fuer sauberen Export
    export_kwargs = {
        "filepath": output_path,
        "ignore_warning": True,
        "export_invisibles": True,
        "export_only_selections": False,
        "armature_object_name": armature.name,
    }
    print(f"[INFO] Export-Parameter: {export_kwargs}")

    ctx = _get_context_override()
    try:
        if ctx:
            with bpy.context.temp_override(**ctx):
                result = bpy.ops.export_scene.vrm(**export_kwargs)
        else:
            result = bpy.ops.export_scene.vrm(**export_kwargs)
        if result != {"FINISHED"}:
            raise RuntimeError(f"VRM Export Status: {result}")
    except Exception as e:
        print(f"[WARN] Export mit vollen Optionen fehlgeschlagen: {e}")

        # === DIAGNOSE: Addon-interne Fehler auslesen ===
        print("[INFO] === EXPORT FEHLER DIAGNOSE ===")
        try:
            # Check if addon stores errors in extension data
            vrm_ext = armature.data.vrm_addon_extension
            # Scan all VRM addon related properties for error hints
            for attr in dir(vrm_ext):
                if 'error' in attr.lower() or 'message' in attr.lower() or 'warning' in attr.lower():
                    val = getattr(vrm_ext, attr, None)
                    if val:
                        print(f"[ERROR] vrm_ext.{attr} = {val}")
        except Exception:
            pass

        try:
            # Try to find stored validation results in addon modules
            for mod_name, mod in sys.modules.items():
                if 'vrm' not in mod_name.lower():
                    continue
                for attr in dir(mod):
                    obj = getattr(mod, attr, None)
                    if isinstance(obj, list) and len(obj) > 0 and 'error' in attr.lower():
                        print(f"[ERROR] {mod_name}.{attr} = {obj}")
        except Exception:
            pass

        print("[INFO] Versuche minimalen Export...")
        try:
            minimal_kwargs = {"filepath": output_path, "ignore_warning": True}
            if ctx:
                with bpy.context.temp_override(**ctx):
                    result = bpy.ops.export_scene.vrm(**minimal_kwargs)
            else:
                result = bpy.ops.export_scene.vrm(**minimal_kwargs)
            if result != {"FINISHED"}:
                # Last resort: try exporting as GLB with VRM extensions manually
                print(f"[WARN] Minimaler Export auch fehlgeschlagen: {result}")
                print("[INFO] Versuche alternativen Export via glTF...")
                try:
                    gltf_path = output_path.replace('.vrm', '.glb')
                    bpy.ops.export_scene.gltf(
                        filepath=gltf_path,
                        export_format='GLB',
                        use_selection=False,
                        export_apply=True,
                    )
                    if os.path.exists(gltf_path):
                        # Rename .glb to .vrm (VRM is a superset of glTF/GLB)
                        if gltf_path != output_path:
                            os.rename(gltf_path, output_path)
                        print(f"[INFO] GLB-Export erfolgreich, als VRM gespeichert")
                        print(f"[WARN] Die VRM-Datei hat keine VRM-Metadaten! Nur als Fallback gedacht.")
                    else:
                        raise RuntimeError(f"Minimaler Export fehlgeschlagen: {result}")
                except Exception as gltf_err:
                    print(f"[WARN] GLB-Export auch fehlgeschlagen: {gltf_err}")
                    raise RuntimeError(f"Minimaler Export fehlgeschlagen: {result}")
        except TypeError:
            # Aeltere Addon-Version ohne ignore_warning
            result = bpy.ops.export_scene.vrm(filepath=output_path)

    progress(95)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        status(f"VRM erfolgreich erstellt ({size_mb:.1f} MB)")
        print(f"[OK] VRM erstellt: {output_path} ({size_mb:.1f} MB)")

        # Summary
        print("\n=== Zusammenfassung ===")
        print(f"  Armature: {armature.name} ({len(armature.data.bones)} Bones)")
        print(f"  Bone-Mapping: {len(bone_mapping)} VRM Bones")
        print(f"  Shape Keys: {total_keys}")
        if total_keys > 0:
            print(f"  Expressions: {len(auto_map_expressions(shape_keys))}")
        print(f"  Output: {output_path}")
        print(f"  Größe: {size_mb:.1f} MB")
    else:
        raise RuntimeError("VRM-Datei wurde nicht erstellt!")

    progress(100)


# ======================= ENTRY POINT =======================

if __name__ == "__main__":
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  Blend → VRM Converter")
    print(f"  Input:  {args['input']}")
    print(f"  Output: {args['output']}")
    print(f"  Job:    {args['job_id']}")
    print(f"{'='*60}\n")

    try:
        convert(args["input"], args["output"], args["job_id"], args.get("options", {}))
    except Exception as e:
        print(f"\n[FAIL] Konvertierung fehlgeschlagen: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n[OK] Fertig!")
