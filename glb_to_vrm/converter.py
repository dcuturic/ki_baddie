"""
GLB → VRM Converter Script (läuft INNERHALB von Blender headless)
====================================================================
Wird von app.py via subprocess aufgerufen:
  blender --background --factory-startup --python converter.py -- input.glb output.vrm job_id

Features:
  - GLB/glTF Import via Blender
  - VRM Addon auto-install & force-register
  - Humanoid Bone-Mapping (automatisch + bekannte Namens-Patterns)
  - Shape Keys → VRM Expressions (Gesichtsemotionen)
  - Viseme-Mapping fuer Lipsync (A, I, U, E, O + erweiterte Viseme)
  - Eye-Tracking / LookAt Setup (Bone-based oder BlendShape-based)
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
                        return {
                            'window': window,
                            'screen': window.screen,
                            'area': area,
                            'region': region,
                        }
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


# ======================= ARGS =======================

def parse_args():
    """Parse arguments after '--' separator."""
    argv = sys.argv
    if "--" not in argv:
        print("[FAIL] Keine Argumente. Usage: blender -b --factory-startup -P converter.py -- input.glb output.vrm [job_id]")
        sys.exit(1)
    args = argv[argv.index("--") + 1:]
    if len(args) < 2:
        print("[FAIL] Mindestens input und output Pfad noetig")
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

    for mod in addon_utils.modules():
        name = getattr(mod, "__name__", "")
        if "VRM_Addon_for_Blender" in name:
            enabled, _loaded = addon_utils.check(name)
            if not enabled:
                bpy.ops.preferences.addon_enable(module=name)
                bpy.ops.wm.save_userpref()
            print(f"[OK] VRM Add-on aktiv: {name}")
            return name

    if not os.path.exists(VRM_ADDON_ZIP):
        # Suche in Nachbar-Ordnern
        for sibling in ["blend_to_vrm", "fbx_to_vrm"]:
            alt_zip = os.path.join(os.path.dirname(SCRIPT_DIR), sibling, "vrm_addon.zip")
            if os.path.exists(alt_zip):
                import shutil
                shutil.copy2(alt_zip, VRM_ADDON_ZIP)
                print(f"[INFO] vrm_addon.zip von {sibling} kopiert")
                break
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

    candidates = [e for e in os.listdir(addons_dir)
                  if os.path.isdir(os.path.join(addons_dir, e)) and "VRM_Addon_for_Blender" in e]
    if not candidates:
        raise RuntimeError("VRM Add-on Ordner nach Installation nicht gefunden")

    candidates.sort(key=len, reverse=True)
    folder = candidates[0]

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
    """Force re-register um sicherzustellen dass alle Operatoren verfuegbar sind."""
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
    """Convert camelCase to snake_case."""
    import re
    return re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name).lower()


def _normalize_for_match(name):
    """Normalize bone/pattern name: strip suffixes/prefixes, unify separators."""
    n = name.lower().strip()
    for suffix in ["_jnt", ".jnt", "_bone", ".bone", "_bn", ".bn", "_joint", ".joint",
                    "_def", ".def", "_sk", "_end", ".end", "_null", ".null"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
            break
    for prefix in ["def-", "def_", "org-", "org_", "mch-", "mch_", "jnt_", "jnt-",
                    "bn_", "bone_", "bip_", "bip01_", "bip01 ",
                    "j_", "sk_", "b_", "mixamorig:", "mixamorig_",
                    "valvebip01_", "rig_"]:
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    n = n.replace(".", "_").replace("-", "_").replace(" ", "_")
    while "__" in n:
        n = n.replace("__", "_")
    return n.strip("_")


def _compact(name):
    """Remove ALL separators for ultra-fuzzy comparison."""
    return name.replace("_", "").replace(".", "").replace("-", "").replace(" ", "")


VRM_BONE_PATTERNS = {
    "hips": ["hips", "pelvis", "hip", "cog", "root", "center"],
    "spine": ["spine", "spine1", "spine_01", "spine_1", "spine.001", "spine.01"],
    "chest": ["chest", "spine2", "spine_02", "spine_2", "spine.002", "spine.02",
              "upper_body", "upperbody"],
    "upperChest": ["upper_chest", "spine3", "spine_03", "spine_3", "spine.003", "spine.03",
                   "upper_body2", "upper_body_2"],
    "neck": ["neck", "neck_01", "neck_1"],
    "head": ["head"],

    "leftShoulder": ["shoulder.l", "shoulder_l", "leftshoulder", "left_shoulder",
                     "l_shoulder", "clavicle.l", "clavicle_l", "l_clavicle"],
    "rightShoulder": ["shoulder.r", "shoulder_r", "rightshoulder", "right_shoulder",
                      "r_shoulder", "clavicle.r", "clavicle_r", "r_clavicle"],

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

    "leftEye": ["eye.l", "eye_l", "lefteye", "left_eye", "l_eye",
                "eye_left", "eyeball_l", "eyeball.l"],
    "rightEye": ["eye.r", "eye_r", "righteye", "right_eye", "r_eye",
                 "eye_right", "eyeball_r", "eyeball.r"],
    "jaw": ["jaw", "jaw_01", "jaw_bone"],

    # Finger bones (left)
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

    # Finger bones (right)
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

REQUIRED_BONES = ["hips", "spine", "head", "leftUpperArm", "leftLowerArm", "leftHand",
                  "rightUpperArm", "rightLowerArm", "rightHand", "leftUpperLeg", "leftLowerLeg",
                  "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]


def find_bone_match(bone_name: str, patterns: list) -> int:
    """Return match quality: 0=no, 3=exact normalized, 2=exact compact, 1=substring."""
    norm_bone = _normalize_for_match(bone_name)
    compact_bone = _compact(norm_bone)

    best_score = 0
    for pat in patterns:
        norm_pat = _normalize_for_match(pat)
        compact_pat = _compact(norm_pat)

        if norm_bone == norm_pat:
            return 3
        if compact_bone == compact_pat:
            best_score = max(best_score, 2)
            continue
        if len(norm_pat) >= 4 and norm_pat in norm_bone:
            best_score = max(best_score, 1)
        elif len(norm_bone) >= 4 and norm_bone in norm_pat:
            best_score = max(best_score, 1)
        elif len(compact_pat) >= 5 and compact_pat in compact_bone:
            best_score = max(best_score, 1)
        elif len(compact_bone) >= 5 and compact_bone in compact_pat:
            best_score = max(best_score, 1)

    return best_score


def auto_map_bones(armature) -> dict:
    """Automatisch Blender-Bones zu VRM Humanoid Bones mappen."""
    mapping = {}
    all_bones = [b.name for b in armature.data.bones]
    bones_by_name = {b.name: b for b in armature.data.bones}
    used_bones = set()

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

    # === Hierarchy-based fallback ===
    print("[INFO] Pruefe fehlende Pflicht-Bones mit Hierarchie-Analyse...")

    # SPINE fallback
    if "spine" not in mapping and "hips" in mapping:
        hips_bone = bones_by_name.get(mapping["hips"])
        if hips_bone:
            neck_bone_name = mapping.get("neck") or mapping.get("head")
            if neck_bone_name and "chest" in mapping:
                chest_name = mapping["chest"]
                chest_bone = bones_by_name.get(chest_name)
                if chest_bone and chest_bone.parent and chest_bone.parent.name == mapping["hips"]:
                    new_chest = None
                    for child in (chest_bone.children or []):
                        if child.name == neck_bone_name or child.name == mapping.get("neck"):
                            break
                        if child.name not in used_bones:
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
                if hips_bone.children:
                    for child in hips_bone.children:
                        if child.name not in used_bones:
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

    # UPPER ARM fallback
    for side in ["left", "right"]:
        upper_key = f"{side}UpperArm"
        shoulder_key = f"{side}Shoulder"
        lower_key = f"{side}LowerArm"
        if upper_key not in mapping and shoulder_key in mapping:
            shoulder_bone = bones_by_name.get(mapping[shoulder_key])
            lower_bone_name = mapping.get(lower_key)
            if shoulder_bone:
                for child in shoulder_bone.children:
                    if child.name in used_bones and child.name != lower_bone_name:
                        continue
                    if lower_bone_name:
                        if child.name == lower_bone_name:
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
                        mapping[upper_key] = child.name
                        used_bones.add(child.name)
                        print(f"[INFO] {upper_key}-Fallback (Kind von Shoulder): {child.name}")
                        break

    return mapping


def apply_bone_mapping(armature, mapping: dict):
    """VRM Humanoid Bone-Mapping auf das Armature anwenden."""
    if not hasattr(armature.data, "vrm_addon_extension"):
        print("[WARN] VRM Add-on Extension nicht auf Armature gefunden")
        return False

    vrm_ext = armature.data.vrm_addon_extension
    is_vrm1 = getattr(vrm_ext, "spec_version", "0.0") == "1.0"

    if is_vrm1:
        human_bones = vrm_ext.vrm1.humanoid.human_bones
        mapped = 0
        for vrm_name, blender_bone in mapping.items():
            attr_name = _camel_to_snake(vrm_name)
            if hasattr(human_bones, attr_name):
                bone_prop = getattr(human_bones, attr_name)
                if hasattr(bone_prop, "node") and hasattr(bone_prop.node, "bone_name"):
                    bone_prop.node.bone_name = blender_bone
                    mapped += 1
        print(f"[INFO] VRM1 Bone-Mapping: {mapped}/{len(mapping)} Bones gemappt")
        return mapped > 0
    else:
        human_bones_list = vrm_ext.vrm0.humanoid.human_bones
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


# ======================= EXPRESSION / VISEME / EYE MAPPING =======================

# --- Emotion Expressions ---
EXPRESSION_PATTERNS = {
    "happy": ["happy", "smile", "joy", "lachen", "freude", "froh", "grinsen",
              "fcl_all_joy", "fcl_mtl_joy"],
    "angry": ["angry", "anger", "wut", "mad", "fcl_all_angry", "fcl_mtl_angry"],
    "sad": ["sad", "sadness", "trauer", "traurig", "cry", "weinen",
            "fcl_all_sorrow", "fcl_mtl_sorrow"],
    "relaxed": ["relaxed", "relax", "calm", "fcl_all_fun", "fcl_mtl_fun"],
    "surprised": ["surprised", "surprise", "shock", "wow",
                  "fcl_all_surprised", "fcl_mtl_surprised"],
    "neutral": ["neutral", "default", "basis", "normal"],
}

# --- Viseme / Lip Sync Patterns ---
VISEME_PATTERNS = {
    # Japanische AIUEO Viseme (VRM Standard)
    "aa": ["aa", "a", "mouth_a", "vrc.v_aa", "mth_a", "fcl_mtl_a", "fcl_all_a",
           "viseme_aa", "vis_aa", "v_aa", "mouth_open", "mouth_open_a",
           "あ", "jaw_open"],
    "ih": ["ih", "i", "mouth_i", "vrc.v_ih", "mth_i", "fcl_mtl_i", "fcl_all_i",
           "viseme_ih", "vis_ih", "v_ih", "mouth_i", "い"],
    "ou": ["ou", "u", "mouth_u", "vrc.v_ou", "mth_u", "fcl_mtl_u", "fcl_all_u",
           "viseme_ou", "vis_ou", "v_ou", "mouth_u", "う"],
    "ee": ["ee", "e", "mouth_e", "vrc.v_ee", "mth_e", "fcl_mtl_e", "fcl_all_e",
           "viseme_ee", "vis_ee", "v_ee", "mouth_e", "え"],
    "oh": ["oh", "o", "mouth_o", "vrc.v_oh", "mth_o", "fcl_mtl_o", "fcl_all_o",
           "viseme_oh", "vis_oh", "v_oh", "mouth_o", "お"],
}

# --- Extended Visemes (fuer VRM 1.0 / erweiterte Lipsync) ---
EXTENDED_VISEME_PATTERNS = {
    "viseme_pp": ["viseme_pp", "vis_pp", "v_pp", "pp"],
    "viseme_ff": ["viseme_ff", "vis_ff", "v_ff", "ff"],
    "viseme_th": ["viseme_th", "vis_th", "v_th", "th"],
    "viseme_dd": ["viseme_dd", "vis_dd", "v_dd", "dd"],
    "viseme_kk": ["viseme_kk", "vis_kk", "v_kk", "kk"],
    "viseme_ch": ["viseme_ch", "vis_ch", "v_ch", "ch"],
    "viseme_ss": ["viseme_ss", "vis_ss", "v_ss", "ss"],
    "viseme_nn": ["viseme_nn", "vis_nn", "v_nn", "nn"],
    "viseme_rr": ["viseme_rr", "vis_rr", "v_rr", "rr"],
    "viseme_sil": ["viseme_sil", "vis_sil", "v_sil", "sil", "silence"],
}

# --- Blink / Eye Patterns ---
BLINK_PATTERNS = {
    "blink": ["blink", "close_eyes", "eye_close", "eyes_close", "eyes_closed",
              "fcl_eye_close", "fcl_all_close", "eye_blink", "vrc.blink"],
    "blinkLeft": ["blink_l", "blink.l", "wink_l", "left_blink", "blink_left",
                  "fcl_eye_close_l", "eye_blink_left", "wink_left"],
    "blinkRight": ["blink_r", "blink.r", "wink_r", "right_blink", "blink_right",
                   "fcl_eye_close_r", "eye_blink_right", "wink_right"],
}

# --- LookAt Eye BlendShape Patterns ---
LOOKAT_BLENDSHAPE_PATTERNS = {
    "lookUp": ["look_up", "lookup", "eye_up", "eyes_up",
               "fcl_eye_up", "eye_look_up", "vrc.lookupper"],
    "lookDown": ["look_down", "lookdown", "eye_down", "eyes_down",
                 "fcl_eye_down", "eye_look_down", "vrc.looklower"],
    "lookLeft": ["look_left", "lookleft", "eye_left", "eyes_left",
                 "fcl_eye_left", "eye_look_left", "vrc.lookleft"],
    "lookRight": ["look_right", "lookright", "eye_right", "eyes_right",
                  "fcl_eye_right", "eye_look_right", "vrc.lookright"],
}


def collect_shape_keys(armature) -> dict:
    """Alle Shape Keys von allen Meshes finden."""
    shape_keys = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent != armature and obj.find_armature() != armature:
            continue
        if obj.data.shape_keys and obj.data.shape_keys.key_blocks:
            keys = [kb.name for kb in obj.data.shape_keys.key_blocks if kb.name != "Basis"]
            if keys:
                shape_keys[obj.name] = keys
    return shape_keys


def _match_shape_key(key_name: str, patterns: list) -> bool:
    """Pruefe ob ein Shape Key Name zu einem Pattern passt."""
    clean = key_name.lower().strip()
    for pat in patterns:
        if clean == pat.lower() or pat.lower() in clean:
            return True
    return False


def auto_map_expressions(shape_keys: dict) -> dict:
    """Shape Keys zu Emotionen, Viseme, Blinks und LookAt mappen."""
    all_keys = []
    for mesh_name, keys in shape_keys.items():
        for key in keys:
            all_keys.append((mesh_name, key))

    mapping = {}

    # 1. Emotionen
    for expr_name, patterns in EXPRESSION_PATTERNS.items():
        for mesh_name, key_name in all_keys:
            if _match_shape_key(key_name, patterns):
                if expr_name not in mapping:
                    mapping[expr_name] = []
                mapping[expr_name].append((mesh_name, key_name))
                break  # Nur erstes Match pro Expression

    # 2. Viseme / Lipsync (AIUEO)
    for vis_name, patterns in VISEME_PATTERNS.items():
        for mesh_name, key_name in all_keys:
            if _match_shape_key(key_name, patterns):
                if vis_name not in mapping:
                    mapping[vis_name] = []
                mapping[vis_name].append((mesh_name, key_name))
                break

    # 3. Extended Viseme (optional, wenn vorhanden)
    for vis_name, patterns in EXTENDED_VISEME_PATTERNS.items():
        for mesh_name, key_name in all_keys:
            if _match_shape_key(key_name, patterns):
                if vis_name not in mapping:
                    mapping[vis_name] = []
                mapping[vis_name].append((mesh_name, key_name))
                break

    # 4. Blink
    for blink_name, patterns in BLINK_PATTERNS.items():
        for mesh_name, key_name in all_keys:
            if _match_shape_key(key_name, patterns):
                if blink_name not in mapping:
                    mapping[blink_name] = []
                mapping[blink_name].append((mesh_name, key_name))
                break

    # 5. LookAt BlendShapes
    for look_name, patterns in LOOKAT_BLENDSHAPE_PATTERNS.items():
        for mesh_name, key_name in all_keys:
            if _match_shape_key(key_name, patterns):
                if look_name not in mapping:
                    mapping[look_name] = []
                mapping[look_name].append((mesh_name, key_name))
                break

    return mapping


def apply_expression_mapping(armature, expression_mapping: dict):
    """VRM Expression Bindings setzen (Emotionen + Viseme + Blink + LookAt)."""
    if not hasattr(armature.data, "vrm_addon_extension"):
        return

    vrm_ext = armature.data.vrm_addon_extension
    is_vrm1 = getattr(vrm_ext, "spec_version", "0.0") == "1.0"

    if is_vrm1:
        expressions = vrm_ext.vrm1.expressions
        applied = 0
        for expr_name, bindings in expression_mapping.items():
            snake_name = _camel_to_snake(expr_name)
            preset_attr = None
            if hasattr(expressions, "preset"):
                preset = expressions.preset
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
        blend_shape_master = vrm_ext.vrm0.blend_shape_master
        applied = 0
        # VRM 0.x Preset-Name Mapping
        vrm0_map = {
            # Emotionen
            "happy": "joy", "angry": "angry", "sad": "sorrow",
            "relaxed": "fun", "surprised": "surprised",
            # Viseme (AIUEO)
            "aa": "a", "ih": "i", "ou": "u", "ee": "e", "oh": "o",
            # Blink
            "blink": "blink", "blinkLeft": "blink_l", "blinkRight": "blink_r",
            # LookAt
            "lookUp": "lookup", "lookDown": "lookdown",
            "lookLeft": "lookleft", "lookRight": "lookright",
            # Neutral
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


# ======================= EYE / LOOKAT SETUP =======================

def setup_eye_tracking(armature, bone_mapping: dict, expression_mapping: dict):
    """Eye-Tracking / LookAt konfigurieren.
    
    Zwei Modi:
    1. Bone-based: Wenn Eye-Bones gemappt sind (leftEye, rightEye)
    2. BlendShape-based: Wenn lookUp/lookDown/lookLeft/lookRight Shape Keys vorhanden
    
    VSeeFace benutzt primaer bone-based eye tracking.
    """
    if not hasattr(armature.data, "vrm_addon_extension"):
        return

    vrm_ext = armature.data.vrm_addon_extension
    is_vrm1 = getattr(vrm_ext, "spec_version", "0.0") == "1.0"

    has_eye_bones = "leftEye" in bone_mapping and "rightEye" in bone_mapping
    has_lookat_shapes = any(k in expression_mapping for k in ["lookUp", "lookDown", "lookLeft", "lookRight"])

    print(f"[INFO] Eye-Tracking Setup: eye_bones={has_eye_bones}, lookat_shapes={has_lookat_shapes}")

    if is_vrm1:
        # VRM 1.0 LookAt
        if hasattr(vrm_ext.vrm1, 'look_at'):
            look_at = vrm_ext.vrm1.look_at
            if has_eye_bones:
                if hasattr(look_at, 'type'):
                    look_at.type = 'bone'
                print("[OK] VRM1 LookAt: bone-based (Eye-Bones vorhanden)")
            elif has_lookat_shapes:
                if hasattr(look_at, 'type'):
                    look_at.type = 'expression'
                print("[OK] VRM1 LookAt: expression-based (BlendShapes vorhanden)")
            else:
                print("[INFO] Kein Eye-Tracking moeglich (keine Eye-Bones oder LookAt-BlendShapes)")

            # Range limits setzen
            if hasattr(look_at, 'range_map_horizontal_inner'):
                look_at.range_map_horizontal_inner.input_max_value = 90.0
                look_at.range_map_horizontal_inner.output_scale = 10.0
            if hasattr(look_at, 'range_map_horizontal_outer'):
                look_at.range_map_horizontal_outer.input_max_value = 90.0
                look_at.range_map_horizontal_outer.output_scale = 10.0
            if hasattr(look_at, 'range_map_vertical_down'):
                look_at.range_map_vertical_down.input_max_value = 90.0
                look_at.range_map_vertical_down.output_scale = 10.0
            if hasattr(look_at, 'range_map_vertical_up'):
                look_at.range_map_vertical_up.input_max_value = 90.0
                look_at.range_map_vertical_up.output_scale = 10.0
    else:
        # VRM 0.x firstPerson / lookAt
        first_person = vrm_ext.vrm0.first_person
        if has_eye_bones:
            if hasattr(first_person, 'look_at_type_name'):
                first_person.look_at_type_name = 'Bone'
            print("[OK] VRM0 LookAt: Bone-Typ (Eye-Bones vorhanden)")

            # Degree Map fuer Augen-Bewegungsbereich
            for prop_name in ['look_at_horizontal_inner', 'look_at_horizontal_outer',
                              'look_at_vertical_down', 'look_at_vertical_up']:
                deg_map = getattr(first_person, prop_name, None)
                if deg_map is not None:
                    if hasattr(deg_map, 'x_range'):
                        deg_map.x_range = 90.0
                    if hasattr(deg_map, 'y_range'):
                        deg_map.y_range = 10.0

        elif has_lookat_shapes:
            if hasattr(first_person, 'look_at_type_name'):
                first_person.look_at_type_name = 'BlendShape'
            print("[OK] VRM0 LookAt: BlendShape-Typ")

            for prop_name in ['look_at_horizontal_inner', 'look_at_horizontal_outer',
                              'look_at_vertical_down', 'look_at_vertical_up']:
                deg_map = getattr(first_person, prop_name, None)
                if deg_map is not None:
                    if hasattr(deg_map, 'x_range'):
                        deg_map.x_range = 90.0
                    if hasattr(deg_map, 'y_range'):
                        deg_map.y_range = 1.0
        else:
            print("[INFO] Kein Eye-Tracking: Keine Eye-Bones oder LookAt-BlendShapes gefunden")

        # Head Bone fuer firstPerson setzen
        if "head" in bone_mapping:
            if hasattr(first_person, 'first_person_bone'):
                first_person.first_person_bone.bone_name = bone_mapping["head"]
                print(f"[OK] FirstPerson Bone: {bone_mapping['head']}")
            elif hasattr(first_person, 'first_person_bone_offset'):
                pass  # Offset bleibt default

    # Zusammenfassung
    lookat_shapes_found = [k for k in ["lookUp", "lookDown", "lookLeft", "lookRight"] if k in expression_mapping]
    if lookat_shapes_found:
        print(f"[INFO] LookAt BlendShapes: {', '.join(lookat_shapes_found)}")


# ======================= MODEL SETUP =======================

def find_armature():
    """Hauptarmature finden (das mit den meisten Bones)."""
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE" and o.data and len(o.data.bones) > 0]
    if not armatures:
        return None
    armatures.sort(key=lambda a: len(a.data.bones), reverse=True)
    return armatures[0]


def normalize_model(armature):
    """Model auf menschliche Groesse skalieren, zentrieren, Fuesse auf Boden."""
    import mathutils

    TARGET_HEIGHT = 1.6  # Ziel-Hoehe in Metern (typischer VRM Avatar)

    _ensure_object_mode()
    _select_all(True)
    bpy.context.view_layer.objects.active = armature
    ctx = _get_context_override()

    # Schritt 1: Rotation + Scale anwenden (damit Bounds stimmen)
    try:
        if ctx:
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        else:
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        print("[INFO] Transforms applied (rotation + scale)")
    except Exception as e:
        print(f"[WARN] transform_apply: {e}")

    # Schritt 2: Bounds berechnen
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

    print(f"[INFO] Bounds VOR Normalisierung: X[{min_x:.3f},{max_x:.3f}] Y[{min_y:.3f},{max_y:.3f}] Z[{min_z:.3f},{max_z:.3f}]")

    # Schritt 3: Up-Achse bestimmen und Hoehe berechnen
    height_y = max_y - min_y
    height_z = max_z - min_z

    if height_z > height_y * 1.5:
        # Z-up Modell -> Hoehe ist Z-Ausdehnung
        up_axis = 'Z'
        current_height = height_z
        bottom = min_z
        center_horiz_1 = (min_x + max_x) / 2.0
        center_horiz_2 = (min_y + max_y) / 2.0
    else:
        # Y-up Modell (Standard fuer glTF/VRM)
        up_axis = 'Y'
        current_height = height_y
        bottom = min_y
        center_horiz_1 = (min_x + max_x) / 2.0
        center_horiz_2 = (min_z + max_z) / 2.0

    if current_height < 0.001:
        current_height = 1.0

    print(f"[INFO] Up-Achse: {up_axis}, aktuelle Hoehe: {current_height:.3f}")

    # Schritt 4: Skalierungsfaktor berechnen
    scale_factor = TARGET_HEIGHT / current_height
    needs_scale = abs(scale_factor - 1.0) > 0.01  # Nur skalieren wenn noetig (>1% Abweichung)

    if needs_scale:
        print(f"[INFO] Skalierung: {scale_factor:.6f}x (von {current_height:.1f} auf {TARGET_HEIGHT:.1f} m)")

        # Alle Root-Objekte skalieren
        for obj in bpy.data.objects:
            if obj.parent is None:
                obj.scale *= scale_factor

        # Scale anwenden
        bpy.context.view_layer.update()
        _select_all(True)
        bpy.context.view_layer.objects.active = armature
        try:
            if ctx:
                with bpy.context.temp_override(**ctx):
                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            else:
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            print("[INFO] Scale applied")
        except Exception as e:
            print(f"[WARN] scale apply: {e}")

        # Bounds neu berechnen nach Skalierung
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        for obj in all_meshes:
            for corner in obj.bound_box:
                wc = obj.matrix_world @ mathutils.Vector(corner)
                min_x = min(min_x, wc.x); max_x = max(max_x, wc.x)
                min_y = min(min_y, wc.y); max_y = max(max_y, wc.y)
                min_z = min(min_z, wc.z); max_z = max(max_z, wc.z)

        print(f"[INFO] Bounds NACH Skalierung: X[{min_x:.3f},{max_x:.3f}] Y[{min_y:.3f},{max_y:.3f}] Z[{min_z:.3f},{max_z:.3f}]")
    else:
        print(f"[INFO] Skalierung nicht noetig (Hoehe {current_height:.2f} m ist bereits ok)")

    # Schritt 5: Zentrieren + Fuesse auf Boden
    if up_axis == 'Z':
        offset = mathutils.Vector((-(min_x + max_x) / 2.0, -(min_y + max_y) / 2.0, -min_z))
    else:
        offset = mathutils.Vector((-(min_x + max_x) / 2.0, -min_y, -(min_z + max_z) / 2.0))

    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.location += offset

    # Location anwenden
    bpy.context.view_layer.update()
    _select_all(True)
    bpy.context.view_layer.objects.active = armature
    try:
        if ctx:
            with bpy.context.temp_override(**ctx):
                bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
        else:
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
        print("[INFO] Location applied (zentriert + Fuesse auf Boden)")
    except Exception as e:
        print(f"[WARN] location apply: {e}")

    # Finale Hoehe berechnen
    final_min_y = float('inf')
    final_max_y = float('-inf')
    for obj in all_meshes:
        for corner in obj.bound_box:
            wc = obj.matrix_world @ mathutils.Vector(corner)
            if up_axis == 'Z':
                final_min_y = min(final_min_y, wc.z)
                final_max_y = max(final_max_y, wc.z)
            else:
                final_min_y = min(final_min_y, wc.y)
                final_max_y = max(final_max_y, wc.y)

    final_height = final_max_y - final_min_y
    print(f"[OK] Finale Model-Hoehe: {final_height:.3f} m (Ziel: {TARGET_HEIGHT} m)")

    _select_only(armature)


def ensure_rest_pose(armature):
    """Sicherstellen dass Armature in Rest-Pose ist."""
    _ensure_object_mode()
    _select_only(armature)
    if armature.pose:
        for pb in armature.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
            pb.scale = (1, 1, 1)


def setup_vrm_version(armature, vrm_version="0.x"):
    """VRM spec_version setzen."""
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
    """VRM Meta-Daten mit ALLEN Pflichtfeldern setzen."""
    if not hasattr(armature.data, "vrm_addon_extension"):
        return
    vrm_ext = armature.data.vrm_addon_extension
    name = os.path.splitext(filename)[0]

    if vrm_version == "1.0":
        meta = vrm_ext.vrm1.meta
        if hasattr(meta, "vrm_name"):
            meta.vrm_name = name
        if hasattr(meta, "version"):
            meta.version = "1.0"
        if hasattr(meta, "authors"):
            while len(meta.authors) > 0:
                meta.authors.remove(0)
            author = meta.authors.add()
            author.value = "glb_to_vrm"
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
        print(f"[INFO] VRM 1.0 Meta gesetzt: {name}")
    else:
        meta = vrm_ext.vrm0.meta
        if hasattr(meta, "title"):
            meta.title = name
        if hasattr(meta, "version"):
            meta.version = "1.0"
        if hasattr(meta, "author"):
            meta.author = "glb_to_vrm"
        if hasattr(meta, "contact_information"):
            meta.contact_information = ""
        if hasattr(meta, "reference"):
            meta.reference = ""
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
    """Unnoetige Objekte entfernen."""
    _ensure_object_mode()
    to_remove = []
    for obj in bpy.data.objects:
        if obj.type in ("CAMERA", "LIGHT", "SPEAKER"):
            to_remove.append(obj)
        elif obj.type == "EMPTY":
            if any(c.type in ("MESH", "ARMATURE") for c in obj.children):
                continue
            to_remove.append(obj)
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    if to_remove:
        print(f"[INFO] {len(to_remove)} unnoetige Objekte entfernt")

    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if len(armatures) > 1:
        armatures.sort(key=lambda a: len(a.data.bones) if a.data else 0, reverse=True)
        keep = armatures[0]
        for arm in armatures[1:]:
            for child in list(arm.children):
                if child.type == 'MESH':
                    child.parent = keep
            bpy.data.objects.remove(arm, do_unlink=True)
        print(f"[INFO] {len(armatures)-1} Extra-Armatures entfernt, behalten: {keep.name}")


def ensure_mesh_parenting(armature):
    """Sicherstellen dass alle Meshes zum Armature geparented sind."""
    orphan_meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.parent is None]
    for mesh in orphan_meshes:
        mesh.parent = armature
        if not any(m.type == "ARMATURE" for m in mesh.modifiers):
            mod = mesh.modifiers.new(name="Armature", type="ARMATURE")
            mod.object = armature
            print(f"[INFO] Mesh '{mesh.name}' zum Armature geparented + Armature-Modifier")

    for obj in bpy.data.objects:
        if obj.type in ('MESH', 'ARMATURE'):
            sx, sy, sz = obj.scale
            if sx < 0 or sy < 0 or sz < 0:
                obj.scale = (abs(sx), abs(sy), abs(sz))
                print(f"[INFO] Negativer Scale bei '{obj.name}' korrigiert")


def ensure_mesh_materials():
    """Sicherstellen dass alle Meshes mindestens ein Material haben."""
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if obj.data.materials and len(obj.data.materials) > 0:
            has_valid = False
            for i, mat in enumerate(obj.data.materials):
                if mat is not None:
                    has_valid = True
                else:
                    default_mat = bpy.data.materials.get("VRM_Default")
                    if not default_mat:
                        default_mat = bpy.data.materials.new(name="VRM_Default")
                        default_mat.use_nodes = True
                    obj.data.materials[i] = default_mat
                    print(f"[INFO] Mesh '{obj.name}' leerer Material-Slot {i} mit Default gefuellt")
            if has_valid:
                continue
        default_mat = bpy.data.materials.get("VRM_Default")
        if not default_mat:
            default_mat = bpy.data.materials.new(name="VRM_Default")
            default_mat.use_nodes = True
            if default_mat.node_tree:
                bsdf = default_mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        obj.data.materials.append(default_mat)
        print(f"[INFO] Mesh '{obj.name}' hatte kein Material - VRM_Default zugewiesen")


# ======================= GLB / GLTF IMPORT =======================

def import_glb(glb_path: str):
    """GLB/glTF-Datei in Blender importieren."""
    print(f"[INFO] Importiere GLB/glTF: {glb_path}")

    # Szene leeren
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # GLB/glTF Import
    ext = os.path.splitext(glb_path)[1].lower()
    try:
        bpy.ops.import_scene.gltf(filepath=glb_path)
    except Exception as e:
        raise RuntimeError(f"GLB/glTF-Import fehlgeschlagen: {e}")

    bpy.context.view_layer.update()
    _ensure_object_mode()

    obj_count = len(bpy.data.objects)
    mesh_count = sum(1 for o in bpy.data.objects if o.type == 'MESH')
    arm_count = sum(1 for o in bpy.data.objects if o.type == 'ARMATURE')
    sk_count = 0
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.data.shape_keys and obj.data.shape_keys.key_blocks:
            sk_count += len([k for k in obj.data.shape_keys.key_blocks if k.name != "Basis"])

    print(f"[INFO] GLB importiert: {obj_count} Objekte ({mesh_count} Meshes, {arm_count} Armatures, {sk_count} Shape Keys)")

    # Alle Objekt-Typen loggen
    type_counts = {}
    for o in bpy.data.objects:
        type_counts[o.type] = type_counts.get(o.type, 0) + 1
    print(f"[INFO] Objekt-Typen: {type_counts}")

    # Vertex Groups loggen
    all_vg = set()
    for o in bpy.data.objects:
        if o.type == 'MESH':
            for vg in o.vertex_groups:
                all_vg.add(vg.name)
    if all_vg:
        print(f"[INFO] Vertex Groups gefunden: {len(all_vg)}")
        for i, name in enumerate(sorted(all_vg)):
            if i < 40:
                print(f"[INFO]   VG: {name}")
            elif i == 40:
                print(f"[INFO]   ... und {len(all_vg) - 40} weitere")

    # Empties loggen
    empties_found = [o for o in bpy.data.objects if o.type == 'EMPTY']
    if empties_found:
        print(f"[INFO] Empty-Objekte: {len(empties_found)}")
        for e in empties_found[:30]:
            parent_name = e.parent.name if e.parent else "NONE"
            child_names = [c.name for c in e.children][:5]
            pos = e.matrix_world.translation
            print(f"[INFO]   Empty '{e.name}' pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f}) parent={parent_name} children={child_names}")

    if arm_count == 0:
        print("[WARN] Kein Armature im GLB gefunden - versuche automatische Erstellung...")
        armature = create_armature_from_hierarchy()
        if armature:
            print(f"[OK] Armature automatisch erstellt: {armature.name} ({len(armature.data.bones)} Bones)")
        else:
            raise RuntimeError(
                "Kein Armature/Rig in der GLB-Datei gefunden und automatische Erstellung fehlgeschlagen!\n"
                "Das Modell hat kein Skeleton. Bitte ein Modell mit Rig/Skeleton verwenden."
            )
    if mesh_count == 0:
        print("[WARN] Keine Meshes in der GLB-Datei gefunden!")


# ======================= AUTO ARMATURE =======================

def create_armature_from_hierarchy():
    """
    Armature aus vorhandener Scene-Hierarchie erstellen wenn kein Armature importiert wurde.

    Strategien:
      1. Empty-Objekte bilden Bone-Hierarchie (haeufig bei Game-Rips)
      2. Mesh Vertex Groups definieren Bone-Namen + Positionen
      3. Minimales Humanoid-Armature als Fallback (fuer komplett statische Modelle)
    """
    import mathutils
    _ensure_object_mode()

    empties = [o for o in bpy.data.objects if o.type == 'EMPTY']
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']

    # Vertex Groups sammeln
    all_vg_names = set()
    for mesh_obj in meshes:
        for vg in mesh_obj.vertex_groups:
            all_vg_names.add(vg.name)

    print(f"[INFO] Auto-Armature: {len(empties)} Empties, {len(meshes)} Meshes, {len(all_vg_names)} Vertex Groups")

    armature = None
    used_empties_as_bones = False

    # === STRATEGIE 1: Empties mit Hierarchie ===
    # Nur wenn genuegend Empties vorhanden UND mindestens einige davon Bone-artige Namen haben
    # Filtert Scene-Struktur Empties (Sketchfab_model, Node, Root, etc.) heraus
    SCENE_STRUCTURE_NAMES = {
        'sketchfab_model', 'skfb_offset', 'scene', 'root', 'rootnode',
        'armature', 'skeleton', 'gltf_scenerootenode',
    }
    if len(empties) >= 5:
        # Pruefen ob Empties einen Baum bilden
        has_hierarchy = any(
            e.parent and e.parent.type == 'EMPTY'
            for e in empties
        )
        # Pruefen ob Empties bone-artige Namen haben (nicht nur Scene-Struktur)
        bone_like_count = 0
        for e in empties:
            norm = _normalize_for_match(e.name)
            if norm in SCENE_STRUCTURE_NAMES:
                continue
            # Gegen bekannte Bone-Patterns pruefen
            for vrm_bone, patterns in BONE_PATTERNS.items():
                if any(p in norm for p in patterns):
                    bone_like_count += 1
                    break
        if has_hierarchy and bone_like_count >= 2:
            print(f"[INFO] === Strategie 1: Armature aus {len(empties)} Empties ({bone_like_count} bone-like) ===")
            armature = _build_armature_from_empties(empties, meshes, all_vg_names)
            used_empties_as_bones = True
        else:
            reason = []
            if not has_hierarchy:
                reason.append("keine Hierarchie")
            if bone_like_count < 2:
                reason.append(f"nur {bone_like_count} bone-artige Namen")
            print(f"[INFO] Strategie 1 uebersprungen: {', '.join(reason)}")

    # Empties aufraeumen wenn sie NICHT als Bones verwendet wurden
    if not used_empties_as_bones and empties:
        _cleanup_scene_empties(empties)
        # Meshes-Liste aktualisieren (Empties sind weg)
        meshes = [o for o in bpy.data.objects if o.type == 'MESH']

    # === STRATEGIE 2: Vertex Groups -> Armature ===
    if not armature and len(all_vg_names) >= 3:
        print(f"[INFO] === Strategie 2: Armature aus {len(all_vg_names)} Vertex Groups ===")
        armature = _build_armature_from_vgroups(meshes, all_vg_names)

    # === STRATEGIE 3: Minimales Humanoid Armature ===
    if not armature:
        print(f"[INFO] === Strategie 3: Minimales Humanoid-Armature fuer {len(meshes)} Meshes ===")
        armature = _build_minimal_armature(meshes)

    return armature


def _cleanup_scene_empties(empties):
    """Scene-Struktur Empties entfernen und World-Transforms auf Kinder uebertragen."""
    import mathutils
    _ensure_object_mode()

    # Schritt 1: Alle Nicht-Empty-Objekte von Empty-Parents losloesen
    # WICHTIG: World-Matrix muss erhalten bleiben!
    unparented = 0
    for obj in list(bpy.data.objects):
        if obj.type == 'EMPTY':
            continue
        if obj.parent and obj.parent.type == 'EMPTY':
            world_mat = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world_mat
            unparented += 1

    bpy.context.view_layer.update()

    if unparented > 0:
        print(f"[INFO] {unparented} Objekte von Empties losgeloest (World-Transform erhalten)")

    # Schritt 2: Alle Empties entfernen
    removed = 0
    for e in list(empties):
        try:
            bpy.data.objects.remove(e, do_unlink=True)
            removed += 1
        except Exception:
            pass

    bpy.context.view_layer.update()
    print(f"[INFO] {removed} Scene-Struktur Empties entfernt")

    # Schritt 3: Logging der resultierenden Mesh-Positionen
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            pos = obj.matrix_world.translation
            rot = obj.matrix_world.to_euler()
            import math
            print(f"[INFO]   Mesh '{obj.name}' pos=({pos.x:.2f},{pos.y:.2f},{pos.z:.2f}) "
                  f"rot=({math.degrees(rot.x):.0f},{math.degrees(rot.y):.0f},{math.degrees(rot.z):.0f})°")


def _build_armature_from_empties(empties, meshes, vg_names):
    """Armature aus Empty-Objekten erstellen (Game-Rip Skeleton Recovery)."""
    import mathutils
    _ensure_object_mode()

    # Armature erstellen
    arm_data = bpy.data.armatures.new("Generated_Armature")
    arm_obj = bpy.data.objects.new("Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.update()

    _select_only(arm_obj)
    bpy.ops.object.mode_set(mode='EDIT')

    # Empties nach Tiefe sortieren (Roots zuerst)
    def depth(obj, d=0):
        if obj.parent and obj.parent.type == 'EMPTY':
            return depth(obj.parent, d + 1)
        return d

    sorted_empties = sorted(empties, key=lambda e: depth(e))
    bone_map = {}  # empty.name -> edit_bone

    for empty in sorted_empties:
        bone = arm_data.edit_bones.new(empty.name)
        pos = empty.matrix_world.translation.copy()
        bone.head = pos

        # Tail: Richtung zum ersten Kind oder Offset
        child_empties = [c for c in empty.children if c.type == 'EMPTY']
        if child_empties:
            child_pos = child_empties[0].matrix_world.translation
            direction = child_pos - pos
            if direction.length > 0.001:
                bone.tail = pos + direction.normalized() * min(direction.length, 0.1)
            else:
                bone.tail = pos + mathutils.Vector((0, 0.05, 0))
        else:
            # Leaf-Bone
            if empty.parent and empty.parent.type == 'EMPTY':
                parent_pos = empty.parent.matrix_world.translation
                direction = pos - parent_pos
                if direction.length > 0.001:
                    bone.tail = pos + direction.normalized() * 0.03
                else:
                    bone.tail = pos + mathutils.Vector((0, 0.03, 0))
            else:
                bone.tail = pos + mathutils.Vector((0, 0.05, 0))

        # Parent setzen
        if empty.parent and empty.parent.type == 'EMPTY' and empty.parent.name in bone_map:
            bone.parent = bone_map[empty.parent.name]

        bone_map[empty.name] = bone

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    bone_count = len(bone_map)
    print(f"[OK] {bone_count} Bones aus Empties erstellt")

    # Meshes zum Armature parenten
    _parent_meshes_to_armature(arm_obj, meshes, [])

    # Empties entfernen (sind jetzt Bones)
    for empty in empties:
        try:
            bpy.data.objects.remove(empty, do_unlink=True)
        except Exception:
            pass
    print(f"[INFO] {len(empties)} Empties entfernt")

    return arm_obj


def _build_armature_from_vgroups(meshes, vg_names):
    """Armature aus Mesh Vertex Groups erstellen (Positionen aus gewichteten Vertices)."""
    import mathutils
    _ensure_object_mode()

    # Centroids fuer jede Vertex Group berechnen
    bone_positions = {}
    for mesh_obj in meshes:
        if not mesh_obj.data.vertices:
            continue
        for vg in mesh_obj.vertex_groups:
            if vg.name not in bone_positions:
                bone_positions[vg.name] = {"sum": mathutils.Vector((0, 0, 0)), "count": 0}
            for v in mesh_obj.data.vertices:
                for g in v.groups:
                    if g.group == vg.index and g.weight > 0.1:
                        world_pos = mesh_obj.matrix_world @ v.co
                        bone_positions[vg.name]["sum"] += world_pos
                        bone_positions[vg.name]["count"] += 1

    centroids = {}
    for name, data in bone_positions.items():
        if data["count"] > 0:
            centroids[name] = data["sum"] / data["count"]
        else:
            centroids[name] = mathutils.Vector((0, 0, 0))

    if not centroids:
        print("[WARN] Keine Vertex-Positionen fuer Bones berechenbar")
        return None

    print(f"[INFO] {len(centroids)} Bone-Positionen aus Vertex Groups berechnet")

    # Armature erstellen
    arm_data = bpy.data.armatures.new("Generated_Armature")
    arm_obj = bpy.data.objects.new("Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.update()

    _select_only(arm_obj)
    bpy.ops.object.mode_set(mode='EDIT')

    created_bones = {}
    # Bones versuchen hierarchisch zu ordnen
    # Zuerst potentielle Root-Bones (hips/pelvis/root/center)
    root_name = None
    for name in centroids:
        norm = _normalize_for_match(name)
        if any(p in norm for p in ['hip', 'pelvis', 'root', 'center']):
            root_name = name
            break

    for name, pos in centroids.items():
        bone = arm_data.edit_bones.new(name)
        bone.head = pos
        bone.tail = pos + mathutils.Vector((0, 0.05, 0))
        created_bones[name] = bone

    # Einfache Hierarchie: bekannte Knochen an Root/Spine haengen
    if root_name and root_name in created_bones:
        root_bone = created_bones[root_name]
        for name, bone in created_bones.items():
            if name == root_name:
                continue
            norm = _normalize_for_match(name)
            # Spine-Chain ans Root
            if any(p in norm for p in ['spine', 'chest', 'neck', 'head']):
                # Versuche Kette zu bilden
                if 'spine' in norm:
                    bone.parent = root_bone
                elif 'chest' in norm:
                    spine_bone = next((b for n, b in created_bones.items() if 'spine' in _normalize_for_match(n)), root_bone)
                    bone.parent = spine_bone
                elif 'neck' in norm:
                    chest_bone = next((b for n, b in created_bones.items() if 'chest' in _normalize_for_match(n)), root_bone)
                    bone.parent = chest_bone
                elif 'head' in norm:
                    neck_bone = next((b for n, b in created_bones.items() if 'neck' in _normalize_for_match(n)), root_bone)
                    bone.parent = neck_bone
            elif not bone.parent:
                bone.parent = root_bone

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    print(f"[OK] {len(created_bones)} Bones aus Vertex Groups erstellt")

    _parent_meshes_to_armature(arm_obj, meshes, [])
    return arm_obj


def _classify_meshes(meshes):
    """Meshes in Charakter-Meshes und Props (Plattformen, Buecher, etc.) klassifizieren.
    
    Verwendet Overlap-Analyse mit dem Body-Mesh (meiste Vertices) um Props zu erkennen.
    
    Returns:
        (character_meshes, prop_meshes) - zwei Listen
    """
    import mathutils
    
    if len(meshes) <= 1:
        return meshes, []
    
    # Bounding Box fuer jedes Mesh berechnen (per-vertex, world space)
    mesh_info = []
    for m in meshes:
        if not m.data.vertices:
            mesh_info.append({'obj': m, 'verts': 0, 'min': mathutils.Vector((0,0,0)), 'max': mathutils.Vector((0,0,0))})
            continue
        
        min_co = mathutils.Vector((float('inf'), float('inf'), float('inf')))
        max_co = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
        for v in m.data.vertices:
            wc = m.matrix_world @ v.co
            min_co.x = min(min_co.x, wc.x); max_co.x = max(max_co.x, wc.x)
            min_co.y = min(min_co.y, wc.y); max_co.y = max(max_co.y, wc.y)
            min_co.z = min(min_co.z, wc.z); max_co.z = max(max_co.z, wc.z)
        
        dims = max_co - min_co
        sorted_dims = sorted([dims.x, dims.y, dims.z], reverse=True)
        flatness = (sorted_dims[0] * sorted_dims[1]) / max(sorted_dims[2], 0.001) if sorted_dims[2] > 0 else 999
        
        mesh_info.append({
            'obj': m,
            'verts': len(m.data.vertices),
            'min': min_co.copy(),
            'max': max_co.copy(),
            'dims': dims.copy(),
            'sorted_dims': sorted_dims,
            'flatness': flatness,
            'center': (min_co + max_co) / 2,
        })
    
    # Body-Candidate: Mesh mit den meisten Vertices
    mesh_info.sort(key=lambda x: x['verts'], reverse=True)
    body = mesh_info[0]
    
    # Body-Zylinder berechnen: Schmalerer Bereich um das Zentrum des Body-Mesh
    # Fuer den Koerper nehmen wir die mittleren 60% der X-Breite als "Torso-Zylinder"
    body_cx = (body['min'].x + body['max'].x) / 2
    body_half_w = (body['max'].x - body['min'].x) / 2
    torso_half_w = body_half_w * 0.7  # 70% der Body-Breite = Torso
    
    # Up-Achse bestimmen
    body_h_y = body['max'].y - body['min'].y
    body_h_z = body['max'].z - body['min'].z
    body_up = 'Z' if body_h_z > body_h_y else 'Y'
    
    if body_up == 'Z':
        body_height = body_h_z
        body_bottom = body['min'].z
        body_top = body['max'].z
    else:
        body_height = body_h_y
        body_bottom = body['min'].y
        body_top = body['max'].y
    
    character_meshes = []
    prop_meshes = []
    
    for info in mesh_info:
        obj = info['obj']
        if info['verts'] == 0:
            prop_meshes.append(obj)
            continue
        
        # Body-Mesh ist immer Charakter
        if obj == body['obj']:
            character_meshes.append(obj)
            continue
        
        # Overlap-Analyse: Wie viel vom Mesh liegt innerhalb des Body-Zylinders?
        verts_inside = 0
        verts_total = info['verts']
        sample_step = max(1, verts_total // 500)  # Sample max 500 Vertices fuer Performance
        sampled = 0
        
        mesh_obj = info['obj']
        for i, v in enumerate(mesh_obj.data.vertices):
            if i % sample_step != 0:
                continue
            wc = mesh_obj.matrix_world @ v.co
            sampled += 1
            
            # Pruefe ob Vertex innerhalb des Body-Zylinders ist
            h = wc.z if body_up == 'Z' else wc.y
            x_dist = abs(wc.x - body_cx)
            
            in_height = body_bottom - body_height * 0.1 <= h <= body_top + body_height * 0.1
            in_width = x_dist <= torso_half_w * 2.0  # Grosszuegiger Bereich (200% Torso)
            
            if in_height and in_width:
                verts_inside += 1
        
        overlap_pct = (verts_inside / max(sampled, 1)) * 100
        
        # Flatness-Check
        is_flat = info['flatness'] > 30
        
        # Prop-Kriterien:
        # 1. Sehr wenig Overlap mit dem Body-Zylinder UND flach
        is_non_overlapping_flat = overlap_pct < 40 and is_flat
        
        # 2. Breit und flach (groesser als Body in 2 Dimensionen)
        is_platform = (info['sorted_dims'][0] > body['sorted_dims'][0] * 0.4 and
                      info['sorted_dims'][1] > body['sorted_dims'][1] * 0.4 and
                      info['flatness'] > 20 and
                      overlap_pct < 60)
        
        # 3. Sehr wenig Vertices UND gross
        is_simple_large = info['verts'] < 200 and overlap_pct < 50
        
        if is_non_overlapping_flat or is_platform or is_simple_large:
            prop_meshes.append(obj)
            reason = []
            if is_non_overlapping_flat: reason.append(f"non-overlapping flat ({overlap_pct:.0f}% overlap)")
            if is_platform: reason.append(f"platform-like")
            if is_simple_large: reason.append(f"simple+large")
            print(f"[INFO] Prop erkannt: '{obj.name}' ({info['verts']} Verts, "
                  f"Overlap={overlap_pct:.0f}%, Flatness={info['flatness']:.0f}, "
                  f"Grund: {', '.join(reason)})")
        else:
            character_meshes.append(obj)
            print(f"[INFO]   Charakter: '{obj.name}' ({info['verts']} Verts, Overlap={overlap_pct:.0f}%)")
    
    # Sicherheit: Wenn alles als Prop, alles als Charakter nehmen
    if not character_meshes:
        print("[WARN] Keine Charakter-Meshes erkannt, verwende alle Meshes")
        return meshes, []
    
    print(f"[INFO] Mesh-Klassifikation: {len(character_meshes)} Charakter, {len(prop_meshes)} Props")
    
    return character_meshes, prop_meshes


def _analyze_vertex_distribution(meshes, up_axis='Z'):
    """Analysiert die Vertex-Verteilung aller Meshes entlang der Hoehenachse.
    
    Returns dict mit Median-Positionen fuer Koerperregionen:
      - body_center_x, body_center_depth: Mitte des Koerpers horizontal
      - body_bottom, body_top: Unterkante / Oberkante
      - head_center, torso_center, hip_center: vertikale Mittelpunkte der Regionen
      - left_extent, right_extent: seitliche Ausdehnung (fuer Arme)
    """
    import mathutils
    
    # Alle Vertices sammeln (World Space)
    all_verts_height = []
    all_verts_x = []
    all_verts_depth = []
    
    for mesh_obj in meshes:
        for v in mesh_obj.data.vertices:
            wc = mesh_obj.matrix_world @ v.co
            all_verts_x.append(wc.x)
            if up_axis == 'Z':
                all_verts_height.append(wc.z)
                all_verts_depth.append(wc.y)
            else:
                all_verts_height.append(wc.y)
                all_verts_depth.append(wc.z)
    
    if not all_verts_height:
        return None
    
    all_verts_height.sort()
    all_verts_x.sort()
    all_verts_depth.sort()
    
    n = len(all_verts_height)
    
    # Percentile-basierte Analyse (robust gegen Outlier/Props)
    result = {
        'center_x': all_verts_x[n // 2],
        'center_depth': all_verts_depth[n // 2],
        'bottom': all_verts_height[int(n * 0.02)],     # 2. Perzentil (ignoriert Boden-Outlier)
        'top': all_verts_height[int(n * 0.98)],         # 98. Perzentil (ignoriert Spitzen)
        'left_extent': all_verts_x[int(n * 0.02)],
        'right_extent': all_verts_x[int(n * 0.98)],
        'depth_front': all_verts_depth[int(n * 0.02)],
        'depth_back': all_verts_depth[int(n * 0.98)],
    }
    result['height'] = result['top'] - result['bottom']
    
    return result


def _build_minimal_armature(meshes):
    """Minimales Humanoid-Armature fuer statische Modelle erstellen + Smart-Weight."""
    import mathutils
    _ensure_object_mode()

    # Schritt 1: Meshes klassifizieren (Charakter vs Props)
    character_meshes, prop_meshes = _classify_meshes(meshes)

    # Schritt 2: Body-Mesh finden (meiste Vertices) fuer Bone-Platzierung
    body_mesh = max(character_meshes, key=lambda m: len(m.data.vertices) if m.data and m.data.vertices else 0)
    print(f"[INFO] Body-Mesh fuer Bone-Platzierung: '{body_mesh.name}' ({len(body_mesh.data.vertices)} Verts)")

    # Up-Achse aus Body-Mesh bestimmen
    min_co = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_co = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    for v in body_mesh.data.vertices:
        wc = body_mesh.matrix_world @ v.co
        min_co.x = min(min_co.x, wc.x); max_co.x = max(max_co.x, wc.x)
        min_co.y = min(min_co.y, wc.y); max_co.y = max(max_co.y, wc.y)
        min_co.z = min(min_co.z, wc.z); max_co.z = max(max_co.z, wc.z)

    if min_co.x == float('inf'):
        min_co = mathutils.Vector((0, 0, 0))
        max_co = mathutils.Vector((0.5, 1.7, 0.3))

    height_y = max_co.y - min_co.y
    height_z = max_co.z - min_co.z
    up = 'Z' if height_z > height_y else 'Y'

    # Schritt 3: Vertex-Verteilung analysieren (NUR Body-Mesh fuer praezise Bone-Platzierung)
    vdist = _analyze_vertex_distribution([body_mesh], up)
    
    if vdist and vdist['height'] > 0.01:
        cx = vdist['center_x']
        cy = vdist['center_depth']
        base = vdist['bottom']
        height = vdist['height']
        body_width = vdist['right_extent'] - vdist['left_extent']
        print(f"[INFO] Vertex-Analyse: Hoehe={height:.2f}, Breite={body_width:.2f}, "
              f"Center=({cx:.3f},{cy:.3f}), Base={base:.3f}, Up={up}")
    else:
        # Fallback auf Bounding Box
        cx = (min_co.x + max_co.x) / 2
        cy = (min_co.y + max_co.y) / 2 if up == 'Z' else (min_co.z + max_co.z) / 2
        height = height_z if up == 'Z' else height_y
        base = min_co.z if up == 'Z' else min_co.y
        body_width = max_co.x - min_co.x
        if height < 0.01:
            height = 1.7
        print(f"[INFO] Fallback Bounds: Hoehe={height:.2f}, Center=({cx:.3f},{cy:.3f}), Up={up}")

    print(f"[INFO] Model-Bounds: min=({min_co.x:.3f},{min_co.y:.3f},{min_co.z:.3f}) "
          f"max=({max_co.x:.3f},{max_co.y:.3f},{max_co.z:.3f})")
    print(f"[INFO] Up-Achse: {up}, Charakter-Hoehe: {height:.2f}")

    # Armature erstellen
    arm_data = bpy.data.armatures.new("Generated_Armature")
    arm_obj = bpy.data.objects.new("Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.update()

    _select_only(arm_obj)
    bpy.ops.object.mode_set(mode='EDIT')

    bones = {}

    def pos(x, y, z):
        """Position je nach Up-Achse."""
        if up == 'Z':
            return mathutils.Vector((x, y, z))
        else:
            return mathutils.Vector((x, z, y))

    def add_bone(name, head, tail, parent=None):
        b = arm_data.edit_bones.new(name)
        b.head = head
        b.tail = tail
        if parent and parent in bones:
            b.parent = bones[parent]
        bones[name] = b

    # Proportionen (humanoid standard)
    hips_h = base + height * 0.48
    spine_h = base + height * 0.55
    chest_h = base + height * 0.65
    upper_chest_h = base + height * 0.72
    neck_h = base + height * 0.82
    head_h = base + height * 0.87
    head_top_h = base + height * 0.98

    add_bone("hips", pos(cx, cy, hips_h), pos(cx, cy, spine_h))
    add_bone("spine", pos(cx, cy, spine_h), pos(cx, cy, chest_h), "hips")
    add_bone("chest", pos(cx, cy, chest_h), pos(cx, cy, upper_chest_h), "spine")
    add_bone("upperChest", pos(cx, cy, upper_chest_h), pos(cx, cy, neck_h), "chest")
    add_bone("neck", pos(cx, cy, neck_h), pos(cx, cy, head_h), "upperChest")
    add_bone("head", pos(cx, cy, head_h), pos(cx, cy, head_top_h), "neck")

    # Eye Bones
    eye_width = height * 0.03
    eye_h = base + height * 0.92
    eye_depth = cy - height * 0.04 if up == 'Z' else cy
    eye_tail_off = max(height * 0.01, 0.02)
    add_bone("leftEye", pos(cx + eye_width, eye_depth, eye_h), pos(cx + eye_width, eye_depth - eye_tail_off, eye_h), "head")
    add_bone("rightEye", pos(cx - eye_width, eye_depth, eye_h), pos(cx - eye_width, eye_depth - eye_tail_off, eye_h), "head")

    # Arme - basierend auf tatsaechlicher Koerperbreite
    shoulder_width = max(body_width * 0.35, height * 0.08) if body_width > 0 else height * 0.12
    upper_arm_len = height * 0.14
    lower_arm_len = height * 0.13

    for side, sign in [("left", 1), ("right", -1)]:
        sx = cx + sign * shoulder_width
        ua_end = sx + sign * upper_arm_len
        la_end = ua_end + sign * lower_arm_len
        hand_end = la_end + sign * height * 0.03

        arm_h = upper_chest_h - height * 0.02

        add_bone(f"{side}Shoulder", pos(cx, cy, arm_h + height * 0.01), pos(sx, cy, arm_h), "upperChest")
        add_bone(f"{side}UpperArm", pos(sx, cy, arm_h), pos(ua_end, cy, arm_h), f"{side}Shoulder")
        add_bone(f"{side}LowerArm", pos(ua_end, cy, arm_h), pos(la_end, cy, arm_h), f"{side}UpperArm")
        add_bone(f"{side}Hand", pos(la_end, cy, arm_h), pos(hand_end, cy, arm_h), f"{side}LowerArm")

    # Beine
    leg_spread = max(body_width * 0.15, height * 0.04) if body_width > 0 else height * 0.06
    knee_h = base + height * 0.25
    foot_h = base + height * 0.03
    toe_offset = height * 0.04
    bone_tail_min = max(height * 0.01, 0.02)

    for side, sign in [("left", 1), ("right", -1)]:
        lx = cx + sign * leg_spread

        add_bone(f"{side}UpperLeg", pos(lx, cy, hips_h), pos(lx, cy, knee_h), "hips")
        add_bone(f"{side}LowerLeg", pos(lx, cy, knee_h), pos(lx, cy, foot_h), f"{side}UpperLeg")
        if up == 'Z':
            add_bone(f"{side}Foot", pos(lx, cy - toe_offset, foot_h), pos(lx, cy - toe_offset * 2, foot_h), f"{side}LowerLeg")
            add_bone(f"{side}Toes", pos(lx, cy - toe_offset * 2, foot_h), pos(lx, cy - toe_offset * 2.5, foot_h), f"{side}Foot")
        else:
            add_bone(f"{side}Foot", pos(lx, cy, foot_h), pos(lx, cy, foot_h - bone_tail_min), f"{side}LowerLeg")
            add_bone(f"{side}Toes", pos(lx, cy, foot_h - bone_tail_min), pos(lx, cy, foot_h - bone_tail_min * 2), f"{side}Foot")

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    print(f"[OK] Minimales Humanoid-Armature erstellt: {len(bones)} Bones")

    # Meshes mit Smart-Weight parenten (Charakter zuerst, dann Props)
    _parent_meshes_to_armature(arm_obj, character_meshes, prop_meshes, up_axis=up)

    return arm_obj


def _parent_meshes_to_armature(armature, character_meshes, prop_meshes=None, up_axis='Z'):
    """Meshes zum Armature parenten mit intelligenter Gewichtung.
    
    Strategie:
      1. Body-Mesh identifizieren (meiste Vertices unter den Charakter-Meshes)
      2. Body-Mesh per Anatomie-Region-Weights gewichten (Vertex-Hoehe -> Bone-Zuordnung)
      3. Alle anderen Charakter-Meshes: Weights vom Body per DataTransfer uebertragen
      4. Props: An naechsten Bone (meistens hips) binden
    """
    import mathutils
    _ensure_object_mode()

    if prop_meshes is None:
        prop_meshes = []
    
    all_meshes = list(character_meshes) + list(prop_meshes)
    
    # Bone-Positionen sammeln (Head-Position jedes Bones)
    bone_positions = {}
    for bone in armature.data.bones:
        # Head position in world space
        bone_positions[bone.name] = armature.matrix_world @ bone.head_local
    
    # Bone-Hoehen-Bereiche definieren (fuer Anatomie-basierte Zuweisung)
    bone_heights = {}
    for bname, bpos in bone_positions.items():
        h = bpos.z if up_axis == 'Z' else bpos.y
        bone_heights[bname] = h
    
    # 1. Body-Mesh identifizieren (meiste Vertices unter den Charakter-Meshes)
    body_mesh = None
    body_verts = 0
    for m in character_meshes:
        nv = len(m.data.vertices) if m.data and m.data.vertices else 0
        if nv > body_verts:
            body_verts = nv
            body_mesh = m
    
    other_char_meshes = [m for m in character_meshes if m != body_mesh]
    
    if body_mesh:
        print(f"[INFO] Body-Mesh: '{body_mesh.name}' ({body_verts} Vertices)")
        print(f"[INFO] Weitere Charakter-Meshes: {len(other_char_meshes)}")
        if prop_meshes:
            print(f"[INFO] Props: {len(prop_meshes)} ({', '.join(p.name for p in prop_meshes)})")
    
    # 2. Alle Meshes zum Armature parenten
    for mesh_obj in all_meshes:
        world_mat = mesh_obj.matrix_world.copy()
        mesh_obj.parent = armature
        mesh_obj.matrix_world = world_mat
        
        has_arm_mod = any(m.type == 'ARMATURE' for m in mesh_obj.modifiers)
        if not has_arm_mod:
            mod = mesh_obj.modifiers.new("Armature", 'ARMATURE')
            mod.object = armature
    
    # 3. Body-Mesh mit anatomie-basierter Gewichtung
    if body_mesh:
        _assign_anatomical_weights(body_mesh, armature, bone_positions, up_axis)
        
        # Pruefen ob Body-Mesh brauchbare Weights hat
        has_good_weights = False
        if body_mesh.vertex_groups:
            for vg in body_mesh.vertex_groups:
                # Pruefe ob mind. 10% der Vertices gewichtet sind
                weighted = 0
                for v in body_mesh.data.vertices:
                    for g in v.groups:
                        if g.group == vg.index and g.weight > 0.01:
                            weighted += 1
                            break
                if weighted > len(body_mesh.data.vertices) * 0.1:
                    has_good_weights = True
                    break
        
        if not has_good_weights:
            print(f"[WARN] Anatomische Weights unzureichend, verwende Distance-Based Fallback")
            _assign_weights_by_distance(body_mesh, armature, bone_positions, up_axis)
    
    # 4. Andere Charakter-Meshes: Weights vom Body uebertragen
    for mesh_obj in other_char_meshes:
        if body_mesh:
            success = _transfer_weights_from_body(body_mesh, mesh_obj, armature)
            if not success:
                _assign_weights_by_distance(mesh_obj, armature, bone_positions, up_axis)
        else:
            _assign_weights_by_distance(mesh_obj, armature, bone_positions, up_axis)
    
    # 5. Props: Einfach an hips binden (bewegen sich mit dem Koerper)
    for mesh_obj in prop_meshes:
        # Alle Vertices zu hips
        hips_vg = mesh_obj.vertex_groups.get("hips")
        if not hips_vg:
            hips_vg = mesh_obj.vertex_groups.new(name="hips")
        all_indices = [v.index for v in mesh_obj.data.vertices]
        if all_indices:
            hips_vg.add(all_indices, 1.0, 'REPLACE')
        print(f"[OK] Prop '{mesh_obj.name}': alle {len(all_indices)} Vertices -> hips")
    
    total = len(all_meshes)
    print(f"[OK] Alle {total} Meshes zum Armature geparented + gewichtet")
    _ensure_object_mode()


def _assign_anatomical_weights(mesh_obj, armature, bone_positions, up_axis='Z'):
    """Seiten-bewusste anatomische Gewichtung: Jeder Vertex wird den Bones zugewiesen,
    die auf der gleichen Koerperseite liegen und geometrisch am naechsten sind."""
    import mathutils
    
    # Alte Vertex Groups entfernen
    mesh_obj.vertex_groups.clear()
    
    # Vertex Groups fuer alle Bones erstellen
    vg_map = {}
    for bone in armature.data.bones:
        vg = mesh_obj.vertex_groups.new(name=bone.name)
        vg_map[bone.name] = vg
    
    # Armature-Zentrum berechnen (X-Achse)
    all_bone_x = []
    for bone in armature.data.bones:
        head_world = armature.matrix_world @ bone.head_local
        all_bone_x.append(head_world.x)
    armature_center_x = sum(all_bone_x) / len(all_bone_x) if all_bone_x else 0.0
    
    # Bone-Infos mit Seiten-Klassifikation sammeln
    # Links/Rechts/Mitte bestimmen + welche Bones NICHT gemischt werden duerfen
    SIDE_BONES = {
        'left': ['leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
                 'leftUpperArm', 'leftLowerArm', 'leftHand', 'leftShoulder', 'leftEye'],
        'right': ['rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes',
                  'rightUpperArm', 'rightLowerArm', 'rightHand', 'rightShoulder', 'rightEye'],
    }
    
    bone_info = []
    for bone in armature.data.bones:
        head_world = armature.matrix_world @ bone.head_local
        tail_world = armature.matrix_world @ bone.tail_local
        center = (head_world + tail_world) / 2
        length = (tail_world - head_world).length
        
        # Seite bestimmen
        side = 'center'
        if bone.name in SIDE_BONES['left']:
            side = 'left'
        elif bone.name in SIDE_BONES['right']:
            side = 'right'
        
        bone_info.append({
            'name': bone.name,
            'center': center,
            'head': head_world,
            'tail': tail_world,
            'length': length,
            'side': side,
        })
    
    # Jeden Vertex gewichten (seiten-bewusst)
    weighted_count = 0
    for v in mesh_obj.data.vertices:
        wc = mesh_obj.matrix_world @ v.co
        
        # Seite des Vertex bestimmen (relativ zum Armature-Zentrum)
        x_offset = wc.x - armature_center_x
        # Toleranzzone: Vertices nah an der Mitte koennen beide Seiten nutzen
        side_tolerance = abs(x_offset) < (armature.data.bones[0].length * 0.3 if armature.data.bones else 0.01)
        vertex_side = 'center' if side_tolerance else ('left' if x_offset > 0 else 'right')
        
        # Distanz zu jedem Bone berechnen mit Seiten-Penalty
        distances = []
        for bi in bone_info:
            d = _point_to_segment_dist(wc, bi['head'], bi['tail'])
            
            # Seiten-Penalty: Erhoehe Distanz fuer gegenueberliegende Seite
            if bi['side'] != 'center' and vertex_side != 'center':
                if bi['side'] != vertex_side:
                    d *= 10.0  # Starker Penalty fuer falsche Seite
            
            distances.append((bi['name'], d))
        
        distances.sort(key=lambda x: x[1])
        
        # Top 3 naechsten Bones mit Smooth Falloff
        min_dist = max(distances[0][1], 0.001) if distances else 1.0
        
        weights = []
        for bname, dist in distances[:3]:
            dist = max(dist, 0.001)
            relative_dist = dist / min_dist
            if relative_dist > 4.0:
                break
            w = 1.0 / (relative_dist * relative_dist)
            weights.append((bname, w))
        
        # Normalisieren
        total_w = sum(w for _, w in weights)
        if total_w > 0:
            for bname, w in weights:
                normalized_w = w / total_w
                if normalized_w > 0.01:
                    vg_map[bname].add([v.index], normalized_w, 'REPLACE')
            weighted_count += 1
    
    print(f"[OK] Anatomische Weights fuer '{mesh_obj.name}': {weighted_count}/{len(mesh_obj.data.vertices)} Vertices -> {len(bone_info)} Bones")


def _point_to_segment_dist(point, seg_start, seg_end):
    """Berechnet die kuerzeste Distanz von einem Punkt zu einem Liniensegment."""
    import mathutils
    
    line = seg_end - seg_start
    line_len_sq = line.length_squared
    
    if line_len_sq < 0.000001:
        return (point - seg_start).length
    
    t = max(0, min(1, (point - seg_start).dot(line) / line_len_sq))
    projection = seg_start + t * line
    return (point - projection).length


def _transfer_weights_from_body(body_mesh, target_mesh, armature):
    """Vertex Weights vom Body-Mesh auf ein anderes Mesh uebertragen via DataTransfer Modifier."""
    _ensure_object_mode()
    
    try:
        # Vertex Groups auf Target erstellen (gleiche wie Body)
        for vg in body_mesh.vertex_groups:
            if vg.name not in [g.name for g in target_mesh.vertex_groups]:
                target_mesh.vertex_groups.new(name=vg.name)
        
        # DataTransfer Modifier
        _select_only(target_mesh)
        bpy.context.view_layer.objects.active = target_mesh
        
        mod = target_mesh.modifiers.new("WeightTransfer", 'DATA_TRANSFER')
        mod.object = body_mesh
        mod.use_vert_data = True
        mod.data_types_verts = {'VGROUP_WEIGHTS'}
        mod.vert_mapping = 'POLYINTERP_NEAREST'  # Naechstes Polygon interpoliert
        
        # Modifier anwenden
        bpy.ops.object.modifier_apply(modifier=mod.name)
        
        # Pruefen wie viele Vertices Weights bekommen haben
        weighted = 0
        for v in target_mesh.data.vertices:
            if v.groups:
                weighted += 1
        
        pct = (weighted / len(target_mesh.data.vertices) * 100) if target_mesh.data.vertices else 0
        print(f"[OK] Weight-Transfer '{body_mesh.name}' -> '{target_mesh.name}': "
              f"{weighted}/{len(target_mesh.data.vertices)} Vertices ({pct:.0f}%)")
        
        return weighted > 0
        
    except Exception as e:
        print(f"[WARN] Weight-Transfer fehlgeschlagen fuer '{target_mesh.name}': {e}")
        return False


def _assign_weights_by_distance(mesh_obj, armature, bone_positions, up_axis='Z'):
    """Fallback-Gewichtung mit Seiten-Awareness basierend auf Punkt-zu-Bone-Segment Distanz."""
    import mathutils
    
    if not mesh_obj.data.vertices:
        return
    
    # Alte Vertex Groups entfernen
    mesh_obj.vertex_groups.clear()
    
    # Vertex Groups fuer alle Bones erstellen
    vg_map = {}
    for bone in armature.data.bones:
        vg = mesh_obj.vertex_groups.new(name=bone.name)
        vg_map[bone.name] = vg
    
    # Armature-Zentrum berechnen
    all_bone_x = [armature.matrix_world @ bone.head_local for bone in armature.data.bones]
    armature_center_x = sum(p.x for p in all_bone_x) / len(all_bone_x) if all_bone_x else 0.0
    
    LEFT_BONES = {'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
                  'leftUpperArm', 'leftLowerArm', 'leftHand', 'leftShoulder', 'leftEye'}
    RIGHT_BONES = {'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes',
                   'rightUpperArm', 'rightLowerArm', 'rightHand', 'rightShoulder', 'rightEye'}
    
    # Bone-Segmente mit Seiten-Info sammeln
    bone_segments = []
    for bone in armature.data.bones:
        head = armature.matrix_world @ bone.head_local
        tail = armature.matrix_world @ bone.tail_local
        side = 'left' if bone.name in LEFT_BONES else ('right' if bone.name in RIGHT_BONES else 'center')
        bone_segments.append((bone.name, head, tail, side))
    
    for v in mesh_obj.data.vertices:
        wc = mesh_obj.matrix_world @ v.co
        
        # Vertex-Seite bestimmen
        x_off = wc.x - armature_center_x
        tol = armature.data.bones[0].length * 0.3 if armature.data.bones else 0.01
        v_side = 'center' if abs(x_off) < tol else ('left' if x_off > 0 else 'right')
        
        distances = []
        for bname, head, tail, bside in bone_segments:
            d = _point_to_segment_dist(wc, head, tail)
            # Seiten-Penalty
            if bside != 'center' and v_side != 'center' and bside != v_side:
                d *= 10.0
            distances.append((bname, d))
        
        distances.sort(key=lambda x: x[1])
        
        min_dist = max(distances[0][1], 0.001)
        weights = []
        for bname, dist in distances[:3]:
            rel = max(dist, 0.001) / min_dist
            if rel > 4.0:
                break
            w = 1.0 / (rel * rel)
            weights.append((bname, w))
        
        total_w = sum(w for _, w in weights)
        if total_w > 0:
            for bname, w in weights:
                nw = w / total_w
                if nw > 0.01:
                    vg_map[bname].add([v.index], nw, 'REPLACE')
    
    print(f"[OK] Distance-Weights fuer '{mesh_obj.name}': {len(mesh_obj.data.vertices)} Vertices -> {len(bone_segments)} Bones")


# ======================= VALIDATION =======================

def validate_vrm_model(armature):
    """VRM-Validierung vor Export."""
    vrm_ext = armature.data.vrm_addon_extension
    is_vrm1 = getattr(vrm_ext, "spec_version", "0.0") == "1.0"

    print(f"\n{'='*50}")
    print(f"  VRM VALIDIERUNG (spec_version={getattr(vrm_ext, 'spec_version', '?')})")
    print(f"{'='*50}")

    arm_count = sum(1 for o in bpy.data.objects if o.type == 'ARMATURE')
    mesh_count = sum(1 for o in bpy.data.objects if o.type == 'MESH')
    print(f"[CHECK] Armatures: {arm_count} (braucht genau 1)")
    print(f"[CHECK] Meshes: {mesh_count} (braucht mindestens 1)")

    if mesh_count == 0:
        print("[ERROR] KEIN MESH GEFUNDEN!")
    if arm_count > 1:
        print("[ERROR] MEHRERE ARMATURES!")

    # Bone mapping check
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
                level = "OK" if bone_name else "FEHLT"
                print(f"  [{level}] {req_bone} -> {bone_name or 'NICHT ZUGEWIESEN'}")
    else:
        human_bones_list = vrm_ext.vrm0.humanoid.human_bones
        print(f"\n[CHECK] VRM 0.x Human Bones ({len(human_bones_list)} Eintraege):")
        mapped_vrm_bones = set()
        for hb in human_bones_list:
            bone_name = hb.node.bone_name if hasattr(hb.node, 'bone_name') else '?'
            vrm_name = hb.bone if hasattr(hb, 'bone') else '?'
            actual_exists = bone_name in [b.name for b in armature.data.bones] if bone_name else False
            level = "OK" if (bone_name and actual_exists) else "FEHLT"
            print(f"  [{level}] {vrm_name} -> {bone_name}")
            if vrm_name:
                mapped_vrm_bones.add(vrm_name)
        for req in REQUIRED_BONES:
            if req not in mapped_vrm_bones:
                print(f"  [FEHLT!] Pflicht-Bone '{req}' nicht gemappt!")

    # Meta check
    if is_vrm1:
        meta = vrm_ext.vrm1.meta
        print(f"\n[CHECK] Meta: name={getattr(meta, 'vrm_name', '?')}")
    else:
        meta = vrm_ext.vrm0.meta
        print(f"\n[CHECK] Meta: title={getattr(meta, 'title', '?')}, author={getattr(meta, 'author', '?')}")

    # Material check
    print(f"\n[CHECK] Materialien:")
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if not obj.data.materials:
                print(f"[ERROR] Mesh '{obj.name}' hat KEIN Material!")
            else:
                for i, mat in enumerate(obj.data.materials):
                    if mat is None:
                        print(f"[ERROR] Mesh '{obj.name}' Material-Slot {i} leer!")

    # Duplicate bone check (VRM0)
    if not is_vrm1:
        bone_names_used = []
        for hb in vrm_ext.vrm0.humanoid.human_bones:
            bn = hb.node.bone_name if hasattr(hb.node, 'bone_name') else ''
            if bn in bone_names_used:
                print(f"[ERROR] Bone '{bn}' wird MEHRFACH verwendet!")
            bone_names_used.append(bn)

    print(f"{'='*50}\n")


# ======================= MAIN CONVERSION =======================

def convert(input_path: str, output_path: str, job_id: str, options: dict = None):
    """Hauptkonvertierungs-Pipeline: GLB/glTF -> VRM."""
    if options is None:
        options = {}

    vrm_version = options.get("vrm_version", "0.x")
    do_normalize = options.get("normalize", True)
    do_expressions = options.get("expression_mapping", "auto") != "none"
    do_eye_tracking = options.get("eye_tracking", True)
    do_lipsync = options.get("lipsync", True)
    print(f"[INFO] Optionen: VRM={vrm_version}, Normalize={do_normalize}, "
          f"Expressions={do_expressions}, EyeTracking={do_eye_tracking}, Lipsync={do_lipsync}")

    # VRM Addon laden
    status("VRM Add-on wird geladen...")
    module_name = ensure_vrm_addon()
    force_register_vrm(module_name)

    # VRM Operatoren pruefen
    vrm_ops = {}
    for cat_name in ["export_scene", "import_scene", "vrm"]:
        cat = getattr(bpy.ops, cat_name, None)
        if cat:
            names = [n for n in dir(cat) if "vrm" in n.lower()]
            if names:
                vrm_ops[cat_name] = names
    print(f"[INFO] VRM Operatoren: {vrm_ops}")

    if "export_scene" not in vrm_ops or "vrm" not in vrm_ops["export_scene"]:
        raise RuntimeError("VRM Export-Operator nicht verfuegbar!")

    # GLB importieren
    status("GLB/glTF wird importiert...")
    progress(20)
    import_glb(input_path)
    progress(30)

    # 3D Viewport sicherstellen
    found_3d = False
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                found_3d = True
                break
    if not found_3d:
        for window in bpy.context.window_manager.windows:
            if window.screen.areas:
                window.screen.areas[0].type = 'VIEW_3D'
                print("[INFO] Area zu VIEW_3D umgewandelt")
                break

    # Szene aufraeumen
    status("Szene wird aufgeraeumt...")
    clean_scene_for_export()
    progress(35)

    # Armature finden
    armature = find_armature()
    if not armature:
        raise RuntimeError("Kein Armature/Rig nach Import gefunden!")
    print(f"[INFO] Armature: {armature.name} ({len(armature.data.bones)} Bones)")

    # Bones listen
    status("Bones werden analysiert...")
    progress(40)
    print("[INFO] Alle Bones im Armature:")
    for bone in armature.data.bones:
        parent_name = bone.parent.name if bone.parent else "ROOT"
        norm = _normalize_for_match(bone.name)
        print(f"[INFO]   - {bone.name} (parent: {parent_name}) [normalized: {norm}]")

    # VRM Version setzen
    status("VRM-Version wird konfiguriert...")
    setup_vrm_version(armature, vrm_version)

    # Mesh Parenting
    ensure_mesh_parenting(armature)

    # Normalisieren
    if do_normalize:
        status("Model wird normalisiert...")
        progress(45)
        normalize_model(armature)
    else:
        print("[INFO] Normalisierung uebersprungen")
        progress(45)

    # Rest Pose
    ensure_rest_pose(armature)
    progress(50)

    # Bone Mapping
    status("Bone-Mapping wird erstellt...")
    progress(55)
    bone_mapping = auto_map_bones(armature)

    print(f"[INFO] Bone-Mapping ({len(bone_mapping)} gefunden):")
    for vrm_name, blender_bone in sorted(bone_mapping.items()):
        print(f"[INFO]   {vrm_name} -> {blender_bone}")

    missing = [b for b in REQUIRED_BONES if b not in bone_mapping]
    if missing:
        print(f"[WARN] Fehlende Pflicht-Bones: {missing}")

    # Bone Mapping anwenden
    status("Bone-Mapping wird angewendet...")
    progress(60)
    apply_bone_mapping(armature, bone_mapping)

    # Shape Keys + Expressions + Viseme + Blink + LookAt
    status("Shape Keys werden analysiert...")
    progress(65)
    shape_keys = collect_shape_keys(armature)
    total_keys = sum(len(v) for v in shape_keys.values())
    print(f"[INFO] Shape Keys gesamt: {total_keys} auf {len(shape_keys)} Meshes")

    for mesh_name, keys in shape_keys.items():
        print(f"[INFO]   {mesh_name}: {', '.join(keys[:15])}" + (" ..." if len(keys) > 15 else ""))

    expr_mapping = {}
    if total_keys > 0 and (do_expressions or do_lipsync):
        status("Expressions + Viseme + Blink werden gemappt...")
        progress(70)
        expr_mapping = auto_map_expressions(shape_keys)

        # Kategorisiert ausgeben
        emotions = {k: v for k, v in expr_mapping.items() if k in EXPRESSION_PATTERNS}
        visemes = {k: v for k, v in expr_mapping.items() if k in VISEME_PATTERNS}
        ext_visemes = {k: v for k, v in expr_mapping.items() if k in EXTENDED_VISEME_PATTERNS}
        blinks = {k: v for k, v in expr_mapping.items() if k in BLINK_PATTERNS}
        lookats = {k: v for k, v in expr_mapping.items() if k in LOOKAT_BLENDSHAPE_PATTERNS}

        print(f"[INFO] === Expression-Mapping Zusammenfassung ===")
        print(f"[INFO]   Emotionen:       {len(emotions)} ({', '.join(emotions.keys()) if emotions else 'keine'})")
        print(f"[INFO]   Viseme (AIUEO):  {len(visemes)} ({', '.join(visemes.keys()) if visemes else 'keine'})")
        print(f"[INFO]   Extended Viseme: {len(ext_visemes)} ({', '.join(ext_visemes.keys()) if ext_visemes else 'keine'})")
        print(f"[INFO]   Blink:           {len(blinks)} ({', '.join(blinks.keys()) if blinks else 'keine'})")
        print(f"[INFO]   LookAt Shapes:   {len(lookats)} ({', '.join(lookats.keys()) if lookats else 'keine'})")

        if do_lipsync and not visemes:
            print("[WARN] Lipsync aktiviert aber keine Viseme-Shape-Keys (A/I/U/E/O) gefunden!")
            print("[WARN] Lipsync benoetigt mindestens 'aa', 'ih', 'ou', 'ee', 'oh' Shape Keys")

        apply_expression_mapping(armature, expr_mapping)
        progress(75)
    else:
        print("[INFO] Keine Shape Keys oder Expressions deaktiviert")
        progress(75)

    # Eye Tracking Setup
    if do_eye_tracking:
        status("Eye-Tracking wird konfiguriert...")
        progress(78)
        setup_eye_tracking(armature, bone_mapping, expr_mapping)
    else:
        print("[INFO] Eye-Tracking uebersprungen")

    # VRM Meta
    status("VRM Meta-Daten werden gesetzt...")
    progress(80)
    set_vrm_meta(armature, os.path.basename(input_path), vrm_version)

    # Materialien pruefen
    status("Materialien werden geprueft...")
    ensure_mesh_materials()

    # Validierung
    status("VRM wird validiert...")
    progress(82)
    validate_vrm_model(armature)

    # Export
    _ensure_object_mode()
    _select_only(armature)

    status("VRM wird exportiert...")
    progress(85)
    print(f"[INFO] Exportiere nach: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
        print(f"[WARN] Export fehlgeschlagen: {e}")
        print("[INFO] Versuche minimalen Export...")
        try:
            minimal_kwargs = {"filepath": output_path, "ignore_warning": True}
            if ctx:
                with bpy.context.temp_override(**ctx):
                    result = bpy.ops.export_scene.vrm(**minimal_kwargs)
            else:
                result = bpy.ops.export_scene.vrm(**minimal_kwargs)
            if result != {"FINISHED"}:
                # Fallback: GLB export
                print(f"[WARN] Minimaler Export fehlgeschlagen: {result}")
                print("[INFO] Versuche GLB-Fallback...")
                gltf_path = output_path.replace('.vrm', '.glb')
                bpy.ops.export_scene.gltf(
                    filepath=gltf_path,
                    export_format='GLB',
                    use_selection=False,
                    export_apply=True,
                )
                if os.path.exists(gltf_path):
                    if gltf_path != output_path:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        os.rename(gltf_path, output_path)
                    print("[INFO] GLB-Export als VRM gespeichert (Fallback)")
                else:
                    raise RuntimeError(f"Export fehlgeschlagen: {result}")
        except TypeError:
            result = bpy.ops.export_scene.vrm(filepath=output_path)

    progress(95)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        status(f"VRM erfolgreich erstellt ({size_mb:.1f} MB)")
        print(f"[OK] VRM erstellt: {output_path} ({size_mb:.1f} MB)")

        # Zusammenfassung
        print("\n=== Zusammenfassung ===")
        print(f"  Armature: {armature.name} ({len(armature.data.bones)} Bones)")
        print(f"  Bone-Mapping: {len(bone_mapping)} VRM Bones")
        print(f"  Shape Keys: {total_keys}")
        if expr_mapping:
            vis_count = sum(1 for k in expr_mapping if k in VISEME_PATTERNS or k in EXTENDED_VISEME_PATTERNS)
            emo_count = sum(1 for k in expr_mapping if k in EXPRESSION_PATTERNS)
            eye_count = sum(1 for k in expr_mapping if k in BLINK_PATTERNS or k in LOOKAT_BLENDSHAPE_PATTERNS)
            print(f"  Emotionen: {emo_count}")
            print(f"  Viseme (Lipsync): {vis_count}")
            print(f"  Eye/Blink: {eye_count}")
        has_eyes = "leftEye" in bone_mapping and "rightEye" in bone_mapping
        print(f"  Eye-Bones: {'ja' if has_eyes else 'nein'}")
        print(f"  Output: {output_path}")
        print(f"  Groesse: {size_mb:.1f} MB")
    else:
        raise RuntimeError("VRM-Datei wurde nicht erstellt!")

    progress(100)


# ======================= ENTRY POINT =======================

if __name__ == "__main__":
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  GLB/glTF -> VRM Converter")
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
