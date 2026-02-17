# app.py (FULL AUTO + FORCE REGISTER)
# Run (PowerShell):
#   & "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" -b --factory-startup -P .\app.py
#
# Folders:
#   ./glb/  -> input .glb
#   ./vrm/  -> output .vrm
#   ./vrm_addon.zip -> VRM_Addon_for_Blender-3_19_4.zip (renamed is fine)

import os
import sys
import importlib
import traceback
import bpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "glb")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "vrm")
VRM_ADDON_ZIP = os.path.join(SCRIPT_DIR, "vrm_addon.zip")


def _blender_addons_dir():
    p = bpy.utils.user_resource("SCRIPTS", path="addons")
    if p:
        os.makedirs(p, exist_ok=True)
    return p


def ensure_vrm_addon_installed_and_enabled():
    """
    Installiert & aktiviert das VRM Add-on aus vrm_addon.zip.
    (Fix: '-' im Ordnernamen wird zu '_' umbenannt.)
    Gibt den finalen Modulnamen zurück.
    """
    import addon_utils

    # 1) Wenn schon vorhanden (egal enabled oder nicht), merken wir uns den Namen
    existing = None
    for mod in addon_utils.modules():
        name = getattr(mod, "__name__", "")
        if "VRM_Addon_for_Blender" in name:
            existing = name
            enabled, _loaded = addon_utils.check(name)
            if not enabled:
                print(f"[INFO] VRM add-on found but not enabled: {name} -> enabling")
                bpy.ops.preferences.addon_enable(module=name)
                bpy.ops.wm.save_userpref()
            else:
                print(f"[OK] VRM add-on already enabled: {name}")
            return name

    # 2) Nicht vorhanden -> installiere aus ZIP
    if not os.path.exists(VRM_ADDON_ZIP):
        raise RuntimeError(f"vrm_addon.zip not found: {VRM_ADDON_ZIP}")

    print("[INFO] Installing VRM Add-on from zip...")
    bpy.ops.preferences.addon_install(filepath=VRM_ADDON_ZIP)

    addons_dir = _blender_addons_dir()
    if not addons_dir:
        raise RuntimeError("Could not locate Blender user addons directory.")

    # 3) Kandidaten suchen
    candidates = []
    for entry in os.listdir(addons_dir):
        full = os.path.join(addons_dir, entry)
        if os.path.isdir(full) and "VRM_Addon_for_Blender" in entry:
            candidates.append(entry)

    if not candidates:
        raise RuntimeError("VRM add-on folder not found after installation.")

    candidates.sort(key=len, reverse=True)
    folder = candidates[0]

    # 4) '-' fixen
    fixed_folder = folder.replace("-", "_")
    if fixed_folder != folder:
        src = os.path.join(addons_dir, folder)
        dst = os.path.join(addons_dir, fixed_folder)
        if not os.path.exists(dst):
            print(f"[INFO] Renaming addon folder: {folder} -> {fixed_folder}")
            os.rename(src, dst)
        folder = fixed_folder

    module_name = folder
    print(f"[INFO] Enabling VRM Add-on module: {module_name}")
    bpy.ops.preferences.addon_enable(module=module_name)
    bpy.ops.wm.save_userpref()
    print("[OK] VRM Add-on installed & enabled.")

    return module_name


def force_import_and_register(module_name: str):
    """
    WICHTIGER FIX:
    Manche Setups aktivieren das Add-on, aber registrieren die Operatoren nicht.
    Hier importieren wir das Modul explizit und rufen register() auf.
    Falls Translations schon registriert sind, wird erst unregister() aufgerufen.
    """
    addons_dir = _blender_addons_dir()
    if not addons_dir:
        raise RuntimeError("No addons_dir")

    # Parent von addons_dir muss in sys.path sein, damit import klappt:
    parent = os.path.dirname(addons_dir)
    if parent not in sys.path:
        sys.path.append(parent)

    try:
        mod = importlib.import_module(module_name)
        mod = importlib.reload(mod)

        if hasattr(mod, "register"):
            # Erst unregister() um doppelte Registrierung zu vermeiden
            if hasattr(mod, "unregister"):
                try:
                    mod.unregister()
                    print(f"[INFO] unregister() executed for: {module_name}")
                except Exception:
                    pass

            try:
                mod.register()
                print(f"[OK] Forced register() executed for: {module_name}")
            except Exception as e:
                if "translations" in str(e).lower():
                    try:
                        bpy.app.translations.unregister(module_name)
                    except Exception:
                        pass
                    try:
                        mod.register()
                        print(f"[OK] register() succeeded after clearing translations")
                    except Exception:
                        print(f"[FAIL] register() crashed for {module_name}:")
                        traceback.print_exc()
                else:
                    print(f"[FAIL] register() crashed for {module_name}:")
                    traceback.print_exc()
        else:
            print(f"[WARN] Module has no register(): {module_name}")

    except Exception:
        print(f"[FAIL] Could not import module {module_name}:")
        traceback.print_exc()


def list_vrm_ops():
    found = {}
    for cat_name in ["export_scene", "wm", "object"]:
        cat = getattr(bpy.ops, cat_name, None)
        if not cat:
            continue
        names = [n for n in dir(cat) if "vrm" in n.lower()]
        if names:
            found[cat_name] = names
    return found


def reset_scene():
    """Scene leeren ohne Factory-Reset (behaelt Addons aktiv)."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=True)
    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                       bpy.data.images, bpy.data.armatures, bpy.data.cameras,
                       bpy.data.lights, bpy.data.actions, bpy.data.node_groups,
                       bpy.data.curves, bpy.data.collections]:
        for block in list(block_type):
            block_type.remove(block)
    scene = bpy.context.scene
    for col in list(scene.collection.children):
        scene.collection.children.unlink(col)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def import_glb(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    result = bpy.ops.import_scene.gltf(filepath=path)
    if result != {"FINISHED"}:
        raise RuntimeError(f"GLB import failed: {result}")


def find_armature():
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE" and o.data and len(o.data.bones) > 0]
    if not armatures:
        return None
    armatures.sort(key=lambda a: len(a.data.bones), reverse=True)
    return armatures[0]


def normalize_model(arm):
    """Model zentrieren: Fuesse auf Y=0, horizontal zentriert, Transforms applied."""
    import mathutils

    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = arm
    try:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        print("[INFO] Transforms applied (rotation + scale)")
    except Exception as e:
        print(f"[WARN] transform_apply: {e}")

    all_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not all_meshes:
        all_meshes = [arm]

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
    print(f"[INFO] Centering offset: ({offset.x:.3f}, {offset.y:.3f}, {offset.z:.3f})")

    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.location += offset

    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active = arm
    try:
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
        print("[INFO] Location applied (centered, feet on floor)")
    except Exception as e:
        print(f"[WARN] location apply: {e}")

    bpy.ops.object.select_all(action='DESELECT')
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm


def export_vrm(path: str):
    # Nach force_register sollte das existieren:
    result = bpy.ops.export_scene.vrm(filepath=path)
    if result != {"FINISHED"}:
        raise RuntimeError(f"VRM export failed: {result}")


def convert_all():
    if not os.path.exists(INPUT_DIR):
        raise RuntimeError(f"Input folder not found: {INPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".glb")]
    if not files:
        print(f"[INFO] No .glb files found in: {INPUT_DIR}")
        return

    ok = fail = skip = 0

    for f in files:
        glb_path = os.path.join(INPUT_DIR, f)
        vrm_path = os.path.join(OUTPUT_DIR, os.path.splitext(f)[0] + ".vrm")

        print(f"\n[INFO] Converting: {f}")
        try:
            reset_scene()
            import_glb(glb_path)

            arm = find_armature()
            if not arm:
                print("[SKIP] No ARMATURE (rig) found.")
                skip += 1
                continue

            print(f"[INFO] Found armature: {arm.name} (bones: {len(arm.data.bones)})")

            # Model zentrieren und ausrichten
            normalize_model(arm)

            # active selection
            bpy.ops.object.select_all(action="DESELECT")
            arm.select_set(True)
            bpy.context.view_layer.objects.active = arm

            export_vrm(vrm_path)
            print(f"[OK] {vrm_path}")
            ok += 1

        except Exception as e:
            print(f"[FAIL] {f}: {e}")
            traceback.print_exc()
            fail += 1

    print("\n=== Summary ===")
    print("OK:", ok)
    print("SKIP:", skip)
    print("FAIL:", fail)
    print("Output:", OUTPUT_DIR)


def main():
    module_name = ensure_vrm_addon_installed_and_enabled()
    force_import_and_register(module_name)

    ops = list_vrm_ops()
    print("[DEBUG] VRM operators now:", ops if ops else "(none)")

    if not ops:
        raise RuntimeError(
            "VRM operators are still missing after forced register(). "
            "Then the add-on is not compatible with this Blender build or crashed during register(). "
            "Check the traceback above."
        )

    convert_all()
    print("\nAll done.")


if __name__ == "__main__":
    main()
