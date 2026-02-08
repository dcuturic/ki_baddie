from tkinter import *
from tkinter import messagebox
from tkinter import simpledialog
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse
import keyboard
import shutil
import math
import os
import json

# -------------------------
# App setup
# -------------------------
root = Tk()
root.resizable(0, 0)
root.title("VroidPoser v1.0")
root.geometry("910x500+500+200")

POSES_DIR = "Poses"
MOVES_DIR = "Moves"
POSE_EXTS = (".txt", ".json")

os.makedirs("Resources", exist_ok=True)
os.makedirs(POSES_DIR, exist_ok=True)
os.makedirs(MOVES_DIR, exist_ok=True)

try:
    root.iconphoto(False, PhotoImage(file="Resources/icon.png"))
except Exception:
    pass

canvas = Canvas(root, width=910, height=500)
canvas.place(relx=0, rely=0)

ip = ""
osc = 0
port = 0

radius = 15
joyRange = 80
deadzone = 4
rotTarget = None
target = None

hotkeys = {}
listen = False
pressed = []

checkVar = IntVar()
staging = []
travel = [0, 0, 0]
destination = [0, 0, 0]
offset = [0, 0, 0]

L_LEG = {"LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes"}
R_LEG = {"RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"}
BODY = {"RightEye", "LeftEye", "Head", "Neck", "Chest", "Spine", "Hips"}

rArm = ["RightHand", "RightLowerArm", "RightUpperArm", "RightShoulder"]
lArm = ["LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand"]

rHand = ["RightThumbIntermediate", "RightThumbDistal", "RightIndexProximal", "RightIndexIntermediate",
         "RightIndexDistal", "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
         "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
         "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal"]

lHand = ["LeftThumbIntermediate", "LeftThumbDistal", "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
         "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
         "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
         "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal"]

rLeg = ["RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"]
lLeg = ["LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes"]
body = ["RightEye", "LeftEye", "Head", "Neck", "Chest", "Spine", "Hips"]

handR = [[290, 170], [290, 110], [230, 170], [230, 110], [230, 50], [170, 170], [170, 110], [170, 50],
         [110, 170], [110, 110], [110, 50], [50, 170], [50, 110], [50, 50]]
handL = [[410, 170], [410, 110], [470, 170], [470, 110], [470, 50], [530, 170], [530, 110], [530, 50],
         [590, 170], [590, 110], [590, 50], [650, 170], [650, 110], [650, 50]]

armR = [[110, 260], [170, 260], [230, 260], [290, 260]]
armL = [[410, 260], [470, 260], [530, 260], [590, 260]]

legR = [[740, 230], [740, 290], [740, 350], [740, 410]]
legL = [[860, 230], [860, 290], [860, 350], [860, 410]]
yBod = [[740, 50], [860, 50], [800, 50], [800, 110], [800, 170], [800, 230], [800, 300]]

names = [rArm, lArm, rHand, lHand, rLeg, lLeg, body]
startcoords = [armR, armL, handR, handL, legR, legL, yBod]

joints = {}
for lists in names:
    for nm in lists:
        joints[nm] = startcoords[names.index(lists)][lists.index(nm)] + [0]
moved = joints.copy()

POSEJSON_TO_VSEEFACE = {
    "hips": "Hips",
    "spine": "Spine",
    "chest": "Chest",
    "upperChest": "Chest",
    "neck": "Neck",
    "head": "Head",
    "leftEye": "LeftEye",
    "rightEye": "RightEye",
    "leftUpperLeg": "LeftUpperLeg",
    "leftLowerLeg": "LeftLowerLeg",
    "leftFoot": "LeftFoot",
    "leftToes": "LeftToes",
    "rightUpperLeg": "RightUpperLeg",
    "rightLowerLeg": "RightLowerLeg",
    "rightFoot": "RightFoot",
    "rightToes": "RightToes",
    "leftShoulder": "LeftShoulder",
    "leftUpperArm": "LeftUpperArm",
    "leftLowerArm": "LeftLowerArm",
    "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder",
    "rightUpperArm": "RightUpperArm",
    "rightLowerArm": "RightLowerArm",
    "rightHand": "RightHand",

    # Fingers: JSON keys from VRM pose often use "...Proximal/Intermediate/Distal"
    "leftThumbMetacarpal": "LeftThumbMetacarpal",
    "leftThumbProximal": "LeftThumbIntermediate",
    "leftThumbDistal": "LeftThumbDistal",

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

    "rightThumbMetacarpal": "RightThumbMetacarpal",
    "rightThumbProximal": "RightThumbIntermediate",
    "rightThumbDistal": "RightThumbDistal",

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

current_json_quat = {k: [0.0, 0.0, 0.0, 1.0] for k in joints.keys()}

# UI checkboxes can stay, but quaternion transform is now FIXED and independent
json_swap_lr_var  = IntVar(value=0)
json_mirror_x_var = IntVar(value=0)
json_swap_yz_var  = IntVar(value=0)
json_swap_xz_var  = IntVar(value=0)
json_neg_w_var    = IntVar(value=0)
json_thumb_fix_var= IntVar(value=1)

# -------------------------
# Math helpers
# -------------------------
def create_circle(x, y, r, canvasname, **kwargs):
    return canvasname.create_oval(x - r, y - r, x + r, y + r, **kwargs)

def quat_norm(q):
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
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

# -------------------------
# Correct VRM(JSON/Unity) -> VMC(VSeeFace/OpenVR) transforms
# -------------------------
def _swap_lr_bone(name: str) -> str:
    # optional, if user wants mirroring by swapping bones
    if json_swap_lr_var.get() != 1:
        return name
    if name.startswith("Left"):
        return "Right" + name[4:]
    if name.startswith("Right"):
        return "Left" + name[5:]
    return name

def _transform_quat(q):
    """
    FIXED conversion:
    VRM / Unity: X right, Y up, Z forward
    VMC / VSeeFace: X right, Y up, Z backward

    The robust conversion for rotations is:
      invert Z and W (equivalent to coordinate handedness flip on forward axis)
    """
    qx, qy, qz, qw = q
    qz = -qz
    qw = -qw
    return quat_norm([qx, qy, qz, qw])

def _transform_pos(pos):
    """
    Position conversion: forward axis flip
    """
    x, y, z = pos
    return [x, y, -z]

# -------------------------
# UI elements (must exist before readcfg)
# -------------------------
ipLabel = Label(root, text="IP:", anchor="w")
ipLabel.place(x=310, y=70, width=30, height=20)
ipEntry = Entry(root)
ipEntry.place(x=340, y=70, width=150, height=20)

portLabel = Label(root, text="Port:", anchor="w")
portLabel.place(x=500, y=70, width=40, height=20)
portEntry = Entry(root)
portEntry.place(x=540, y=70, width=70, height=20)

poseList = Listbox(root)
poseList.place(x=400, y=340, width=90, height=110)

stagingList = Listbox(root)
stagingList.place(x=500, y=340, width=90, height=110)

moveList = Listbox(root)
moveList.place(x=600, y=340, width=90, height=110)

# -------------------------
# File helpers
# -------------------------
def find_pose_path(base: str) -> str:
    p_txt = os.path.join(POSES_DIR, base + ".txt")
    p_json = os.path.join(POSES_DIR, base + ".json")
    if os.path.exists(p_txt):
        return p_txt
    if os.path.exists(p_json):
        return p_json
    return ""

def find_move_pose_path(motion: str, pose_name: str) -> str:
    folder = os.path.join(MOVES_DIR, motion)
    if pose_name.lower().endswith(".txt") or pose_name.lower().endswith(".json"):
        p1 = os.path.join(folder, pose_name)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(POSES_DIR, pose_name)
        if os.path.exists(p2):
            return p2
        return ""
    p1 = os.path.join(folder, pose_name + ".txt")
    p2 = os.path.join(folder, pose_name + ".json")
    p3 = os.path.join(POSES_DIR, pose_name + ".txt")
    p4 = os.path.join(POSES_DIR, pose_name + ".json")
    for p in (p1, p2, p3, p4):
        if os.path.exists(p):
            return p
    return ""

# -------------------------
# Config
# -------------------------
def readcfg():
    global ip, port, osc, target, offset
    target = None

    cfg_path = os.path.join("Resources", "config.cfg")

    if not ip:
        ip = "127.0.0.1"
    if not port:
        port = 39539
    osc = 0 if osc is None else osc

    ipEntry.delete(0, END)
    ipEntry.insert(0, ip)
    portEntry.delete(0, END)
    portEntry.insert(0, str(port))

    if not os.path.exists(cfg_path):
        return

    try:
        with open(cfg_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except Exception:
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue
        split = line.split("=")

        if split[0] == "ip" and len(split) >= 2:
            ip = str(split[1])
            ipEntry.delete(0, END)
            ipEntry.insert(0, ip)

        elif split[0] == "port" and len(split) >= 2:
            try:
                port = int(split[1])
            except Exception:
                port = 39539
            portEntry.delete(0, END)
            portEntry.insert(0, str(port))

        elif split[0] == "osc" and len(split) >= 2:
            try:
                osc = int(split[1])
            except Exception:
                osc = 0

        elif split[0] == "offset" and len(split) >= 4:
            try:
                offset = [float(split[1]), float(split[2]), float(split[3])]
            except Exception:
                offset = [0.0, 0.0, 0.0]

        else:
            # hotkey: name=combo=poseList|moveList
            if len(split) == 3:
                name, combo, boxname = split[0], split[1], split[2]

                if boxname == "poseList":
                    if find_pose_path(name):
                        hotkeys[name] = [
                            combo,
                            keyboard.add_hotkey(combo, lambda n=name: hotkeyactivated(n, poseList)),
                            boxname
                        ]
                else:
                    if os.path.exists(os.path.join(MOVES_DIR, name)):
                        hotkeys[name] = [
                            combo,
                            keyboard.add_hotkey(combo, lambda n=name: hotkeyactivated(n, moveList)),
                            boxname
                        ]

# -------------------------
# OSC
# -------------------------
def enablecoms():
    global ip, port
    if checkVar.get() == 1:
        ip = str(ipEntry.get()).strip()
        try:
            port = int(str(portEntry.get()).strip())
        except Exception:
            port = 39539
            portEntry.delete(0, END)
            portEntry.insert(0, str(port))

        osc_startup()
        osc_udp_client(ip, port, "VroidPoser")
    else:
        osc_terminate()

def sendosc_quat(bone, px, py, pz, qx, qy, qz, qw):
    if checkVar.get() != 1:
        return
    msg = oscbuildparse.OSCMessage(
        "/VMC/Ext/Bone/Pos",
        None,
        [bone, float(px), float(py), float(pz), float(qx), float(qy), float(qz), float(qw)]
    )
    osc_send(msg, "VroidPoser")
    osc_process()

def sendosc_smallquat(bone, x, y, z):
    if checkVar.get() != 1:
        return
    if bone == "Hips":
        msg = oscbuildparse.OSCMessage(
            "/VMC/Ext/Bone/Pos",
            None,
            [bone,
             float(destination[0] + offset[0]),
             float(destination[1] + offset[1]),
             float(destination[2] + offset[2]),
             float(x), float(y), float(z), float(1)]
        )
    else:
        msg = oscbuildparse.OSCMessage(
            "/VMC/Ext/Bone/Pos",
            None,
            [bone, 0.0, 0.0, 0.0, float(x), float(y), float(z), float(1)]
        )
    osc_send(msg, "VroidPoser")
    osc_process()

def sieve(joint, q, w, rot):
    if joint in lLeg:
        sendosc_smallquat(joint, w, rot, q)
    elif joint in rLeg:
        sendosc_smallquat(joint, -1 * w, rot, q)
    elif joint in body and "Eye" not in joint:
        sendosc_smallquat(joint, -1 * w, -1 * rot, -1 * q)
    else:
        sendosc_smallquat(joint, rot, q, w)

# -------------------------
# GUI draw & interactive joystick
# -------------------------
def rotate(bone, rot):
    global moved
    if bone is None:
        return
    multiplier = -1 if "Left" in bone else 1
    q = -4 * ((moved[bone][0] - joints[bone][0]) / joyRange)
    w = -4 * ((moved[bone][1] - joints[bone][1]) / joyRange) * multiplier
    rot = float(rot)
    moved[bone][2] = rot
    rot2 = (1 * rot) ** 3
    sieve(bone, q, w, rot2)

def draw():
    canvas.delete("all")
    for objs in joints.keys():
        if objs in names[0]:
            color = "red"
        elif objs in names[1]:
            color = "green"
        elif objs in names[2]:
            color = "blue"
        elif objs in names[3]:
            color = "indigo"
        elif objs in names[4]:
            color = "violet"
        elif objs in names[5]:
            color = "maroon"
        else:
            color = "black"

        create_circle(joints[objs][0], joints[objs][1], joyRange / 2, canvas, width=2)

        if objs == target:
            canvas.create_line(joints[target][0], joints[target][1], moved[target][0], moved[target][1])
            create_circle(moved[target][0], moved[target][1], 5, canvas, fill=color, width=0)
            multiplier = -1 if "Left" in target else 1
            q = -4 * ((moved[target][0] - joints[target][0]) / joyRange)
            w = -4 * ((moved[target][1] - joints[target][1]) / joyRange) * multiplier
            rotv = moved[target][2] ** 3
            sieve(target, q, w, rotv)
        else:
            canvas.create_line(joints[objs][0], joints[objs][1], moved[objs][0], moved[objs][1])
            create_circle(moved[objs][0], moved[objs][1], radius, canvas, fill=color, width=0)

def refresh():
    poseList.delete(0, END)
    stagingList.delete(0, END)
    moveList.delete(0, END)

    if os.path.isdir(POSES_DIR):
        bases = set()
        for entry in os.listdir(POSES_DIR):
            p = os.path.join(POSES_DIR, entry)
            if os.path.isfile(p):
                ext = os.path.splitext(entry)[1].lower()
                if ext in POSE_EXTS:
                    bases.add(os.path.splitext(entry)[0])
        for b in sorted(bases):
            poseList.insert(poseList.size(), b)

    for pose in staging:
        stagingList.insert(stagingList.size(), pose)

    for x in os.walk(MOVES_DIR + "/"):
        if x[0] != MOVES_DIR + "/":
            moveList.insert(moveList.size(), x[0].split('/')[1])

def reset():
    global moved, travel, destination, current_json_quat
    sure = messagebox.askyesno("Resetting...", "Are you sure you want to reset all joints?")
    if sure != 1:
        return

    destination[:] = [0, 0, 0]
    travel[:] = [0, 0, 0]
    for i in joints.keys():
        sendosc_quat(i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        current_json_quat[i] = [0.0, 0.0, 0.0, 1.0]

    moved = joints.copy()
    draw()

# -------------------------
# Pose parsing / applying
# -------------------------
def parse_pose_txt(path: str):
    speed_val = speedSlider.get()
    dest = travel.copy()
    targets = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            split = line.split(",")
            if split[0] == "speed" and len(split) >= 2:
                try:
                    speed_val = float(split[1].rstrip("\n"))
                except Exception:
                    speed_val = speedSlider.get()
            elif split[0] == "travel" and len(split) >= 4:
                try:
                    dest = [float(split[1]), float(split[2]), float(split[3])]
                except Exception:
                    dest = travel.copy()
            else:
                if len(split) >= 4:
                    try:
                        bone = str(split[0])
                        q = float(split[1])
                        w = float(split[2])
                        r = float(split[3].rstrip("\n"))
                        targets[bone] = [q, w, r]
                    except Exception:
                        pass
    return speed_val, dest, targets

def apply_pose_txt(speed_val, new_dest, targets):
    global moved, travel, destination
    tempdict = {}
    x = 0.0
    step = max(0.001, float(speed_val) / 100.0)
    while x <= 1.000001:
        destination[:] = [
            travel[0] + (float(new_dest[0]) - travel[0]) * x,
            travel[1] + (float(new_dest[1]) - travel[1]) * x,
            travel[2] + (float(new_dest[2]) - travel[2]) * x
        ]
        for bone, vals in targets.items():
            if bone not in joints:
                continue
            q1, w1, r1 = float(vals[0]), float(vals[1]), float(vals[2])
            multiplier = -1 if "Left" in bone else 1
            a = ((q1 * joyRange) / (-4)) + joints[bone][0]
            b = ((w1 * joyRange) / (-4 * multiplier)) + joints[bone][1]
            tempdict[bone] = [a, b, r1]
            q0 = -4 * ((moved[bone][0] - joints[bone][0]) / joyRange)
            w0 = -4 * ((moved[bone][1] - joints[bone][1]) / joyRange) * multiplier
            hor = q0 + (q1 - q0) * x
            ver = w0 + (w1 - w0) * x
            rotv = moved[bone][2] + (r1 - moved[bone][2]) * x
            rotv = (1 * rotv) ** 3
            sieve(bone, hor, ver, rotv)
        x += step

    for k in moved:
        if k in tempdict:
            moved[k] = tempdict[k]
    travel[:] = destination[:]
    draw()

def parse_pose_json(path: str):
    speed_val = speedSlider.get()
    dest = travel.copy()
    targets = {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pose = data.get("pose", {})
    if not isinstance(pose, dict):
        return speed_val, dest, targets

    hips = pose.get("hips", {})
    if isinstance(hips, dict):
        pos = hips.get("position", None)
        if isinstance(pos, list) and len(pos) == 3:
            try:
                dest = _transform_pos([float(pos[0]), float(pos[1]), float(pos[2])])
            except Exception:
                dest = travel.copy()

    tmp = {}
    for json_name, payload in pose.items():
        vseeface_bone = POSEJSON_TO_VSEEFACE.get(json_name)
        if not vseeface_bone:
            continue

        vseeface_bone = _swap_lr_bone(vseeface_bone)
        if vseeface_bone not in joints:
            continue

        rot = (payload or {}).get("rotation", [0, 0, 0, 1])
        if not isinstance(rot, list) or len(rot) != 4:
            continue
        try:
            qx, qy, qz, qw = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
        except Exception:
            continue

        tmp[vseeface_bone] = _transform_quat([qx, qy, qz, qw])

    # Thumb fix optional
    if json_thumb_fix_var.get() == 1:
        for side in ("Left", "Right"):
            meta = side + "ThumbMetacarpal"
            prox = side + "ThumbIntermediate"
            if meta not in tmp and prox in tmp and meta in joints:
                tmp[meta] = tmp[prox]

    targets.update(tmp)
    return speed_val, dest, targets

def apply_pose_json(speed_val, new_dest, targets):
    global travel, destination, current_json_quat
    x = 0.0
    step = max(0.001, float(speed_val) / 100.0)
    bones = list(targets.keys())

    while x <= 1.000001:
        destination[:] = [
            travel[0] + (float(new_dest[0]) - travel[0]) * x,
            travel[1] + (float(new_dest[1]) - travel[1]) * x,
            travel[2] + (float(new_dest[2]) - travel[2]) * x
        ]

        for bone in bones:
            q0 = current_json_quat.get(bone, [0.0, 0.0, 0.0, 1.0])
            q1 = targets[bone]
            qt = quat_slerp(q0, q1, x)

            if bone == "Hips":
                sendosc_quat(
                    bone,
                    float(destination[0] + offset[0]),
                    float(destination[1] + offset[1]),
                    float(destination[2] + offset[2]),
                    qt[0], qt[1], qt[2], qt[3]
                )
            else:
                sendosc_quat(bone, 0.0, 0.0, 0.0, qt[0], qt[1], qt[2], qt[3])

        x += step

    for bone in bones:
        current_json_quat[bone] = targets[bone]
    travel[:] = destination[:]

def posepicker(path: str):
    if not os.path.exists(path):
        return
    if path.lower().endswith(".json"):
        sp, dest, targets = parse_pose_json(path)
        speedSlider.set(float(sp))
        apply_pose_json(float(sp), dest, targets)
    else:
        sp, dest, targets = parse_pose_txt(path)
        speedSlider.set(float(sp))
        apply_pose_txt(float(sp), dest, targets)

# -------------------------
# List actions
# -------------------------
def delete():
    global staging
    select = poseList.curselection()
    if str(select) == "()":
        return
    base = str(poseList.get(select[0]))
    sure = messagebox.askyesno("", "Are you sure you want to delete the " + base + " pose?")
    if sure is True:
        if base in staging:
            staging.remove(base)
        p1 = os.path.join(POSES_DIR, base + ".txt")
        p2 = os.path.join(POSES_DIR, base + ".json")
        if os.path.exists(p1):
            os.remove(p1)
        if os.path.exists(p2):
            os.remove(p2)
        reset()
        refresh()

def save():
    pose = simpledialog.askstring("", "Enter pose name:")
    if pose is None:
        return
    file = open(os.path.join(POSES_DIR, pose + ".txt"), "w", encoding="utf-8")
    file.write("speed," + str(speedSlider.get()) + "\n")
    file.write("travel," + str(travel[0]) + "," + str(travel[1]) + "," + str(travel[2]) + "\n")
    for item in moved.keys():
        multiplier = -1 if "Left" in item else 1
        q = -4 * (moved[item][0] - joints[item][0]) / joyRange
        w = -4 * (moved[item][1] - joints[item][1]) / joyRange * multiplier
        file.write(str(item) + "," + str(q) + "," + str(w) + "," + str(moved[item][2]) + "\n")
    file.close()
    refresh()

def clear(event):
    global staging
    if event is True:
        staging = []
        refresh()
    else:
        select = stagingList.curselection()
        if str(select) != "()":
            staging.pop(select[0])
            refresh()

def snag():
    get = poseList.curselection()
    if str(get) == "()":
        return
    staging.append(str(poseList.get(get[0])))
    refresh()

def createanimation():
    motion = simpledialog.askstring("", "Enter animation name:")
    if motion is None:
        return
    os.makedirs(os.path.join(MOVES_DIR, motion), exist_ok=True)
    file = open(os.path.join(MOVES_DIR, motion, "animate.txt"), "w", encoding="utf-8")
    for pose in staging:
        src = find_pose_path(pose)
        if not src:
            continue
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(MOVES_DIR, motion, pose + ext)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        file.write(pose + ext + "\n")
    file.close()
    moveList.insert(0, motion)

def reposition(up_dir):
    select = stagingList.curselection()
    if str(select) == "()":
        return
    idx = select[0]
    if up_dir:
        if idx - 1 >= 0:
            staging.insert(idx - 1, staging[idx])
            staging.pop(idx + 1)
    else:
        if idx + 1 < len(staging):
            staging.insert(idx + 2, staging[idx])
            staging.pop(idx)
    refresh()

def deleteanim():
    select = moveList.curselection()
    if str(select) == "()":
        return
    sure = messagebox.askyesno("", "Are you sure you want to delete the " + str(moveList.get(select[0])) + " Motion?")
    if sure is True:
        shutil.rmtree(os.path.join(MOVES_DIR, str(moveList.get(select[0]))), ignore_errors=True)
        reset()
        refresh()

# -------------------------
# Mouse interaction
# -------------------------
def left_click(event):
    global target, rotTarget
    for objs in moved.keys():
        if math.dist([event.x, event.y], [moved[objs][0], moved[objs][1]]) < radius:
            target = objs
            rotTarget = objs
            rotSlider.set(float(moved[target][2]))
            break

def release(event):
    global target
    target = None
    draw()

def move_mouse(event):
    global moved
    if target is None:
        return
    if math.dist([joints[target][0], joints[target][1]], [event.x, event.y]) > joyRange / 2:
        d = math.dist([event.x, event.y], [joints[target][0], joints[target][1]])
        b = (joyRange * (event.y - joints[target][1])) / (2 * d)
        a = (joyRange * (event.x - joints[target][0])) / (2 * d)
        moved[target][0] = joints[target][0] + a
        moved[target][1] = joints[target][1] + b
    elif math.dist([joints[target][0], joints[target][1]], [event.x, event.y]) <= deadzone:
        moved[target] = joints[target]
    else:
        moved[target] = [event.x, event.y, moved[target][2]]
    draw()

def advanced(show):
    root.geometry("910x500" if show else "700x500")

# -------------------------
# Picker / hotkeys
# -------------------------
def picker(event, listbox):
    if listbox == "test":
        for pose in staging:
            path = find_pose_path(str(pose))
            if path:
                posepicker(path)
        return

    listbox.selection_clear(0, END)
    listbox.selection_set(listbox.nearest(event.y))
    listbox.activate(listbox.nearest(event.y))

    if listbox == poseList:
        select = poseList.curselection()
        if str(select) != "()":
            base = str(poseList.get(select[0]))
            path = find_pose_path(base)
            if path:
                posepicker(path)

    elif listbox == moveList:
        select = moveList.curselection()
        if str(select) != "()":
            motion = str(moveList.get(select[0]))
            anim = os.path.join(MOVES_DIR, motion, "animate.txt")
            if os.path.exists(anim):
                with open(anim, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                for line in lines:
                    p = find_move_pose_path(motion, line)
                    if p:
                        posepicker(p)

def hotkeyactivated(base, listbox):
    if listbox == poseList:
        path = find_pose_path(base)
        if path:
            posepicker(path)
    elif listbox == moveList:
        anim = os.path.join(MOVES_DIR, base, "animate.txt")
        if os.path.exists(anim):
            with open(anim, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            for line in lines:
                p = find_move_pose_path(base, line)
                if p:
                    posepicker(p)

def keylogger(event):
    global pressed, listen
    key = event.name
    if key not in pressed and listen is True:
        pressed.append(key)

def addhotkey(thing, listbox):
    global pressed, listen
    listen = True
    messagebox.showinfo(title=None, message="Input your hotkey combo and press enter or click OK. To clear a hotkey, just continue.")
    listen = False

    if len(pressed) > 0:
        if pressed[-1] == "enter":
            pressed.pop()

        if len(pressed) != 0:
            combo = "+".join(pressed)
            pressed.clear()
            box = "poseList" if listbox == poseList else "moveList"

            if thing not in hotkeys:
                hotkeys[thing] = [combo, keyboard.add_hotkey(combo, lambda n=thing, lb=listbox: hotkeyactivated(n, lb)), box]
            else:
                keyboard.clear_hotkey(hotkeys[thing][1])
                hotkeys.pop(thing)
                hotkeys[thing] = [combo, keyboard.add_hotkey(combo, lambda n=thing, lb=listbox: hotkeyactivated(n, lb)), box]
        else:
            if thing in hotkeys:
                keyboard.clear_hotkey(hotkeys[thing][1])
                hotkeys.pop(thing)
    elif thing in hotkeys:
        keyboard.clear_hotkey(hotkeys[thing][1])
        hotkeys.pop(thing)

keyboard.on_press(lambda r: keylogger(r))

m = Menu(root, tearoff=0)

def do_popup(event, listbox):
    listbox.selection_clear(0, END)
    listbox.selection_set(listbox.nearest(event.y))
    listbox.activate(listbox.nearest(event.y))
    try:
        m.delete(0, END)
        if listbox == poseList:
            m.add_command(label="Delete", command=delete)
            m.add_command(label="Add to Staging", command=snag)
            select = listbox.curselection()
            if select != "()":
                motion = listbox.get(select[0])
                label = hotkeys[motion][0] if motion in hotkeys else "Assign Hotkey"
                m.add_command(label=label, command=lambda mm=motion, lb=listbox: addhotkey(mm, lb))
        elif listbox == stagingList:
            m.add_command(label="Remove", command=lambda: clear(False))
            m.add_command(label="Move Up", command=lambda: reposition(True))
            m.add_command(label="Move Down", command=lambda: reposition(False))
            m.add_command(label="Clear Staging", command=lambda: clear(True))
            m.add_command(label="Test", command=lambda: picker(None, "test"))
        elif listbox == moveList:
            m.add_command(label="Delete", command=deleteanim)
            select = listbox.curselection()
            if select != "()":
                motion = listbox.get(select[0])
                label = hotkeys[motion][0] if motion in hotkeys else "Assign Hotkey"
                m.add_command(label=label, command=lambda mm=motion, lb=listbox: addhotkey(mm, lb))
        m.tk_popup(event.x_root, event.y_root)
    finally:
        m.grab_release()

# -------------------------
# UI
# -------------------------
poseLabel = Label(root, text="Poses:", anchor="w")
poseLabel.place(x=400, y=320, width=50, height=20)

stagingLabel = Label(root, text="Staging:", anchor="w")
stagingLabel.place(x=500, y=320, width=50, height=20)

moveLabel = Label(root, text="Motions:", anchor="w")
moveLabel.place(x=600, y=320, width=50, height=20)

enable = Checkbutton(root, text="Send Pose Data", variable=checkVar, onvalue=1, command=enablecoms)
enable.place(x=220, y=430, width=120, height=20)

resetButton = Button(root, text="Reset Joints", command=reset)
resetButton.place(x=220, y=390, width=90, height=20)

saveButton = Button(root, text="Save Pose", command=save)
saveButton.place(x=400, y=460, width=80, height=20)

rotLabel = Label(root, text="Limb Rotation:", anchor="w")
rotLabel.place(x=10, y=340, width=100, height=20)

rotSlider = Scale(root, from_=-4, to=4, resolution=0.0001, orient=HORIZONTAL)
rotSlider.configure(command=lambda r: rotate(rotTarget, r))
rotSlider.place(x=100, y=320, width=200)

speedLabel = Label(root, text="Pose Speed", anchor="w")
speedLabel.place(x=10, y=390, width=100, height=20)

speedSlider = Scale(root, from_=0.1, to=5, resolution=0.1, orient=HORIZONTAL)
speedSlider.place(x=100, y=370, width=120)
speedSlider.set(2.5)

infoLabel = Label(root, text="| Copyleft 2021 NeilioClown | Covered by GPL-3.0 |", anchor="center")
infoLabel.place(x=10, y=460, width=300, height=20)

moveUp = Button(root, text="⬆️", command=lambda: reposition(True))
moveUp.place(x=500, y=460, width=40, height=20)

moveDown = Button(root, text="⬇️", command=lambda: reposition(False))
moveDown.place(x=540, y=460, width=40, height=20)

addAnim = Button(root, text="Save Motion", command=createanimation)
addAnim.place(x=600, y=460, width=80, height=20)

advancedButton = Button(root, text="Advanced", command=lambda: advanced(True))
advancedButton.place(x=310, y=30, width=80, height=20)

standardButton = Button(root, text="Standard", command=lambda: advanced(False))
standardButton.place(x=760, y=460, width=80, height=20)

unitEntry = Entry(root)
unitEntry.place(x=840, y=130, width=60, height=20)

try:
    p1 = PhotoImage(file="Resources/Up.png")
    p2 = PhotoImage(file="Resources/Down.png")
    p3 = PhotoImage(file="Resources/Left.png")
    p4 = PhotoImage(file="Resources/Right.png")
except Exception:
    p1 = p2 = p3 = p4 = None

def trip(direct):
    global destination, travel
    try:
        unit = float(unitEntry.get()) if unitEntry.get().strip() else 0.01
    except Exception:
        unit = 0.01

    if direct == "Up":
        destination[1] += unit
    elif direct == "Down":
        destination[1] -= unit
    elif direct == "Left":
        destination[0] += unit
    elif direct == "Right":
        destination[0] -= unit
    elif direct == "Front":
        destination[2] += unit
    elif direct == "Back":
        destination[2] -= unit

    q = -4 * (moved["Hips"][0] - joints["Hips"][0]) / joyRange
    w = -4 * (moved["Hips"][1] - joints["Hips"][1]) / joyRange
    sendosc_smallquat("Hips", -1 * w, -1 * moved["Hips"][2], -1 * q)
    travel = destination.copy()

if p1 and p2 and p3 and p4:
    Button(root, image=p1, repeatdelay=100, repeatinterval=100, command=lambda: trip("Up")).place(x=720, y=110, width=20, height=20)
    Button(root, image=p2, repeatdelay=100, repeatinterval=100, command=lambda: trip("Down")).place(x=720, y=150, width=20, height=20)
    Button(root, image=p3, repeatdelay=100, repeatinterval=100, command=lambda: trip("Left")).place(x=700, y=130, width=20, height=20)
    Button(root, image=p4, repeatdelay=100, repeatinterval=100, command=lambda: trip("Right")).place(x=740, y=130, width=20, height=20)
    Button(root, image=p1, repeatdelay=100, repeatinterval=100, command=lambda: trip("Back")).place(x=860, y=110, width=20, height=20)
    Button(root, image=p2, repeatdelay=100, repeatinterval=100, command=lambda: trip("Front")).place(x=860, y=150, width=20, height=20)

# JSON panel (keep only options that make sense)
jsonPanel = LabelFrame(root, text="JSON Pose Options", padx=6, pady=4)
jsonPanel.place(x=10, y=10, width=290, height=160)
Checkbutton(jsonPanel, text="Swap Left/Right (mirror)", variable=json_swap_lr_var).grid(row=0, column=0, sticky="w")
Checkbutton(jsonPanel, text="Thumb Fix", variable=json_thumb_fix_var).grid(row=1, column=0, sticky="w")

def rerun_current_pose():
    sel = poseList.curselection()
    if str(sel) == "()":
        return
    base = str(poseList.get(sel[0]))
    path = find_pose_path(base)
    if path and path.lower().endswith(".json"):
        posepicker(path)

Button(jsonPanel, text="Apply + Re-test JSON", command=rerun_current_pose).grid(row=2, column=0, sticky="w", pady=(8, 0))

# -------------------------
# Terminate
# -------------------------
def terminate():
    sure = messagebox.askyesno("Exiting...", "Are you sure you want to quit?")
    if sure != 1:
        return

    cfg_path = os.path.join("Resources", "config.cfg")
    with open(cfg_path, "w", encoding="utf-8") as file:
        file.write("ip=" + str(ipEntry.get()) + "\n")
        file.write("port=" + str(portEntry.get()) + "\n")
        file.write("osc=" + str(checkVar.get()) + "\n")
        file.write("offset=" + str(offset[0]) + "=" + str(offset[1]) + "=" + str(offset[2]) + "\n")
        for hot in hotkeys.keys():
            file.write(hot + "=" + hotkeys[hot][0] + "=" + hotkeys[hot][2] + "\n")

    root.destroy()

def load_initial():
    readcfg()
    refresh()
    draw()
    checkVar.set(osc)
    enablecoms()

# -------------------------
# Bindings
# -------------------------
poseList.bind("<Button-3>", lambda r: do_popup(r, poseList))
stagingList.bind("<Button-3>", lambda r: do_popup(r, stagingList))
moveList.bind("<Button-3>", lambda r: do_popup(r, moveList))

poseList.bind("<Button-1>", lambda r: picker(r, poseList))
moveList.bind("<Button-1>", lambda r: picker(r, moveList))

canvas.bind("<Button 1>", left_click)
canvas.bind("<ButtonRelease 1>", release)
canvas.bind("<Motion>", move_mouse)

root.protocol("WM_DELETE_WINDOW", terminate)

# Start
load_initial()
root.mainloop()
