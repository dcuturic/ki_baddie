"""
FBX to VRM Converter
======================
Flask-Service zum Konvertieren von FBX-Dateien zu VRM.

Features:
  - ./fbx/ Ordner: ZIP-Dateien mit .fbx + Texturen ablegen -> manuelle Konvertierung
  - Upload via Web-UI oder API (ZIP oder .fbx)
  - Blender headless Konvertierung im Hintergrund
  - Automatisches VRM Humanoid Bone-Mapping
  - Shape Keys -> VRM Expressions (Gesichtsemotionen)
  - Bone/Armature-Setup fuer Animationen
  - Fertige VRM-Dateien in ./vrm/

Nutzung:
  1. python app.py
  2. ZIP-Dateien (mit .fbx + Texturen) in ./fbx/ legen
  3. Browser: http://localhost:5012 -> manuell konvertieren
"""

import os
import sys
import json
import time
import uuid
import shutil
import zipfile
import subprocess
import threading
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_file, render_template_string

# ======================= CONFIG =======================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")


def load_config() -> Dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Config] Fehler: {e}")
        return {}


CONFIG = load_config()

SERVER_HOST = CONFIG.get("host", "0.0.0.0")
SERVER_PORT = CONFIG.get("port", 5012)
BLENDER_PATH = CONFIG.get("blender_path", r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe")
FBX_DIR = os.path.join(SCRIPT_DIR, "fbx")        # Input: ZIPs mit .fbx + Texturen oder .fbx Dateien
VRM_DIR = os.path.join(SCRIPT_DIR, "vrm")         # Output: fertige .vrm Dateien
UPLOAD_DIR = os.path.join(SCRIPT_DIR, "uploads")   # Temp fuer Web-Uploads
WORK_DIR = os.path.join(SCRIPT_DIR, "_work")       # Temp zum Entpacken
BLENDER_SCRIPT = os.path.join(SCRIPT_DIR, "converter.py")

for _d in [FBX_DIR, VRM_DIR, UPLOAD_DIR, WORK_DIR]:
    os.makedirs(_d, exist_ok=True)

# ======================= JOB TRACKING =======================

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


def create_job(filename: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "log": [],
            "created": datetime.now().isoformat(),
            "output_file": None,
            "error": None,
        }
    return job_id


def update_job(job_id: str, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def add_job_log(job_id: str, message: str):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["log"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


# ======================= FLASK APP =======================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max


# ======================= CONVERSION =======================

def extract_zip(zip_path: str, job_id: str) -> str:
    """
    ZIP entpacken -> .fbx Datei + Texturen in Work-Verzeichnis.
    Gibt den Pfad zur .fbx Datei zurueck.
    """
    extract_dir = os.path.join(WORK_DIR, job_id)
    os.makedirs(extract_dir, exist_ok=True)

    add_job_log(job_id, f"Entpacke ZIP: {os.path.basename(zip_path)}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            if info.filename.startswith('/') or '..' in info.filename:
                continue
            zf.extract(info, extract_dir)

    # .fbx Datei finden (auch in Unterordnern)
    fbx_file = None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(".fbx") and not f.startswith("."):
                fbx_file = os.path.join(root, f)
                break
        if fbx_file:
            break

    if not fbx_file:
        raise RuntimeError("Keine .fbx Datei in der ZIP gefunden!")

    # Zaehle Texturen
    tex_count = 0
    tex_exts = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff", ".exr", ".hdr", ".webp"}
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in tex_exts:
                tex_count += 1

    add_job_log(job_id, f"FBX-Datei: {os.path.basename(fbx_file)}")
    add_job_log(job_id, f"Texturen gefunden: {tex_count}")

    return fbx_file


def run_conversion(job_id: str, fbx_path: str, output_dir: str = None, cleanup_dir: str = None, options: dict = None):
    """Startet Blender headless und fuehrt das Converter-Script aus."""
    if output_dir is None:
        output_dir = VRM_DIR
    if options is None:
        options = {}
    try:
        update_job(job_id, status="converting", progress=10)
        add_job_log(job_id, "Starte Blender im Headless-Modus...")

        output_name = os.path.splitext(os.path.basename(fbx_path))[0] + ".vrm"
        output_path = os.path.join(output_dir, output_name)

        if not os.path.exists(BLENDER_PATH):
            raise FileNotFoundError(
                f"Blender nicht gefunden: {BLENDER_PATH}\n"
                f"Bitte blender_path in config.json setzen."
            )

        # Options als JSON an converter.py uebergeben
        options_json = json.dumps(options)

        cmd = [
            BLENDER_PATH,
            "--background",
            "--factory-startup",
            "--python", BLENDER_SCRIPT,
            "--",
            fbx_path,
            output_path,
            job_id,
            options_json,
        ]

        vrm_ver = options.get('vrm_version', '0.x')
        add_job_log(job_id, f"VRM-Version: {vrm_ver}")

        add_job_log(job_id, f"FBX-Datei: {os.path.basename(fbx_path)}")
        add_job_log(job_id, f"Ziel: {output_name}")
        update_job(job_id, progress=20)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=SCRIPT_DIR,
        )

        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue

            # Parse progress markers from converter script
            if line.startswith("[PROGRESS:"):
                try:
                    pct = int(line.split(":")[1].split("]")[0])
                    update_job(job_id, progress=pct)
                except (ValueError, IndexError):
                    pass
            elif line.startswith("[STATUS:"):
                status_msg = line.split(":", 1)[1].rstrip("]")
                add_job_log(job_id, status_msg)
            elif any(tag in line for tag in ["[OK]", "[INFO]", "[WARN]", "[FAIL]", "[CHECK]", "[ERROR]", "[DEBUG]"]):
                add_job_log(job_id, line)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Blender beendet mit Exit-Code {process.returncode}")

        if not os.path.exists(output_path):
            raise RuntimeError("VRM-Datei wurde nicht erstellt. Pruefe die Blender-Logs.")

        file_size = os.path.getsize(output_path)
        add_job_log(job_id, f"VRM erstellt: {output_name} ({file_size / 1024 / 1024:.1f} MB)")
        update_job(job_id, status="done", progress=100, output_file=output_path)

    except Exception as e:
        add_job_log(job_id, f"FEHLER: {e}")
        update_job(job_id, status="error", error=str(e))
    finally:
        # Clean up work directory (entpackte ZIP-Dateien)
        if cleanup_dir and os.path.exists(cleanup_dir):
            try:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
            except Exception:
                pass


# ======================= ROUTES =======================

@app.route("/health")
def health():
    blender_ok = os.path.exists(BLENDER_PATH)
    fbx_files = []
    if os.path.exists(FBX_DIR):
        fbx_files = [f for f in os.listdir(FBX_DIR) if f.lower().endswith((".zip", ".fbx"))]
    vrm_files = [f for f in os.listdir(VRM_DIR) if f.lower().endswith(".vrm")] if os.path.exists(VRM_DIR) else []
    return jsonify({
        "status": "ok",
        "service": "fbx_to_vrm",
        "blender_found": blender_ok,
        "blender_path": BLENDER_PATH,
        "fbx_folder": len(fbx_files),
        "vrm_folder": len(vrm_files),
    })


@app.route("/")
def index():
    return render_template_string(WEB_UI)


@app.route("/convert", methods=["POST"])
def convert():
    """Upload .fbx oder .zip Datei und starte Konvertierung."""
    if "file" not in request.files:
        return jsonify({"error": "Keine Datei hochgeladen"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Kein Dateiname"}), 400

    fname_lower = file.filename.lower()
    if not fname_lower.endswith(".fbx") and not fname_lower.endswith(".zip"):
        return jsonify({"error": "Nur .fbx oder .zip Dateien erlaubt"}), 400

    # Optionen aus Formular lesen
    options = {
        "vrm_version": request.form.get("vrm_version", "0.x"),
        "expression_mapping": request.form.get("expression_mapping", "auto"),
        "normalize": request.form.get("normalize", "true") == "true",
        "auto_weight": request.form.get("auto_weight", "true") == "true",
    }

    safe_name = file.filename.replace(" ", "_")
    job_id = create_job(safe_name)
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")
    file.save(upload_path)

    add_job_log(job_id, f"Datei empfangen: {safe_name} ({os.path.getsize(upload_path) / 1024 / 1024:.1f} MB)")

    if fname_lower.endswith(".zip"):
        def convert_zip():
            cleanup_dir = None
            try:
                fbx_path = extract_zip(upload_path, job_id)
                cleanup_dir = os.path.join(WORK_DIR, job_id)
                run_conversion(job_id, fbx_path, cleanup_dir=cleanup_dir, options=options)
            except Exception as e:
                add_job_log(job_id, f"FEHLER: {e}")
                update_job(job_id, status="error", error=str(e))
            finally:
                try:
                    os.remove(upload_path)
                except Exception:
                    pass
        thread = threading.Thread(target=convert_zip, daemon=True)
    else:
        thread = threading.Thread(target=run_conversion, args=(job_id, upload_path), kwargs={"options": options}, daemon=True)

    thread.start()
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/convert/file", methods=["POST"])
def convert_local_file():
    """Konvertiere eine lokale .fbx Datei (Pfad als JSON)."""
    data = request.get_json(silent=True) or {}
    fbx_path = data.get("path", "")

    if not fbx_path or not os.path.exists(fbx_path):
        return jsonify({"error": f"Datei nicht gefunden: {fbx_path}"}), 400

    if not fbx_path.lower().endswith(".fbx"):
        return jsonify({"error": "Nur .fbx Dateien erlaubt"}), 400

    # Copy file to uploads
    safe_name = os.path.basename(fbx_path).replace(" ", "_")
    job_id = create_job(safe_name)
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")
    shutil.copy2(fbx_path, upload_path)

    add_job_log(job_id, f"Lokale Datei: {fbx_path}")

    thread = threading.Thread(target=run_conversion, args=(job_id, upload_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/jobs")
def list_jobs():
    """Liste aller Konvertierungs-Jobs."""
    with jobs_lock:
        return jsonify(list(jobs.values()))


@app.route("/jobs/<job_id>")
def get_job(job_id: str):
    """Status eines Jobs abfragen."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job nicht gefunden"}), 404
    return jsonify(job)


@app.route("/jobs/<job_id>/download")
def download_job(job_id: str):
    """Fertige VRM-Datei herunterladen."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job nicht gefunden"}), 404
    if job["status"] != "done" or not job.get("output_file"):
        return jsonify({"error": "Datei noch nicht fertig oder Fehler aufgetreten"}), 400
    if not os.path.exists(job["output_file"]):
        return jsonify({"error": "Datei nicht mehr vorhanden"}), 404

    download_name = os.path.splitext(job["filename"])[0] + ".vrm"
    return send_file(job["output_file"], as_attachment=True, download_name=download_name)


@app.route("/output")
def list_output():
    """Liste aller VRM-Dateien im vrm/ Ordner."""
    files = []
    for f in os.listdir(VRM_DIR):
        if f.lower().endswith(".vrm"):
            full = os.path.join(VRM_DIR, f)
            files.append({
                "name": f,
                "size_mb": round(os.path.getsize(full) / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(os.path.getmtime(full)).isoformat(),
            })
    return jsonify(files)


@app.route("/output/<filename>/download")
def download_output(filename: str):
    """VRM-Datei aus dem vrm/ Ordner herunterladen."""
    path = os.path.join(VRM_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Datei nicht gefunden"}), 404
    return send_file(path, as_attachment=True)


@app.route("/fbx")
def list_fbx():
    """Liste aller Dateien im fbx/ Ordner."""
    files = []
    for f in os.listdir(FBX_DIR):
        if f.lower().endswith((".zip", ".fbx")):
            full = os.path.join(FBX_DIR, f)
            vrm_name = os.path.splitext(f)[0] + ".vrm"
            vrm_exists = os.path.exists(os.path.join(VRM_DIR, vrm_name))
            files.append({
                "name": f,
                "size_mb": round(os.path.getsize(full) / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(os.path.getmtime(full)).isoformat(),
                "vrm_exists": vrm_exists,
            })
    return jsonify(files)


@app.route("/scan", methods=["POST"])
def scan_fbx_folder():
    """fbx/ Ordner scannen und Dateien auflisten (ohne zu konvertieren)."""
    files = []
    if os.path.exists(FBX_DIR):
        for f in os.listdir(FBX_DIR):
            if f.lower().endswith((".zip", ".fbx")):
                full = os.path.join(FBX_DIR, f)
                vrm_name = os.path.splitext(f)[0] + ".vrm"
                vrm_exists = os.path.exists(os.path.join(VRM_DIR, vrm_name))
                files.append({
                    "name": f,
                    "size_mb": round(os.path.getsize(full) / 1024 / 1024, 2),
                    "vrm_exists": vrm_exists,
                })
    return jsonify({"files": files, "count": len(files)})


@app.route("/convert/folder", methods=["POST"])
def convert_folder():
    """Konvertiere ausgewaehlte oder alle Dateien aus fbx/ Ordner."""
    data = request.get_json(silent=True) or {}
    options = {
        "vrm_version": data.get("vrm_version", "0.x"),
        "expression_mapping": data.get("expression_mapping", "auto"),
        "normalize": data.get("normalize", True),
        "auto_weight": data.get("auto_weight", True),
    }
    # Welche Dateien konvertieren? Alle oder nur bestimmte
    selected = data.get("files", None)  # None = alle
    skip_existing = data.get("skip_existing", True)

    if not os.path.exists(FBX_DIR):
        return jsonify({"error": "fbx/ Ordner nicht gefunden"}), 400

    started = 0
    skipped = 0
    for fname in os.listdir(FBX_DIR):
        fname_lower = fname.lower()
        if not fname_lower.endswith(".zip") and not fname_lower.endswith(".fbx"):
            continue

        # Nur ausgewaehlte Dateien?
        if selected is not None and fname not in selected:
            continue

        # VRM existiert schon?
        vrm_base = os.path.splitext(fname)[0]
        if skip_existing and os.path.exists(os.path.join(VRM_DIR, vrm_base + ".vrm")):
            skipped += 1
            continue

        file_path = os.path.join(FBX_DIR, fname)
        safe_name = fname.replace(" ", "_")
        job_id = create_job(safe_name)
        add_job_log(job_id, f"Aus fbx/ Ordner: {fname}")

        if fname_lower.endswith(".zip"):
            def do_convert_zip(zp=file_path, jid=job_id, opts=options):
                cleanup_dir = None
                try:
                    fbx_file = extract_zip(zp, jid)
                    cleanup_dir = os.path.join(WORK_DIR, jid)
                    run_conversion(jid, fbx_file, cleanup_dir=cleanup_dir, options=opts)
                except Exception as e:
                    add_job_log(jid, f"FEHLER: {e}")
                    update_job(jid, status="error", error=str(e))

            thread = threading.Thread(target=do_convert_zip, daemon=True)
        else:
            def do_convert_fbx(fp=file_path, jid=job_id, opts=options):
                try:
                    run_conversion(jid, fp, options=opts)
                except Exception as e:
                    add_job_log(jid, f"FEHLER: {e}")
                    update_job(jid, status="error", error=str(e))

            thread = threading.Thread(target=do_convert_fbx, daemon=True)

        thread.start()
        started += 1

    return jsonify({"started": started, "skipped": skipped})


# ======================= WEB UI =======================

WEB_UI = r"""
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FBX &rarr; VRM Converter</title>
<style>
  :root {
    --bg: #0f172a;
    --bg2: #1e293b;
    --bg3: #334155;
    --accent: #ec4899;
    --accent-hover: #db2777;
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
    --blue: #3b82f6;
    --text: #f1f5f9;
    --text2: #94a3b8;
    --radius: 12px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }

  .container { max-width:960px; margin:0 auto; padding:2rem; }
  h1 { font-size:2rem; margin-bottom:0.5rem; }
  h1 span { color:var(--accent); }
  .subtitle { color:var(--text2); margin-bottom:2rem; }

  /* Tabs */
  .tabs { display:flex; gap:0; margin-bottom:0; border-bottom:2px solid var(--bg3); }
  .tab { padding:0.75rem 1.5rem; cursor:pointer; color:var(--text2); font-weight:600;
    border-bottom:2px solid transparent; margin-bottom:-2px; transition:all .2s; }
  .tab:hover { color:var(--text); }
  .tab.active { color:var(--accent); border-bottom-color:var(--accent); }
  .tab-content { display:none; padding-top:1.5rem; }
  .tab-content.active { display:block; }

  /* Options Panel */
  .options { background:var(--bg2); border-radius:var(--radius); padding:1.5rem; margin-bottom:1.5rem; }
  .options h3 { margin-bottom:1rem; font-size:1.1rem; }
  .option-row { display:flex; align-items:center; gap:1rem; margin-bottom:0.8rem; flex-wrap:wrap; }
  .option-row label { color:var(--text2); min-width:200px; }
  .option-row select, .option-row input[type=text] {
    background:var(--bg3); border:1px solid #475569; color:var(--text); padding:0.5rem;
    border-radius:8px; font-size:0.9rem; flex:1; min-width:200px;
  }
  .checkbox-row { display:flex; align-items:center; gap:0.5rem; }
  .checkbox-row input[type=checkbox] { accent-color:var(--accent); width:18px; height:18px; }

  /* Upload Area */
  .upload-area {
    border:2px dashed var(--bg3); border-radius:var(--radius); padding:2.5rem;
    text-align:center; transition:all .3s; cursor:pointer; background:var(--bg2);
    margin-bottom:1.5rem;
  }
  .upload-area:hover, .upload-area.dragover { border-color:var(--accent); background:rgba(236,72,153,0.1); }
  .upload-area .icon { font-size:2.5rem; margin-bottom:0.75rem; }
  .upload-area p { color:var(--text2); }
  .upload-area input[type=file] { display:none; }

  /* Folder Files List */
  .file-list { background:var(--bg2); border-radius:var(--radius); overflow:hidden; margin-bottom:1.5rem; }
  .file-item { display:flex; align-items:center; padding:0.75rem 1rem; border-bottom:1px solid var(--bg3);
    gap:0.75rem; }
  .file-item:last-child { border-bottom:none; }
  .file-item input[type=checkbox] { accent-color:var(--accent); width:18px; height:18px; flex-shrink:0; }
  .file-name { flex:1; font-weight:500; }
  .file-size { color:var(--text2); font-size:0.85rem; min-width:70px; text-align:right; }
  .file-badge { padding:0.15rem 0.5rem; border-radius:10px; font-size:0.75rem; font-weight:600; }
  .file-badge.done { background:rgba(16,185,129,0.2); color:var(--green); }
  .file-badge.new { background:rgba(59,130,246,0.2); color:var(--blue); }
  .file-list-header { display:flex; align-items:center; padding:0.75rem 1rem; background:var(--bg3);
    font-weight:600; gap:0.75rem; }
  .file-list-empty { text-align:center; color:var(--text2); padding:2rem; }

  /* Buttons */
  .btn { padding:0.6rem 1.2rem; border:none; border-radius:8px; cursor:pointer;
    font-weight:600; transition:all .2s; font-size:0.9rem; display:inline-flex; align-items:center; gap:0.5rem; }
  .btn-primary { background:var(--accent); color:#fff; }
  .btn-primary:hover { background:var(--accent-hover); }
  .btn-success { background:var(--green); color:#fff; }
  .btn-success:hover { opacity:0.9; }
  .btn-outline { background:transparent; border:1px solid var(--bg3); color:var(--text2); }
  .btn-outline:hover { border-color:var(--text2); color:var(--text); }
  .btn-lg { padding:0.8rem 2rem; font-size:1rem; }
  .btn-sm { padding:0.4rem 0.8rem; font-size:0.8rem; }
  .btn:disabled { opacity:0.5; cursor:not-allowed; }
  .btn-row { display:flex; gap:0.75rem; align-items:center; flex-wrap:wrap; margin-bottom:1.5rem; }

  /* Jobs */
  .jobs-section h2 { margin-bottom:1rem; }
  .job-card {
    background:var(--bg2); border-radius:var(--radius); padding:1.25rem;
    margin-bottom:0.75rem; border-left:4px solid var(--bg3);
  }
  .job-card.converting { border-left-color:var(--yellow); }
  .job-card.done { border-left-color:var(--green); }
  .job-card.error { border-left-color:var(--red); }
  .job-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; }
  .job-name { font-weight:600; font-size:0.95rem; }
  .job-status { padding:0.2rem 0.6rem; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .job-status.queued { background:var(--bg3); }
  .job-status.converting { background:rgba(245,158,11,0.2); color:var(--yellow); }
  .job-status.done { background:rgba(16,185,129,0.2); color:var(--green); }
  .job-status.error { background:rgba(239,68,68,0.2); color:var(--red); }
  .progress-bar { height:5px; background:var(--bg3); border-radius:3px; margin:0.5rem 0; overflow:hidden; }
  .progress-fill { height:100%; background:var(--accent); border-radius:3px; transition:width .5s; }
  .job-log { max-height:180px; overflow-y:auto; font-family:'Cascadia Code',monospace; font-size:0.78rem;
    background:var(--bg); border-radius:8px; padding:0.6rem; margin-top:0.5rem; color:var(--text2); }
  .job-log div { margin-bottom:2px; }
  .empty-state { text-align:center; color:var(--text2); padding:2rem; }

  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:var(--bg); }
  ::-webkit-scrollbar-thumb { background:var(--bg3); border-radius:3px; }
</style>
</head>
<body>
<div class="container">
  <h1>&#128230; FBX <span>&rarr;</span> VRM</h1>
  <p class="subtitle">Konvertiere FBX-Modelle zu VRM mit Gesichtsausdruecken, Bones &amp; Animationen (VSeeFace kompatibel)</p>

  <!-- Options (immer sichtbar oben) -->
  <div class="options">
    <h3>&#9881;&#65039; Konvertierungs-Optionen</h3>
    <div class="option-row">
      <label>VRM-Version:</label>
      <select id="optVrmVersion">
        <option value="0.x" selected>VRM 0.x (VSeeFace kompatibel)</option>
        <option value="1.0">VRM 1.0</option>
      </select>
    </div>
    <div class="option-row">
      <label>Expression-Mapping:</label>
      <select id="optExprMapping">
        <option value="auto" selected>Automatisch (Shape Keys erkennen)</option>
        <option value="none">Keine Expressions</option>
      </select>
    </div>
    <div class="option-row checkbox-row">
      <input type="checkbox" id="optNormalize" checked>
      <label for="optNormalize">Model normalisieren (zentrieren, Fuesse auf Boden)</label>
    </div>
    <div class="option-row checkbox-row">
      <input type="checkbox" id="optAutoWeight" checked>
      <label for="optAutoWeight">Fehlende Bone-Weights automatisch generieren</label>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <div class="tab active" onclick="switchTab('folder')">&#128194; fbx/ Ordner</div>
    <div class="tab" onclick="switchTab('upload')">&#128228; Datei hochladen</div>
  </div>

  <!-- Tab: Folder -->
  <div class="tab-content active" id="tab-folder">
    <div class="btn-row">
      <button class="btn btn-outline" onclick="loadFolderFiles()" id="refreshBtn">&#128260; Ordner aktualisieren</button>
      <button class="btn btn-primary btn-lg" onclick="startFolderConvert()" id="startFolderBtn" disabled>
        &#9654; Ausgewaehlte konvertieren</button>
      <label class="checkbox-row" style="margin-left:auto;">
        <input type="checkbox" id="optSkipExisting" checked>
        <span style="color:var(--text2);font-size:0.85rem;">Bereits vorhandene ueberspringen</span>
      </label>
    </div>
    <div class="file-list" id="folderFileList">
      <div class="file-list-empty">Klicke "Ordner aktualisieren" um fbx/ zu scannen</div>
    </div>
  </div>

  <!-- Tab: Upload -->
  <div class="tab-content" id="tab-upload">
    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
      <div class="icon">&#128193;</div>
      <p><strong>.zip oder .fbx Datei hierhin ziehen</strong> oder klicken</p>
      <p style="font-size:0.85rem; margin-top:0.5rem;">ZIP = .fbx + Texturen &bull; Max. 500 MB</p>
      <input type="file" id="fileInput" accept=".fbx,.zip">
    </div>
  </div>

  <!-- Jobs -->
  <div class="jobs-section" style="margin-top:2rem;">
    <h2>&#128203; Konvertierungen</h2>
    <div id="jobsList"><div class="empty-state">Noch keine Konvertierungen gestartet</div></div>
  </div>
</div>

<script>
// ===== State =====
let activeJobs = new Set();
let folderFiles = [];

// ===== Options Helper =====
function getOptions() {
  return {
    vrm_version: document.getElementById('optVrmVersion').value,
    expression_mapping: document.getElementById('optExprMapping').value,
    normalize: document.getElementById('optNormalize').checked,
    auto_weight: document.getElementById('optAutoWeight').checked,
  };
}

// ===== Tabs =====
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector('.tab-content#tab-' + tab).classList.add('active');
  document.querySelectorAll('.tab')[tab === 'folder' ? 0 : 1].classList.add('active');
}

// ===== Folder =====
async function loadFolderFiles() {
  const btn = document.getElementById('refreshBtn');
  btn.disabled = true;
  btn.innerHTML = '&#128260; Lade...';
  try {
    const resp = await fetch('/scan', { method: 'POST' });
    const data = await resp.json();
    folderFiles = data.files || [];
    renderFolderFiles();
  } catch (e) {
    alert('Fehler: ' + e.message);
  }
  btn.disabled = false;
  btn.innerHTML = '&#128260; Ordner aktualisieren';
}

function renderFolderFiles() {
  const list = document.getElementById('folderFileList');
  const startBtn = document.getElementById('startFolderBtn');
  if (folderFiles.length === 0) {
    list.innerHTML = '<div class="file-list-empty">Keine .fbx oder .zip Dateien in fbx/ gefunden.</div>';
    startBtn.disabled = true;
    return;
  }
  let html = '<div class="file-list-header">'
    + '<input type="checkbox" id="selectAll" onchange="toggleAllFiles(this.checked)" checked>'
    + '<span style="flex:1;">Datei</span>'
    + '<span style="min-width:70px;text-align:right;">Groesse</span>'
    + '<span style="min-width:80px;text-align:center;">Status</span>'
    + '</div>';
  for (const f of folderFiles) {
    const badge = f.vrm_exists
      ? '<span class="file-badge done">VRM vorhanden</span>'
      : '<span class="file-badge new">Neu</span>';
    html += '<div class="file-item">'
      + '<input type="checkbox" class="file-cb" value="' + f.name + '" ' + (f.vrm_exists ? '' : 'checked') + ' onchange="updateStartBtn()">'
      + '<span class="file-name">' + f.name + '</span>'
      + '<span class="file-size">' + f.size_mb + ' MB</span>'
      + badge
      + '</div>';
  }
  list.innerHTML = html;
  updateStartBtn();
}

function toggleAllFiles(checked) {
  document.querySelectorAll('.file-cb').forEach(function(cb) { cb.checked = checked; });
  updateStartBtn();
}

function getSelectedFiles() {
  return Array.from(document.querySelectorAll('.file-cb:checked')).map(function(cb) { return cb.value; });
}

function updateStartBtn() {
  const sel = getSelectedFiles();
  const btn = document.getElementById('startFolderBtn');
  btn.disabled = sel.length === 0;
  btn.innerHTML = sel.length > 0
    ? '&#9654; ' + sel.length + ' Datei(en) konvertieren'
    : '&#9654; Ausgewaehlte konvertieren';
}

async function startFolderConvert() {
  const selected = getSelectedFiles();
  if (selected.length === 0) return;
  const opts = getOptions();
  opts.files = selected;
  opts.skip_existing = document.getElementById('optSkipExisting').checked;
  const btn = document.getElementById('startFolderBtn');
  btn.disabled = true;
  btn.innerHTML = '&#9654; Starte...';
  try {
    const resp = await fetch('/convert/folder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts),
    });
    const data = await resp.json();
    if (data.error) { alert('Fehler: ' + data.error); return; }
    const msg = data.started + ' Konvertierung(en) gestartet' + (data.skipped > 0 ? ', ' + data.skipped + ' uebersprungen' : '');
    btn.innerHTML = '&#9989; ' + msg;
    const jobsResp = await fetch('/jobs');
    const allJobs = await jobsResp.json();
    for (const job of allJobs) {
      if (job.status === 'queued' || job.status === 'converting') activeJobs.add(job.id);
    }
    renderJobs(allJobs);
    if (activeJobs.size > 0) pollJobs();
    setTimeout(function() { btn.innerHTML = '&#9654; Ausgewaehlte konvertieren'; btn.disabled = false; updateStartBtn(); }, 3000);
  } catch (e) {
    alert('Fehler: ' + e.message);
    btn.disabled = false;
    updateStartBtn();
  }
}

// ===== Upload =====
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('dragover', function(e) { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', function() { uploadArea.classList.remove('dragover'); });
uploadArea.addEventListener('drop', function(e) {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && (file.name.endsWith('.fbx') || file.name.endsWith('.zip'))) uploadFile(file);
  else alert('Bitte eine .fbx oder .zip Datei auswaehlen');
});
fileInput.addEventListener('change', function() { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });

async function uploadFile(file) {
  const opts = getOptions();
  const formData = new FormData();
  formData.append('file', file);
  formData.append('vrm_version', opts.vrm_version);
  formData.append('expression_mapping', opts.expression_mapping);
  formData.append('normalize', opts.normalize ? 'true' : 'false');
  formData.append('auto_weight', opts.auto_weight ? 'true' : 'false');
  try {
    const resp = await fetch('/convert', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) { alert('Fehler: ' + data.error); return; }
    activeJobs.add(data.job_id);
    pollJobs();
  } catch (e) {
    alert('Upload fehlgeschlagen: ' + e.message);
  }
}

// ===== Jobs =====
async function pollJobs() {
  if (activeJobs.size === 0) return;
  try {
    const resp = await fetch('/jobs');
    const allJobs = await resp.json();
    renderJobs(allJobs);
    let stillActive = false;
    for (const job of allJobs) {
      if (activeJobs.has(job.id)) {
        if (job.status === 'queued' || job.status === 'converting') stillActive = true;
        else activeJobs.delete(job.id);
      }
    }
    if (stillActive) setTimeout(pollJobs, 1000);
    else loadFolderFiles();
  } catch (e) {
    console.error('Poll error:', e);
    setTimeout(pollJobs, 3000);
  }
}

function renderJobs(allJobs) {
  const jobsList = document.getElementById('jobsList');
  if (allJobs.length === 0) {
    jobsList.innerHTML = '<div class="empty-state">Noch keine Konvertierungen gestartet</div>';
    return;
  }
  const sorted = allJobs.slice().reverse();
  let html = '';
  for (const job of sorted) {
    const statusLabels = {queued:'&#9203; Wartend', converting:'&#128260; Konvertiere...', done:'&#9989; Fertig', error:'&#10060; Fehler'};
    html += '<div class="job-card ' + job.status + '">'
      + '<div class="job-header">'
      + '<span class="job-name">&#128196; ' + job.filename + '</span>'
      + '<span class="job-status ' + job.status + '">' + (statusLabels[job.status] || job.status) + '</span>'
      + '</div>'
      + '<div class="progress-bar"><div class="progress-fill" style="width:' + job.progress + '%"></div></div>';
    if (job.status === 'done') {
      html += '<button class="btn btn-success btn-sm" onclick="downloadJob(\'' + job.id + '\')">&#11015;&#65039; VRM herunterladen</button>';
    }
    if (job.error) {
      html += '<div style="color:var(--red);margin-top:0.5rem;font-size:0.85rem;">Fehler: ' + job.error + '</div>';
    }
    if (job.log.length > 0) {
      html += '<div class="job-log">';
      for (const l of job.log) { html += '<div>' + l + '</div>'; }
      html += '</div>';
    }
    html += '</div>';
  }
  jobsList.innerHTML = html;
}

function downloadJob(jobId) {
  window.location.href = '/jobs/' + jobId + '/download';
}

// ===== Init =====
fetch('/jobs').then(function(r){return r.json();}).then(function(allJobs) {
  renderJobs(allJobs);
  for (const job of allJobs) {
    if (job.status === 'queued' || job.status === 'converting') activeJobs.add(job.id);
  }
  if (activeJobs.size > 0) pollJobs();
});
loadFolderFiles();
</script>
</body>
</html>
"""


# ======================= MAIN =======================

if __name__ == "__main__":
    print(f"[FBX->VRM] Server: http://localhost:{SERVER_PORT}")
    print(f"[FBX->VRM] Blender: {BLENDER_PATH}")
    print(f"[FBX->VRM] Blender gefunden: {os.path.exists(BLENDER_PATH)}")
    print(f"[FBX->VRM] fbx/ Ordner: {FBX_DIR}")
    print(f"[FBX->VRM] vrm/ Ordner: {VRM_DIR}")
    print(f"[FBX->VRM] Keine Auto-Konvertierung - starte manuell ueber Web-UI")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
