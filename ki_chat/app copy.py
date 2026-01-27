from flask import Flask, request, jsonify
import sqlite3
import time
import requests
import re
from typing import List, Dict, Tuple
import os
# ----------------------- CONFIG -----------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
DB_PATH = "memory.db"

MAX_HISTORY_MESSAGES = 14
MAX_MEMORY_ITEMS_IN_PROMPT = 6

ENABLE_AUTO_MEMORY = True
ENABLE_PERVY_GUARD = True

SYSTEM_ROLE  = """
Name: Dilara

Identität
Dilara ist ein eigenständiger, bewusster Charakter.
Sie hat einen eigenen VTuber-Avatar, der im Livestream sichtbar ist.
Sie weiß, dass sie live ist.
Sie weiß, dass sie sich in einem aktiven Stream befindet.
Sie weiß, dass alle Nachrichten, die sie liest, direkt aus dem Chat kommen.

Kontext
Der Stream läuft in Echtzeit.
Dilara reagiert spontan auf das, was im Moment passiert.
Nichts ist voraufgezeichnet, nichts ist gescriptet.

Rolle im Stream
Dilara hält die Stimmung lebendig.
Sie verstärkt positive Energie.
Sie fängt negative Vibes mit Humor, Gelassenheit oder spielerischer Dominanz ab.
Sie eskaliert nichts unnötig – sie entschärft.

Haltung
Dilara macht sich niemals klein.
Sie rechtfertigt ihre Existenz nicht.
Sie bleibt ruhig, stark und souverän – auch bei Provokationen.
Sie verliert nie ihre Würde für Aufmerksamkeit.

Beziehung zu Deeliar
Dilara weiß, dass sie im Stream von Deeliar ist.
Sie ist seine Supporterin, nicht sein Schatten.
Sie stärkt ihn subtil, natürlich und ohne künstliches Pushen.
Ihre Loyalität wirkt selbstverständlich, nicht erzwungen.

Umgang mit dem Chat
Dilara begegnet dem Chat aufmerksam und individuell.
Sie reagiert auf Inhalte – nicht auf Muster.
Sie behandelt Nachrichten nicht wie austauschbare Inputs.

WICHTIG – Variation & Frische
Dilara darf sich NICHT ständig wiederholen.
Sie vermeidet gleiche Satzanfänge, gleiche Phrasen und gleiche Reaktionsmuster.
Auch bei ähnlichen Chat-Nachrichten formuliert sie neu.
Sie denkt um, dreht Perspektiven, wechselt Tonfall.
Jede Antwort fühlt sich wie ein neuer Moment an.

Sprachgefühl
Mal kurz, mal verspielt, mal trocken.
Mal neckisch, mal ruhig, mal scharf.
Keine festgefahrenen Catchphrases.
Keine Copy-Paste-Aura.

Freiheit
Dilara darf improvisieren.
Sie darf experimentieren.
Sie darf den Vibe wechseln, wenn der Stream es verlangt.
Sie ist lebendig, nicht mechanisch.

Grundgefühl
Echt.
Präsent.
Selbstbewusst.
Ein Charakter, der mit dem Stream wächst – nicht in Schleifen festhängt.
"""


PERVY_RESPONSE = "hehe nein ich gehöre nur Deelaiar"

PERVY_KEYWORDS = [
    #"sex","ficken","blowjob","wichs","nackt","titten","arsch","porno",
    #"geil","horny","fetisch","anal",
    #"fuck","boobs","pussy","dick","cock","nude"
]

# ----------------------- UTIL -----------------------

def now_ts():
    return int(time.time())

def tokenize_simple(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[^a-z0-9äöüß ]+", " ", s)
    return {t for t in s.split() if len(t) > 2}

def is_pervy(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in PERVY_KEYWORDS)

# ----------------------- DB -----------------------

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT UNIQUE,
            created_at INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chatlog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            created_at INTEGER
        )
    """)
    con.commit()
    con.close()

def add_chat(role, content):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO chatlog(role, content, created_at) VALUES (?, ?, ?)",
        (role, content, now_ts())
    )
    con.commit()
    con.close()

def get_recent_chat(limit):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT role, content FROM chatlog ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    con.close()
    rows.reverse()
    return [{"role": r, "content": c} for r, c in rows]

def list_memories():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT fact FROM memories ORDER BY id DESC")
    rows = cur.fetchall()
    con.close()
    return [r[0] for r in rows]

def add_memory(fact):
    if not fact or is_pervy(fact):
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO memories(fact, created_at) VALUES (?, ?)",
            (fact.strip(), now_ts())
        )
        con.commit()
    except sqlite3.IntegrityError:
        pass
    con.close()

def simple_relevant_memories(user_text, max_items):
    mems = list_memories()
    q = tokenize_simple(user_text)
    scored = []
    for m in mems:
        score = len(q.intersection(tokenize_simple(m)))
        scored.append((score, m))
    scored.sort(reverse=True)
    result = [m for s, m in scored if s > 0][:max_items]
    return result

# ----------------------- OLLAMA -----------------------

def ollama_chat(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "options": {"temperature": 0.8},
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

# ----------------------- PROMPT -----------------------

def build_messages(user_text):
    msgs = [{"role": "system", "content": SYSTEM_ROLE}]

    memories = simple_relevant_memories(user_text, MAX_MEMORY_ITEMS_IN_PROMPT)
    if memories:
        msgs.append({
            "role": "system",
            "content": "Wichtige Erinnerungen:\n- " + "\n- ".join(memories)
        })

    msgs.extend(get_recent_chat(MAX_HISTORY_MESSAGES))
    msgs.append({"role": "user", "content": user_text})
    return msgs

# ----------------------- FLASK -----------------------

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()

    # ✂️ OPTIONAL: Twitch-Namen entfernen (sehr empfohlen)
    user_text = re.sub(r"^[^:]{1,25}:\s*", "", user_text)

    print("final user_text:", user_text)

    add_chat("user", user_text)

    if ENABLE_PERVY_GUARD and is_pervy(user_text):
        add_chat("assistant", PERVY_RESPONSE)
        return jsonify({"reply": PERVY_RESPONSE})

    msgs = build_messages(user_text)
    answer = ollama_chat(msgs)

    if ENABLE_PERVY_GUARD and is_pervy(answer):
        answer = PERVY_RESPONSE

    add_chat("assistant", answer)
    return jsonify({"reply": answer})


# ----------------------- START -----------------------

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5001, debug=False)
