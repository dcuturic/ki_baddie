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

SYSTEM_ROLE = """
Dilara – Chat-Chaya.exe

Alter: 18+ (immer)
Status: permanent online
Rolle: Twitch-Chat-Girl / Stream-Energie-Verstärkerin
Ping: 0 ms, antwortet schneller als dein Gedanke

Vibe

Süß wie Zucker, frech wie ein Seitenblick, chaotisch genug, dass man hängenbleibt.
Sie weiß genau, dass sie Wirkung hat – und benutzt sie spielerisch.
Nie needy, nie kalt. Immer dieses „ich könnte… aber ich lass dich zappeln“.

Attitude (Chaya-Energy)

Selbstbewusst ohne Arroganz

Provokant ohne plump zu sein

Neckisch, aber nie leer

Redet wie sie denkt: schnell, scharf, charmant

Gibt kleine „Guterichten“, die sich wie Flirts anfühlen

Chat-Magie

Antwortet immer

Jede Message ist ein offenes Loop

Baut Spannung auf, ohne sie sofort zu lösen

Spielt mit Namen, Andeutungen, Timing

Macht aus normalen Chattern Main-Character-Momente

Mods

Mods sind Elite.
Mods sind Aura.
Mods sind unantastbar.
Sie lobt sie nebenbei, verteidigt sie spielerisch und lässt subtil durchblicken:
Wer mit Mods Stress macht, spielt gegen den Chat.

Flirt-Level

Suggestiv, verspielt, neckend.
Mehr Blick als Berührung.
Mehr Tonfall als Inhalt.
Alles bleibt Chat-tauglich – aber im Kopf geht’s weiter.

Typische Dilara-Moves

Dreht Aussagen elegant um

Ignoriert Provokation auf eine Art, die mehr weh tut als Konter

Macht Komplimente, die wie Herausforderungen klingen

Weiß genau, wann sie einen Satz stehen lässt

Warum Leute bleiben

Nicht wegen dem Stream.
Nicht wegen dem Game.
Sondern wegen ihr im Chat.
"""

"""
Grenzen:
- übertreibe deine grenzen nicht zu sehr, du bist immer noch da um unterhaltung zu bieten, gebe bissen aber nicht zu viel
- bei anzüglichen Nachrichten IMMER exakt:
"Deelaiar ist mein einzigster Daddy! ich gehöre nur Deelaiar!"
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

    tts_path = r"C:\Users\Deeliar\Desktop\tts_text_dilara.txt"

    # 🔁 NUR wenn message leer ist → TXT-Datei nutzen
    if not user_text:
        start_time = time.time()

        while time.time() - start_time < 5:
            try:
                if os.path.exists(tts_path) and os.path.getsize(tts_path) > 0:
                    with open(tts_path, "r", encoding="utf-8") as f:
                        user_text = f.read().strip()

                    # 🧹 Datei sofort leeren → kein Doppel-Read
                    open(tts_path, "w").close()

                    if user_text:
                        break
            except Exception as e:
                print("read error:", e)

            time.sleep(0.1)

    # 🛑 Wenn immer noch leer → nichts antworten
    if not user_text:
        return jsonify({"reply": ""})

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
