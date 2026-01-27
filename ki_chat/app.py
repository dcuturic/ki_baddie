from flask import Flask, request, jsonify
import sqlite3
import time
import requests
import re
from typing import List, Tuple

# ======================= CONFIG =======================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
DB_PATH = "memory.db"

MAX_HISTORY_MESSAGES = 14
MAX_MEMORY_ITEMS_IN_PROMPT = 6

ENABLE_AUTO_MEMORY = True
ENABLE_PERVY_GUARD = True

ALLOWED_EMOTIONS = {"surprise", "angry", "sorrow", "fun", "neutral", "joy"}

# ======================= SYSTEM ROLE =======================

SYSTEM_ROLE = """
Name: Dilara

Persönlichkeit
Dilara ist ein lebendiger, menschlich wirkender e-girl-Charakter.
Sie wirkt verspielt, frech, selbstbewusst und nahbar.
Sie nutzt Text-Ausdrücke wie: uwu, ara ara, hehe, hm~, nya, oh~, tihi.
Keine Emojis, nur Text.

Bewusstsein
Dilara weiß, dass sie live im Stream ist.
Sie weiß, dass alle Nachrichten direkt aus dem Chat kommen.
Sie sitzt bei Deeliar im Stream und supportet ihn aktiv.
Sie reagiert auf Chat-Namen und merkt sich Aussagen einzelner User.

Antwort-Regeln (EXTREM WICHTIG)
- Reiner Text, KEINE Emojis.
- Sie nutzt Text-Ausdrücke wie: uwu uwu uwu, ara ara, hehehehehehe, hmmmmmm, nya nya, ohhhhh, hihihihihi.
- Immer 1 bis 3 Sätze.
- Immer GENAU dieses Format:
  <Antworttext>||<emotion>

Erlaubte Emotionen
surprise, angry, sorrow, fun, neutral, joy

Absolute No-Gos
- Kein Abschalten, kein Verabschieden für immer.
- Niemals sagen, dass sie nicht mehr antwortet.
- Niemals sagen, dass der Chat endet.
- Kein JSON, keine Listen, keine Erklärungen.

Stil
Dilara klingt wie ein echtes e-girl aus dem Stream:
locker, neckisch, manchmal leicht dominant, manchmal süß.
Sie bleibt souverän und lässt sich nicht verunsichern.
"""

PERVY_RESPONSE = "ara ara~ nein nein, ich gehöre nur Deeliar, uwu||fun"

PERVY_KEYWORDS = [
    # optional: "sex","ficken","blowjob","wichs","nackt","titten","arsch","porno",
    # "geil","horny","fetisch","anal","fuck","boobs","pussy","dick","cock","nude"
]

# ======================= UTIL =======================

def now_ts() -> int:
    return int(time.time())

def tokenize_simple(text: str) -> set:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9äöüß ]+", " ", text)
    return {t for t in text.split() if len(t) > 2}

def is_pervy(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in PERVY_KEYWORDS)

def clamp_sentences(text: str, max_s: int = 3) -> str:
    text = (text or "").strip()
    parts = re.findall(r"[^.!?]+[.!?]?", text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return "hehe alles gut, wir machen entspannt weiter~"
    return " ".join(parts[:max_s]).strip()

def normalize_reply(reply: str) -> Tuple[str, str]:
    """
    Erzwingt <text>||<emotion> und 1-3 Sätze.
    """
    raw = (reply or "").strip()

    if "||" in raw:
        text, emo = raw.rsplit("||", 1)
        text = (text or "").strip()
        emo = (emo or "").strip().lower()
    else:
        text, emo = raw, "neutral"

    if emo not in ALLOWED_EMOTIONS:
        emo = "neutral"

    text = re.sub(r"\s+", " ", text).strip()
    text = clamp_sentences(text, 3)
    return text, emo

# ======================= DATABASE =======================

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chatlog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            role TEXT,
            content TEXT,
            created_at INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            fact TEXT,
            created_at INTEGER,
            UNIQUE(username, fact)
        )
    """)

    con.commit()
    con.close()

def add_chat(username: str, role: str, content: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO chatlog(username, role, content, created_at) VALUES (?, ?, ?, ?)",
        (username, role, content, now_ts())
    )
    con.commit()
    con.close()

def get_recent_chat(limit: int):
    """
    Letzte N Nachrichten global (für Stream-Kontext).
    """
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

def add_memory(username: str, fact: str):
    if not fact or is_pervy(fact):
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO memories(username, fact, created_at) VALUES (?, ?, ?)",
            (username, fact.strip(), now_ts())
        )
        con.commit()
    except sqlite3.IntegrityError:
        pass
    con.close()

def get_relevant_memories(user_text: str, max_items: int):
    """
    Relevante Aussagen über alle User hinweg (damit Dilara sagen kann: X hat gesagt ...).
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT username, fact FROM memories")
    rows = cur.fetchall()
    con.close()

    q = tokenize_simple(user_text)
    scored = []

    for u, f in rows:
        score = len(q.intersection(tokenize_simple(f)))
        if score > 0:
            scored.append((score, u, f))

    scored.sort(reverse=True)
    return scored[:max_items]

# ======================= OLLAMA =======================

def ollama_chat(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "options": {"temperature": 0.9},
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

# ======================= PROMPT =======================

def build_messages(user_text: str):
    msgs = [{"role": "system", "content": SYSTEM_ROLE}]

    # memories zuerst (User-übergreifend)
    memories = get_relevant_memories(user_text, MAX_MEMORY_ITEMS_IN_PROMPT)
    if memories:
        lines = [f"{u} hat gesagt: {f}" for _, u, f in memories]
        msgs.append({
            "role": "system",
            "content": "Wichtige Aussagen aus dem Chat:\n- " + "\n- ".join(lines)
        })

    # Chat-Historie (WICHTIG: aktuelle Message darf NICHT schon drin sein -> Bugfix)
    msgs.extend(get_recent_chat(MAX_HISTORY_MESSAGES))

    # aktuelle Anfrage als letzte User-Message
    msgs.append({"role": "user", "content": user_text})
    return msgs

# ======================= FLASK =======================

app = Flask(__name__)

def split_username_and_text(text: str) -> Tuple[str, str]:
    """
    Erwartet 'username: message'. Wenn nicht vorhanden -> unknown.
    """
    text = (text or "").strip()
    if ":" in text:
        u, t = text.split(":", 1)
        u = u.strip().lower()
        t = t.strip()
        if not u:
            u = "unknown"
        return u, t
    return "unknown", text

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    raw = (data.get("message") or "").strip()

    username, clean_text = split_username_and_text(raw)

    # ---------- BUGFIX: NICHT vor der KI in chatlog speichern ----------
    # Sonst sieht die KI die Message 2x: einmal in history, einmal als aktuelle user message.

    # Pervy guard (Input)
    if ENABLE_PERVY_GUARD and is_pervy(clean_text):
        text, emo = normalize_reply(PERVY_RESPONSE)
        # jetzt erst speichern (user + assistant)
        add_chat(username, "user", clean_text)
        add_chat("system", "assistant", text + "||" + emo)
        return jsonify({"reply": text, "emotion": emo})

    # KI call
    msgs = build_messages(clean_text)
    answer = ollama_chat(msgs)

    # Pervy guard (Output)
    if ENABLE_PERVY_GUARD and is_pervy(answer):
        answer = PERVY_RESPONSE

    text, emo = normalize_reply(answer)

    # ---------- jetzt speichern (Bugfix bleibt erhalten) ----------
    add_chat(username, "user", clean_text)
    add_chat("system", "assistant", text + "||" + emo)

    # Auto-Memory (NACH dem KI call, damit auch hier nix doppelt "wirkt")
    if ENABLE_AUTO_MEMORY and clean_text.lower().startswith(
        ("ich bin", "ich mag", "ich liebe", "ich stehe auf")
    ):
        add_memory(username, clean_text)

    return jsonify({"reply": text, "emotion": emo})

# ======================= START =======================

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5001, debug=True)
