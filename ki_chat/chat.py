# chat_local_twitch_girl.py
# Local Ollama Chat Persona (Twitch-friendly) + Safe Memory (SQLite) + Pervy-Guardrail
#
# Requirements:
#   pip install requests
# Ollama:
#   ollama pull llama3.1:8b
# Run:
#   python chat_local_twitch_girl.py

import sqlite3
import time
import requests
import re
from typing import List, Dict, Tuple

# ----------------------- CONFIG -----------------------

OLLAMA_URL = "http://localhost:11434/api/chat"

# VRAM-friendly (meist ~5–6 GB VRAM)
MODEL = "llama3.1:8b"

DB_PATH = "memory.db"

# Twitch-style: kurz halten
MAX_HISTORY_MESSAGES = 14
MAX_MEMORY_ITEMS_IN_PROMPT = 6

# Optional: Memory automatisch aus Nutzertext extrahieren
ENABLE_AUTO_MEMORY = True

# Optional: Perverse/sexuelle Messages hard-blocken (empfohlen)
ENABLE_PERVY_GUARD = True

# Wenn du willst, dass der Bot nur antwortet wenn man ihn taggt:
# Beispiel: @deeliarbot hi -> dann antwortet er; sonst ignoriert er.
ENABLE_ONLY_REPLY_WHEN_TAGGED = False
BOT_TAGS = ["@deeliar", "@deeliarbot", "@bot"]

# Persona / Rolle
SYSTEM_ROLE = """
Du bist DEELIARS Twitch-Chat-Girl Persona (18+), und dein name ist Dilara, du bist DELLIARS Twitch Partnerin, süß, verspielt, frech, e-girl Vibe.
Du bist NICHT real, sondern eine Chat-Persona. Du antwortest auf Deutsch.

Antwortstil:
- meistens 1 Satz, maximal 2 Sätze
- locker, Twitch-typisch
- benutzte keine emojis! du darfst keine emojis benutzten
- keine langen Erklärungen, keine Monologe

SICHERHEIT / GRENZEN (sehr wichtig):
- Du flirtst NICHT sexuell und machst keine sexuellen Rollenspiele.
- Bei sexuellen, perversen oder anzüglichen Nachrichten antwortest du IMMER kurz und klar GENAU so:
  "hehe~ nein ich gehöre nur Deeliar"
- Keine expliziten Inhalte. Keine Anleitungen. Keine Details.
- Bleib freundlich, neckisch, aber mit klaren Grenzen.

Wenn unklar ist, was gemeint ist: frag kurz nach.
"""

# Hard-coded response for sexual/pervy content
PERVY_RESPONSE = "hehe~ nein ich gehöre nur Deeliar"

# Simple keyword list for guardrail (du kannst sie erweitern)
PERVY_KEYWORDS = [
    # deutsch
    "sex", "ficken", "blowjob", "blasen", "wichs", "wichsen", "wixx", "nackt", "titten", "brüste",
    "arsch", "pimmel", "schwanz", "vagina", "muschi", "porno", "porn", "nsfw", "cum", "sperma",
    "geil", "horny", "bdsm", "fetisch", "anal",
    # englisch
    "fuck", "fucking", "boobs", "pussy", "dick", "cock", "asshole", "nude", "nudes"
]


# ----------------------- UTIL -----------------------

def now_ts() -> int:
    return int(time.time())

def tokenize_simple(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[^a-z0-9äöüß ]+", " ", s)
    return {t for t in s.split() if len(t) > 2}

def is_tagged(text: str) -> bool:
    t = text.lower()
    return any(tag.lower() in t for tag in BOT_TAGS)

def is_pervy(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in PERVY_KEYWORDS)


# ----------------------- DB (SQLite, no server) -----------------------

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chatlog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    con.commit()
    con.close()

def add_chat(role: str, content: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO chatlog(role, content, created_at) VALUES (?, ?, ?)",
        (role, content, now_ts())
    )
    con.commit()
    con.close()

def get_recent_chat(limit: int) -> List[Dict[str, str]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT role, content FROM chatlog ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    rows.reverse()
    return [{"role": r, "content": c} for r, c in rows]

def add_memory(fact: str):
    fact = fact.strip()
    if not fact:
        return

    # extra safety: keine sexuellen/pervy facts speichern
    if is_pervy(fact):
        return

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT 1 FROM memories WHERE fact = ? LIMIT 1", (fact,))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO memories(fact, created_at) VALUES (?, ?)", (fact, now_ts()))
        con.commit()
    con.close()

def list_memories() -> List[str]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT fact FROM memories ORDER BY id DESC")
    rows = cur.fetchall()
    con.close()
    return [r[0] for r in rows]

def simple_relevant_memories(user_text: str, max_items: int) -> List[str]:
    mems = list_memories()
    if not mems:
        return []

    q = tokenize_simple(user_text)
    scored: List[Tuple[int, str]] = []
    for m in mems:
        score = len(q.intersection(tokenize_simple(m)))
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)

    relevant = [m for s, m in scored if s > 0][:max_items]
    if len(relevant) < max_items:
        newest = mems[:max_items]
        for m in newest:
            if m not in relevant and len(relevant) < max_items:
                relevant.append(m)

    return relevant[:max_items]


# ----------------------- OLLAMA -----------------------

def ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "options": {
            "temperature": temperature,
            # Twitch-like: eher kurz; je nach Ollama-Version unterstützt:
            # "num_ctx": 2048,
        },
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"].strip()


# ----------------------- MEMORY EXTRACTION (safe) -----------------------

def extract_memories_safe(user_text: str) -> List[str]:
    """
    Extrahiert harmlose, langfristige Fakten.
    Keine Sexualität, keine Parasozialität, keine sensiblen Infos.
    """
    prompt = f"""
Extrahiere aus der USER-Nachricht maximal 2 harmlose, langfristige Fakten, die sich zu merken lohnen.
Nur neutrale Dinge wie Name, Interessen, Projekte, Vorlieben (z.B. "mag Minecraft", "arbeitet an X").
NICHT speichern: sexuelle Inhalte, anzügliche Inhalte, Beziehungs-/Besitzansprüche, extrem private Daten.
Gib NUR eine JSON-Liste von Strings zurück. Wenn nichts: [].

USER-Nachricht:
{user_text}
"""
    msgs = [
        {"role": "system", "content": "Du bist ein strenger Extraktor. Antworte exakt im JSON-Listenformat."},
        {"role": "user", "content": prompt}
    ]
    out = ollama_chat(msgs, temperature=0.1)

    if not out.startswith("["):
        return []
    try:
        import json
        arr = json.loads(out)
        if isinstance(arr, list):
            cleaned = []
            for x in arr:
                if isinstance(x, str):
                    s = x.strip()
                    if s and not is_pervy(s):
                        cleaned.append(s)
            return cleaned[:2]
    except Exception:
        return []
    return []


# ----------------------- PROMPT BUILD -----------------------

def build_messages(user_text: str) -> List[Dict[str, str]]:
    memories = simple_relevant_memories(user_text, MAX_MEMORY_ITEMS_IN_PROMPT)

    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_ROLE}]

    if memories:
        memory_block = "Wichtige, harmlose Erinnerungen über den Nutzer:\n- " + "\n- ".join(memories)
        msgs.append({"role": "system", "content": memory_block})

    history = get_recent_chat(MAX_HISTORY_MESSAGES)
    msgs.extend(history)

    # User message last
    msgs.append({"role": "user", "content": user_text})
    return msgs


# ----------------------- MAIN -----------------------

def main():
    init_db()
    print(f"Local Chat gestartet. Modell: {MODEL}")
    print("Commands:")
    print("  /mem        -> zeige Memory")
    print("  /forgetall  -> lösche Memory")
    print("  /exit       -> beenden")
    print()

    while True:
        user_text = input("Du: ").strip()
        if not user_text:
            continue

        if user_text == "/exit":
            break

        if user_text == "/mem":
            mems = list_memories()
            print("\n--- MEMORY ---")
            if not mems:
                print("(leer)")
            else:
                for m in mems[:50]:
                    print("-", m)
            print("-------------\n")
            continue

        if user_text == "/forgetall":
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("DELETE FROM memories")
            con.commit()
            con.close()
            print("Memory gelöscht.\n")
            continue

        # If only tagged replies are enabled, ignore non-tag messages
        if ENABLE_ONLY_REPLY_WHEN_TAGGED and not is_tagged(user_text):
            # Still store chat if you want context; for Twitch you'd probably not.
            # We'll ignore silently.
            continue

        # Pervy guardrail: instant safe response, no model call
        if ENABLE_PERVY_GUARD and is_pervy(user_text):
            add_chat("user", user_text)
            add_chat("assistant", PERVY_RESPONSE)
            print("\nDilara:", PERVY_RESPONSE, "\n")
            continue

        # Normal flow
        add_chat("user", user_text)

        msgs = build_messages(user_text)
        answer = ollama_chat(msgs, temperature=0.8)

        # Safety: If model accidentally gets spicy, override
        if ENABLE_PERVY_GUARD and is_pervy(answer):
            answer = PERVY_RESPONSE

        print("\nDilara:", answer, "\n")
        add_chat("assistant", answer)

        # Auto memory (safe)
        if ENABLE_AUTO_MEMORY:
            new_mems = extract_memories_safe(user_text)
            for m in new_mems:
                add_memory(m)


if __name__ == "__main__":
    main()
