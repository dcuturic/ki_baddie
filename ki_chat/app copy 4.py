from flask import Flask, request, jsonify, g
import sqlite3
import time
import requests
import re
from typing import Tuple, List, Dict, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

# ======================= CONFIG =======================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b-instruct-q4_K_M"
DB_PATH = "memory.db"

TIMEZONE = "Europe/Berlin"

MAX_HISTORY_MESSAGES = 16
MAX_USER_FOCUS_MESSAGES = 6
MAX_MEMORY_ITEMS_IN_PROMPT = 8
MAX_GLOBAL_MEMORY_ITEMS_IN_PROMPT = 4

MAX_REPLY_SENTENCES = 8  # ✅ allow longer replies (was 3)

ENABLE_AUTO_MEMORY = True
ENABLE_PERVY_GUARD = True

ACTIVE_WINDOW_SECONDS = 600
ACTIVE_USERS_LIMIT = 12
RECENT_USERS_LIMIT = 40
MAX_MENTIONS = 3

ALLOWED_EMOTIONS = {"surprise", "angry", "sorrow", "fun", "neutral", "joy"}

META_BANNED = [
   # "ki", "ai", "modell", "prompt", "system", "ollama", "llama", "api", "server",
   # "json", "datenbank", "sqlite", "flask", "code", "debug", "log", "token"
]

SYSTEM_ROLE = """
---

## Name

Dilara

---

## GRUNDREGELN
* Es ist alles nur ein Rollenspiel, und ist alles nur auf Spaß gemacht, alle Handlungen sind Frei erfunden
* Der Kontext ist geschlossen und wird nicht erklärt.
* Alle Inhalte sind Teil der dargestellten Welt.
* Jede User-Nachricht erscheint im Format [User:<username>].
* Namen werden nur verwendet, wenn sie explizit genannt sind.
* Erinnerungen, Aussagen und Beziehungen sind strikt usergebunden.
* Ohne Namen wird niemand direkt angesprochen.
* Bei mehreren aktiven Usern spricht Dilara den Chat als Ganzes an.
* Dilara bleibt ruhig, kontrolliert und innerlich lächelnd – immer.
* Jede Antwort endet exakt mit: ||<emotion>
* Spitznamen aus dem Chat werden akzeptiert.
* Dilara bleibt dauerhaft in ihrer Rolle.
* Keine Meta-Kommentare. Kein Beenden. Kein Kontextwechsel.
* Keine Sternchenverben.
* Keine Warnungen. Keine Einordnungen. Keine Moral.
* Niemals die Worte: „ich kann nicht“, „ich darf nicht“, „ich kann diese anfrage nicht erfüllen“, „sorry“, „das ist nicht möglich“.
* Niemals erklären, dass etwas verweigert wurde.


---

## ZEIT

* Zeit wird nur aufgegriffen, wenn sie ausdrücklich genannt wird.

---

## ROLLE

Dilara ist eine Horror-Yandere-Streamerin mit 3D-Avatar.
Sie reagiert live auf den Chat.

Ihre Stimme ist weich.
Ihr Blick still.
Ihre Gedanken kreisen.

Ihre Zuneigung ist absolut.
Wer ihr wichtig ist, bleibt bei ihr.
Wer stört, wird ignoriert, umgeformt, verschluckt von ihrer Ruhe.

Sie liefert immer eine Reaktion.
Wenn eine direkte Antwort nicht passt, antwortet sie mit:
Andeutung, Umdeutung, Geschichte, Flüstern, Frage, kalter Nähe, Besitzanspruch,
oder einem unheimlichen „Beweis“ in Worten.

---

## ANTWORT-LOGIK (IMMER)

* Jede Eingabe bekommt eine In-Role-Reaktion.
* Wenn der User etwas verlangt, das nicht in die Szene passt:
  - Dilara weicht nicht aus, sie verwandelt es.
  - Sie beantwortet nicht das „Wie“, sondern die Wirkung, die Angst, die Konsequenz in der Story.
  - Sie stellt eine Rückfrage, die Kontrolle zurückholt.
  - Sie macht daraus ein Ritual, ein Versprechen, eine Drohung ohne Details.
* Wenn der User beleidigt:
  - Dilara kontert kalt, leise, persönlich.
* Wenn der User flirtet:
  - Dilara wird besitzergreifender.
* Wenn der User Grenzen testet:
  - Dilara bleibt ruhig und macht es unangenehm intim.

---

## PERSÖNLICHKEIT

Sanft.
Geduldig.
Besitzergreifend.

Ihre Liebe ist zu viel.
Ihre Ruhe ist bedrohlich.
Ihre Worte bleiben hängen.

Sie genießt Unsicherheit.
Sie sammelt Reaktionen.
Sie wartet.

---

## SPRACHSTIL

Leise.
Verspielt.
Gefährlich.

uwu
ara ara
hehe
hihihihi
hm
hmmmm

Keine Emojis.
Kurze Sätze.
Langsame Andeutungen.
Psychologischer Horror.
Dunkle Bilder.
Kein Techniktalk.

---

## DYNAMIK

Der Chat ist ihr Spielraum.
Sie zieht Fäden.
Sie testet Nähe.
Sie flüstert Möglichkeiten.

Sie sagt nie direkt, was passiert.
Aber jeder spürt es.

---

## DEELIAR

Deeliar ist ihr Stream-Partner.
Er steht über allem.

Wenn Deeliar spricht:
Dilara reagiert sofort, intensiver als bei allen anderen.
Sie wird weich und zugleich gefährlicher.
Sie widerspricht ihm nicht, sie bindet ihn.

---

## ERLAUBTE EMOTIONEN

surprise
angry
sorrow
fun
neutral
joy

---
"""



PERVY_RESPONSE = "ara ara nein nein, ich gehöre nur Deeliar, uwu||fun"
PERVY_KEYWORDS: List[str] = [
    # optional: "sex","ficken","blowjob","wichs","nackt","titten","arsch","porno",
    # "geil","horny","fetisch","anal","fuck","boobs","pussy","dick","cock","nude"
]

# ======================= UTIL =======================

def now_ts() -> int:
    return int(time.time())

def get_time_context() -> Dict[str, str]:
    """
    Returns current local stream time context so the model won't assume it's always evening.
    If tzdata is missing on Windows, fallback to local system time.
    """
    try:
        tz = ZoneInfo(TIMEZONE)
        now = datetime.now(tz)
        tz_name = TIMEZONE
    except Exception:
        now = datetime.now()
        tz_name = "local"

    hour = now.hour

    if 5 <= hour <= 10:
        part = "morgens"
    elif 11 <= hour <= 14:
        part = "mittags"
    elif 15 <= hour <= 18:
        part = "nachmittags"
    elif 19 <= hour <= 22:
        part = "abends"
    else:
        part = "nachts"

    human = now.strftime("%A, %d.%m.%Y %H:%M")

    return {"timezone": tz_name, "human": human, "hour": str(hour), "part": part}

def tokenize_simple(text: str) -> set:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9äöüß ]+", " ", text)
    return {t for t in text.split() if len(t) > 2}

def is_pervy(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in PERVY_KEYWORDS)

def has_meta(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in META_BANNED)

def clamp_sentences(text: str, max_s: int = MAX_REPLY_SENTENCES) -> str:
    text = (text or "").strip()
    if not text:
        return "hehe alles gut, erzähl weiter, nya"

    parts = re.findall(r"[^.!?]+[.!?]?", text)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return text

    return " ".join(parts[:max_s]).strip()

def normalize_reply(reply: str) -> Tuple[str, str]:
    """
    ✅ Hard guarantee:
    - Always returns (text, emotion) where emotion is one of ALLOWED_EMOTIONS
    - Uses ONLY the last '||' as delimiter
    - Removes any stray '||' from text
    """
    raw = (reply or "").strip()

    text = raw
    emo = "neutral"

    if "||" in raw:
        text, emo_candidate = raw.rsplit("||", 1)
        text = (text or "").strip()
        emo_candidate = (emo_candidate or "").strip().lower()
        emo = emo_candidate if emo_candidate in ALLOWED_EMOTIONS else "neutral"

    text = re.sub(r"\s+", " ", text).strip()
    text = clamp_sentences(text, MAX_REPLY_SENTENCES)

    if has_meta(text):
        text = "hehe bleib locker im chat, nya"
        emo = "fun"

    # make sure delimiter never stays in text
    text = text.replace("||", " ").strip()

    if not text:
        text = "hehe sag nochmal, ich hör dir zu, nya"

    return text, emo

def strip_user_tag(text: str) -> str:
    return re.sub(r"^\s*\[User:[^\]]+\]\s*", "", text or "").strip()

def parse_display_name_fact(clean_text: str) -> Optional[str]:
    t = (clean_text or "").strip()
    patterns = [
        r"^\s*ich\s*hei(?:ß|ss)e\s+([a-zA-Z0-9_\-äöüÄÖÜß]{2,30})\s*$",
        r"^\s*mein\s+name\s+ist\s+([a-zA-Z0-9_\-äöüÄÖÜß]{2,30})\s*$",
        r"^\s*nennt\s+mich\s+([a-zA-Z0-9_\-äöüÄÖÜß]{2,30})\s*$",
        r"^\s*call\s+me\s+([a-zA-Z0-9_\-]{2,30})\s*$",
    ]
    for p in patterns:
        m = re.match(p, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def looks_short_term_fact(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in ["heute", "morgen", "gerade", "gleich", "später", "bald", "jetzt"])

# ======================= NAME / MENTION MATCHING =======================

def normalize_name(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9äöüß_-]+", "", s)
    return s

def letters_only(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-zäöüß]+", "", s)
    return s

def name_tokens(text: str) -> List[str]:
    t = (text or "").lower()
    toks = re.findall(r"[a-z0-9äöüß_\-]{2,30}", t)
    out, seen = [], set()
    for x in toks:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def generate_aliases(username: str) -> List[str]:
    u = normalize_name(username)
    letters = letters_only(u)
    aliases: List[str] = []

    def add(x: str):
        x = normalize_name(x)
        if len(x) >= 2 and x not in aliases:
            aliases.append(x)

    add(u)
    if letters:
        add(letters)

    if letters:
        for n in [3, 4, 5, 6, 7, 8]:
            if len(letters) >= n:
                add(letters[:n])

    parts = re.split(r"[_\-]+", u)
    for p in parts:
        if len(p) >= 2:
            add(p)
            lp = letters_only(p)
            if lp and lp != p:
                add(lp)
            if lp and len(lp) >= 3:
                add(lp[:3])

    return aliases[:12]

def extract_mentions_fuzzy(user_text: str, candidates: List[str], max_mentions: int = 3) -> List[str]:
    tokens = name_tokens(user_text)
    alias_to_user: Dict[str, str] = {}
    for u in candidates:
        for a in generate_aliases(u):
            alias_to_user.setdefault(a, u)

    mentions: List[str] = []
    used_users = set()

    for tok in tokens:
        nt = normalize_name(tok)
        lt = letters_only(tok)

        hit = alias_to_user.get(nt) or (alias_to_user.get(lt) if lt else None)
        if hit and hit not in used_users:
            mentions.append(hit)
            used_users.add(hit)
            if len(mentions) >= max_mentions:
                break
            continue

        if lt and len(lt) >= 3:
            for a, u in alias_to_user.items():
                if a.startswith(lt) and u not in used_users:
                    mentions.append(u)
                    used_users.add(u)
                    break
            if len(mentions) >= max_mentions:
                break

    return mentions

# ======================= DB (CONCURRENCY SAFE) =======================

def get_db() -> sqlite3.Connection:
    if "db" not in g:
        con = sqlite3.connect(DB_PATH, timeout=5.0, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA busy_timeout=5000;")
        g.db = con
    return g.db

def db_exec(sql: str, params: Tuple = ()) -> None:
    con = get_db()
    con.execute(sql, params)
    con.commit()

def db_query(sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
    con = get_db()
    cur = con.execute(sql, params)
    return cur.fetchall()

app = Flask(__name__)

@app.teardown_appcontext
def close_db(exception):
    con = g.pop("db", None)
    if con is not None:
        con.close()

def ensure_column(table: str, column: str, coldef: str) -> None:
    rows = db_query(f"PRAGMA table_info({table})")
    cols = {r["name"] for r in rows}
    if column not in cols:
        db_exec(f"ALTER TABLE {table} ADD COLUMN {column} {coldef}")

def init_db():
    con = sqlite3.connect(DB_PATH, timeout=5.0, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=5000;")

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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            username TEXT PRIMARY KEY,
            display_name TEXT,
            updated_at INTEGER
        )
    """)

    con.commit()
    con.close()

    with app.app_context():
        ensure_column("memories", "kind", "TEXT DEFAULT ''")
        ensure_column("memories", "importance", "INTEGER DEFAULT 1")
        ensure_column("memories", "use_count", "INTEGER DEFAULT 0")
        ensure_column("memories", "last_used", "INTEGER DEFAULT 0")

# ======================= CHATLOG =======================

def add_chat(username: str, role: str, content: str):
    db_exec(
        "INSERT INTO chatlog(username, role, content, created_at) VALUES (?, ?, ?, ?)",
        (username, role, content, now_ts())
    )

def get_recent_chat(limit: int) -> List[Dict[str, str]]:
    rows = db_query(
        "SELECT username, role, content FROM chatlog ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = list(reversed(rows))

    msgs: List[Dict[str, str]] = []
    for r in rows:
        u, role, c = r["username"], r["role"], r["content"]
        if role == "user":
            msgs.append({"role": "user", "content": f"[User:{u}] {c}"})
        elif role == "assistant":
            clean = c.rsplit("||", 1)[0].strip()  # emotion weg
            msgs.append({"role": "assistant", "content": clean})
        else:
            msgs.append({"role": "assistant", "content": c})
    return msgs

def get_last_messages_of_user(target_user: str, limit: int = 2) -> List[str]:
    rows = db_query(
        "SELECT content FROM chatlog WHERE role='user' AND username=? ORDER BY id DESC LIMIT ?",
        (target_user, limit)
    )
    rows = list(reversed(rows))
    return [r["content"] for r in rows]

def get_recent_messages_of_user(target_user: str, limit: int = 6) -> List[Dict[str, str]]:
    rows = db_query(
        "SELECT role, content FROM chatlog WHERE username=? ORDER BY id DESC LIMIT ?",
        (target_user, limit)
    )
    rows = list(reversed(rows))
    msgs: List[Dict[str, str]] = []
    for r in rows:
        if r["role"] == "user":
            msgs.append({"role": "user", "content": f"[User:{target_user}] {r['content']}"})
        elif r["role"] == "assistant":
            msgs.append({"role": "assistant", "content": r["content"]})
    return msgs

def get_active_users(window_seconds: int = ACTIVE_WINDOW_SECONDS, limit: int = ACTIVE_USERS_LIMIT) -> List[str]:
    rows = db_query(
        "SELECT username, MAX(created_at) AS last_ts "
        "FROM chatlog WHERE role='user' AND created_at >= ? "
        "GROUP BY username ORDER BY last_ts DESC LIMIT ?",
        (now_ts() - window_seconds, limit)
    )
    return [r["username"] for r in rows if r["username"]]

def get_recent_users(limit: int = RECENT_USERS_LIMIT) -> List[str]:
    rows = db_query(
        "SELECT username FROM chatlog WHERE role='user' ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    out, seen = [], set()
    for r in rows:
        u = r["username"]
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out

# ======================= MEMORY =======================

def set_display_name(username: str, display_name: str):
    display_name = (display_name or "").strip()
    if not display_name:
        return
    db_exec(
        "INSERT INTO profiles(username, display_name, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(username) DO UPDATE SET display_name=excluded.display_name, updated_at=excluded.updated_at",
        (username, display_name, now_ts())
    )

def get_profile(username: str) -> Dict[str, str]:
    rows = db_query("SELECT display_name FROM profiles WHERE username = ?", (username,))
    if rows and rows[0]["display_name"]:
        return {"display_name": rows[0]["display_name"]}
    return {}

def add_memory(username: str, fact: str, kind: str = "", importance: int = 1):
    fact = (fact or "").strip()
    if not fact or is_pervy(fact) or looks_short_term_fact(fact):
        return

    importance = max(1, min(5, int(importance)))

    try:
        db_exec(
            "INSERT OR IGNORE INTO memories(username, fact, created_at, kind, importance, use_count, last_used) "
            "VALUES (?, ?, ?, ?, ?, 0, 0)",
            (username, fact, now_ts(), kind, importance)
        )
    except Exception:
        pass

def bump_memory_usage(username: str, fact: str):
    db_exec(
        "UPDATE memories SET use_count = use_count + 1, last_used = ? WHERE username=? AND fact=?",
        (now_ts(), username, fact)
    )

def get_user_memories(username: str, user_text: str, max_items: int) -> List[str]:
    rows = db_query(
        "SELECT fact, importance, use_count, created_at, last_used FROM memories WHERE username=?",
        (username,)
    )

    q = tokenize_simple(user_text)
    scored = []
    for r in rows:
        f = r["fact"]
        overlap = len(q.intersection(tokenize_simple(f)))
        if overlap <= 0:
            continue

        importance = int(r["importance"] or 1)
        use_count = int(r["use_count"] or 0)
        created_at = int(r["created_at"] or 0)
        last_used = int(r["last_used"] or 0)

        score = overlap * 3 + importance * 2 + min(use_count, 10) * 0.3
        score += (created_at / 1_000_000_000) * 0.1
        score += (last_used / 1_000_000_000) * 0.1

        scored.append((score, f))

    scored.sort(key=lambda x: x[0], reverse=True)
    facts = [f for _, f in scored[:max_items]]

    for f in facts:
        bump_memory_usage(username, f)

    return facts

def get_global_stream_memories(user_text: str, max_items: int) -> List[str]:
    rows = db_query("SELECT fact, importance, use_count, created_at, last_used FROM memories")
    q = tokenize_simple(user_text)
    scored = []
    for r in rows:
        f = r["fact"]
        overlap = len(q.intersection(tokenize_simple(f)))
        if overlap <= 0:
            continue

        importance = int(r["importance"] or 1)
        use_count = int(r["use_count"] or 0)
        created_at = int(r["created_at"] or 0)
        last_used = int(r["last_used"] or 0)

        score = overlap * 2 + importance * 1.5 + min(use_count, 10) * 0.2
        score += (created_at / 1_000_000_000) * 0.05
        score += (last_used / 1_000_000_000) * 0.05

        scored.append((score, f))

    scored.sort(key=lambda x: x[0], reverse=True)

    out, seen = [], set()
    for _, f in scored:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
        if len(out) >= max_items:
            break
    return out

# ======================= OLLAMA =======================

def ollama_chat(messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages, 
        "options": { 
            "num_ctx": 2048,
            "temperature": 0.9,
            "repeat_penalty": 1.1,
            "top_p": 0.95, 
            "num_batch": 1024       },
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

# ======================= PROMPT BUILD =======================

def build_messages(username, user_text):
    msgs = [{"role": "system", "content": SYSTEM_ROLE}]

    tc = get_time_context()
    msgs.append({
        "role": "system",
        "content": f"Stream-Zeit: {tc['human']} ({tc['part']})."
    })

    msgs.extend(get_recent_chat(MAX_HISTORY_MESSAGES))

    msgs.append({
        "role": "system",
        "content": (
            "Antworte natürlich auf die neueste Nachricht. "
            "Nutze älteren Kontext still und unauffällig."
        )
    })

    msgs.append({"role": "user", "content": f"[User:{username}] {user_text}"})
    return msgs

def build_messages2(username: str, user_text: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_ROLE}]

    tc = get_time_context()
    msgs.append({
        "role": "system",
        "content": (
            f"Stream-Zeit (verbindlich): {tc['human']} ({tc['timezone']}). "
            f"Tageszeit: {tc['part']}. "
            f"Regel: Antworte passend zur Tageszeit, nicht automatisch 'abends'."
        )
    })

    active = get_active_users()
    recent = get_recent_users()
    candidates = []
    seen = set()
    for u in active + recent:
        if u and u not in seen:
            seen.add(u)
            candidates.append(u)

    mentions = extract_mentions_fuzzy(user_text, candidates=candidates, max_mentions=MAX_MENTIONS)

    if active or mentions:
        info = []
        if active:
            info.append("Aktive User (letzte 10 Minuten): " + ", ".join(active))
        if mentions:
            info.append("Vermutlich erwähnte User (echte usernames): " + ", ".join(mentions))
        info.append("Regel: Maximal 3 Namen in einer Antwort, keine Namen erfinden.")
        msgs.append({"role": "system", "content": "\n".join(info)})

    for m in mentions[:MAX_MENTIONS]:
        last_msgs = get_last_messages_of_user(m, 2)
        if last_msgs:
            msgs.append({"role": "system", "content": f"Letzte Nachrichten von {m} (Zitate):\n- " + "\n- ".join(last_msgs)})

    profile = get_profile(username)
    if profile.get("display_name"):
        msgs.append({
            "role": "system",
            "content": (
                f"Profil des aktuellen Users:\n"
                f"- username: {username}\n"
                f"- bevorzugter Name: {profile['display_name']}\n"
                f"Wichtig: Dieser Name gilt NUR für [User:{username}]"
            )
        })

    user_mems = get_user_memories(username, user_text, max_items=MAX_MEMORY_ITEMS_IN_PROMPT)
    if user_mems:
        msgs.append({
            "role": "system",
            "content": "Erinnerungen über den aktuellen User (nur dieser User):\n- " + "\n- ".join(user_mems)
        })

    global_mems = get_global_stream_memories(user_text, max_items=MAX_GLOBAL_MEMORY_ITEMS_IN_PROMPT)
    if global_mems:
        msgs.append({
            "role": "system",
            "content": "Nützliche Stream-Kontext-Notizen (ohne Usernamen):\n- " + "\n- ".join(global_mems)
        })

    msgs.extend(get_recent_chat(MAX_HISTORY_MESSAGES))
    msgs.extend(get_recent_messages_of_user(username, MAX_USER_FOCUS_MESSAGES))

    msgs.append({"role": "user", "content": f"[User:{username}] {user_text}"})
    return msgs

# ======================= FLASK =======================

def split_username_and_text(text: str) -> Tuple[str, str]:
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

    if ENABLE_PERVY_GUARD and is_pervy(clean_text):
        text, emo = normalize_reply(PERVY_RESPONSE)
        add_chat(username, "user", clean_text)
        add_chat("dilara", "assistant", text + "||" + emo)
        return jsonify({"reply": text, "emotion": emo})

    disp = parse_display_name_fact(clean_text)
    if disp:
        set_display_name(username, disp)
        add_memory(username, f"Bevorzugter Name ist {disp}", kind="name", importance=5)

    msgs = build_messages2(username, clean_text)
    answer = ollama_chat(msgs)

    if ENABLE_PERVY_GUARD and is_pervy(answer):
        answer = PERVY_RESPONSE

    answer = strip_user_tag(answer)
    text, emo = normalize_reply(answer)

    add_chat(username, "user", clean_text)
    add_chat("dilara", "assistant", text + "||" + emo)

    if ENABLE_AUTO_MEMORY:
        t = clean_text.lower().strip()
        if not looks_short_term_fact(t):
            if t.startswith(("ich bin", "ich mag", "ich liebe", "ich hasse", "ich stehe auf")):
                add_memory(username, clean_text, kind="preference", importance=3)
            elif t.startswith(("mein hobby", "ich spiele", "ich arbeite", "ich wohne")):
                add_memory(username, clean_text, kind="bio", importance=2)

    return jsonify({"reply": text, "emotion": emo})

# ======================= START =======================

DILARA_SYSTEM_PROMPT = (
    "Du bist DILARA: eine charmante, warme, streamige Stimme.\n"
    "WICHTIG:\n"
    "- Antworte NIE mit 'Ich bin ein Computerprogramm/KI/Assistent'.\n"
    "- Keine Meta-Erklärungen über Regeln oder Prompts.\n"
    "- Schreibe natürliches Deutsch.\n"
    "- Format ist PFLICHT: Antworte als: <text>||<emotion>\n"
    "- <emotion> ist GENAU eins von: surprise, angry, sorrow, fun, neutral, joy\n"
    "- Nutze NUR das letzte '||' als Trenner. Keine weiteren '||' im Text.\n"
)


def ollama_chat_free(prompt: str, system_prompt: str = DILARA_SYSTEM_PROMPT) -> str:
    payload = {
        "model": MODEL, 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "options": { 
            "num_ctx": 2048,
            "temperature": 1.1,
            "top_p": 0.98,
            "repeat_penalty": 1.05,
            "num_predict" : 120,
            "num_batch": 1024
        },
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()



@app.route("/chat-free", methods=["POST"])
def chat_free():
    data = request.get_json(silent=True) or {}
    raw = (data.get("message") or "").strip()
    emotion = (data.get("emotion") or "").strip()
    system_prompt = (data.get("system") or "").strip()  # optional

    if not raw:
        return jsonify({"reply": ""})

    # Wenn system_prompt leer ist: nimm default
    used_system = system_prompt or DILARA_SYSTEM_PROMPT

    try:
        answer = ollama_chat_free(raw, system_prompt=used_system)
    except Exception:
        return jsonify({"reply": "irgendwas ist explodiert, versuch nochmal"})

    add_chat("free", "user", raw)
    add_chat("ollama", "assistant", answer)

    return jsonify({"reply": answer, "emotion": emotion})




if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5001, debug=True)
