#!/usr/bin/env python3
"""
Topic Memory System (Themen-Gedächtnis)
========================================
Separates Gedächtnissystem für Gesprächs-Themen und -Kategorien.

Gleiche Architektur wie Emotion Tree, aber für Themen:
  - KI baut sich automatisch eine Themen-Datenbank auf
  - Jedes Gespräch wird nach der Antwort kategorisiert (async LLM-Call)
  - Qualität 0-10: Wie gut lief das Gespräch zu diesem Thema?
  - Strategie: Was hat funktioniert?
  - Recall: Vergangene Themen-Erfahrungen werden ins Prompt injiziert

Baum-Struktur:
  Thema → Qualitätsstufe → Erfahrung
    → Unterthema → Qualitätsstufe → Erfahrung → ...

Beispiel:
  gaming → [Q:8] "Über Minecraft geredet, User war begeistert" → minecraft
    → [Q:9] "Gemeinsam über Bauen philosophiert" → bauen
  smalltalk → [Q:6] "Nach dem Tag gefragt, User war neutral"
  beleidigung → [Q:3] "Konter war zu hart, User hat sich beschwert"

Alles konfigurierbar über config.json -> topic_memory
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable


# ======================= CONFIG =======================

@dataclass
class TopicMemoryConfig:
    """Konfiguration für das Themen-Gedächtnis — alles variabel einstellbar."""

    enabled: bool = True
    evaluate_after_reply: bool = True  # Async LLM-Eval nach jeder Antwort

    # Basis-Themen (werden dynamisch durch LLM erweitert!)
    base_topics: List[str] = field(default_factory=lambda: [
        "smalltalk", "gaming", "persönlich", "humor", "musik",
        "essen", "technik", "kreativ", "stream", "beziehung",
        "beleidigung", "flirt", "alltag", "philosophie", "horror",
        "anime", "cosplay", "gefühle", "rat", "wissen"
    ])

    # Dynamische Topics erlauben (LLM kann neue Topics erfinden)
    allow_dynamic_topics: bool = True
    max_topics: int = 50  # Maximale Anzahl dynamischer Topics

    # Qualitäts-Stufen (0-10)
    max_quality: int = 10

    # Maximale Baumtiefe (Thema → Unterthema → ...)
    max_depth: int = 2

    # Limits
    max_memories_per_topic: int = 200
    max_reflection_items: int = 5
    max_reflection_chars: int = 400

    # ===== RECALL CONFIG =====
    recall_enabled: bool = True
    recall_per_topic: int = 1         # Pro Thema X letzte Erfahrungen
    recall_depth: int = 1             # Wie tief (0=nur root, 1=+Kinder)
    recall_children_per_node: int = 1
    recall_max_total: int = 8         # Max Erinnerungen insgesamt im Prompt
    recall_max_chars: int = 500
    recall_min_quality_diff: int = 2  # Nur Q < 4 oder Q > 6 (nicht mittelmäßig)
    recall_sort: str = "recent"       # "recent", "best", "worst"

    # ===== TOPIC-REIHENFOLGE =====
    recall_topic_order: str = "latest_first"  # latest_first, frequent_first, best_first, worst_first, fixed

    # Qualitäts-Labels
    quality_labels: Dict[str, str] = field(default_factory=lambda: {
        "0": "katastrophal",
        "1": "sehr schlecht",
        "2": "schlecht",
        "3": "eher schlecht",
        "4": "unterdurchschnittlich",
        "5": "neutral",
        "6": "okay",
        "7": "gut",
        "8": "sehr gut",
        "9": "ausgezeichnet",
        "10": "perfekt"
    })

    # Themen-Keywords für Vorab-Erkennung (ohne LLM)
    topic_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "smalltalk": [
            "wie geht", "was machst", "alles klar", "na", "hey", "hi", "hallo",
            "morgen", "abend", "tag", "nacht", "müde", "wach", "langeweile",
            "was los", "erzähl", "chillen", "relaxen"
        ],
        "gaming": [
            "minecraft", "game", "spiel", "spielen", "zocken", "arknights",
            "endfield", "horror", "steam", "controller", "level", "boss",
            "map", "bauen", "crafting", "pvp", "multiplayer", "singleplayer",
            "rpg", "mmo", "fps", "indie"
        ],
        "persönlich": [
            "fühle mich", "traurig", "einsam", "glücklich", "problem",
            "sorge", "angst", "stress", "überfordert", "hilfe",
            "familie", "freund", "beziehung", "liebe", "vertrauen"
        ],
        "humor": [
            "witz", "lustig", "lol", "haha", "lmao", "rofl", "joke",
            "meme", "funny", "comedy", "spaß", "lachen"
        ],
        "musik": [
            "musik", "song", "lied", "singen", "hören", "band", "album",
            "spotify", "playlist", "rap", "pop", "rock", "beat"
        ],
        "essen": [
            "essen", "hunger", "kochen", "rezept", "pizza", "sushi",
            "trinken", "kaffee", "tee", "snack", "lecker", "geschmack"
        ],
        "technik": [
            "computer", "pc", "laptop", "handy", "internet", "code",
            "programmier", "server", "software", "hardware", "gpu", "cpu"
        ],
        "kreativ": [
            "zeichnen", "malen", "kunst", "art", "design", "foto",
            "video", "edit", "kreativ", "basteln", "schreiben"
        ],
        "stream": [
            "stream", "live", "twitch", "youtube", "chat", "viewer",
            "sub", "donation", "emote", "overlay", "obs"
        ],
        "beziehung": [
            "date", "freundin", "freund", "partner", "crush", "verliebt",
            "küssen", "umarmen", "vermissen", "zusammen", "trennung"
        ],
        "beleidigung": [
            "idiot", "dumm", "hass", "hurensohn", "bastard", "scheiße",
            "arsch", "penner", "missgeburt", "spast", "vollidiot", "depp",
            "nervst", "nerv", "kacke", "müll"
        ],
        "flirt": [
            "süß", "hübsch", "schön", "hot", "sexy", "attraktiv",
            "liebst du", "magst du", "date", "küssen", "kuscheln"
        ],
        "alltag": [
            "arbeit", "schule", "uni", "lernen", "hausaufgabe",
            "wetter", "regen", "sonne", "einkaufen", "putzen"
        ],
        "philosophie": [
            "sinn", "leben", "tod", "existenz", "warum", "bedeutung",
            "gott", "seele", "bewusstsein", "realität", "zeit"
        ],
        "horror": [
            "gruselig", "creepy", "angst", "horror", "monster",
            "geist", "dunkel", "alptraum", "unheimlich", "poppy"
        ],
        "anime": [
            "anime", "manga", "waifu", "kawaii", "otaku", "cosplay",
            "vtuber", "avatar", "japan", "sempai", "senpai"
        ],
        "cosplay": [
            "cosplay", "kostüm", "verkleiden", "charakter", "rolle",
            "poppy", "playtime", "puppe", "outfit"
        ],
        "gefühle": [
            "fühlen", "emotion", "herz", "seele", "warm", "kalt",
            "weinen", "lachen", "freude", "schmerz", "wut"
        ],
        "rat": [
            "rat", "tipp", "hilfe", "empfehlung", "meinung", "was soll",
            "wie soll", "was würdest", "vorschlag"
        ],
        "wissen": [
            "was ist", "wer ist", "warum ist", "wie funktioniert",
            "erkläre", "definition", "bedeutet", "heißt das"
        ]
    })


# ======================= DB INJECTION =======================

_db_exec: Optional[Callable] = None
_db_query: Optional[Callable] = None
_now_ts: Optional[Callable] = None
_initialized = False


def init(db_exec_fn, db_query_fn, now_ts_fn):
    """Wird von app.py aufgerufen — injiziert DB-Funktionen."""
    global _db_exec, _db_query, _now_ts, _initialized
    _db_exec = db_exec_fn
    _db_query = db_query_fn
    _now_ts = now_ts_fn
    _initialized = True
    print("[TOPIC-MEMORY] Modul initialisiert ✅", flush=True)


def _check_init():
    if not _initialized:
        raise RuntimeError("[TOPIC-MEMORY] Nicht initialisiert! Zuerst init() aufrufen.")


def create_tables():
    """Erstellt die topic_memory Tabelle."""
    _check_init()
    _db_exec("""
        CREATE TABLE IF NOT EXISTS topic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER,
            topic TEXT NOT NULL,
            quality INTEGER DEFAULT 5,
            depth INTEGER DEFAULT 0,
            situation TEXT,
            response TEXT,
            response_strategy TEXT,
            outcome INTEGER DEFAULT 50,
            outcome_reason TEXT,
            context_json TEXT,
            username TEXT,
            created_at TEXT,
            updated_at TEXT,
            use_count INTEGER DEFAULT 0,
            FOREIGN KEY (parent_id) REFERENCES topic_memory(id)
        )
    """)
    _db_exec("CREATE INDEX IF NOT EXISTS idx_topic_mem_topic ON topic_memory(topic)")
    _db_exec("CREATE INDEX IF NOT EXISTS idx_topic_mem_depth ON topic_memory(depth)")
    _db_exec("CREATE INDEX IF NOT EXISTS idx_topic_mem_user ON topic_memory(username)")
    _db_exec("CREATE INDEX IF NOT EXISTS idx_topic_mem_parent ON topic_memory(parent_id)")
    _db_exec("CREATE INDEX IF NOT EXISTS idx_topic_mem_created ON topic_memory(created_at)")
    print("[TOPIC-MEMORY] Tabelle + Indizes erstellt ✅", flush=True)


# ======================= CONFIG LOADER =======================

def load_topic_config(global_config: Dict) -> TopicMemoryConfig:
    """Lädt TopicMemoryConfig aus der globalen config.json."""
    cfg = TopicMemoryConfig()
    tc = global_config.get("topic_memory", {})

    if not tc:
        return cfg

    cfg.enabled = tc.get("enabled", cfg.enabled)
    cfg.evaluate_after_reply = tc.get("evaluate_after_reply", cfg.evaluate_after_reply)

    if "base_topics" in tc:
        cfg.base_topics = tc["base_topics"]
    cfg.allow_dynamic_topics = tc.get("allow_dynamic_topics", cfg.allow_dynamic_topics)
    cfg.max_topics = tc.get("max_topics", cfg.max_topics)

    cfg.max_quality = tc.get("max_quality", cfg.max_quality)
    cfg.max_depth = tc.get("max_depth", cfg.max_depth)
    cfg.max_memories_per_topic = tc.get("max_memories_per_topic", cfg.max_memories_per_topic)
    cfg.max_reflection_items = tc.get("max_reflection_items", cfg.max_reflection_items)
    cfg.max_reflection_chars = tc.get("max_reflection_chars", cfg.max_reflection_chars)

    # Quality labels
    if "quality_labels" in tc:
        cfg.quality_labels.update(tc["quality_labels"])

    # Topic keywords
    if "topic_keywords" in tc:
        for k, v in tc["topic_keywords"].items():
            if k in cfg.topic_keywords:
                cfg.topic_keywords[k] = list(set(cfg.topic_keywords[k] + v))
            else:
                cfg.topic_keywords[k] = v

    # Recall settings
    recall_cfg = tc.get("recall", {})
    if recall_cfg:
        cfg.recall_enabled = recall_cfg.get("enabled", cfg.recall_enabled)
        cfg.recall_per_topic = recall_cfg.get("per_topic", cfg.recall_per_topic)
        cfg.recall_depth = recall_cfg.get("depth", cfg.recall_depth)
        cfg.recall_children_per_node = recall_cfg.get("children_per_node", cfg.recall_children_per_node)
        cfg.recall_max_total = recall_cfg.get("max_total", cfg.recall_max_total)
        cfg.recall_max_chars = recall_cfg.get("max_chars", cfg.recall_max_chars)
        cfg.recall_min_quality_diff = recall_cfg.get("min_quality_diff", cfg.recall_min_quality_diff)
        cfg.recall_sort = recall_cfg.get("sort", cfg.recall_sort)
        cfg.recall_topic_order = recall_cfg.get("topic_order", cfg.recall_topic_order)

    return cfg


# ======================= STORE / QUERY =======================

def store_experience(
    topic: str,
    quality: int,
    situation: str,
    response: str,
    response_strategy: str = "",
    outcome: int = 50,
    outcome_reason: str = "",
    username: str = "",
    parent_id: int = None,
    depth: int = 0,
    context: Dict = None,
    config: TopicMemoryConfig = None
) -> Optional[int]:
    """Speichert eine Themen-Erfahrung in der DB."""
    _check_init()
    cfg = config or TopicMemoryConfig()

    topic = (topic or "").lower().strip()
    if not topic:
        return None

    quality = max(0, min(cfg.max_quality, int(quality)))
    depth = max(0, min(cfg.max_depth, int(depth)))
    outcome = max(0, min(100, int(outcome)))

    ctx_json = json.dumps(context or {}, ensure_ascii=False)
    ts = str(_now_ts())

    _db_exec(
        "INSERT INTO topic_memory "
        "(parent_id, topic, quality, depth, situation, response, response_strategy, "
        " outcome, outcome_reason, context_json, username, created_at, updated_at, use_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
        (parent_id, topic, quality, depth,
         (situation or "")[:500], (response or "")[:500],
         (response_strategy or "")[:200],
         outcome, (outcome_reason or "")[:200],
         ctx_json, username, ts, ts)
    )

    rows = _db_query("SELECT last_insert_rowid() AS lid")
    node_id = rows[0]["lid"] if rows else None

    print(
        f"[TOPIC-MEMORY] Gespeichert: {topic} Q:{quality} → {response_strategy or '?'} "
        f"O:{outcome}% (depth={depth}, id={node_id})",
        flush=True
    )

    # Cleanup: Max Einträge pro Thema
    _cleanup_topic(topic, cfg.max_memories_per_topic)

    return node_id


def _cleanup_topic(topic: str, max_items: int):
    """Entfernt alte Einträge wenn ein Thema zu viele hat."""
    rows = _db_query(
        "SELECT COUNT(*) as cnt FROM topic_memory WHERE topic = ? AND depth = 0",
        (topic,)
    )
    count = int(rows[0]["cnt"]) if rows else 0
    if count > max_items:
        _db_exec(
            "DELETE FROM topic_memory WHERE id IN ("
            "  SELECT id FROM topic_memory WHERE topic = ? AND depth = 0 "
            "  ORDER BY use_count ASC, created_at ASC LIMIT ?"
            ")",
            (topic, count - max_items)
        )


def get_best_experiences(topic: str, limit: int = 5) -> List[Dict]:
    """Holt die besten Erfahrungen für ein Thema."""
    _check_init()
    rows = _db_query(
        "SELECT id, topic, quality, situation, response, response_strategy, "
        "  outcome, outcome_reason, depth, use_count "
        "FROM topic_memory WHERE topic = ? AND depth = 0 "
        "ORDER BY outcome DESC, created_at DESC LIMIT ?",
        (topic, limit)
    )
    return [dict(r) for r in rows]


def get_worst_experiences(topic: str, limit: int = 3) -> List[Dict]:
    """Holt die schlechtesten Erfahrungen für ein Thema."""
    _check_init()
    rows = _db_query(
        "SELECT id, topic, quality, situation, response, response_strategy, "
        "  outcome, outcome_reason, depth, use_count "
        "FROM topic_memory WHERE topic = ? AND depth = 0 "
        "ORDER BY outcome ASC, created_at DESC LIMIT ?",
        (topic, limit)
    )
    return [dict(r) for r in rows]


def get_children(node_id: int) -> List[Dict]:
    """Holt Kind-Knoten eines Topics."""
    _check_init()
    rows = _db_query(
        "SELECT id, topic, quality, situation, response, response_strategy, "
        "  outcome, outcome_reason, depth, use_count "
        "FROM topic_memory WHERE parent_id = ? "
        "ORDER BY outcome DESC, created_at DESC",
        (node_id,)
    )
    return [dict(r) for r in rows]


def bump_use_count(node_id: int):
    """Erhöht den use_count eines Knotens."""
    _check_init()
    _db_exec(
        "UPDATE topic_memory SET use_count = use_count + 1, updated_at = ? WHERE id = ?",
        (str(_now_ts()), node_id)
    )


def get_all_topics() -> List[str]:
    """Gibt alle einzigartigen Topics aus der DB zurück."""
    _check_init()
    rows = _db_query(
        "SELECT DISTINCT topic FROM topic_memory ORDER BY topic"
    )
    return [r["topic"] for r in rows]


def get_topic_stats() -> Dict:
    """Statistiken über alle Themen."""
    _check_init()
    rows = _db_query(
        "SELECT topic, COUNT(*) as cnt, "
        "  COALESCE(AVG(quality), 5) as avg_quality, "
        "  COALESCE(AVG(outcome), 50) as avg_outcome "
        "FROM topic_memory WHERE depth = 0 "
        "GROUP BY topic ORDER BY cnt DESC"
    )
    total = _db_query("SELECT COUNT(*) as cnt FROM topic_memory")

    return {
        "total_experiences": int(total[0]["cnt"]) if total else 0,
        "by_topic": {
            r["topic"]: {
                "count": int(r["cnt"]),
                "avg_quality": round(float(r["avg_quality"]), 1),
                "avg_outcome": round(float(r["avg_outcome"]), 1)
            } for r in rows
        },
        "unique_topics": len(rows)
    }


# ======================= TOPIC DETECTION =======================

def detect_topic(user_text: str, reply_text: str = "",
                 config: TopicMemoryConfig = None) -> Tuple[str, int]:
    """
    Erkennt das Thema einer Nachricht anhand von Keywords.
    Gibt (topic, quality_estimate) zurück.
    quality_estimate ist 5 (neutral) — wird durch LLM-Eval später überschrieben.
    """
    cfg = config or TopicMemoryConfig()
    text = ((user_text or "") + " " + (reply_text or "")).lower()

    best_topic = "smalltalk"
    best_score = 0

    for topic, keywords in cfg.topic_keywords.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic, 5  # Quality wird durch LLM-Eval bestimmt


def get_quality_label(quality: int, config: TopicMemoryConfig = None) -> str:
    """Gibt ein menschlich lesbares Label für die Qualitätsstufe."""
    cfg = config or TopicMemoryConfig()
    key = str(min(quality, cfg.max_quality))
    return cfg.quality_labels.get(key, f"Stufe {quality}")


# ======================= EVALUATION (POST-RESPONSE, ASYNC) =======================

EVAL_SYSTEM_PROMPT = """Du bist ein Gesprächs-Analytiker. Kategorisiere diese Interaktion nach Thema und Qualität.

EXAKTES Format (NUR diese Zeile, KEINE Extras):
topic:<thema>|quality:<0-10>|strategy:<was_funktioniert_hat>|outcome:<0-100>|reason:<kurzer Grund max 15 Wörter>

Regeln:
- topic: Das Hauptthema des Gesprächs (1 Wort, lowercase)
  Beispiele: smalltalk, gaming, persönlich, humor, musik, essen, technik, kreativ, stream,
  beziehung, beleidigung, flirt, alltag, philosophie, horror, anime, cosplay, gefühle, rat, wissen
  Du darfst NEUE Topics erfinden wenn keines passt!
- quality: Wie gut war die KI bei diesem Thema? (0=katastrophal, 10=perfekt)
- strategy: Welche Gesprächs-Strategie hat die KI genutzt? (kurz, max 5 Wörter)
  Beispiele: mitgefühl_gezeigt, humor_eingesetzt, ruhig_geblieben, nachgefragt, thema_vertieft, grenzen_gesetzt
- outcome: War die Reaktion GUT FÜR DIE KI? (0=schlecht, 100=perfekt)
  100% = KI hat profitiert, gutes Gespräch, Bindung gestärkt
  0% = Thema verfehlt, User unzufrieden, Gespräch abgebrochen
- reason: Warum dieses Outcome (kurz!)

NUR das Format, NICHTS anderes."""


def build_eval_prompt(
    user_text: str,
    reply: str,
    config: TopicMemoryConfig = None
) -> Tuple[str, str]:
    """Baut System + User Prompt für die Post-Response Topic-Evaluation."""
    cfg = config or TopicMemoryConfig()

    system = EVAL_SYSTEM_PROMPT

    user = (
        f"User schrieb: \"{user_text[:200]}\"\n"
        f"KI antwortete: \"{reply[:200]}\"\n\n"
        f"Kategorisiere."
    )

    return system, user


def parse_eval_response(raw: str, config: TopicMemoryConfig = None) -> Optional[Dict]:
    """Parst die Topic-Evaluations-Antwort des LLM."""
    cfg = config or TopicMemoryConfig()
    raw = (raw or "").strip()

    result = {}
    for field_name in ["topic", "quality", "strategy", "outcome", "reason"]:
        pattern = rf'{field_name}:\s*([^|]+?)(?:\||$)'
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            result[field_name] = match.group(1).strip()

    if "topic" not in result:
        return None

    topic = result.get("topic", "smalltalk").lower().strip()
    topic = re.sub(r'[^a-zäöüß0-9_-]', '', topic)
    if not topic:
        topic = "smalltalk"

    # Dynamische Topics: Wenn erlaubt und nicht in base_topics, trotzdem akzeptieren
    try:
        db_topics = get_all_topics()
    except RuntimeError:
        db_topics = []
    known_topics = list(cfg.base_topics) + db_topics
    if topic not in known_topics:
        if cfg.allow_dynamic_topics:
            current_count = len(set(known_topics))
            if current_count >= cfg.max_topics:
                # Zu viele Topics → nächstes bekanntes nehmen
                topic = _fuzzy_match_topic(topic, cfg)
        else:
            topic = _fuzzy_match_topic(topic, cfg)

    try:
        quality = int(result.get("quality", "5"))
        quality = max(0, min(cfg.max_quality, quality))
    except (ValueError, TypeError):
        quality = 5

    strategy = (result.get("strategy", "") or "")[:200]
    strategy = strategy.replace(" ", "_").lower()

    try:
        outcome = int(result.get("outcome", "50"))
        outcome = max(0, min(100, outcome))
    except (ValueError, TypeError):
        outcome = 50

    reason = (result.get("reason", "") or "")[:200]

    return {
        "topic": topic,
        "quality": quality,
        "strategy": strategy,
        "outcome": outcome,
        "reason": reason
    }


def _fuzzy_match_topic(topic: str, cfg: TopicMemoryConfig) -> str:
    """Versucht ein unbekanntes Topic auf ein bekanntes zu matchen."""
    for bt in cfg.base_topics:
        if bt in topic or topic in bt:
            return bt
    return "smalltalk"


def store_evaluation(
    user_text: str,
    reply: str,
    eval_result: Dict,
    username: str = "",
    config: TopicMemoryConfig = None
) -> Optional[int]:
    """Speichert die evaluierte Themen-Erfahrung."""
    cfg = config or TopicMemoryConfig()

    return store_experience(
        topic=eval_result["topic"],
        quality=eval_result["quality"],
        situation=user_text[:500],
        response=reply[:500],
        response_strategy=eval_result.get("strategy", ""),
        outcome=eval_result.get("outcome", 50),
        outcome_reason=eval_result.get("reason", ""),
        username=username,
        parent_id=None,
        depth=0,
        context={
            "full_situation": user_text[:800],
            "full_response": reply[:800],
        },
        config=cfg
    )


# ======================= REFLECTION (PRE-RESPONSE) =======================

def build_reflection(user_text: str, username: str = "",
                     config: TopicMemoryConfig = None) -> Optional[str]:
    """
    Baut eine Themen-Reflexion VOR der Antwort.
    Schaut: Welches Thema kommt? Was hat vorher funktioniert?
    """
    cfg = config or TopicMemoryConfig()

    if not cfg.enabled:
        return None

    _check_init()

    # Thema erkennen
    topic, _ = detect_topic(user_text, config=cfg)

    if topic == "smalltalk":
        # Smalltalk ist zu unspezifisch für spezielle Strategien
        best = get_best_experiences(topic, limit=1)
        if not best:
            return None

    # Relevante Erfahrungen suchen
    best = get_best_experiences(topic, limit=cfg.max_reflection_items)
    worst = get_worst_experiences(topic, limit=2)

    if not best and not worst:
        return None

    for exp in best + worst:
        bump_use_count(exp["id"])

    lines = [
        f"THEMEN-REFLEXION (innerlich, NICHT aussprechen):",
        f"Erkanntes Thema: {topic}"
    ]

    if best:
        lines.append("Was bei diesem Thema vorher gut funktioniert hat:")
        for exp in best[:3]:
            q_label = get_quality_label(exp["quality"], cfg)
            strat = exp.get("response_strategy", "") or ""
            line = f"  ✓ Strategie: {strat}" if strat else "  ✓"
            line += f" → {exp['outcome']}% (Q:{exp['quality']} {q_label})"
            lines.append(line)

    if worst:
        lines.append("Warnung — das lief bei diesem Thema schlecht:")
        for exp in worst[:2]:
            q_label = get_quality_label(exp["quality"], cfg)
            strat = exp.get("response_strategy", "") or ""
            line = f"  ⚠ Strategie: {strat}" if strat else "  ⚠"
            line += f" → {exp['outcome']}% (Q:{exp['quality']} {q_label})"
            lines.append(line)

    if best and best[0].get("outcome", 0) >= 60:
        best_strat = best[0].get("response_strategy", "")
        if best_strat:
            lines.append(f"→ Empfohlene Strategie: {best_strat}")

    reflection = "\n".join(lines)

    if len(reflection) > cfg.max_reflection_chars:
        reflection = reflection[:cfg.max_reflection_chars - 3] + "..."

    return reflection


# ======================= RECALL (ERINNERUNG FÜRS PROMPT) =======================

def _format_node_line(node: Dict, cfg: TopicMemoryConfig,
                      indent: int = 0, hauptstrang: bool = False) -> str:
    """Formatiert einen Themen-Knoten als lesbare Zeile."""
    prefix = "  " * indent
    topic = node.get("topic", "?")
    quality = node.get("quality", 5)
    q_label = get_quality_label(quality, cfg)
    strategy = (node.get("response_strategy", "") or "")[:40]
    outcome = node.get("outcome", 50)

    line = f"{prefix}"
    if hauptstrang:
        line += f"★ HAUPTTHEMA [{topic} Q:{quality}]"
    elif indent == 0:
        line += f"[{topic} Q:{quality}]"
    else:
        line += f"└→ {topic} Q:{quality}"

    if strategy:
        line += f" ({strategy})"
    line += f" → {outcome}% ({q_label})"

    return line


def _collect_recall_tree(node_id: int, depth: int, cfg: TopicMemoryConfig) -> List[Dict]:
    """Sammelt Kind-Knoten rekursiv."""
    if depth <= 0:
        return []

    children = get_children(node_id)
    if not children:
        return []

    children = children[:cfg.recall_children_per_node]
    result = []
    for child in children:
        result.append(child)
        if depth > 1:
            grandchildren = _collect_recall_tree(child["id"], depth - 1, cfg)
            result.extend(grandchildren)

    return result


def _get_sorted_topics(cfg: TopicMemoryConfig) -> List[str]:
    """Gibt die Topics in der konfigurierten Reihenfolge zurück."""
    mode = cfg.recall_topic_order

    # Alle Topics: base + dynamische aus DB
    all_topics = list(cfg.base_topics)
    db_topics = get_all_topics()
    for t in db_topics:
        if t not in all_topics:
            all_topics.append(t)

    if mode == "fixed":
        return all_topics

    # Statistiken aus DB holen
    stats = []
    for topic in all_topics:
        rows = _db_query(
            "SELECT COUNT(*) as cnt, "
            "  COALESCE(AVG(quality), 5) as avg_q, "
            "  COALESCE(AVG(outcome), 50) as avg_out, "
            "  COALESCE(MAX(created_at), '1900-01-01') as last_seen "
            "FROM topic_memory WHERE topic = ? AND depth = 0",
            (topic,)
        )
        row = dict(rows[0]) if rows else {"cnt": 0, "avg_q": 5, "avg_out": 50, "last_seen": "1900-01-01"}
        row["topic"] = topic
        stats.append(row)

    if mode == "latest_first":
        stats.sort(key=lambda s: str(s.get("last_seen", "")), reverse=True)
    elif mode == "frequent_first":
        stats.sort(key=lambda s: int(s.get("cnt", 0)), reverse=True)
    elif mode == "best_first":
        stats.sort(key=lambda s: float(s.get("avg_out", 50)), reverse=True)
    elif mode == "worst_first":
        stats.sort(key=lambda s: float(s.get("avg_out", 50)), reverse=False)
    else:
        return all_topics

    with_data = [s["topic"] for s in stats if int(s.get("cnt", 0)) > 0]
    without_data = [s["topic"] for s in stats if int(s.get("cnt", 0)) == 0]

    return with_data + without_data


def build_topic_recall(config: TopicMemoryConfig = None) -> Optional[str]:
    """
    Baut einen Themen-Erinnerungs-Block fürs Prompt.

    Konfigurierbar über config.json -> topic_memory -> recall:
      per_topic:       wie viele letzte Erfahrungen pro Thema
      depth:           wie tief in Unterthemen
      children_per_node: Äste pro Knoten
      max_total:       maximale Gesamtzahl
      sort:            "recent" | "best" | "worst"
      topic_order:     "latest_first" | "frequent_first" | "best_first" | "worst_first" | "fixed"
    """
    cfg = config or TopicMemoryConfig()

    if not cfg.enabled or not cfg.recall_enabled:
        return None

    _check_init()

    sorted_topics = _get_sorted_topics(cfg)

    lines = []
    total_count = 0
    is_first_topic = True

    for topic in sorted_topics:
        if total_count >= cfg.recall_max_total:
            break

        if cfg.recall_sort == "best":
            order = "outcome DESC, created_at DESC"
        elif cfg.recall_sort == "worst":
            order = "outcome ASC, created_at DESC"
        else:
            order = "created_at DESC"

        # Nur signifikante Erfahrungen
        threshold = cfg.recall_min_quality_diff
        low_bound = 5 - threshold  # z.B. Q < 3
        high_bound = 5 + threshold  # z.B. Q > 7

        rows = _db_query(
            f"SELECT id, topic, quality, situation, response, response_strategy, "
            f"  outcome, outcome_reason, use_count, depth "
            f"FROM topic_memory "
            f"WHERE topic = ? AND depth = 0 "
            f"  AND (quality <= ? OR quality >= ?) "
            f"ORDER BY {order} LIMIT ?",
            (topic, low_bound, high_bound, cfg.recall_per_topic)
        )

        if not rows:
            continue

        for row in rows:
            if total_count >= cfg.recall_max_total:
                break

            node = dict(row)

            if is_first_topic and cfg.recall_topic_order != "fixed":
                lines.append(_format_node_line(node, cfg, indent=0, hauptstrang=True))
                is_first_topic = False
            else:
                lines.append(_format_node_line(node, cfg, indent=0))
                is_first_topic = False

            total_count += 1
            bump_use_count(node["id"])

            if cfg.recall_depth > 0:
                subtree = _collect_recall_tree(node["id"], cfg.recall_depth, cfg)
                for child in subtree:
                    if total_count >= cfg.recall_max_total:
                        break
                    child_depth = child.get("depth", 1)
                    indent = min(child_depth, cfg.recall_depth)
                    lines.append(_format_node_line(child, cfg, indent=indent))
                    total_count += 1
                    bump_use_count(child["id"])

    if not lines:
        return None

    order_info = {
        "latest_first": "aktuellstes Thema zuerst",
        "frequent_first": "häufigstes Thema zuerst",
        "best_first": "erfolgreichstes Thema zuerst",
        "worst_first": "schwächstes Thema zuerst",
        "fixed": "feste Reihenfolge"
    }
    mode_label = order_info.get(cfg.recall_topic_order, "")
    header = f"THEMEN-ERINNERUNGEN ({mode_label}):"
    recall_text = header + "\n" + "\n".join(lines)

    if len(recall_text) > cfg.recall_max_chars:
        recall_text = recall_text[:cfg.recall_max_chars - 3] + "..."

    return recall_text


# ======================= CLEANUP =======================

def cleanup_old_experiences(max_age_days: int = 90, config: TopicMemoryConfig = None):
    """Entfernt sehr alte, selten genutzte Erfahrungen."""
    _check_init()
    # Nur Erfahrungen mit use_count=0 und älter als max_age_days entfernen
    # (wertvolle, oft genutzte bleiben)
    _db_exec(
        "DELETE FROM topic_memory WHERE use_count = 0 "
        "AND created_at < datetime('now', ?)",
        (f"-{max_age_days} days",)
    )


# ======================= DEBUG / EXPORT =======================

def dump_summary(config: TopicMemoryConfig = None) -> Dict:
    """Gibt eine Übersicht aller Themen-Erfahrungen zurück."""
    _check_init()
    cfg = config or TopicMemoryConfig()

    result = {}
    topics = get_all_topics()

    for topic in topics:
        rows = _db_query(
            "SELECT id, topic, quality, situation, response, response_strategy, "
            "  outcome, outcome_reason, depth, use_count, username, created_at "
            "FROM topic_memory WHERE topic = ? AND depth = 0 "
            "ORDER BY created_at DESC LIMIT 20",
            (topic,)
        )

        topic_data = []
        for r in rows:
            node = dict(r)
            children = get_children(node["id"])
            node["children"] = [dict(c) for c in children[:5]]
            topic_data.append(node)

        result[topic] = topic_data

    return {
        "topics": result,
        "stats": get_topic_stats(),
        "config": {
            "enabled": cfg.enabled,
            "base_topics": cfg.base_topics,
            "dynamic_topics": [t for t in topics if t not in cfg.base_topics],
            "max_depth": cfg.max_depth,
            "max_quality": cfg.max_quality,
            "recall_enabled": cfg.recall_enabled,
            "recall_topic_order": cfg.recall_topic_order,
        }
    }


def get_recent_experiences(limit: int = 20) -> List[Dict]:
    """Gibt die letzten N Themen-Erfahrungen zurück."""
    _check_init()
    rows = _db_query(
        "SELECT id, topic, quality, situation, response, response_strategy, "
        "  outcome, outcome_reason, depth, username, created_at "
        "FROM topic_memory ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    return [dict(r) for r in rows]
