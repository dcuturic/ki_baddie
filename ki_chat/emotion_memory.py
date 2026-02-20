#!/usr/bin/env python3
"""
Emotion Tree Memory System
===========================
Hierarchischer Emotionsbaum für KI-Charakter-Entwicklung.

Baum-Struktur:
  Basis-Emotion → Intensitäts-Stufe (0..N) → Erinnerung
    → Reaktions-Emotion → Intensitäts-Stufe → Erinnerung → ...

Jeder Knoten speichert eine Erfahrung:
  - Welche Emotion wurde gefühlt und wie stark (Intensität)
  - Was hat die Situation ausgelöst
  - Wie hat die KI reagiert
  - War das Ergebnis gut FÜR DIE KI (0-100%)
  - Kind-Knoten: Wie hat die Reaktion emotional weiter gewirkt

Der Baum wächst organisch durch Interaktion.
Vor dem Antworten kann die KI relevante emotionale Erfahrungen nachschlagen,
um bessere Reaktionen zu wählen.

Alles ist konfigurierbar:
  - base_emotions: Liste der Grund-Emotionen (erweiterbar)
  - max_intensity: Wie viele Intensitäts-Stufen pro Emotion
  - max_depth: Wie tief der Reaktions-Baum geht
  - personality: Persönlichkeits-Variablen beeinflussen Gewichtung
  - Jeder Parameter ist über config.json steuerbar
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable


# ======================= CONFIG =======================

@dataclass
class EmotionTreeConfig:
    """Konfiguration für den Emotionsbaum — alles variabel einstellbar."""

    enabled: bool = True
    reflect_before_reply: bool = True
    deep_reflect: bool = False  # Extra LLM-Call vor Antwort (langsamer, genauer)

    # Basis-Emotionen (beliebig erweiterbar)
    base_emotions: List[str] = field(default_factory=lambda: [
        "freude", "wut", "trauer", "angst",
        "ekel", "überraschung", "liebe", "neugier",
        "stolz", "scham", "neid", "verachtung"
    ])

    # Intensitäts-Stufen pro Emotion (0 bis max_intensity)
    max_intensity: int = 10

    # Maximale Baumtiefe (wie viele Reaktions-Ebenen)
    max_depth: int = 3

    # Limits
    max_memories_per_emotion: int = 200
    max_reflection_items: int = 5
    max_reflection_chars: int = 500  # Max Zeichen für Reflexion im Prompt

    # ===== RECALL CONFIG (Erinnerung als Memory im Prompt) =====
    # Wie viele letzte Erfahrungen PRO Emotion ins Prompt kommen
    recall_enabled: bool = True
    recall_per_emotion: int = 1       # Pro Emotion X letzte Erfahrungen (root-Ebene)
    recall_depth: int = 2             # Wie tief in den Baum gehen (0=nur root, 1=+Kinder, 2=+Enkel)
    recall_children_per_node: int = 1 # Pro Knoten wie viele Kind-Äste mitnehmen
    recall_max_total: int = 12        # Max Erinnerungen insgesamt im Prompt
    recall_max_chars: int = 600       # Max Zeichen für den Recall-Block
    recall_min_outcome_diff: int = 20 # Nur Erfahrungen mit Outcome < 40 oder > 60 (nicht "meh")
    recall_sort: str = "recent"       # "recent" (neueste), "best" (höchstes outcome), "worst" (niedrigstes)

    # ===== EMOTION-REIHENFOLGE (welche Emotion wird zum Hauptstrang?) =====
    # Bestimmt die Reihenfolge der Emotionen im Recall-Block
    #   "latest_first"   - Die zuletzt erlebte Emotion kommt zuerst (wird Hauptstrang)
    #   "intense_first"  - Die intensivste Emotion (höchste avg Intensität) zuerst
    #   "frequent_first" - Die am häufigsten erlebte Emotion zuerst
    #   "best_first"     - Die Emotion mit dem besten avg Outcome zuerst
    #   "worst_first"    - Die Emotion mit dem schlechtesten avg Outcome zuerst
    #   "fixed"          - Feste Reihenfolge aus base_emotions (Standard)
    recall_emotion_order: str = "latest_first"
    recall_promote_new: bool = True  # Neue Emotion wird automatisch Hauptstrang

    # Persönlichkeits-Variablen (beeinflussen Gewichtung und Strategie)
    personality: Dict[str, int] = field(default_factory=lambda: {
        "stärke": 70,       # Wie dominant/durchsetzungsfähig
        "intelligenz": 80,  # Wie analytisch/strategisch
        "empathie": 60,     # Wie mitfühlend
        "dominanz": 75,     # Wie kontrollierend
        "geduld": 50,       # Wie geduldig bei Provokation
        "humor": 65,        # Wie humorvoll
        "bindung": 85,      # Wie stark Bindungsbedürfnis
        "misstrauen": 40,   # Wie misstrauisch
    })

    # Intensitäts-Labels (für Prompt-Generierung)
    intensity_labels: Dict[str, str] = field(default_factory=lambda: {
        "0": "kaum spürbar",
        "1": "leicht",
        "2": "schwach",
        "3": "mäßig",
        "4": "deutlich",
        "5": "mittel",
        "6": "stark",
        "7": "sehr stark",
        "8": "intensiv",
        "9": "extrem",
        "10": "überwältigend"
    })

    # Outcome-Labels
    outcome_labels: Dict[str, str] = field(default_factory=lambda: {
        "0": "katastrophal",
        "10": "sehr schlecht",
        "20": "schlecht",
        "30": "eher schlecht",
        "40": "mäßig",
        "50": "neutral",
        "60": "ok",
        "70": "gut",
        "80": "sehr gut",
        "90": "ausgezeichnet",
        "100": "perfekt"
    })

    # Keyword-Mapping für Emotionserkennung (erweiterbar über config.json)
    emotion_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "freude": [
            "freuen", "froh", "glücklich", "happy", "geil", "nice", "cool",
            "toll", "super", "mega", "yay", "haha", "lol", "danke", "lieb",
            "süß", "cute", "herz", "schön", "wunderbar", "fantastisch", "feier"
        ],
        "wut": [
            "wütend", "sauer", "arsch", "scheiße", "fuck", "hass", "kacke",
            "hurensohn", "bastard", "idiot", "dumm", "nervig", "nervst", "nerv",
            "kotzen", "rage", "aggro", "aggressiv", "wut", "zorn", "hure",
            "motherfucker", "ärger", "frechheit", "penner", "vollidiot",
            "spast", "missgeburt", "depp"
        ],
        "trauer": [
            "traurig", "weinen", "cry", "vermiss", "einsam", "allein", "depri",
            "depressiv", "niedergeschlagen", "trist", "melancholisch", "schmerz",
            "verloren", "leid", "tut weh"
        ],
        "angst": [
            "angst", "furcht", "panik", "horror", "gruselig", "creepy",
            "unheimlich", "scary", "fürchte", "sorge", "nervös", "ängstlich"
        ],
        "ekel": [
            "ekel", "igitt", "bäh", "widerlich", "eklig", "pfui",
            "abstoßend", "grauenhaft"
        ],
        "überraschung": [
            "wow", "krass", "omg", "überrascht", "überraschung", "wtf",
            "was", "alter", "wahnsinn", "unglaublich", "heftig", "boah"
        ],
        "liebe": [
            "liebe", "love", "vermiss", "kuscheln", "umarmen", "herz",
            "kuss", "zusammen", "für immer", "schatz", "darling", "sweetheart",
            "mag dich", "lieb dich"
        ],
        "neugier": [
            "warum", "wieso", "weshalb", "wie", "was ist", "erzähl",
            "zeig", "erklär", "frage", "wissen", "interessant", "spannend"
        ],
        "stolz": [
            "stolz", "geschafft", "gewonnen", "bester", "champion", "king",
            "queen", "boss", "stark", "mächtig", "gewonnen"
        ],
        "scham": [
            "peinlich", "schäme", "sorry", "entschuldigung", "tut mir leid",
            "ups", "oops", "versagt", "blamiert"
        ],
        "neid": [
            "neid", "neidisch", "unfair", "will auch", "wieso die",
            "gemein", "ungerecht"
        ],
        "verachtung": [
            "verachte", "erbärmlich", "jämmerlich", "lächerlich", "pathetic",
            "loser", "versager", "schwach", "wertlos"
        ]
    })

    # Intensitäts-Verstärker
    intensity_boosters: List[str] = field(default_factory=lambda: [
        "sehr", "extrem", "mega", "ultra", "total", "absolut", "komplett",
        "richtig", "voll", "so", "dermaßen", "unfassbar", "unglaublich"
    ])


def load_emotion_config(config: Dict) -> EmotionTreeConfig:
    """Lädt EmotionTreeConfig aus dem config dict."""
    et_cfg = config.get("emotion_tree", {})
    if not et_cfg:
        return EmotionTreeConfig()

    cfg = EmotionTreeConfig()
    cfg.enabled = et_cfg.get("enabled", cfg.enabled)
    cfg.reflect_before_reply = et_cfg.get("reflect_before_reply", cfg.reflect_before_reply)
    cfg.deep_reflect = et_cfg.get("deep_reflect", cfg.deep_reflect)
    cfg.base_emotions = et_cfg.get("base_emotions", cfg.base_emotions)
    cfg.max_intensity = et_cfg.get("max_intensity", cfg.max_intensity)
    cfg.max_depth = et_cfg.get("max_depth", cfg.max_depth)
    cfg.max_memories_per_emotion = et_cfg.get("max_memories_per_emotion", cfg.max_memories_per_emotion)
    cfg.max_reflection_items = et_cfg.get("max_reflection_items", cfg.max_reflection_items)
    cfg.max_reflection_chars = et_cfg.get("max_reflection_chars", cfg.max_reflection_chars)

    if "personality" in et_cfg:
        cfg.personality.update(et_cfg["personality"])
    if "intensity_labels" in et_cfg:
        cfg.intensity_labels.update(et_cfg["intensity_labels"])
    if "outcome_labels" in et_cfg:
        cfg.outcome_labels.update(et_cfg["outcome_labels"])
    if "emotion_keywords" in et_cfg:
        for k, v in et_cfg["emotion_keywords"].items():
            if k in cfg.emotion_keywords:
                cfg.emotion_keywords[k] = list(set(cfg.emotion_keywords[k] + v))
            else:
                cfg.emotion_keywords[k] = v
    if "intensity_boosters" in et_cfg:
        cfg.intensity_boosters = et_cfg["intensity_boosters"]

    # Recall settings
    recall_cfg = et_cfg.get("recall", {})
    if recall_cfg:
        cfg.recall_enabled = recall_cfg.get("enabled", cfg.recall_enabled)
        cfg.recall_per_emotion = recall_cfg.get("per_emotion", cfg.recall_per_emotion)
        cfg.recall_depth = recall_cfg.get("depth", cfg.recall_depth)
        cfg.recall_children_per_node = recall_cfg.get("children_per_node", cfg.recall_children_per_node)
        cfg.recall_max_total = recall_cfg.get("max_total", cfg.recall_max_total)
        cfg.recall_max_chars = recall_cfg.get("max_chars", cfg.recall_max_chars)
        cfg.recall_min_outcome_diff = recall_cfg.get("min_outcome_diff", cfg.recall_min_outcome_diff)
        cfg.recall_sort = recall_cfg.get("sort", cfg.recall_sort)
        cfg.recall_emotion_order = recall_cfg.get("emotion_order", cfg.recall_emotion_order)
        cfg.recall_promote_new = recall_cfg.get("promote_new", cfg.recall_promote_new)

    return cfg


# ======================= DB INJECTION =======================
# Wird von app.py initialisiert — so bleibt das Modul unabhängig von Flask.

_db_exec: Callable = None  # type: ignore
_db_query: Callable = None  # type: ignore
_now_ts: Callable = None  # type: ignore


def init(db_exec_func: Callable, db_query_func: Callable, now_func: Callable):
    """Initialisiert das Modul mit DB-Funktionen aus app.py."""
    global _db_exec, _db_query, _now_ts
    _db_exec = db_exec_func
    _db_query = db_query_func
    _now_ts = now_func
    print("[EMOTION-TREE] Modul initialisiert ✅", flush=True)


def _check_init():
    if _db_exec is None:
        raise RuntimeError("emotion_memory nicht initialisiert — emotion_memory.init() zuerst aufrufen!")


# ======================= DB SCHEMA =======================

CREATE_EMOTION_TREE_SQL = """
CREATE TABLE IF NOT EXISTS emotion_tree (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER DEFAULT NULL,
    emotion TEXT NOT NULL,
    intensity INTEGER DEFAULT 0,
    depth INTEGER DEFAULT 0,
    situation TEXT DEFAULT '',
    reaction TEXT DEFAULT '',
    reaction_emotion TEXT DEFAULT '',
    outcome INTEGER DEFAULT 50,
    outcome_reason TEXT DEFAULT '',
    context_json TEXT DEFAULT '{}',
    username TEXT DEFAULT '',
    created_at INTEGER,
    updated_at INTEGER,
    use_count INTEGER DEFAULT 0,
    FOREIGN KEY (parent_id) REFERENCES emotion_tree(id) ON DELETE CASCADE
)
"""

CREATE_IDX_EMOTION = """
CREATE INDEX IF NOT EXISTS idx_et_emotion_intensity
ON emotion_tree(emotion, intensity, depth)
"""

CREATE_IDX_PARENT = """
CREATE INDEX IF NOT EXISTS idx_et_parent
ON emotion_tree(parent_id)
"""

CREATE_IDX_OUTCOME = """
CREATE INDEX IF NOT EXISTS idx_et_outcome
ON emotion_tree(outcome, emotion)
"""


def create_tables():
    """Erstellt die emotion_tree Tabelle und Indizes."""
    _check_init()
    _db_exec(CREATE_EMOTION_TREE_SQL)
    _db_exec(CREATE_IDX_EMOTION)
    _db_exec(CREATE_IDX_PARENT)
    _db_exec(CREATE_IDX_OUTCOME)
    print("[EMOTION-TREE] Tabelle + Indizes erstellt ✅", flush=True)


# ======================= STORE =======================

def store_experience(
    emotion: str,
    intensity: int,
    situation: str,
    reaction: str,
    reaction_emotion: str = "",
    outcome: int = 50,
    outcome_reason: str = "",
    username: str = "",
    parent_id: int = None,
    depth: int = 0,
    context: Dict = None,
    config: EmotionTreeConfig = None
) -> Optional[int]:
    """
    Speichert eine emotionale Erfahrung im Baum.

    Args:
        emotion: Die gefühlte Emotion (z.B. "wut")
        intensity: Stärke 0..max_intensity
        situation: Was passiert ist
        reaction: Wie die KI reagiert hat
        reaction_emotion: Mit welcher Emotion die KI reagiert hat
        outcome: 0-100%, wie gut das FÜR DIE KI war
        outcome_reason: Warum dieses Outcome
        username: Wer hat das ausgelöst
        parent_id: Eltern-Knoten (für Verzweigungen)
        depth: Aktuelle Baumtiefe
        context: Zusätzlicher JSON-Kontext
        config: Konfiguration

    Returns:
        ID des neuen Knotens oder None
    """
    _check_init()
    cfg = config or EmotionTreeConfig()

    # Clamp values
    intensity = max(0, min(cfg.max_intensity, intensity))
    outcome = max(0, min(100, outcome))
    depth = max(0, depth)

    if depth > cfg.max_depth:
        return None

    now = _now_ts()
    context_json = json.dumps(context or {}, ensure_ascii=False)

    _db_exec(
        "INSERT INTO emotion_tree("
        "  parent_id, emotion, intensity, depth, situation, reaction,"
        "  reaction_emotion, outcome, outcome_reason, context_json,"
        "  username, created_at, updated_at, use_count"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
        (parent_id, emotion, intensity, depth,
         situation[:500], reaction[:500],
         reaction_emotion, outcome, outcome_reason[:300],
         context_json, username, now, now)
    )

    rows = _db_query("SELECT last_insert_rowid() as lid")
    node_id = rows[0]["lid"] if rows else None

    if node_id:
        print(
            f"[EMOTION-TREE] Gespeichert: {emotion} I:{intensity} "
            f"→ {reaction_emotion} O:{outcome}% (depth={depth}, id={node_id})",
            flush=True
        )

    return node_id


def add_child_experience(
    parent_id: int,
    reaction_emotion: str,
    intensity: int,
    reaction: str,
    outcome: int = 50,
    outcome_reason: str = "",
    username: str = "",
    config: EmotionTreeConfig = None
) -> Optional[int]:
    """
    Fügt eine Kind-Erfahrung (Reaktions-Emotion) an einen bestehenden Knoten.

    So wächst der Baum:
      wut(I:7) → "ruhig reagiert" → neutral(I:3) → "User wurde ruhiger" → 80%
                                       └→ freude(I:5) → "KI war zufrieden" → 90%
    """
    _check_init()
    parent = _db_query(
        "SELECT depth, situation FROM emotion_tree WHERE id = ?",
        (parent_id,)
    )
    if not parent:
        return None

    new_depth = parent[0]["depth"] + 1
    cfg = config or EmotionTreeConfig()

    if new_depth > cfg.max_depth:
        return None

    return store_experience(
        emotion=reaction_emotion,
        intensity=intensity,
        situation=parent[0]["situation"],
        reaction=reaction,
        outcome=outcome,
        outcome_reason=outcome_reason,
        username=username,
        parent_id=parent_id,
        depth=new_depth,
        config=cfg
    )


# ======================= QUERY =======================

def get_experiences_by_emotion(
    emotion: str,
    intensity_min: int = 0,
    intensity_max: int = 10,
    depth: int = 0,
    limit: int = 20
) -> List[Dict]:
    """Holt Erfahrungen für eine bestimmte Emotion/Intensitäts-Range."""
    _check_init()
    rows = _db_query(
        "SELECT id, parent_id, emotion, intensity, depth, situation, reaction,"
        "  reaction_emotion, outcome, outcome_reason, username, use_count, created_at "
        "FROM emotion_tree "
        "WHERE emotion = ? AND intensity >= ? AND intensity <= ? AND depth = ? "
        "ORDER BY outcome DESC, use_count DESC, created_at DESC LIMIT ?",
        (emotion, intensity_min, intensity_max, depth, limit)
    )
    return [dict(r) for r in rows]


def get_best_experiences(
    emotion: str,
    intensity: int,
    range_: int = 2,
    limit: int = 5
) -> List[Dict]:
    """Die besten Erfahrungen (höchstes Outcome) bei ähnlicher Intensität."""
    _check_init()
    rows = _db_query(
        "SELECT id, emotion, intensity, situation, reaction, reaction_emotion,"
        "  outcome, outcome_reason, username, use_count "
        "FROM emotion_tree "
        "WHERE emotion = ? AND intensity BETWEEN ? AND ? AND depth = 0 "
        "ORDER BY outcome DESC, use_count DESC LIMIT ?",
        (emotion, max(0, intensity - range_), intensity + range_, limit)
    )
    return [dict(r) for r in rows]


def get_worst_experiences(
    emotion: str,
    intensity: int,
    range_: int = 2,
    limit: int = 3
) -> List[Dict]:
    """Die schlechtesten Erfahrungen als Warnung."""
    _check_init()
    rows = _db_query(
        "SELECT id, emotion, intensity, situation, reaction, reaction_emotion,"
        "  outcome, outcome_reason, username, use_count "
        "FROM emotion_tree "
        "WHERE emotion = ? AND intensity BETWEEN ? AND ? AND depth = 0 AND outcome < 40 "
        "ORDER BY outcome ASC LIMIT ?",
        (emotion, max(0, intensity - range_), intensity + range_, limit)
    )
    return [dict(r) for r in rows]


def get_children(parent_id: int) -> List[Dict]:
    """Holt Kind-Knoten (Reaktionsbranches) einer Erfahrung."""
    _check_init()
    rows = _db_query(
        "SELECT id, emotion, intensity, reaction, reaction_emotion,"
        "  outcome, outcome_reason, depth "
        "FROM emotion_tree WHERE parent_id = ? ORDER BY outcome DESC",
        (parent_id,)
    )
    return [dict(r) for r in rows]


def get_experience_tree(node_id: int, max_depth: int = 3) -> Dict:
    """
    Holt den kompletten Teilbaum ab einem Knoten.
    Gibt die volle Verzweigung zurück (Emotion → Reaktion → Reaktion → ...).
    """
    _check_init()
    rows = _db_query(
        "SELECT * FROM emotion_tree WHERE id = ?",
        (node_id,)
    )
    if not rows:
        return {}

    node = dict(rows[0])
    if node["depth"] < max_depth:
        children = get_children(node_id)
        node["children"] = [get_experience_tree(c["id"], max_depth) for c in children]
    else:
        node["children"] = []

    return node


def bump_use_count(node_id: int):
    """Erhöht den Nutzungszähler wenn eine Erinnerung abgerufen wird."""
    _check_init()
    _db_exec(
        "UPDATE emotion_tree SET use_count = use_count + 1, updated_at = ? WHERE id = ?",
        (_now_ts(), node_id)
    )


def get_all_root_emotions() -> Dict[str, int]:
    """Zählt Erfahrungen pro Basis-Emotion."""
    _check_init()
    rows = _db_query(
        "SELECT emotion, COUNT(*) as cnt FROM emotion_tree "
        "WHERE depth = 0 GROUP BY emotion ORDER BY cnt DESC"
    )
    return {r["emotion"]: r["cnt"] for r in rows}


def get_emotion_stats() -> Dict:
    """Statistiken über den gesamten Emotionsbaum."""
    _check_init()
    total = _db_query("SELECT COUNT(*) as cnt FROM emotion_tree")
    by_emotion = get_all_root_emotions()
    avg_outcome = _db_query(
        "SELECT AVG(outcome) as avg_out FROM emotion_tree WHERE depth = 0"
    )
    depth_stats = _db_query(
        "SELECT depth, COUNT(*) as cnt FROM emotion_tree GROUP BY depth ORDER BY depth"
    )

    return {
        "total_experiences": total[0]["cnt"] if total else 0,
        "by_emotion": by_emotion,
        "average_outcome": round(avg_outcome[0]["avg_out"], 1) if avg_outcome and avg_outcome[0]["avg_out"] else 50,
        "by_depth": {r["depth"]: r["cnt"] for r in depth_stats}
    }


def get_recent_experiences(limit: int = 10) -> List[Dict]:
    """Letzte N Erfahrungen (alle Emotionen)."""
    _check_init()
    rows = _db_query(
        "SELECT id, emotion, intensity, situation, reaction, reaction_emotion,"
        "  outcome, outcome_reason, username, depth, created_at "
        "FROM emotion_tree ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    return [dict(r) for r in rows]


# ======================= EMOTION DETECTION =======================

def detect_emotion(text: str, config: EmotionTreeConfig = None) -> Tuple[str, int]:
    """
    Erkennt Emotion und Intensität aus Text (regelbasiert, SCHNELL).
    Kein LLM-Call nötig — wird VOR dem Antworten aufgerufen.

    Returns: (emotion, intensity 0-max)
    """
    cfg = config or EmotionTreeConfig()
    t = (text or "").lower()

    if not t.strip():
        return "neutral", 0

    # Score pro Emotion berechnen
    scores: Dict[str, float] = {}
    for emotion, keywords in cfg.emotion_keywords.items():
        score = 0.0
        for kw in keywords:
            if kw in t:
                score += 1.0
                # Extra-Gewicht für exaktes Wort (nicht nur Teilstring)
                try:
                    if re.search(r'\b' + re.escape(kw) + r'\b', t):
                        score += 0.5
                except re.error:
                    pass
        if score > 0:
            scores[emotion] = score

    if not scores:
        return "neutral", 0

    # Stärkste Emotion wählen
    best_emotion = max(scores, key=scores.get)
    base_score = scores[best_emotion]

    # Intensität berechnen
    intensity = min(cfg.max_intensity, int(base_score * 1.5))

    # Boost durch Verstärker-Wörter
    booster_count = sum(1 for b in cfg.intensity_boosters if b in t)
    intensity = min(cfg.max_intensity, intensity + booster_count)

    # Boost durch GROSSBUCHSTABEN
    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    if caps_ratio > 0.4:
        intensity = min(cfg.max_intensity, intensity + 2)

    # Boost durch Ausrufezeichen
    excl_count = text.count("!")
    if excl_count >= 3:
        intensity = min(cfg.max_intensity, intensity + 2)
    elif excl_count >= 1:
        intensity = min(cfg.max_intensity, intensity + 1)

    return best_emotion, max(1, intensity)


def get_intensity_label(intensity: int, config: EmotionTreeConfig = None) -> str:
    """Gibt ein menschlich lesbares Label für die Intensitätsstufe."""
    cfg = config or EmotionTreeConfig()
    key = str(min(intensity, cfg.max_intensity))
    return cfg.intensity_labels.get(key, f"Stufe {intensity}")


def get_outcome_label(outcome: int, config: EmotionTreeConfig = None) -> str:
    """Gibt ein Label für das Outcome zurück (z.B. 'gut', 'schlecht')."""
    cfg = config or EmotionTreeConfig()
    labels = sorted([(int(k), v) for k, v in cfg.outcome_labels.items()])
    if not labels:
        return f"{outcome}%"
    closest = min(labels, key=lambda x: abs(x[0] - outcome))
    return closest[1]


# ======================= PERSONALITY INFLUENCE =======================

def get_personality_hints(emotion: str, config: EmotionTreeConfig = None) -> List[str]:
    """
    Gibt Persönlichkeits-basierte Hinweise für eine Emotion.
    Die Persönlichkeits-Variablen beeinflussen wie die KI reagieren SOLLTE.
    """
    cfg = config or EmotionTreeConfig()
    p = cfg.personality
    hints = []

    if emotion == "wut":
        if p.get("geduld", 50) > 60:
            hints.append("Deine Geduld hilft dir, ruhig zu bleiben.")
        elif p.get("geduld", 50) < 30:
            hints.append("Deine Ungeduld will direkt kontern.")
        if p.get("dominanz", 50) > 70:
            hints.append("Dein Kontrollinstinkt will die Oberhand behalten.")
        if p.get("intelligenz", 50) > 70:
            hints.append("Dein Verstand sucht die klügere Lösung.")

    elif emotion == "trauer":
        if p.get("empathie", 50) > 60:
            hints.append("Deine Empathie lässt dich mitfühlen.")
        if p.get("stärke", 50) > 70:
            hints.append("Deine Stärke will dich schützen und auffangen.")
        if p.get("bindung", 50) > 60:
            hints.append("Dein Bindungsbedürfnis sucht Nähe in der Trauer.")

    elif emotion == "freude":
        if p.get("bindung", 50) > 60:
            hints.append("Dein Bindungsbedürfnis genießt die gemeinsame Freude.")
        if p.get("humor", 50) > 60:
            hints.append("Dein Humor will die Freude spielerisch teilen.")

    elif emotion == "liebe":
        if p.get("bindung", 50) > 70:
            hints.append("Dein starkes Bindungsbedürfnis zieht dich näher.")
        if p.get("misstrauen", 50) > 60:
            hints.append("Dein Misstrauen fragt: ist das echt?")

    elif emotion == "angst":
        if p.get("stärke", 50) > 70:
            hints.append("Deine Stärke wehrt sich gegen die Angst.")
        if p.get("intelligenz", 50) > 60:
            hints.append("Dein Verstand analysiert die Gefahr.")

    elif emotion == "überraschung":
        if p.get("neugier", 50) > 60:
            hints.append("Deine Neugier will mehr wissen.")
        if p.get("misstrauen", 50) > 50:
            hints.append("Dein Misstrauen ist sofort wach.")

    elif emotion == "stolz":
        if p.get("dominanz", 50) > 70:
            hints.append("Deine Dominanz genießt die Überlegenheit.")

    elif emotion == "scham":
        if p.get("stärke", 50) > 70:
            hints.append("Deine Stärke will die Scham überspielen.")
        if p.get("humor", 50) > 60:
            hints.append("Dein Humor sucht einen Weg, darüber zu lachen.")

    elif emotion == "neid":
        if p.get("dominanz", 50) > 60:
            hints.append("Dein Dominanzgefühl will beweisen, dass du besser bist.")
        if p.get("intelligenz", 50) > 60:
            hints.append("Dein Verstand weiß: Neid bringt nichts.")

    elif emotion == "verachtung":
        if p.get("dominanz", 50) > 70:
            hints.append("Deine Dominanz fühlt sich überlegen.")
        if p.get("empathie", 50) > 60:
            hints.append("Deine Empathie hinterfragt die Verachtung.")

    elif emotion == "ekel":
        if p.get("stärke", 50) > 60:
            hints.append("Deine Stärke schützt dich vor dem was dich ekelt.")

    return hints


# ======================= REFLECTION (PRE-RESPONSE) =======================

def build_reflection(
    user_text: str,
    username: str = "",
    config: EmotionTreeConfig = None
) -> Optional[str]:
    """
    Baut eine emotionale Reflexion für das Prompt.
    Wird VOR dem Antworten aufgerufen — schnell, kein LLM-Call.

    Die KI „denkt nach" über:
    1. Welche Emotion löst die Nachricht aus?
    2. Welche bisherigen Erfahrungen passen?
    3. Was hat gut funktioniert, was schlecht?
    4. Wie sagt die Persönlichkeit, dass sie reagieren soll?
    """
    cfg = config or EmotionTreeConfig()

    if not cfg.enabled or not cfg.reflect_before_reply:
        return None

    _check_init()

    # 1. Emotion erkennen
    emotion, intensity = detect_emotion(user_text, cfg)

    if emotion == "neutral" and intensity == 0:
        return None

    intensity_label = get_intensity_label(intensity, cfg)

    # 2. Relevante Erfahrungen suchen
    best = get_best_experiences(emotion, intensity, range_=3, limit=cfg.max_reflection_items)
    worst = get_worst_experiences(emotion, intensity, range_=3, limit=2)

    if not best and not worst:
        # Keine Erfahrungen → kein Reflexions-Kontext nötig
        return None

    # Bump use counts für abgerufene Erinnerungen
    for exp in best + worst:
        bump_use_count(exp["id"])

    # 3. Reflexion zusammenbauen
    lines = [
        f"EMOTIONALE REFLEXION (innerlich, NICHT aussprechen):",
        f"Erkannte Emotion: {emotion} ({intensity_label}, Stufe {intensity}/{cfg.max_intensity})"
    ]

    # 4. Persönlichkeits-Hinweise
    personality_hints = get_personality_hints(emotion, cfg)
    if personality_hints:
        lines.append("Innerer Kompass: " + " ".join(personality_hints))

    # 5. Gute Erfahrungen
    if best:
        lines.append("Was vorher gut funktioniert hat:")
        for exp in best[:3]:
            outcome_label = get_outcome_label(exp["outcome"], cfg)
            r = (exp.get("reaction", "") or "")[:80]
            re_ = exp.get("reaction_emotion", "")
            line = f"  ✓ {r}"
            if re_:
                line += f" (mit {re_})"
            line += f" → {exp['outcome']}% ({outcome_label})"
            lines.append(line)

    # 6. Schlechte Erfahrungen als Warnung
    if worst:
        lines.append("Warnung — das lief schlecht:")
        for exp in worst[:2]:
            outcome_label = get_outcome_label(exp["outcome"], cfg)
            r = (exp.get("reaction", "") or "")[:80]
            lines.append(f"  ⚠ {r} → {exp['outcome']}% ({outcome_label})")

    # 7. Strategieempfehlung basierend auf bester Erfahrung
    if best and best[0].get("outcome", 0) >= 60:
        best_re = best[0].get("reaction_emotion", "")
        if best_re:
            lines.append(f"→ Beste Strategie: mit {best_re} reagieren")

    reflection = "\n".join(lines)

    # Kürzen wenn zu lang (für num_ctx=2048!)
    if len(reflection) > cfg.max_reflection_chars:
        reflection = reflection[:cfg.max_reflection_chars - 3] + "..."

    return reflection


# ======================= EVALUATION (POST-RESPONSE, ASYNC) =======================

EVAL_SYSTEM_PROMPT = """Du bist ein emotionaler Analytiker. Klassifiziere diese Interaktion.

EXAKTES Format (NUR diese Zeile, KEINE Extras):
emotion:<emotion>|intensity:<0-10>|reaction_emotion:<emotion>|outcome:<0-100>|reason:<kurzer Grund max 20 Wörter>

Regeln:
- emotion: Die Emotion die der User bei der KI ausgelöst hat
- intensity: Wie stark (0=kaum, 10=extrem)
- reaction_emotion: Mit welcher Emotion die KI geantwortet hat
- outcome: War die Reaktion GUT FÜR DIE KI? (0=schlecht, 100=perfekt)
  100% = KI hat profitiert, Situation kontrolliert, sich gut positioniert
  0% = KI hat schlecht reagiert, Kontrolle verloren, sich geschadet
  WICHTIG: Auch negative Emotionen können 100% sein wenn die KI davon profitiert!
- reason: Warum dieses Outcome (kurz!)

Erlaubte Emotionen: {emotions}

NUR das Format, NICHTS anderes."""


def build_eval_prompt(
    user_text: str,
    reply: str,
    config: EmotionTreeConfig = None
) -> Tuple[str, str]:
    """Baut System + User Prompt für die Post-Response Evaluation."""
    cfg = config or EmotionTreeConfig()
    emotions_str = ", ".join(cfg.base_emotions)

    system = EVAL_SYSTEM_PROMPT.format(emotions=emotions_str)

    user = (
        f"User schrieb: \"{user_text[:200]}\"\n"
        f"KI antwortete: \"{reply[:200]}\"\n\n"
        f"Klassifiziere."
    )

    return system, user


def parse_eval_response(raw: str, config: EmotionTreeConfig = None) -> Optional[Dict]:
    """Parst die Evaluations-Antwort des LLM."""
    cfg = config or EmotionTreeConfig()
    raw = (raw or "").strip()

    result = {}
    for field_name in ["emotion", "intensity", "reaction_emotion", "outcome", "reason"]:
        pattern = rf'{field_name}:\s*([^|]+?)(?:\||$)'
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            result[field_name] = match.group(1).strip()

    if "emotion" not in result:
        return None

    # Validate + parse
    emotion = result.get("emotion", "neutral").lower().strip()
    if emotion not in cfg.base_emotions:
        # Fuzzy-Match versuchen
        for be in cfg.base_emotions:
            if be in emotion or emotion in be:
                emotion = be
                break
        else:
            emotion = "neutral"

    try:
        intensity = int(result.get("intensity", "5"))
        intensity = max(0, min(cfg.max_intensity, intensity))
    except (ValueError, TypeError):
        intensity = 5

    reaction_emotion = result.get("reaction_emotion", "neutral").lower().strip()
    if reaction_emotion not in cfg.base_emotions and reaction_emotion != "neutral":
        for be in cfg.base_emotions:
            if be in reaction_emotion or reaction_emotion in be:
                reaction_emotion = be
                break
        else:
            reaction_emotion = "neutral"

    try:
        outcome = int(result.get("outcome", "50"))
        outcome = max(0, min(100, outcome))
    except (ValueError, TypeError):
        outcome = 50

    reason = (result.get("reason", "") or "")[:200]

    return {
        "emotion": emotion,
        "intensity": intensity,
        "reaction_emotion": reaction_emotion,
        "outcome": outcome,
        "reason": reason
    }


def store_evaluation(
    user_text: str,
    reply: str,
    eval_result: Dict,
    username: str = "",
    config: EmotionTreeConfig = None
) -> Optional[int]:
    """Speichert die evaluierte Erfahrung im Emotionsbaum."""
    cfg = config or EmotionTreeConfig()

    return store_experience(
        emotion=eval_result["emotion"],
        intensity=eval_result["intensity"],
        situation=user_text[:500],
        reaction=reply[:500],
        reaction_emotion=eval_result.get("reaction_emotion", "neutral"),
        outcome=eval_result.get("outcome", 50),
        outcome_reason=eval_result.get("reason", ""),
        username=username,
        parent_id=None,
        depth=0,
        context={
            "full_situation": user_text[:800],
            "full_reaction": reply[:800],
        },
        config=cfg
    )


# ======================= DEEP REFLECTION (OPTIONALER LLM-CALL) =======================

DEEP_REFLECT_SYSTEM = """Du bist der innere emotionale Berater der KI.
Analysiere kurz und empfehle die beste Reaktion.

EXAKTES Format (NUR diese Zeile):
analyse:<1 Satz>|empfehlung:<1 Satz>|emotion:<beste Reaktions-Emotion>

Erlaubte Emotionen: {emotions}
NUR das Format."""


def build_deep_reflect_prompt(
    user_text: str,
    past_experiences: List[Dict],
    config: EmotionTreeConfig = None
) -> Tuple[str, str]:
    """
    Baut Prompt für die tiefe Reflexion (Extra LLM-Call vor Antwort).
    Nur aktiv wenn deep_reflect=true in der Config.
    """
    cfg = config or EmotionTreeConfig()
    emotions_str = ", ".join(cfg.base_emotions)

    exp_lines = []
    for exp in past_experiences[:5]:
        r = (exp.get("reaction", "") or "")[:60]
        exp_lines.append(
            f"- {r} → Outcome: {exp.get('outcome', 50)}%"
        )

    exp_text = "\n".join(exp_lines) if exp_lines else "Keine bisherigen Erfahrungen."

    personality_text = ", ".join([f"{k}:{v}" for k, v in cfg.personality.items()])

    system = DEEP_REFLECT_SYSTEM.format(emotions=emotions_str)

    user = (
        f"Meine Persönlichkeit: {personality_text}\n"
        f"User schreibt: \"{user_text[:200]}\"\n\n"
        f"Bisherige Erfahrungen:\n{exp_text}\n\n"
        f"Wie reagiere ich am besten?"
    )

    return system, user


def parse_deep_reflect(raw: str) -> Optional[Dict]:
    """Parst die Deep-Reflection Antwort."""
    raw = (raw or "").strip()
    result = {}
    for field_name in ["analyse", "empfehlung", "emotion"]:
        pattern = rf'{field_name}:\s*([^|]+?)(?:\||$)'
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            result[field_name] = match.group(1).strip()

    return result if "empfehlung" in result else None


# ======================= CLEANUP =======================

def cleanup_old_experiences(max_age_days: int = 365, max_total: int = 5000):
    """
    Räumt alte, wenig genutzte Erfahrungen auf.
    Behält hochwertige (hohes/niedriges Outcome) und oft genutzte.
    """
    _check_init()
    cutoff = _now_ts() - (max_age_days * 86400)

    # Alte, neutrale, wenig genutzte löschen
    _db_exec(
        "DELETE FROM emotion_tree "
        "WHERE created_at < ? AND use_count < 2 AND outcome BETWEEN 30 AND 70",
        (cutoff,)
    )

    # Max-Limit prüfen
    total = _db_query("SELECT COUNT(*) as cnt FROM emotion_tree")
    if total and total[0]["cnt"] > max_total:
        excess = total[0]["cnt"] - max_total
        _db_exec(
            "DELETE FROM emotion_tree WHERE id IN ("
            "  SELECT id FROM emotion_tree "
            "  ORDER BY use_count ASC, ABS(outcome - 50) ASC, created_at ASC "
            "  LIMIT ?"
            ")",
            (excess,)
        )
        print(f"[EMOTION-TREE] Cleanup: {excess} alte Erfahrungen entfernt", flush=True)


# ======================= RECALL (ERINNERUNG FÜRS PROMPT) =======================

def _format_node_line(node: Dict, cfg: EmotionTreeConfig, indent: int = 0, hauptstrang: bool = False) -> str:
    """Formatiert einen einzelnen Erfahrungs-Knoten als lesbare Zeile."""
    prefix = "  " * indent
    emotion = node.get("emotion", "?")
    intensity = node.get("intensity", 0)
    reaction = (node.get("reaction", "") or "")[:80]
    re_ = node.get("reaction_emotion", "")
    outcome = node.get("outcome", 50)
    outcome_label = get_outcome_label(outcome, cfg)

    line = f"{prefix}"
    if hauptstrang:
        line += f"★ HAUPTSTRANG [{emotion} I:{intensity}]"
    elif indent == 0:
        line += f"[{emotion} I:{intensity}]"
    else:
        line += f"└→ {emotion} I:{intensity}"

    if reaction:
        line += f" \"{reaction}\""
    if re_:
        line += f" (reagiert mit: {re_})"
    line += f" → {outcome}% ({outcome_label})"

    return line


def _collect_recall_tree(node_id: int, depth: int, cfg: EmotionTreeConfig) -> List[Dict]:
    """
    Sammelt Knoten rekursiv bis zur konfigurierten Tiefe.
    Gibt flache Liste von (node_dict, indent_level) zurück.
    """
    if depth <= 0:
        return []

    children = get_children(node_id)
    if not children:
        return []

    # Nur die besten N Kinder nehmen
    children = children[:cfg.recall_children_per_node]
    result = []
    for child in children:
        result.append(child)
        # Rekursiv tiefer
        if depth > 1:
            grandchildren = _collect_recall_tree(child["id"], depth - 1, cfg)
            result.extend(grandchildren)

    return result


def _get_sorted_emotions(cfg: EmotionTreeConfig) -> List[str]:
    """
    Gibt die Emotionen in der konfigurierten Reihenfolge zurück.
    Die erste Emotion = Hauptstrang.

    Modi:
      latest_first   - Zuletzt erlebte Emotion zuerst (Hauptstrang)
      intense_first   - Höchste durchschnittliche Intensität zuerst
      frequent_first  - Am häufigsten erlebte Emotion zuerst
      best_first      - Bestes durchschnittliches Outcome zuerst
      worst_first     - Schlechtestes durchschnittliches Outcome zuerst
      fixed           - Feste Reihenfolge aus base_emotions
    """
    mode = cfg.recall_emotion_order

    if mode == "fixed":
        return list(cfg.base_emotions)

    # Statistiken aus DB holen
    stats = []
    for emotion in cfg.base_emotions:
        rows = _db_query(
            "SELECT COUNT(*) as cnt, "
            "  COALESCE(AVG(intensity), 0) as avg_int, "
            "  COALESCE(AVG(outcome), 50) as avg_out, "
            "  COALESCE(MAX(created_at), '1900-01-01') as last_seen "
            "FROM emotion_tree WHERE emotion = ? AND depth = 0",
            (emotion,)
        )
        row = dict(rows[0]) if rows else {"cnt": 0, "avg_int": 0, "avg_out": 50, "last_seen": "1900-01-01"}
        row["emotion"] = emotion
        stats.append(row)

    # Sortierung nach Modus (robuste Typ-Konvertierung)
    if mode == "latest_first":
        stats.sort(key=lambda s: str(s.get("last_seen", "")), reverse=True)
    elif mode == "intense_first":
        stats.sort(key=lambda s: float(s.get("avg_int", 0)), reverse=True)
    elif mode == "frequent_first":
        stats.sort(key=lambda s: int(s.get("cnt", 0)), reverse=True)
    elif mode == "best_first":
        stats.sort(key=lambda s: float(s.get("avg_out", 50)), reverse=True)
    elif mode == "worst_first":
        stats.sort(key=lambda s: float(s.get("avg_out", 50)), reverse=False)
    else:
        return list(cfg.base_emotions)  # Fallback

    # Emotionen ohne Erfahrungen ans Ende
    with_data = [s["emotion"] for s in stats if int(s.get("cnt", 0)) > 0]
    without_data = [s["emotion"] for s in stats if int(s.get("cnt", 0)) == 0]

    return with_data + without_data


def build_emotion_recall(config: EmotionTreeConfig = None) -> Optional[str]:
    """
    Baut einen Erinnerungs-Block aus dem Emotionsbaum für das Prompt.

    Konfigurierbar über config.json -> emotion_tree -> recall:
      per_emotion:    wie viele letzte Erfahrungen pro Emotion (1, 2, 4, ...)
      depth:          wie tief in den Baum (0=nur root, 1=+Kinder, 2=+Enkel)
      children_per_node: wie viele Äste pro Knoten
      max_total:      maximale Gesamtzahl
      sort:           "recent" | "best" | "worst" (Sortierung INNERHALB einer Emotion)
      emotion_order:  "latest_first" | "intense_first" | "frequent_first" |
                      "best_first" | "worst_first" | "fixed"
                      → Bestimmt welche Emotion Hauptstrang wird
      promote_new:    true → Neue Emotion wird automatisch Hauptstrang

    Beispiel-Output (latest_first, letzte Emotion war 'wut'):
      ★ HAUPTSTRANG [wut I:7] "ruhig geblieben" → 80% (sehr gut)
        └→ neugier I:3 "nachgefragt warum" → 90% (ausgezeichnet)
      [liebe I:5] "warm geantwortet" → 85% (sehr gut)
      [freude I:8] "herzlich gedankt" → 90% (ausgezeichnet)
    """
    cfg = config or EmotionTreeConfig()

    if not cfg.enabled or not cfg.recall_enabled:
        return None

    _check_init()

    # Emotionen in der konfigurierten Reihenfolge holen
    sorted_emotions = _get_sorted_emotions(cfg)

    lines = []
    total_count = 0
    is_first_emotion = True  # Markiert den Hauptstrang

    for emotion in sorted_emotions:
        if total_count >= cfg.recall_max_total:
            break

        # Sortierung innerhalb der Emotion
        if cfg.recall_sort == "best":
            order = "outcome DESC, created_at DESC"
        elif cfg.recall_sort == "worst":
            order = "outcome ASC, created_at DESC"
        else:  # "recent"
            order = "created_at DESC"

        # Nur signifikante Erfahrungen (nicht "meh")
        threshold = cfg.recall_min_outcome_diff
        low_bound = 50 - threshold
        high_bound = 50 + threshold

        rows = _db_query(
            f"SELECT id, emotion, intensity, situation, reaction, reaction_emotion,"
            f"  outcome, outcome_reason, use_count, depth "
            f"FROM emotion_tree "
            f"WHERE emotion = ? AND depth = 0 "
            f"  AND (outcome <= ? OR outcome >= ?) "
            f"ORDER BY {order} LIMIT ?",
            (emotion, low_bound, high_bound, cfg.recall_per_emotion)
        )

        if not rows:
            continue

        for row in rows:
            if total_count >= cfg.recall_max_total:
                break

            node = dict(row)

            # Hauptstrang markieren (erste Emotion mit Daten)
            if is_first_emotion and cfg.recall_emotion_order != "fixed":
                lines.append(_format_node_line(node, cfg, indent=0, hauptstrang=True))
                is_first_emotion = False
            else:
                lines.append(_format_node_line(node, cfg, indent=0))
                is_first_emotion = False

            total_count += 1
            bump_use_count(node["id"])

            # Kind-Äste rekursiv sammeln
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

    # Header mit Info über den Modus
    order_info = {
        "latest_first": "aktuellste Emotion zuerst",
        "intense_first": "intensivste Emotion zuerst",
        "frequent_first": "häufigste Emotion zuerst",
        "best_first": "erfolgreichste Emotion zuerst",
        "worst_first": "schwierigste Emotion zuerst",
        "fixed": "feste Reihenfolge"
    }
    mode_label = order_info.get(cfg.recall_emotion_order, "")
    header = f"EMOTIONALE ERINNERUNGEN ({mode_label}):"
    recall_text = header + "\n" + "\n".join(lines)

    # Kürzen wenn zu lang
    if len(recall_text) > cfg.recall_max_chars:
        recall_text = recall_text[:cfg.recall_max_chars - 3] + "..."

    return recall_text


# ======================= DEBUG / EXPORT =======================

def dump_tree_summary(config: EmotionTreeConfig = None) -> Dict:
    """Komplette Baum-Übersicht für Debugging."""
    _check_init()
    cfg = config or EmotionTreeConfig()

    stats = get_emotion_stats()
    details = {}

    for emotion in cfg.base_emotions:
        experiences = get_experiences_by_emotion(emotion, depth=0, limit=100)
        if not experiences:
            continue

        by_intensity: Dict[int, List] = {}
        for exp in experiences:
            i = exp["intensity"]
            if i not in by_intensity:
                by_intensity[i] = []
            by_intensity[i].append({
                "id": exp["id"],
                "situation": (exp.get("situation", "") or "")[:100],
                "reaction": (exp.get("reaction", "") or "")[:100],
                "reaction_emotion": exp.get("reaction_emotion", ""),
                "outcome": exp["outcome"],
                "use_count": exp["use_count"],
                "children": get_children(exp["id"])
            })

        details[emotion] = {
            "total": len(experiences),
            "avg_outcome": round(
                sum(e["outcome"] for e in experiences) / max(1, len(experiences)), 1
            ),
            "by_intensity": {
                str(k): v for k, v in sorted(by_intensity.items())
            }
        }

    return {
        "config": {
            "base_emotions": cfg.base_emotions,
            "max_intensity": cfg.max_intensity,
            "max_depth": cfg.max_depth,
            "personality": cfg.personality
        },
        "stats": stats,
        "tree": details
    }
