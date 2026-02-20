#!/usr/bin/env python3
"""
Generic Memory Bank Engine (Generisches Gedächtnis-System)
============================================================
Ein generisches System für beliebig viele Memory-Datenbanken.

Jede Bank wird über memory_banks.json konfiguriert:
  - Name + Tabelle (eindeutig)
  - Kategorien + Keywords zur automatischen Erkennung
  - Bewertungsskala (0-N)
  - Recall: Erinnerungen ins Prompt injizieren
  - Reflexion: Vor-Antwort-Analyse
  - Eval: Nach-Antwort LLM-Bewertung (async)
  - Meta-Variablen (optional, z.B. Persönlichkeit)

Alle Banks teilen dasselbe DB-Schema:
  id, parent_id, category, rating, depth, situation, response,
  response_strategy, outcome, outcome_reason, context_json,
  username, created_at, updated_at, use_count

Konfiguration: memory_banks.json (Array von Bank-Definitionen)
Jeder Eintrag im Array = eine eigene Datenbank/Tabelle.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable


# ======================= CONFIG =======================

@dataclass
class MemoryBankConfig:
    """Konfiguration für eine einzelne Memory Bank — alles variabel."""

    # ===== Identität =====
    name: str = ""               # Eindeutiger Name: "emotion", "topic", "social", etc.
    table: str = ""              # SQLite Tabellenname, z.B. "bank_emotion"
    enabled: bool = True
    evaluate_after_reply: bool = True  # Async LLM-Eval nach jeder Antwort

    # ===== Labels =====
    label: str = "Kategorie"          # Anzeigename: "Emotion", "Thema", etc.
    rating_label: str = "Bewertung"   # "Intensität", "Qualität", etc.
    rating_max: int = 10

    # ===== Kategorien =====
    categories: List[str] = field(default_factory=list)
    keywords: Dict[str, List[str]] = field(default_factory=dict)
    allow_dynamic: bool = True
    max_categories: int = 50

    # ===== Baum-Struktur =====
    max_depth: int = 2
    max_items: int = 200  # Max Einträge pro Kategorie

    # ===== Reflexion =====
    reflection_enabled: bool = True
    reflection_header: str = ""       # Auto-generiert wenn leer
    reflection_detected: str = ""     # "Erkannte Emotion:" / "Erkanntes Thema:"
    reflection_best: str = ""         # Label für gute Erfahrungen
    reflection_worst: str = ""        # Label für schlechte Erfahrungen
    reflection_recommend: str = ""    # Label für Empfehlung
    reflection_max_items: int = 5
    reflection_max_chars: int = 400

    # ===== Recall =====
    recall_enabled: bool = True
    recall_header: str = ""           # Auto-generiert wenn leer
    recall_hauptstrang: str = ""      # "★ HAUPTEMOTION", "★ HAUPTTHEMA", etc.
    recall_per_category: int = 1
    recall_depth: int = 1
    recall_children: int = 1          # children_per_node
    recall_max_total: int = 8
    recall_max_chars: int = 500
    recall_min_diff: int = 2          # Min Abweichung von neutral
    recall_sort: str = "recent"       # "recent", "best", "worst"
    recall_order: str = "latest_first"  # latest_first, frequent_first, best_first, worst_first, fixed

    # ===== Eval Prompt =====
    eval_system_prompt: str = ""      # Custom, sonst auto-generiert

    # ===== Rating Labels =====
    rating_labels: Dict[str, str] = field(default_factory=dict)

    # ===== Meta-Variablen (optional, z.B. Persönlichkeit) =====
    meta_variables: Dict[str, int] = field(default_factory=dict)


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


def _check_init():
    if not _initialized:
        raise RuntimeError("[MEMORY-BANK] Nicht initialisiert! Zuerst init() aufrufen.")


def _safe_table(name: str) -> str:
    """Sanitize table name for SQL injection safety."""
    clean = re.sub(r'[^a-z0-9_]', '', (name or "").lower().strip())
    return clean or "bank_default"


# ======================= TABLE CREATION =======================

def create_table(cfg: MemoryBankConfig):
    """Erstellt die Tabelle + Indizes für eine Bank."""
    _check_init()
    tbl = _safe_table(cfg.table)

    _db_exec(f"""
        CREATE TABLE IF NOT EXISTS {tbl} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER,
            category TEXT NOT NULL,
            rating INTEGER DEFAULT 5,
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
            FOREIGN KEY (parent_id) REFERENCES {tbl}(id)
        )
    """)
    _db_exec(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_cat ON {tbl}(category)")
    _db_exec(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_depth ON {tbl}(depth)")
    _db_exec(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_user ON {tbl}(username)")
    _db_exec(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_parent ON {tbl}(parent_id)")
    _db_exec(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_created ON {tbl}(created_at)")
    print(f"[MEMORY-BANK:{cfg.name}] Tabelle '{tbl}' + Indizes erstellt ✅", flush=True)


# ======================= CONFIG LOADER =======================

def load_banks(json_path: str) -> List[MemoryBankConfig]:
    """Lädt alle Bank-Konfigurationen aus der JSON-Datei."""
    if not os.path.exists(json_path):
        print(f"[MEMORY-BANK] {json_path} nicht gefunden, keine Banks geladen.", flush=True)
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[MEMORY-BANK] Fehler beim Laden von {json_path}: {e}", flush=True)
        return []

    if not isinstance(data, list):
        print(f"[MEMORY-BANK] {json_path} muss ein Array sein!", flush=True)
        return []

    banks = []
    for entry in data:
        try:
            bank = _load_single_bank(entry)
            banks.append(bank)
            print(f"[MEMORY-BANK] Bank '{bank.name}' geladen (enabled={bank.enabled})", flush=True)
        except Exception as e:
            print(f"[MEMORY-BANK] Fehler bei Bank-Config: {e}", flush=True)

    return banks


def _load_single_bank(data: Dict) -> MemoryBankConfig:
    """Lädt eine einzelne Bank-Config aus einem Dict."""
    cfg = MemoryBankConfig()

    # Identität
    cfg.name = (data.get("name") or "").strip()
    if not cfg.name:
        raise ValueError("Bank braucht einen 'name'!")
    cfg.table = data.get("table") or f"bank_{cfg.name}"
    cfg.enabled = data.get("enabled", True)
    cfg.evaluate_after_reply = data.get("evaluate_after_reply", True)

    # Labels
    cfg.label = data.get("label", cfg.label)
    cfg.rating_label = data.get("rating_label", cfg.rating_label)
    cfg.rating_max = int(data.get("rating_max", cfg.rating_max))

    # Kategorien
    if "categories" in data:
        cfg.categories = list(data["categories"])
    if "keywords" in data:
        cfg.keywords = dict(data["keywords"])
    cfg.allow_dynamic = data.get("allow_dynamic", cfg.allow_dynamic)
    cfg.max_categories = int(data.get("max_categories", cfg.max_categories))

    # Baum-Struktur
    cfg.max_depth = int(data.get("max_depth", cfg.max_depth))
    cfg.max_items = int(data.get("max_items", cfg.max_items))

    # Reflexion
    refl = data.get("reflection", {})
    cfg.reflection_enabled = refl.get("enabled", cfg.reflection_enabled)
    cfg.reflection_header = refl.get("header", "")
    cfg.reflection_detected = refl.get("detected_label", "")
    cfg.reflection_best = refl.get("best_label", "")
    cfg.reflection_worst = refl.get("worst_label", "")
    cfg.reflection_recommend = refl.get("recommend_label", "")
    cfg.reflection_max_items = int(refl.get("max_items", cfg.reflection_max_items))
    cfg.reflection_max_chars = int(refl.get("max_chars", cfg.reflection_max_chars))

    # Recall
    recall = data.get("recall", {})
    cfg.recall_enabled = recall.get("enabled", cfg.recall_enabled)
    cfg.recall_header = recall.get("header", "")
    cfg.recall_hauptstrang = recall.get("hauptstrang", "")
    cfg.recall_per_category = int(recall.get("per_category", cfg.recall_per_category))
    cfg.recall_depth = int(recall.get("depth", cfg.recall_depth))
    cfg.recall_children = int(recall.get("children_per_node", cfg.recall_children))
    cfg.recall_max_total = int(recall.get("max_total", cfg.recall_max_total))
    cfg.recall_max_chars = int(recall.get("max_chars", cfg.recall_max_chars))
    cfg.recall_min_diff = int(recall.get("min_diff", cfg.recall_min_diff))
    cfg.recall_sort = recall.get("sort", cfg.recall_sort)
    cfg.recall_order = recall.get("order", cfg.recall_order)

    # Eval
    cfg.eval_system_prompt = data.get("eval_system_prompt", "")

    # Rating Labels
    if "rating_labels" in data:
        cfg.rating_labels = dict(data["rating_labels"])

    # Meta-Variablen
    if "meta_variables" in data:
        cfg.meta_variables = {k: int(v) for k, v in data["meta_variables"].items()}

    # Auto-generate defaults für fehlende Labels
    _fill_defaults(cfg)

    return cfg


def _fill_defaults(cfg: MemoryBankConfig):
    """Füllt fehlende Labels mit sinnvollen Defaults."""
    name_upper = cfg.name.upper()
    label = cfg.label

    if not cfg.reflection_header:
        cfg.reflection_header = f"{name_upper}-REFLEXION (innerlich, NICHT aussprechen):"
    if not cfg.reflection_detected:
        cfg.reflection_detected = f"Erkannte {label}"
    if not cfg.reflection_best:
        cfg.reflection_best = f"Was bei dieser {label} vorher gut funktioniert hat:"
    if not cfg.reflection_worst:
        cfg.reflection_worst = f"Warnung — das lief bei dieser {label} schlecht:"
    if not cfg.reflection_recommend:
        cfg.reflection_recommend = "→ Empfohlene Strategie:"
    if not cfg.recall_header:
        cfg.recall_header = f"{name_upper}-ERINNERUNGEN"
    if not cfg.recall_hauptstrang:
        cfg.recall_hauptstrang = f"★ HAUPT-{name_upper}"

    # Default rating labels
    if not cfg.rating_labels:
        step = cfg.rating_max / 10.0
        default_labels = [
            "nicht spürbar", "kaum merklich", "schwach", "leicht",
            "unter mittel", "mittel", "merklich", "deutlich",
            "stark", "sehr stark", "extrem"
        ]
        for i in range(cfg.rating_max + 1):
            idx = min(int(i / max(step, 0.1)), 10)
            cfg.rating_labels[str(i)] = default_labels[idx]


# ======================= STORE / QUERY =======================

def store_experience(
    category: str,
    rating: int,
    situation: str,
    response: str,
    response_strategy: str = "",
    outcome: int = 50,
    outcome_reason: str = "",
    username: str = "",
    parent_id: int = None,
    depth: int = 0,
    context: Dict = None,
    config: MemoryBankConfig = None
) -> Optional[int]:
    """Speichert eine Erfahrung in einer Bank."""
    _check_init()
    cfg = config or MemoryBankConfig()
    tbl = _safe_table(cfg.table)

    category = (category or "").lower().strip()
    if not category:
        return None

    rating = max(0, min(cfg.rating_max, int(rating)))
    depth = max(0, min(cfg.max_depth, int(depth)))
    outcome = max(0, min(100, int(outcome)))

    ctx_json = json.dumps(context or {}, ensure_ascii=False)
    ts = str(_now_ts())

    _db_exec(
        f"INSERT INTO {tbl} "
        "(parent_id, category, rating, depth, situation, response, response_strategy, "
        " outcome, outcome_reason, context_json, username, created_at, updated_at, use_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
        (parent_id, category, rating, depth,
         (situation or "")[:500], (response or "")[:500],
         (response_strategy or "")[:200],
         outcome, (outcome_reason or "")[:200],
         ctx_json, username, ts, ts)
    )

    rows = _db_query("SELECT last_insert_rowid() AS lid")
    node_id = rows[0]["lid"] if rows else None

    print(
        f"[BANK:{cfg.name}] Gespeichert: {category} R:{rating} → {response_strategy or '?'} "
        f"O:{outcome}% (depth={depth}, id={node_id})",
        flush=True
    )

    _cleanup_category(category, cfg.max_items, cfg)
    return node_id


def _cleanup_category(category: str, max_items: int, cfg: MemoryBankConfig):
    """Entfernt alte Einträge wenn eine Kategorie zu viele hat."""
    tbl = _safe_table(cfg.table)
    rows = _db_query(
        f"SELECT COUNT(*) as cnt FROM {tbl} WHERE category = ? AND depth = 0",
        (category,)
    )
    count = int(rows[0]["cnt"]) if rows else 0
    if count > max_items:
        _db_exec(
            f"DELETE FROM {tbl} WHERE id IN ("
            f"  SELECT id FROM {tbl} WHERE category = ? AND depth = 0 "
            f"  ORDER BY use_count ASC, created_at ASC LIMIT ?"
            f")",
            (category, count - max_items)
        )


def get_best(category: str, limit: int, cfg: MemoryBankConfig) -> List[Dict]:
    """Holt die besten Erfahrungen für eine Kategorie."""
    _check_init()
    tbl = _safe_table(cfg.table)
    rows = _db_query(
        f"SELECT id, category, rating, situation, response, response_strategy, "
        f"  outcome, outcome_reason, depth, use_count "
        f"FROM {tbl} WHERE category = ? AND depth = 0 "
        f"ORDER BY outcome DESC, created_at DESC LIMIT ?",
        (category, limit)
    )
    return [dict(r) for r in rows]


def get_worst(category: str, limit: int, cfg: MemoryBankConfig) -> List[Dict]:
    """Holt die schlechtesten Erfahrungen für eine Kategorie."""
    _check_init()
    tbl = _safe_table(cfg.table)
    rows = _db_query(
        f"SELECT id, category, rating, situation, response, response_strategy, "
        f"  outcome, outcome_reason, depth, use_count "
        f"FROM {tbl} WHERE category = ? AND depth = 0 "
        f"ORDER BY outcome ASC, created_at DESC LIMIT ?",
        (category, limit)
    )
    return [dict(r) for r in rows]


def get_children(node_id: int, cfg: MemoryBankConfig) -> List[Dict]:
    """Holt Kind-Knoten."""
    _check_init()
    tbl = _safe_table(cfg.table)
    rows = _db_query(
        f"SELECT id, category, rating, situation, response, response_strategy, "
        f"  outcome, outcome_reason, depth, use_count "
        f"FROM {tbl} WHERE parent_id = ? "
        f"ORDER BY outcome DESC, created_at DESC",
        (node_id,)
    )
    return [dict(r) for r in rows]


def bump_use(node_id: int, cfg: MemoryBankConfig):
    """Erhöht den use_count."""
    _check_init()
    tbl = _safe_table(cfg.table)
    _db_exec(
        f"UPDATE {tbl} SET use_count = use_count + 1, updated_at = ? WHERE id = ?",
        (str(_now_ts()), node_id)
    )


def get_all_categories(cfg: MemoryBankConfig) -> List[str]:
    """Gibt alle einzigartigen Kategorien aus der DB zurück."""
    _check_init()
    tbl = _safe_table(cfg.table)
    rows = _db_query(f"SELECT DISTINCT category FROM {tbl} ORDER BY category")
    return [r["category"] for r in rows]


def get_stats(cfg: MemoryBankConfig) -> Dict:
    """Statistiken über eine Bank."""
    _check_init()
    tbl = _safe_table(cfg.table)
    rows = _db_query(
        f"SELECT category, COUNT(*) as cnt, "
        f"  COALESCE(AVG(rating), 5) as avg_rating, "
        f"  COALESCE(AVG(outcome), 50) as avg_outcome "
        f"FROM {tbl} WHERE depth = 0 "
        f"GROUP BY category ORDER BY cnt DESC"
    )
    total = _db_query(f"SELECT COUNT(*) as cnt FROM {tbl}")

    return {
        "bank": cfg.name,
        "total_experiences": int(total[0]["cnt"]) if total else 0,
        "by_category": {
            r["category"]: {
                "count": int(r["cnt"]),
                "avg_rating": round(float(r["avg_rating"]), 1),
                "avg_outcome": round(float(r["avg_outcome"]), 1)
            } for r in rows
        },
        "unique_categories": len(rows)
    }


# ======================= DETECTION =======================

def detect_category(text: str, reply_text: str = "",
                    config: MemoryBankConfig = None) -> Tuple[str, int]:
    """
    Erkennt die Kategorie einer Nachricht anhand von Keywords.
    Gibt (category, rating_estimate) zurück.
    rating_estimate ist rating_max/2 (neutral) — wird durch LLM-Eval überschrieben.
    """
    cfg = config or MemoryBankConfig()
    combined = ((text or "") + " " + (reply_text or "")).lower()

    default_cat = cfg.categories[0] if cfg.categories else "unknown"
    best_cat = default_cat
    best_score = 0

    for cat, kws in cfg.keywords.items():
        score = sum(1 for kw in kws if kw in combined)
        if score > best_score:
            best_score = score
            best_cat = cat

    neutral_rating = cfg.rating_max // 2
    return best_cat, neutral_rating


def get_rating_label(rating: int, cfg: MemoryBankConfig) -> str:
    """Gibt ein menschlich lesbares Label für die Bewertungsstufe."""
    key = str(min(rating, cfg.rating_max))
    return cfg.rating_labels.get(key, f"Stufe {rating}")


# ======================= EVALUATION (POST-RESPONSE, ASYNC) =======================

def _default_eval_system_prompt(cfg: MemoryBankConfig) -> str:
    """Generiert einen Default-Eval-Prompt aus der Bank-Config."""
    cats = ", ".join(cfg.categories[:20])
    return (
        f"Du bist ein Analytiker für {cfg.label}. Bewerte diese Interaktion.\n\n"
        f"EXAKTES Format (NUR diese Zeile, KEINE Extras):\n"
        f"category:<{cfg.label.lower()}>|rating:<0-{cfg.rating_max}>|"
        f"strategy:<was_funktioniert_hat>|outcome:<0-100>|reason:<kurzer Grund max 15 Wörter>\n\n"
        f"Regeln:\n"
        f"- category: {cfg.label} der Interaktion (1 Wort, lowercase)\n"
        f"  Bekannte: {cats}\n"
        f"  Du darfst NEUE erfinden wenn keines passt!\n"
        f"- rating: {cfg.rating_label} (0=minimal, {cfg.rating_max}=maximal)\n"
        f"- strategy: Welche Strategie hat die KI genutzt? (kurz, max 5 Wörter)\n"
        f"  Beispiele: mitgefühl_gezeigt, humor_eingesetzt, ruhig_geblieben, nachgefragt, thema_vertieft\n"
        f"- outcome: War die Reaktion GUT FÜR DIE KI? (0=schlecht, 100=perfekt)\n"
        f"  100% = KI hat profitiert, gutes Gespräch, Bindung gestärkt\n"
        f"  0% = Verfehlt, User unzufrieden, schlecht gelaufen\n"
        f"- reason: Warum dieses Outcome (kurz!)\n\n"
        f"NUR das Format, NICHTS anderes."
    )


def build_eval_prompt(
    user_text: str,
    reply: str,
    config: MemoryBankConfig = None
) -> Tuple[str, str]:
    """Baut System + User Prompt für die Post-Response Evaluation."""
    cfg = config or MemoryBankConfig()

    system = cfg.eval_system_prompt or _default_eval_system_prompt(cfg)

    user = (
        f"User schrieb: \"{user_text[:200]}\"\n"
        f"KI antwortete: \"{reply[:200]}\"\n\n"
        f"Bewerte."
    )

    return system, user


def parse_eval_response(raw: str, config: MemoryBankConfig = None) -> Optional[Dict]:
    """Parst die Evaluations-Antwort des LLM."""
    cfg = config or MemoryBankConfig()
    raw = (raw or "").strip()

    result = {}
    for field_name in ["category", "rating", "strategy", "outcome", "reason"]:
        pattern = rf'{field_name}:\s*([^|]+?)(?:\||$)'
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            result[field_name] = match.group(1).strip()

    if "category" not in result:
        return None

    category = result.get("category", "").lower().strip()
    category = re.sub(r'[^a-zäöüß0-9_\-]', '', category)
    if not category:
        category = cfg.categories[0] if cfg.categories else "unknown"

    # Dynamische Kategorien
    try:
        db_cats = get_all_categories(cfg)
    except RuntimeError:
        db_cats = []
    known = list(cfg.categories) + db_cats

    if category not in known:
        if cfg.allow_dynamic:
            if len(set(known)) >= cfg.max_categories:
                category = _fuzzy_match(category, cfg)
        else:
            category = _fuzzy_match(category, cfg)

    try:
        rating = int(result.get("rating", str(cfg.rating_max // 2)))
        rating = max(0, min(cfg.rating_max, rating))
    except (ValueError, TypeError):
        rating = cfg.rating_max // 2

    strategy = (result.get("strategy", "") or "")[:200]
    strategy = strategy.replace(" ", "_").lower()

    try:
        outcome = int(result.get("outcome", "50"))
        outcome = max(0, min(100, outcome))
    except (ValueError, TypeError):
        outcome = 50

    reason = (result.get("reason", "") or "")[:200]

    return {
        "category": category,
        "rating": rating,
        "strategy": strategy,
        "outcome": outcome,
        "reason": reason
    }


def _fuzzy_match(category: str, cfg: MemoryBankConfig) -> str:
    """Versucht eine unbekannte Kategorie auf eine bekannte zu matchen."""
    for bt in cfg.categories:
        if bt in category or category in bt:
            return bt
    return cfg.categories[0] if cfg.categories else "unknown"


def store_evaluation(
    user_text: str,
    reply: str,
    eval_result: Dict,
    username: str = "",
    config: MemoryBankConfig = None
) -> Optional[int]:
    """Speichert die evaluierte Erfahrung."""
    cfg = config or MemoryBankConfig()

    return store_experience(
        category=eval_result["category"],
        rating=eval_result["rating"],
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
                     config: MemoryBankConfig = None) -> Optional[str]:
    """
    Baut eine Reflexion VOR der Antwort.
    Schaut: Welche Kategorie kommt? Was hat vorher funktioniert?
    """
    cfg = config or MemoryBankConfig()

    if not cfg.enabled or not cfg.reflection_enabled:
        return None

    _check_init()

    category, _ = detect_category(user_text, config=cfg)

    best = get_best(category, limit=cfg.reflection_max_items, cfg=cfg)
    worst = get_worst(category, limit=2, cfg=cfg)

    if not best and not worst:
        return None

    for exp in best + worst:
        bump_use(exp["id"], cfg)

    lines = [
        cfg.reflection_header,
        f"{cfg.reflection_detected}: {category}"
    ]

    # Meta-Variablen einblenden (z.B. Persönlichkeit)
    if cfg.meta_variables:
        meta_str = ", ".join(f"{k}:{v}" for k, v in cfg.meta_variables.items())
        lines.append(f"Kontext-Variablen: {meta_str}")

    if best:
        lines.append(cfg.reflection_best)
        for exp in best[:3]:
            r_label = get_rating_label(exp["rating"], cfg)
            strat = exp.get("response_strategy", "") or ""
            line = f"  ✓ Strategie: {strat}" if strat else "  ✓"
            line += f" → {exp['outcome']}% (R:{exp['rating']} {r_label})"
            lines.append(line)

    if worst:
        lines.append(cfg.reflection_worst)
        for exp in worst[:2]:
            r_label = get_rating_label(exp["rating"], cfg)
            strat = exp.get("response_strategy", "") or ""
            line = f"  ⚠ Strategie: {strat}" if strat else "  ⚠"
            line += f" → {exp['outcome']}% (R:{exp['rating']} {r_label})"
            lines.append(line)

    if best and best[0].get("outcome", 0) >= 60:
        best_strat = best[0].get("response_strategy", "")
        if best_strat:
            lines.append(f"{cfg.reflection_recommend} {best_strat}")

    reflection = "\n".join(lines)

    if len(reflection) > cfg.reflection_max_chars:
        reflection = reflection[:cfg.reflection_max_chars - 3] + "..."

    return reflection


# ======================= RECALL =======================

def _format_node_line(node: Dict, cfg: MemoryBankConfig,
                      indent: int = 0, hauptstrang: bool = False) -> str:
    """Formatiert einen Knoten als lesbare Zeile."""
    prefix = "  " * indent
    category = node.get("category", "?")
    rating = node.get("rating", 5)
    r_label = get_rating_label(rating, cfg)
    strategy = (node.get("response_strategy", "") or "")[:40]
    outcome = node.get("outcome", 50)

    line = prefix
    if hauptstrang:
        line += f"{cfg.recall_hauptstrang} [{category} R:{rating}]"
    elif indent == 0:
        line += f"[{category} R:{rating}]"
    else:
        line += f"└→ {category} R:{rating}"

    if strategy:
        line += f" ({strategy})"
    line += f" → {outcome}% ({r_label})"

    return line


def _collect_recall_tree(node_id: int, depth: int, cfg: MemoryBankConfig) -> List[Dict]:
    """Sammelt Kind-Knoten rekursiv."""
    if depth <= 0:
        return []

    children = get_children(node_id, cfg)
    if not children:
        return []

    children = children[:cfg.recall_children]
    result = []
    for child in children:
        result.append(child)
        if depth > 1:
            grandchildren = _collect_recall_tree(child["id"], depth - 1, cfg)
            result.extend(grandchildren)

    return result


def _get_sorted_categories(cfg: MemoryBankConfig) -> List[str]:
    """Gibt die Kategorien in der konfigurierten Reihenfolge zurück."""
    mode = cfg.recall_order
    tbl = _safe_table(cfg.table)

    # Alle Kategorien: config + dynamische aus DB
    all_cats = list(cfg.categories)
    try:
        db_cats = get_all_categories(cfg)
    except RuntimeError:
        db_cats = []
    for c in db_cats:
        if c not in all_cats:
            all_cats.append(c)

    if mode == "fixed":
        return all_cats

    # Statistiken aus DB holen
    stats = []
    for cat in all_cats:
        rows = _db_query(
            f"SELECT COUNT(*) as cnt, "
            f"  COALESCE(AVG(rating), 5) as avg_r, "
            f"  COALESCE(AVG(outcome), 50) as avg_out, "
            f"  COALESCE(MAX(created_at), '1900-01-01') as last_seen "
            f"FROM {tbl} WHERE category = ? AND depth = 0",
            (cat,)
        )
        row = dict(rows[0]) if rows else {"cnt": 0, "avg_r": 5, "avg_out": 50, "last_seen": "1900-01-01"}
        row["category"] = cat
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
        return all_cats

    with_data = [s["category"] for s in stats if int(s.get("cnt", 0)) > 0]
    without_data = [s["category"] for s in stats if int(s.get("cnt", 0)) == 0]

    return with_data + without_data


def build_recall(config: MemoryBankConfig = None) -> Optional[str]:
    """
    Baut einen Erinnerungs-Block fürs Prompt.

    Konfigurierbar über memory_banks.json -> recall:
      per_category:      wie viele letzte Erfahrungen pro Kategorie
      depth:             wie tief in Unterkategorien
      children_per_node: Äste pro Knoten
      max_total:         maximale Gesamtzahl
      sort:              "recent" | "best" | "worst"
      order:             "latest_first" | "frequent_first" | "best_first" | "worst_first" | "fixed"
    """
    cfg = config or MemoryBankConfig()
    tbl = _safe_table(cfg.table)

    if not cfg.enabled or not cfg.recall_enabled:
        return None

    _check_init()

    sorted_cats = _get_sorted_categories(cfg)

    lines = []
    total_count = 0
    is_first = True

    for cat in sorted_cats:
        if total_count >= cfg.recall_max_total:
            break

        if cfg.recall_sort == "best":
            order = "outcome DESC, created_at DESC"
        elif cfg.recall_sort == "worst":
            order = "outcome ASC, created_at DESC"
        else:
            order = "created_at DESC"

        # Nur signifikante Erfahrungen (nicht mittelmäßig)
        neutral = cfg.rating_max // 2
        low_bound = neutral - cfg.recall_min_diff
        high_bound = neutral + cfg.recall_min_diff

        rows = _db_query(
            f"SELECT id, category, rating, situation, response, response_strategy, "
            f"  outcome, outcome_reason, use_count, depth "
            f"FROM {tbl} "
            f"WHERE category = ? AND depth = 0 "
            f"  AND (rating <= ? OR rating >= ?) "
            f"ORDER BY {order} LIMIT ?",
            (cat, low_bound, high_bound, cfg.recall_per_category)
        )

        if not rows:
            continue

        for row in rows:
            if total_count >= cfg.recall_max_total:
                break

            node = dict(row)

            if is_first and cfg.recall_order != "fixed":
                lines.append(_format_node_line(node, cfg, indent=0, hauptstrang=True))
                is_first = False
            else:
                lines.append(_format_node_line(node, cfg, indent=0))
                is_first = False

            total_count += 1
            bump_use(node["id"], cfg)

            if cfg.recall_depth > 0:
                subtree = _collect_recall_tree(node["id"], cfg.recall_depth, cfg)
                for child in subtree:
                    if total_count >= cfg.recall_max_total:
                        break
                    child_depth = child.get("depth", 1)
                    indent = min(child_depth, cfg.recall_depth)
                    lines.append(_format_node_line(child, cfg, indent=indent))
                    total_count += 1
                    bump_use(child["id"], cfg)

    if not lines:
        return None

    order_info = {
        "latest_first": "aktuellste zuerst",
        "frequent_first": "häufigste zuerst",
        "best_first": "erfolgreichste zuerst",
        "worst_first": "schwächste zuerst",
        "fixed": "feste Reihenfolge"
    }
    mode_label = order_info.get(cfg.recall_order, "")
    header = f"{cfg.recall_header} ({mode_label}):"
    recall_text = header + "\n" + "\n".join(lines)

    if len(recall_text) > cfg.recall_max_chars:
        recall_text = recall_text[:cfg.recall_max_chars - 3] + "..."

    return recall_text


# ======================= CLEANUP =======================

def cleanup_old(max_age_days: int = 90, cfg: MemoryBankConfig = None):
    """Entfernt sehr alte, selten genutzte Erfahrungen."""
    _check_init()
    if not cfg:
        return
    tbl = _safe_table(cfg.table)
    _db_exec(
        f"DELETE FROM {tbl} WHERE use_count = 0 "
        f"AND created_at < datetime('now', ?)",
        (f"-{max_age_days} days",)
    )


# ======================= DEBUG / EXPORT =======================

def dump_summary(cfg: MemoryBankConfig) -> Dict:
    """Gibt eine Übersicht aller Erfahrungen einer Bank zurück."""
    _check_init()
    tbl = _safe_table(cfg.table)

    result = {}
    categories = get_all_categories(cfg)

    for cat in categories:
        rows = _db_query(
            f"SELECT id, category, rating, situation, response, response_strategy, "
            f"  outcome, outcome_reason, depth, use_count, username, created_at "
            f"FROM {tbl} WHERE category = ? AND depth = 0 "
            f"ORDER BY created_at DESC LIMIT 20",
            (cat,)
        )

        cat_data = []
        for r in rows:
            node = dict(r)
            children = get_children(node["id"], cfg)
            node["children"] = [dict(c) for c in children[:5]]
            cat_data.append(node)

        result[cat] = cat_data

    return {
        "bank": cfg.name,
        "table": cfg.table,
        "categories": result,
        "stats": get_stats(cfg),
        "config": {
            "enabled": cfg.enabled,
            "label": cfg.label,
            "rating_label": cfg.rating_label,
            "rating_max": cfg.rating_max,
            "categories": cfg.categories,
            "dynamic_categories": [c for c in categories if c not in cfg.categories],
            "max_depth": cfg.max_depth,
            "recall_enabled": cfg.recall_enabled,
            "recall_order": cfg.recall_order,
            "meta_variables": cfg.meta_variables
        }
    }


def get_recent(limit: int, cfg: MemoryBankConfig) -> List[Dict]:
    """Gibt die letzten N Erfahrungen einer Bank zurück."""
    _check_init()
    tbl = _safe_table(cfg.table)
    rows = _db_query(
        f"SELECT id, category, rating, situation, response, response_strategy, "
        f"  outcome, outcome_reason, depth, username, created_at "
        f"FROM {tbl} ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    return [dict(r) for r in rows]
