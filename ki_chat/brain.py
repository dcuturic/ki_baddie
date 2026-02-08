#!/usr/bin/env python3
import sqlite3
import json
import sys
import time
from typing import Any, Dict, List, Tuple

DEFAULT_TABLES = ["chatlog", "memories", "profiles", "thoughts"]

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=5000;")
    return con

def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None

def get_table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [r["name"] for r in rows]

def get_create_sql(con: sqlite3.Connection, table: str) -> str:
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone()
    return (row["sql"] if row and row["sql"] else "").strip()

def export_db(db_path: str, out_json_path: str, tables: List[str] = None) -> None:
    tables = tables or DEFAULT_TABLES
    con = connect(db_path)

    exported: Dict[str, Any] = {
        "meta": {
            "exported_at": int(time.time()),
            "db_path": db_path,
            "format_version": 1
        },
        "tables": {}
    }

    for t in tables:
        if not table_exists(con, t):
            continue

        cols = get_table_columns(con, t)
        create_sql = get_create_sql(con, t)

        rows = con.execute(f"SELECT * FROM {t}").fetchall()
        data = []
        for r in rows:
            # store as dict preserving column names
            item = {}
            for c in cols:
                item[c] = r[c]
            data.append(item)

        exported["tables"][t] = {
            "create_sql": create_sql,
            "columns": cols,
            "rows": data
        }

    con.close()

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)

    print(f"[OK] Exported to: {out_json_path}")
    print(f"[OK] Tables: {', '.join(exported['tables'].keys())}")

def wipe_tables(con: sqlite3.Connection, tables: List[str]) -> None:
    for t in tables:
        if table_exists(con, t):
            con.execute(f"DELETE FROM {t}")
    con.commit()

def create_table_if_missing(con: sqlite3.Connection, table: str, create_sql: str) -> None:
    if table_exists(con, table):
        return
    if not create_sql:
        raise RuntimeError(f"Missing create_sql for table '{table}' (cannot create).")
    con.execute(create_sql)
    con.commit()

def insert_rows(con: sqlite3.Connection, table: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    if not rows:
        return

    # only insert columns that actually exist (for safety if schema changed)
    existing_cols = set(get_table_columns(con, table))
    cols = [c for c in columns if c in existing_cols]

    if not cols:
        return

    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"

    values = []
    for r in rows:
        values.append(tuple(r.get(c) for c in cols))

    con.executemany(sql, values)
    con.commit()

def import_db(db_path: str, in_json_path: str, wipe: bool = False) -> None:
    with open(in_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tables_obj = payload.get("tables") or {}
    if not isinstance(tables_obj, dict):
        raise RuntimeError("Invalid JSON: payload['tables'] must be an object.")

    con = connect(db_path)

    # create missing tables first
    for t, tdata in tables_obj.items():
        create_sql = (tdata.get("create_sql") or "").strip()
        create_table_if_missing(con, t, create_sql)

    if wipe:
        wipe_tables(con, list(tables_obj.keys()))

    # insert data
    for t, tdata in tables_obj.items():
        rows = tdata.get("rows") or []
        cols = tdata.get("columns") or []
        if not isinstance(rows, list) or not isinstance(cols, list):
            continue
        insert_rows(con, t, rows, cols)

    con.close()
    print(f"[OK] Imported from: {in_json_path}")
    print(f"[OK] Tables: {', '.join(tables_obj.keys())}")
    if wipe:
        print("[OK] Wipe was enabled (tables cleared before import).")

def usage() -> None:
    print(
        "Usage:\n"
        "  python db_json_tool.py export <db_path> <out.json>\n"
        "  python db_json_tool.py import <db_path> <in.json> [--wipe]\n"
        "\nExamples:\n"
        "  python db_json_tool.py export memory.db dump.json\n"
        "  python db_json_tool.py import memory.db dump.json --wipe\n"
    )

def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "export":
        if len(sys.argv) != 4:
            usage()
            sys.exit(1)
        db_path = sys.argv[2]
        out_path = sys.argv[3]
        export_db(db_path, out_path)
        return

    if cmd == "import":
        if len(sys.argv) < 4:
            usage()
            sys.exit(1)
        db_path = sys.argv[2]
        in_path = sys.argv[3]
        wipe = ("--wipe" in sys.argv[4:])
        import_db(db_path, in_path, wipe=wipe)
        return

    usage()
    sys.exit(1)

if __name__ == "__main__":
    main()
