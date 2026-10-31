"""SQLite schema DDL initialization, column migration, and structured slot row scoring."""
from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..core.models import MemoryNode
from .phrase_builder import (
    _CN_TO_CANONICAL_APP,
    _CN_TO_CANONICAL_BIZ,
    _extract_ymd,
    time_matches_window,
)

VECTOR_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS vector (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT UNIQUE NOT NULL,
    owner_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT '',
    memory_type TEXT NOT NULL,
    subtype TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    status TEXT NOT NULL,
    validity TEXT NOT NULL,
    confidence REAL NOT NULL,
    heat REAL NOT NULL DEFAULT 0.0,
    last_heat_at TEXT NOT NULL DEFAULT '',
    last_reinforced_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    app_col TEXT NOT NULL DEFAULT '',
    biz_col TEXT NOT NULL DEFAULT '',
    scene_col TEXT NOT NULL DEFAULT '',
    time_window_col TEXT NOT NULL DEFAULT '',
    alias_col TEXT NOT NULL DEFAULT '',
    target_col TEXT NOT NULL DEFAULT '',
    phrase_text TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    search_text TEXT NOT NULL DEFAULT '',
    embedding_json TEXT NOT NULL DEFAULT '[]'
)
"""

MIGRATION_COLUMNS: List[Tuple[str, str]] = [
    ("last_heat_at", "TEXT NOT NULL DEFAULT ''"),
    ("heat", "REAL NOT NULL DEFAULT 0.0"),
    ("app_col", "TEXT NOT NULL DEFAULT ''"),
    ("biz_col", "TEXT NOT NULL DEFAULT ''"),
    ("scene_col", "TEXT NOT NULL DEFAULT ''"),
    ("time_window_col", "TEXT NOT NULL DEFAULT ''"),
    ("alias_col", "TEXT NOT NULL DEFAULT ''"),
    ("target_col", "TEXT NOT NULL DEFAULT ''"),
    ("phrase_text", "TEXT NOT NULL DEFAULT ''"),
]

FTS_TABLE_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    description,
    search_text,
    content='vector',
    content_rowid='rowid'
)
"""


def init_sqlite_schema(conn: sqlite3.Connection) -> None:
    """Initialize SQLite vector and FTS5 tables and run additive column migrations."""
    cur = conn.cursor()
    cur.execute(VECTOR_TABLE_DDL)

    cur.execute("PRAGMA table_info(vector)")
    cols = {row["name"] for row in cur.fetchall()}
    for col_name, col_def in MIGRATION_COLUMNS:
        if col_name not in cols:
            cur.execute(f"ALTER TABLE vector ADD COLUMN {col_name} {col_def}")

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vector_scope
        ON vector(owner_type, user_id, agent_id, subtype)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vector_structured_slots
        ON vector(owner_type, user_id, app_col, biz_col, time_window_col)
        """
    )
    cur.execute(FTS_TABLE_DDL)
    conn.commit()


def extract_structured_columns(node: MemoryNode) -> Dict[str, str]:
    """Extract canonical structured slot columns for SQLite `vector` row projection."""
    raw_app = str(node.content.get("application") or node.content.get("application_name") or "").strip()
    app_col = _CN_TO_CANONICAL_APP.get(raw_app, raw_app)
    raw_biz = str(node.content.get("business") or node.content.get("service_name") or "").strip()
    biz_col = _CN_TO_CANONICAL_BIZ.get(raw_biz, raw_biz)
    scene_col = str(
        node.content.get("scene") or node.content.get("environment") or node.content.get("location") or ""
    ).strip()
    time_window_col = str(node.content.get("time_window") or node.content.get("condition") or "").strip()
    alias_col = str(node.content.get("alias") or node.content.get("source_expression") or "").strip()
    target_col = str(
        node.content.get("target") or node.content.get("value") or node.content.get("action") or ""
    ).strip()

    if node.subtype == "working_record" and not time_window_col:
        time_window_col = _extract_ymd(
            str(node.content.get("created_at") or node.content.get("start_timestamp") or node.created_at or "")
        )

    # Orthogonal separation: if an expression row maps an alias to a time_window (e.g. 午间例行 -> 12:30-13:30),
    # keep app_col and biz_col empty on the expression row so app/biz are resolved via periodic_preference.
    if node.subtype == "expression" and re.search(r"\d{1,2}:\d{2}\s*[-~至到]\s*\d{1,2}:\d{2}", target_col):
        app_col = ""
        biz_col = ""

    return {
        "app_col": app_col,
        "biz_col": biz_col,
        "scene_col": scene_col,
        "time_window_col": time_window_col,
        "alias_col": alias_col,
        "target_col": target_col,
    }


def score_slot_row_match(
    row: sqlite3.Row | Dict[str, Any],
    app_set: Set[str],
    biz_set: Set[str],
    scene_set: Set[str],
    tw_list: Sequence[str],
    relative_date_range: Optional[Tuple[str, str]],
) -> Optional[Tuple[str, float, str]]:
    """Score a single SQLite vector row against structured slot criteria (Orthogonal Scheme A)."""
    mid = row["memory_id"]
    st = row["subtype"]
    r_app = row["app_col"]
    r_biz = row["biz_col"]
    r_scene = row["scene_col"]
    r_tw = row["time_window_col"]

    has_date_range = bool(
        relative_date_range
        and len(relative_date_range) == 2
        and relative_date_range[0]
        and relative_date_range[1]
    )

    app_hit = bool(app_set and r_app and r_app in app_set)
    biz_hit = bool(biz_set and r_biz and r_biz in biz_set)
    scene_hit = bool(scene_set and r_scene and any(sc in r_scene or r_scene in sc for sc in scene_set))
    tw_hit = bool(tw_list and r_tw and any(time_matches_window(qt, r_tw) for qt in tw_list))
    date_hit = bool(
        st == "working_record"
        and has_date_range
        and r_tw
        and len(r_tw) == 10
        and relative_date_range is not None
        and relative_date_range[0] <= r_tw <= relative_date_range[1]
    )

    matched_fields: List[str] = []
    if app_hit:
        matched_fields.append(f"application={r_app}")
    if biz_hit:
        matched_fields.append(f"business={r_biz}")
    if scene_hit:
        matched_fields.append(f"scene={r_scene}")
    if tw_hit:
        matched_fields.append(f"time_window={r_tw}")
    if date_hit and relative_date_range is not None:
        matched_fields.append(f"created_date={r_tw}∈[{relative_date_range[0]}..{relative_date_range[1]}]")

    if not matched_fields:
        return None

    score = 0.0
    if st in ("scene_preference", "environment"):
        if app_hit and biz_hit and scene_hit:
            score = 0.98
        elif (app_hit or biz_hit) and scene_hit:
            score = 0.94
        elif app_hit and biz_hit:
            score = 0.96 if not scene_set else 0.90
        elif scene_hit:
            score = 0.88
        elif app_hit or biz_hit:
            score = 0.85
    elif st in ("preferred_value", "application_business", "experience_parameter"):
        if app_hit and biz_hit:
            score = 0.98 if not scene_set else 0.93
        elif app_hit or biz_hit:
            score = 0.85
    elif st == "working_record":
        if date_hit and (scene_hit or (app_hit and biz_hit)):
            score = 0.96
        elif scene_hit and (app_hit or biz_hit):
            score = 0.94
        elif date_hit:
            score = 0.91
        elif scene_hit:
            score = 0.89
        elif app_hit and biz_hit:
            score = 0.86
        elif app_hit or biz_hit:
            score = 0.82
    elif st == "periodic_preference":
        if app_hit and biz_hit and tw_hit:
            score = 0.96
        elif (app_hit or biz_hit) and tw_hit:
            score = 0.91
        elif app_hit and biz_hit:
            score = 0.90
        elif tw_hit:
            score = 0.86
        elif app_hit or biz_hit:
            score = 0.82
    elif st == "service_to_app":
        score = 0.84 if (app_hit and biz_hit) else 0.78
    else:
        score = 0.80

    if score <= 0:
        return None
    return (mid, score, "SELECT WHERE " + " AND ".join(matched_fields))
