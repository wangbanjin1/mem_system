"""SQLite Vector + FTS5 + Structured Slot secondary index with RLock thread-safety and strict namespace scoping."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.models import MemoryNode
from ..core.namespace import Namespace
from .phrase_builder import (
    _CN_TO_CANONICAL_APP,
    _CN_TO_CANONICAL_BIZ,
    _NOISY_CONTENT_KEYS,
    _TOKEN_RE,
    _cosine_similarity,
    _extract_ymd,
    _time_to_minutes,
    build_pure_phrase_text,
    build_search_text,
    time_matches_window,
    tokenize_for_fts,
)
from .schema_ddl import (
    extract_structured_columns,
    init_sqlite_schema,
    score_slot_row_match,
)

__all__ = [
    "MemoryIndex",
    "SQLiteIndex",
    "_CN_TO_CANONICAL_APP",
    "_CN_TO_CANONICAL_BIZ",
    "_NOISY_CONTENT_KEYS",
    "_TOKEN_RE",
    "_cosine_similarity",
    "_extract_ymd",
    "_time_to_minutes",
    "build_pure_phrase_text",
    "build_search_text",
    "time_matches_window",
    "tokenize_for_fts",
]


class MemoryIndex:
    """SQLite Vector + FTS5 + Structured Slot secondary index with RLock thread-safety and strict namespace scoping."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            init_sqlite_schema(self._conn)

    @staticmethod
    def _build_vector_row_values(
        ns: Namespace,
        node: MemoryNode,
        slot_cols: Dict[str, str],
        pure_phrase: str,
        tokenized_desc: str,
        tokenized_search: str,
        emb_json: str,
    ) -> Tuple[Any, ...]:
        return (
            ns.owner_type,
            ns.user_id,
            ns.agent_id,
            node.memory_type,
            node.subtype,
            node.memory_key,
            node.status,
            node.validity.type,
            node.confidence,
            node.stats.heat,
            node.stats.last_heat_at,
            node.stats.last_reinforced_at,
            node.updated_at,
            slot_cols["app_col"],
            slot_cols["biz_col"],
            slot_cols["scene_col"],
            slot_cols["time_window_col"],
            slot_cols["alias_col"],
            slot_cols["target_col"],
            pure_phrase,
            tokenized_desc,
            tokenized_search,
            emb_json,
        )

    def upsert(self, ns: Namespace, node: MemoryNode, embedding: Optional[Sequence[float]] = None) -> None:
        """Insert or update a MemoryNode and its FTS5 + structured slot projection in SQLite."""
        raw_search = build_search_text(node)
        pure_phrase = build_pure_phrase_text(node)
        tokenized_desc = tokenize_for_fts(node.description)
        tokenized_search = tokenize_for_fts(raw_search)
        emb_json = json.dumps(list(embedding) if embedding is not None else [])
        slot_cols = extract_structured_columns(node)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT rowid, description, search_text, embedding_json FROM vector WHERE memory_id = ?",
                (node.memory_id,),
            )
            existing = cur.fetchone()
            if existing is not None:
                old_rowid = existing["rowid"]
                cur.execute(
                    "INSERT INTO fts(fts, rowid, description, search_text) VALUES('delete', ?, ?, ?)",
                    (old_rowid, existing["description"], existing["search_text"]),
                )
                if embedding is None:
                    emb_json = existing["embedding_json"]
                row_vals = self._build_vector_row_values(
                    ns, node, slot_cols, pure_phrase, tokenized_desc, tokenized_search, emb_json
                )
                cur.execute(
                    """
                    UPDATE vector SET
                        owner_type = ?, user_id = ?, agent_id = ?, memory_type = ?, subtype = ?,
                        memory_key = ?, status = ?, validity = ?, confidence = ?, heat = ?,
                        last_heat_at = ?, last_reinforced_at = ?, updated_at = ?, app_col = ?,
                        biz_col = ?, scene_col = ?, time_window_col = ?, alias_col = ?,
                        target_col = ?, phrase_text = ?, description = ?, search_text = ?, embedding_json = ?
                    WHERE rowid = ?
                    """,
                    (*row_vals, old_rowid),
                )
                rowid = old_rowid
            else:
                row_vals = self._build_vector_row_values(
                    ns, node, slot_cols, pure_phrase, tokenized_desc, tokenized_search, emb_json
                )
                cur.execute(
                    """
                    INSERT INTO vector (
                        memory_id, owner_type, user_id, agent_id, memory_type, subtype,
                        memory_key, status, validity, confidence, heat, last_heat_at,
                        last_reinforced_at, updated_at, app_col, biz_col, scene_col,
                        time_window_col, alias_col, target_col, phrase_text, description, search_text, embedding_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (node.memory_id, *row_vals),
                )
                rowid = cur.lastrowid

            cur.execute(
                "INSERT INTO fts(rowid, description, search_text) VALUES(?, ?, ?)",
                (rowid, tokenized_desc, tokenized_search),
            )
            self._conn.commit()

    def delete(self, memory_id: str) -> bool:
        """Delete from vector and sync FTS5 external content table with 'delete' command."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT rowid, description, search_text FROM vector WHERE memory_id = ?",
                (memory_id,),
            )
            row = cur.fetchone()
            if row is None:
                return False
            cur.execute(
                "INSERT INTO fts(fts, rowid, description, search_text) VALUES('delete', ?, ?, ?)",
                (row["rowid"], row["description"], row["search_text"]),
            )
            cur.execute("DELETE FROM vector WHERE rowid = ?", (row["rowid"],))
            self._conn.commit()
            return True

    def clear_all(self) -> None:
        """Drop and recreate vector and FTS5 tables."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DROP TABLE IF EXISTS fts")
            cur.execute("DROP TABLE IF EXISTS vector")
            self._conn.commit()
            self._init_schema()

    def _scope_where_clause(
        self,
        ns: Namespace,
        subtypes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        table_alias: str = "v",
    ) -> Tuple[str, List[Any]]:
        """Per §1.3: Every SQL query MUST filter by owner_type + user_id (+ agent_id)."""
        prefix = f"{table_alias}." if table_alias else ""
        clauses = [
            f"{prefix}owner_type = ?",
            f"{prefix}user_id = ?",
            f"{prefix}agent_id = ?",
        ]
        params: List[Any] = [ns.owner_type, ns.user_id, ns.agent_id]

        if subtypes:
            placeholders = ",".join("?" for _ in subtypes)
            clauses.append(f"{prefix}subtype IN ({placeholders})")
            params.extend(subtypes)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            clauses.append(f"{prefix}memory_type IN ({placeholders})")
            params.extend(memory_types)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"{prefix}status IN ({placeholders})")
            params.extend(statuses)
        else:
            clauses.append(f"{prefix}status NOT IN ('archived', 'expired', 'superseded')")

        return " AND ".join(clauses), params

    def search_vector(
        self,
        ns: Namespace,
        query_embedding: Sequence[float],
        subtypes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """Search single query vector against scoped embeddings."""
        if not query_embedding:
            return []
        where_sql, params = self._scope_where_clause(
            ns, subtypes=subtypes, memory_types=memory_types, statuses=statuses, table_alias=""
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT memory_id, embedding_json FROM vector WHERE {where_sql}",
                params,
            )
            rows = cur.fetchall()

        scored: List[Tuple[str, float]] = []
        for r in rows:
            try:
                emb = json.loads(r["embedding_json"] or "[]")
            except (ValueError, TypeError):
                emb = []
            if emb:
                sim = max(0.0, _cosine_similarity(query_embedding, emb))
                scored.append((r["memory_id"], sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_vector_multi_windows(
        self,
        ns: Namespace,
        window_embeddings: Sequence[Sequence[float]],
        subtypes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """Compute cross-window Max-Pooling cosine similarity against stored pure phrase embeddings."""
        valid_embs = [w for w in window_embeddings if w]
        if not valid_embs:
            return []
        where_sql, params = self._scope_where_clause(
            ns, subtypes=subtypes, memory_types=memory_types, statuses=statuses, table_alias=""
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT memory_id, embedding_json FROM vector WHERE {where_sql}",
                params,
            )
            rows = cur.fetchall()

        scored: List[Tuple[str, float]] = []
        base_emb = valid_embs[0]
        sub_embs = valid_embs[1:]
        for r in rows:
            try:
                emb = json.loads(r["embedding_json"] or "[]")
            except (ValueError, TypeError):
                emb = []
            if emb:
                base_sim = max(0.0, _cosine_similarity(base_emb, emb))
                win_max = max((_cosine_similarity(w_emb, emb) for w_emb in sub_embs), default=0.0)
                eff_sim = max(base_sim, win_max) if win_max >= 0.48 else base_sim
                if eff_sim > 0.0:
                    scored.append((r["memory_id"], eff_sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_fts(
        self,
        ns: Namespace,
        query: str,
        subtypes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """Run sparse FTS5 keyword recall scoped by namespace."""
        tokenized = tokenize_for_fts(query)
        terms = [t for t in tokenized.split() if t]
        if not terms:
            return []
        unique_terms = list(dict.fromkeys(terms))
        fts_expr = " OR ".join(f'"{t}"' for t in unique_terms[:32])
        where_sql, params = self._scope_where_clause(
            ns, subtypes=subtypes, memory_types=memory_types, statuses=statuses, table_alias="v"
        )

        with self._lock:
            cur = self._conn.cursor()
            sql = f"""
                SELECT v.memory_id, v.description, v.search_text
                FROM fts
                JOIN vector v ON fts.rowid = v.rowid
                WHERE fts MATCH ? AND {where_sql}
            """
            cur.execute(sql, [fts_expr, *params])
            rows = cur.fetchall()

        scored: List[Tuple[str, float]] = []
        for r in rows:
            doc_tokens = set((r["description"] + " " + r["search_text"]).split())
            matched = sum(1 for t in unique_terms if t in doc_tokens)
            overlap_sim = min(1.0, matched / max(1, len(unique_terms)))
            if overlap_sim > 0:
                scored.append((r["memory_id"], overlap_sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def list_scoped_ids(
        self,
        ns: Namespace,
        subtypes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Return all memory IDs matching namespace and status filters."""
        where_sql, params = self._scope_where_clause(
            ns, subtypes=subtypes, memory_types=memory_types, statuses=statuses, table_alias=""
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(f"SELECT memory_id FROM vector WHERE {where_sql}", params)
            return [r["memory_id"] for r in cur.fetchall()]

    def list_active_slot_rows(
        self,
        ns: Namespace,
        subtypes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch active structured slot columns directly from SQLite `vector` table with zero disk I/O."""
        where_sql, params = self._scope_where_clause(
            ns, subtypes=subtypes, memory_types=memory_types, statuses=statuses, table_alias=""
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT memory_id, memory_type, subtype, memory_key, status,
                       app_col, biz_col, scene_col, time_window_col, alias_col, target_col, phrase_text, embedding_json
                FROM vector
                WHERE {where_sql}
                """,
                params,
            )
            return [dict(r) for r in cur.fetchall()]

    def select_by_slots(
        self,
        ns: Namespace,
        applications: Optional[Sequence[str]] = None,
        businesses: Optional[Sequence[str]] = None,
        scenes: Optional[Sequence[str]] = None,
        time_windows: Optional[Sequence[str]] = None,
        relative_date_range: Optional[Tuple[str, str]] = None,
        subtypes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, str]]:
        """Execute deterministic structured SQL SELECT on vector table slot columns (§2.5)."""
        where_sql, params = self._scope_where_clause(
            ns, subtypes=subtypes, memory_types=memory_types, statuses=statuses, table_alias=""
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT memory_id, subtype, memory_key, app_col, biz_col, scene_col, time_window_col, alias_col, target_col
                FROM vector
                WHERE {where_sql}
                """,
                params,
            )
            rows = cur.fetchall()

        app_set = {a.strip() for a in (applications or []) if a and a.strip()}
        biz_set = {b.strip() for b in (businesses or []) if b and b.strip()}
        scene_set = {s.strip() for s in (scenes or []) if s and s.strip()}
        tw_list = [t.strip() for t in (time_windows or []) if t and t.strip()]
        has_date_range = bool(
            relative_date_range
            and len(relative_date_range) == 2
            and relative_date_range[0]
            and relative_date_range[1]
        )

        if not (app_set or biz_set or scene_set or tw_list or has_date_range):
            return []

        results: List[Tuple[str, float, str]] = []
        for r in rows:
            hit = score_slot_row_match(r, app_set, biz_set, scene_set, tw_list, relative_date_range)
            if hit is not None:
                results.append(hit)

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()


SQLiteIndex = MemoryIndex
