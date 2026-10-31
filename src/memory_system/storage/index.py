from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.models import MemoryNode
from ..core.namespace import Namespace


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    try:
        import numpy as np  # type: ignore
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except ImportError:
        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        na = math.sqrt(sum(x * x for x in vec_a))
        nb = math.sqrt(sum(y * y for y in vec_b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return dot / (na * nb)


_NOISY_CONTENT_KEYS = {
    "validity",
    "declaration_mode",
    "category",
    "confidence",
    "owner_type",
    "user_id",
    "tenant_id",
    "source_session_id",
}


def build_pure_phrase_text(node: MemoryNode) -> str:
    """Extract pure 4~12 character right-hand semantic trigger phrase (phrase_text) for high-discriminability embedding."""
    c = node.content or {}
    st = node.subtype

    if st == "expression":
        return str(c.get("phrase_text") or c.get("alias") or c.get("source_expression") or node.memory_key or "").strip()
    if st in ("scene_preference", "environment"):
        return str(c.get("phrase_text") or c.get("scene") or c.get("environment") or "").strip()
    if st == "periodic_preference":
        return str(c.get("phrase_text") or c.get("window_label") or c.get("time_window") or c.get("condition") or "").strip()
    if st == "service_to_app":
        return str(c.get("phrase_text") or c.get("service_cn") or c.get("business") or "").strip()
    if st in ("preferred_value", "application_business", "experience_parameter"):
        if c.get("phrase_text"):
            return str(c["phrase_text"]).strip()
        app_cn = str(c.get("app_cn") or c.get("application") or "").strip()
        svc_cn = str(c.get("service_cn") or c.get("business") or "").strip()
        return f"{app_cn}{svc_cn}".strip()
    if st == "working_record":
        if c.get("phrase_text"):
            return str(c["phrase_text"]).strip()
        loc = str(c.get("location") or c.get("scene") or "").strip()
        svc = str(c.get("service_name") or c.get("business") or "").strip()
        return f"{loc} {svc}".strip() if loc else svc
    if st == "experience":
        if c.get("phrase_text"):
            return str(c["phrase_text"]).strip()
        trig = c.get("trigger_conditions")
        if isinstance(trig, dict):
            kws: List[str] = []
            for tk in ("keywords", "task_keywords", "location_keywords"):
                sub = trig.get(tk)
                if isinstance(sub, list):
                    for item in sub:
                        s_item = str(item).strip()
                        if s_item and s_item not in kws:
                            kws.append(s_item)
            if kws:
                return " ".join(kws[:4])
        return str(c.get("title") or c.get("condition") or node.description or "").strip()
    return str(c.get("phrase_text") or node.description or node.memory_key or "").strip()


def build_search_text(node: MemoryNode) -> str:
    """Build semantic text for FTS5 from node fields and content (zero hardcoded synonym lists)."""
    c = node.content or {}
    st = node.subtype
    pure_phrase = build_pure_phrase_text(node)

    semantic_values: List[str] = []
    if pure_phrase:
        semantic_values.append(pure_phrase)

    if st == "expression":
        alias = str(c.get("alias") or c.get("source_expression") or node.memory_key or "").strip()
        target = str(c.get("target") or "").strip()
        note = str(c.get("note") or "").strip()
        for v in (alias, target, note):
            if v and v not in semantic_values:
                semantic_values.append(v)
    elif st in ("scene_preference", "environment"):
        sc = str(c.get("scene") or c.get("environment") or "").strip()
        app = str(c.get("application") or "").strip()
        biz = str(c.get("business") or "").strip()
        for v in (sc, app, biz):
            if v and v not in semantic_values:
                semantic_values.append(v)
        for k in ("resolution", "max_delay", "latency", "bit_rate", "priority", "note"):
            if c.get(k):
                semantic_values.append(str(c[k]))
    elif st == "working_record":
        loc = str(c.get("location") or c.get("scene") or "").strip()
        app_n = str(c.get("application_name") or c.get("application") or "").strip()
        svc_n = str(c.get("service_name") or c.get("business") or "").strip()
        for v in (loc, app_n, svc_n):
            if v and v not in semantic_values:
                semantic_values.append(v)
        for k in ("start_timestamp", "end_timestamp", "duration", "resolution", "rtt_max", "latency", "summary", "note"):
            if c.get(k):
                semantic_values.append(str(c[k]))
        t_aliases = c.get("temporal_reference_aliases")
        if isinstance(t_aliases, dict):
            for av in t_aliases.values():
                if av:
                    semantic_values.append(str(av))
    elif st == "experience":
        for k in ("title", "condition", "action", "description", "content", "strategy_type", "domain", "rule_type", "causal_rationale"):
            if c.get(k):
                semantic_values.append(str(c[k]))
        trig = c.get("trigger_conditions")
        if isinstance(trig, dict):
            for tk in ("keywords", "location_keywords", "task_keywords"):
                kws = trig.get(tk)
                if isinstance(kws, list):
                    semantic_values.extend(str(x) for x in kws if x)
    else:
        for k, v in c.items():
            if k in _NOISY_CONTENT_KEYS or v is None:
                continue
            if isinstance(v, (str, int, float)):
                semantic_values.append(str(v))
            elif isinstance(v, (list, dict)):
                semantic_values.append(json.dumps(v, ensure_ascii=False))

    parts: List[str] = [
        node.memory_key,
        node.description,
        node.overview,
        node.details,
        " ".join(semantic_values),
    ]
    return " ".join(p for p in parts if p)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,4}")


def tokenize_for_fts(text: str) -> str:
    """Segment Chinese characters and alphanumeric tokens with spaces so SQLite unicode61 FTS5 matches."""
    if not text:
        return ""
    tokens: List[str] = []
    # Add individual CJK characters, CJK bigrams, and alphanumeric words
    for m in re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text):
        chunk = m.group(0)
        if re.match(r"^[A-Za-z0-9_]+$", chunk):
            tokens.append(chunk.lower())
        else:
            for ch in chunk:
                tokens.append(ch)
            for i in range(len(chunk) - 1):
                tokens.append(chunk[i : i + 2])
    return " ".join(tokens)


def _time_to_minutes(hhmm: str) -> Optional[int]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", hhmm.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def time_matches_window(queried_time: str, stored_window: str) -> bool:
    """Check if queried_time ('12:30-13:30' or single time point '12:45') matches stored_window ('12:30-13:30')."""
    if not queried_time or not stored_window:
        return False
    qt = queried_time.strip()
    sw = stored_window.strip()
    if qt == sw:
        return True
    if "-" in sw:
        s_parts = sw.split("-", 1)
        s_start = _time_to_minutes(s_parts[0])
        s_end = _time_to_minutes(s_parts[1])
        if s_start is None or s_end is None:
            return False
        if "-" in qt:
            q_parts = qt.split("-", 1)
            q_start = _time_to_minutes(q_parts[0])
            q_end = _time_to_minutes(q_parts[1])
            if q_start is None or q_end is None:
                return False
            # Overlap check
            if s_end >= s_start and q_end >= q_start:
                return max(s_start, q_start) <= min(s_end, q_end)
        else:
            q_pt = _time_to_minutes(qt)
            if q_pt is None:
                return False
            lead_start = (s_start - 30) % (24 * 60)
            if s_end >= s_start:
                if s_start >= 30:
                    return (s_start - 30) <= q_pt <= s_end
                return q_pt >= lead_start or q_pt <= s_end
            else:
                # Cross-midnight window e.g. 23:00-01:00
                return q_pt >= lead_start or q_pt <= s_end
    return False


_CN_TO_CANONICAL_APP: Dict[str, str] = {
    "抖音": "douyin",
    "快手": "kuaishou",
    "哔哩哔哩": "bilibili",
    "微信": "wechat",
    "腾讯会议": "tencent_meeting",
    "钉钉": "dingtalk",
    "飞书": "feishu",
    "小红书": "xiaohongshu",
    "王者荣耀": "wangzherongyao",
    "高德地图": "gaode",
    "虎牙": "huya",
    "斗鱼直播": "douyu",
    "和平精英": "hepingjingying",
    "QQ": "qq",
    "淘宝": "taobao",
    "拼多多": "pdd",
    "爱奇艺": "iqiyi",
    "优酷": "youku",
    "腾讯视频": "tencentvideo",
    "咪咕": "migu",
}

_CN_TO_CANONICAL_BIZ: Dict[str, str] = {
    "开直播": "openlive",
    "看直播": "watch_live",
    "短视频": "video",
    "视频": "video",
    "视频通话": "video_call",
    "会议": "meeting",
    "游戏": "game",
    "云游戏": "game",
}


def _extract_ymd(ts_str: str) -> str:
    if not ts_str:
        return ""
    m_cn = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", ts_str)
    if m_cn:
        return f"{int(m_cn.group(1)):04d}-{int(m_cn.group(2)):02d}-{int(m_cn.group(3)):02d}"
    m_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", ts_str)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"
    return ""


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
            cur = self._conn.cursor()
            cur.execute(
                """
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
            )
            # Migration check for older databases (§4 Pitfall 7)
            cur.execute("PRAGMA table_info(vector)")
            cols = {row["name"] for row in cur.fetchall()}
            for col_name, col_def in [
                ("last_heat_at", "TEXT NOT NULL DEFAULT ''"),
                ("heat", "REAL NOT NULL DEFAULT 0.0"),
                ("app_col", "TEXT NOT NULL DEFAULT ''"),
                ("biz_col", "TEXT NOT NULL DEFAULT ''"),
                ("scene_col", "TEXT NOT NULL DEFAULT ''"),
                ("time_window_col", "TEXT NOT NULL DEFAULT ''"),
                ("alias_col", "TEXT NOT NULL DEFAULT ''"),
                ("target_col", "TEXT NOT NULL DEFAULT ''"),
                ("phrase_text", "TEXT NOT NULL DEFAULT ''"),
            ]:
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

            # External content FTS5 table synced with vector table (§4 Pitfall 7)
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
                    description,
                    search_text,
                    content='vector',
                    content_rowid='rowid'
                )
                """
            )
            self._conn.commit()

    def upsert(self, ns: Namespace, node: MemoryNode, embedding: Optional[Sequence[float]] = None) -> None:
        raw_search = build_search_text(node)
        pure_phrase = build_pure_phrase_text(node)
        tokenized_desc = tokenize_for_fts(node.description)
        tokenized_search = tokenize_for_fts(raw_search)
        emb_json = json.dumps(list(embedding) if embedding is not None else [])

        raw_app = str(node.content.get("application") or node.content.get("application_name") or "").strip()
        app_col = _CN_TO_CANONICAL_APP.get(raw_app, raw_app)
        raw_biz = str(node.content.get("business") or node.content.get("service_name") or "").strip()
        biz_col = _CN_TO_CANONICAL_BIZ.get(raw_biz, raw_biz)
        scene_col = str(node.content.get("scene") or node.content.get("environment") or node.content.get("location") or "").strip()
        time_window_col = str(node.content.get("time_window") or node.content.get("condition") or "").strip()
        alias_col = str(node.content.get("alias") or node.content.get("source_expression") or "").strip()
        target_col = str(node.content.get("target") or node.content.get("value") or node.content.get("action") or "").strip()

        if node.subtype == "working_record" and not time_window_col:
            time_window_col = _extract_ymd(
                str(node.content.get("created_at") or node.content.get("start_timestamp") or node.created_at or "")
            )

        # Orthogonal separation: if an expression row maps an alias to a time_window (e.g. 午间例行 -> 12:30-13:30),
        # keep app_col and biz_col empty on the expression row so app/biz are strictly resolved via Hop 2 (periodic_preference).
        if node.subtype == "expression" and re.search(r"\d{1,2}:\d{2}\s*[-~至到]\s*\d{1,2}:\d{2}", target_col):
            app_col = ""
            biz_col = ""

        with self._lock:
            cur = self._conn.cursor()
            # If row already exists, delete old FTS5 entry first before updating (§4 Pitfall 7)
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
                cur.execute(
                    """
                    UPDATE vector SET
                        owner_type = ?,
                        user_id = ?,
                        agent_id = ?,
                        memory_type = ?,
                        subtype = ?,
                        memory_key = ?,
                        status = ?,
                        validity = ?,
                        confidence = ?,
                        heat = ?,
                        last_heat_at = ?,
                        last_reinforced_at = ?,
                        updated_at = ?,
                        app_col = ?,
                        biz_col = ?,
                        scene_col = ?,
                        time_window_col = ?,
                        alias_col = ?,
                        target_col = ?,
                        phrase_text = ?,
                        description = ?,
                        search_text = ?,
                        embedding_json = ?
                    WHERE rowid = ?
                    """,
                    (
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
                        app_col,
                        biz_col,
                        scene_col,
                        time_window_col,
                        alias_col,
                        target_col,
                        pure_phrase,
                        tokenized_desc,
                        tokenized_search,
                        emb_json,
                        old_rowid,
                    ),
                )
                rowid = old_rowid
            else:
                cur.execute(
                    """
                    INSERT INTO vector (
                        memory_id, owner_type, user_id, agent_id, memory_type, subtype,
                        memory_key, status, validity, confidence, heat, last_heat_at,
                        last_reinforced_at, updated_at, app_col, biz_col, scene_col,
                        time_window_col, alias_col, target_col, phrase_text, description, search_text, embedding_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.memory_id,
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
                        app_col,
                        biz_col,
                        scene_col,
                        time_window_col,
                        alias_col,
                        target_col,
                        pure_phrase,
                        tokenized_desc,
                        tokenized_search,
                        emb_json,
                    ),
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
            # Exclude archived, expired, superseded by default
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
            except Exception:
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
            except Exception:
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
        tokenized = tokenize_for_fts(query)
        terms = [t for t in tokenized.split() if t]
        if not terms:
            return []
        # Deduplicate terms while preserving order
        seen = set()
        unique_terms = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                unique_terms.append(t)

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
        """Execute deterministic structured SQL SELECT on vector table slot columns (§2.5):
        Returns list of (memory_id, structured_sim, match_reason).
        """
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
            mid = r["memory_id"]
            st = r["subtype"]
            r_app = r["app_col"]
            r_biz = r["biz_col"]
            r_scene = r["scene_col"]
            r_tw = r["time_window_col"]

            matched_fields: List[str] = []
            score = 0.0

            app_hit = bool(app_set and r_app and r_app in app_set)
            biz_hit = bool(biz_set and r_biz and r_biz in biz_set)
            scene_hit = bool(scene_set and r_scene and any(sc in r_scene or r_scene in sc for sc in scene_set))
            tw_hit = bool(tw_list and r_tw and any(time_matches_window(qt, r_tw) for qt in tw_list))
            date_hit = bool(
                st == "working_record"
                and has_date_range
                and r_tw
                and len(r_tw) == 10
                and relative_date_range[0] <= r_tw <= relative_date_range[1]  # type: ignore[index]
            )

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
                continue

            # Orthogonal Scheme A scoring across parameter nodes (scene_preference / preferred_value),
            # working memory records (working_record), and routing pointer nodes (periodic_preference / service_to_app)
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
                if app_hit and biz_hit:
                    score = 0.84
                else:
                    score = 0.78
            else:
                score = 0.80

            if score > 0:
                results.append((mid, score, "SELECT WHERE " + " AND ".join(matched_fields)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def close(self) -> None:
        with self._lock:
            self._conn.close()
