from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory_system.core.models import MemoryNode
from memory_system.core.namespace import agent_namespace, user_namespace
from memory_system.storage.filestore import hash_memory_key, validate_path_segment
from memory_system.storage.index import MemoryIndex


def test_fts5_external_table_upsert_and_delete_no_ghost_hits(tmp_path: Path) -> None:
    """Verify §4 Pitfall 7: FTS5 external content table syncs on update and leaves no ghost rows on delete."""
    idx = MemoryIndex(tmp_path / "memory.db")
    ns = user_namespace("U001")
    node = MemoryNode(
        memory_id="MEM_FTS_1",
        memory_type="preference",
        subtype="application_business",
        memory_key="douyin | live",
        user_id="U001",
        description="抖音户外直播超清配置",
        content={"application": "douyin", "business": "live", "resolution": "1080p"},
    )
    idx.upsert(ns, node, embedding=[1.0, 0.0, 0.0, 0.0])

    hits1 = idx.search_fts(ns, "户外直播")
    assert len(hits1) == 1
    assert hits1[0][0] == "MEM_FTS_1"

    # Update description and verify old keyword no longer ghost-matches
    node.description = "快手室内短视频流畅配置"
    node.content = {"application": "kuaishou", "business": "video", "resolution": "720p"}
    idx.upsert(ns, node, embedding=[0.0, 1.0, 0.0, 0.0])

    assert idx.search_fts(ns, "户外直播") == []
    assert len(idx.search_fts(ns, "室内短视频")) == 1

    # Delete node and verify FTS5 returns empty (no ghost row)
    assert idx.delete("MEM_FTS_1") is True
    assert idx.search_fts(ns, "室内短视频") == []
    idx.close()


def test_migration_adds_last_heat_at_column(tmp_path: Path) -> None:
    """Verify §4 Pitfall 7: older SQLite table missing last_heat_at is migrated via ALTER TABLE."""
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE vector (
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
            last_reinforced_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            search_text TEXT NOT NULL DEFAULT '',
            embedding_json TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    conn.commit()
    conn.close()

    idx = MemoryIndex(db_file)
    ns = user_namespace("U_LEGACY")
    node = MemoryNode(memory_id="MEM_LEG_1", user_id="U_LEGACY", memory_key="douyin | live")
    idx.upsert(ns, node, embedding=[0.5, 0.5])
    res = idx.search_vector(ns, [0.5, 0.5])
    assert len(res) == 1
    idx.close()


def test_index_strict_namespace_isolation(tmp_path: Path) -> None:
    """Verify §1.3: SQL queries strictly filter by owner_type + user_id + agent_id."""
    idx = MemoryIndex(tmp_path / "iso.db")
    ns_u1 = user_namespace("USER_A", tenant_id="T1")
    ns_u2 = user_namespace("USER_B", tenant_id="T1")
    ns_a1 = agent_namespace("USER_A", "AGENT_1", tenant_id="T1")

    idx.upsert(
        ns_u1,
        MemoryNode(memory_id="M_U1", user_id="USER_A", description="抖音直播1080p"),
        embedding=[1.0, 0.0],
    )
    idx.upsert(
        ns_u2,
        MemoryNode(memory_id="M_U2", user_id="USER_B", description="抖音直播1080p"),
        embedding=[1.0, 0.0],
    )
    idx.upsert(
        ns_a1,
        MemoryNode(memory_id="M_A1", owner_type="agent", user_id="USER_A", agent_id="AGENT_1", description="抖音直播1080p"),
        embedding=[1.0, 0.0],
    )

    assert [m for m, _ in idx.search_vector(ns_u1, [1.0, 0.0])] == ["M_U1"]
    assert [m for m, _ in idx.search_vector(ns_u2, [1.0, 0.0])] == ["M_U2"]
    assert [m for m, _ in idx.search_vector(ns_a1, [1.0, 0.0])] == ["M_A1"]
    assert [m for m, _ in idx.search_fts(ns_u1, "抖音直播")] == ["M_U1"]
    idx.close()


def test_path_segment_whitelist_and_hash_memory_key() -> None:
    assert validate_path_segment("USER_001-a.b") == "USER_001-a.b"
    for bad in ("../etc", "user/name", "..", ".", "", "a\\b", "user name"):
        with pytest.raises(ValueError):
            validate_path_segment(bad)

    h = hash_memory_key("douyin | live")
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)
