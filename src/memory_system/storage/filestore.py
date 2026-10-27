from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..core.models import MemoryNode, now_utc_str
from ..core.namespace import Namespace


_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_path_segment(segment: str, field_name: str = "segment") -> str:
    """Validate path segment against strict whitelist ^[A-Za-z0-9._-]+$ to prevent path traversal."""
    s = str(segment or "").strip()
    if not s or s in (".", "..") or not _SAFE_SEGMENT_RE.match(s):
        raise ValueError(f"Unsafe path segment for {field_name}: {segment!r}")
    return s


def hash_memory_key(memory_key: str) -> str:
    """Directory name is sha256(memory_key)[:12], never plaintext key."""
    return hashlib.sha256(memory_key.encode("utf-8")).hexdigest()[:12]


def render_markdown(node: MemoryNode) -> str:
    """Render L0/L1/L2 structured Markdown content.md for a MemoryNode."""
    lines = [
        f"# {node.description or node.memory_key}",
        "",
        "## Overview (L1)",
        f"- **Memory ID**: `{node.memory_id}`",
        f"- **Type**: `{node.memory_type}/{node.subtype}`",
        f"- **Key**: `{node.memory_key}`",
        f"- **Status**: `{node.status}` (version {node.version}, confidence {node.confidence:.2f})",
        f"- **Validity**: `{node.validity.type}`",
    ]
    if node.overview:
        lines.append(f"- **Summary**: {node.overview}")
    lines.append("")
    lines.append("## Details (L2)")
    if node.details:
        lines.append(node.details)
        lines.append("")
    lines.append("```json")
    lines.append(json.dumps(node.content, ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


class FileStore:
    """Authoritative file storage with atomic writes, path whitelisting, and per-key locking."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._meta_lock = threading.RLock()
        self._key_locks: Dict[str, threading.RLock] = {}
        self._history_locks: Dict[str, threading.RLock] = {}

    def get_key_lock(self, ns: Namespace, subtype: str, memory_key: str) -> threading.RLock:
        lock_id = f"{ns.scope_key}:{subtype}:{memory_key}"
        with self._meta_lock:
            if lock_id not in self._key_locks:
                self._key_locks[lock_id] = threading.RLock()
            return self._key_locks[lock_id]

    def _get_history_lock(self, history_path: Path) -> threading.RLock:
        key = str(history_path.resolve())
        with self._meta_lock:
            if key not in self._history_locks:
                self._history_locks[key] = threading.RLock()
            return self._history_locks[key]

    @staticmethod
    def atomic_write_text(target: Path, content: str, retries: int = 5) -> None:
        """Write to .tmp.<pid>.<tid> and os.replace with retries for Windows concurrency."""
        target.parent.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        tid = threading.get_ident()
        tmp_path = target.with_name(f"{target.name}.tmp.{pid}.{tid}")
        tmp_path.write_text(content, encoding="utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                os.replace(tmp_path, target)
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.02 * (attempt + 1))
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        if last_err is not None:
            raise last_err

    def atomic_write_json(self, target: Path, data: Any, retries: int = 5) -> None:
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        self.atomic_write_text(target, text, retries=retries)

    def append_jsonl(self, target: Path, record: Dict[str, Any]) -> None:
        lock = self._get_history_lock(target)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as f:
                f.write(line)

    def namespace_base_dir(self, ns: Namespace) -> Path:
        """Per §1.3: Isolation key is ONLY user_id (+ agent_id for agent experience).
        Paths:
          - users/{user_id}/memory/...
          - users/{user_id}/agent_experience/{agent_id}/...
        """
        uid = validate_path_segment(ns.user_id, "user_id")
        if ns.owner_type == "agent":
            aid = validate_path_segment(ns.agent_id, "agent_id")
            return self.data_root / "users" / uid / "agent_experience" / aid
        return self.data_root / "users" / uid / "memory"

    def node_dir(self, ns: Namespace, memory_type: str, subtype: str, memory_key: str) -> Path:
        mtype = validate_path_segment(memory_type, "memory_type")
        stype = validate_path_segment(subtype, "subtype")
        khash = hash_memory_key(memory_key)
        return self.namespace_base_dir(ns) / mtype / stype / khash

    def save_node(
        self,
        ns: Namespace,
        node: MemoryNode,
        history_entry: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ndir = self.node_dir(ns, node.memory_type, node.subtype, node.memory_key)
        ndir.mkdir(parents=True, exist_ok=True)
        self.atomic_write_json(ndir / "memory.json", node.to_dict())
        self.atomic_write_text(ndir / "content.md", render_markdown(node))
        if history_entry is not None:
            self.append_jsonl(ndir / "history.jsonl", history_entry)
        elif not (ndir / "history.jsonl").exists():
            self.append_jsonl(
                ndir / "history.jsonl",
                {
                    "ts": now_utc_str(),
                    "action": "init",
                    "status": node.status,
                    "version": node.version,
                },
            )
        return ndir

    def load_node_from_dir(self, ndir: Path) -> Optional[MemoryNode]:
        mem_file = ndir / "memory.json"
        if not mem_file.exists():
            return None
        try:
            data = json.loads(mem_file.read_text(encoding="utf-8"))
            return MemoryNode.from_dict(data)
        except Exception:
            return None

    def read_content_md(self, ns: Namespace, node: MemoryNode) -> str:
        ndir = self.node_dir(ns, node.memory_type, node.subtype, node.memory_key)
        md_file = ndir / "content.md"
        if md_file.exists():
            return md_file.read_text(encoding="utf-8")
        return render_markdown(node)

    def read_history(self, ns: Namespace, node: MemoryNode) -> List[Dict[str, Any]]:
        ndir = self.node_dir(ns, node.memory_type, node.subtype, node.memory_key)
        hist_file = ndir / "history.jsonl"
        if not hist_file.exists():
            return []
        records: List[Dict[str, Any]] = []
        for line in hist_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    def find_by_key(
        self,
        ns: Namespace,
        memory_type: str,
        subtype: str,
        memory_key: str,
    ) -> Optional[MemoryNode]:
        ndir = self.node_dir(ns, memory_type, subtype, memory_key)
        node = self.load_node_from_dir(ndir)
        if node is not None and node.memory_key == memory_key:
            return node
        # Fallback scan across subtype directory in case of collision or custom placement
        mtype = validate_path_segment(memory_type, "memory_type")
        stype = validate_path_segment(subtype, "subtype")
        stype_dir = self.namespace_base_dir(ns) / mtype / stype
        if not stype_dir.exists():
            return None
        for child in stype_dir.iterdir():
            if child.is_dir():
                cand = self.load_node_from_dir(child)
                if cand is not None and cand.memory_key == memory_key:
                    return cand
        return None

    def find_by_id(self, ns: Namespace, memory_id: str) -> Optional[MemoryNode]:
        for node in self.iter_nodes(ns):
            if node.memory_id == memory_id:
                return node
        return None

    def iter_nodes(
        self,
        ns: Namespace,
        memory_type: Optional[str] = None,
        subtype: Optional[str] = None,
    ) -> Iterator[MemoryNode]:
        base = self.namespace_base_dir(ns)
        if not base.exists():
            return
        type_dirs = [base / validate_path_segment(memory_type, "memory_type")] if memory_type else [d for d in base.iterdir() if d.is_dir()]
        for tdir in type_dirs:
            if not tdir.exists() or not tdir.is_dir():
                continue
            sub_dirs = [tdir / validate_path_segment(subtype, "subtype")] if subtype else [d for d in tdir.iterdir() if d.is_dir()]
            for sdir in sub_dirs:
                if not sdir.exists() or not sdir.is_dir():
                    continue
                for khash_dir in sdir.iterdir():
                    if khash_dir.is_dir():
                        node = self.load_node_from_dir(khash_dir)
                        if node is not None:
                            yield node

    def iter_all_namespaces_and_nodes(self) -> Iterator[Tuple[Namespace, MemoryNode]]:
        users_dir = self.data_root / "users"
        if not users_dir.exists():
            return
        for udir in sorted(users_dir.iterdir()):
            if not udir.is_dir():
                continue
            uid = udir.name
            user_ns = Namespace(user_id=uid, owner_type="user")
            for node in self.iter_nodes(user_ns):
                yield user_ns, node
            agent_exp_dir = udir / "agent_experience"
            if agent_exp_dir.exists() and agent_exp_dir.is_dir():
                for adir in sorted(agent_exp_dir.iterdir()):
                    if not adir.is_dir():
                        continue
                    agent_ns = Namespace(user_id=uid, owner_type="agent", agent_id=adir.name)
                    for node in self.iter_nodes(agent_ns):
                        yield agent_ns, node

    def delete_node(self, ns: Namespace, node: MemoryNode) -> bool:
        ndir = self.node_dir(ns, node.memory_type, node.subtype, node.memory_key)
        if not ndir.exists():
            return False
        for child in ndir.iterdir():
            if child.is_file():
                child.unlink()
        try:
            ndir.rmdir()
        except OSError:
            pass
        return True

    # Session Archive helpers
    def session_dir(self, user_id: str, session_id: str) -> Path:
        uid = validate_path_segment(user_id, "user_id")
        sid = validate_path_segment(session_id, "session_id")
        return self.data_root / "users" / uid / "sessions" / sid

    def save_session(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        turns: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        sdir = self.session_dir(user_id, session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        hist_path = sdir / "history.jsonl"
        lines = []
        for idx, turn in enumerate(turns, start=1):
            t_copy = dict(turn)
            if "turn" not in t_copy:
                t_copy["turn"] = idx
            lines.append(json.dumps(t_copy, ensure_ascii=False))
        self.atomic_write_text(hist_path, "\n".join(lines) + ("\n" if lines else ""))
        meta = {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "turn_count": len(turns),
            "archived_at": now_utc_str(),
            **(metadata or {}),
        }
        self.atomic_write_json(sdir / "meta.json", meta)
        return sdir
