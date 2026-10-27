from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..core.models import Edge
from ..core.namespace import Namespace
from .filestore import FileStore


class GraphStore:
    """JSONL-backed graph store for memory-event and memory-memory relationships.
    Per §4 Pitfall 9: graph/edges.jsonl full-read -> append -> atomic full-write,
    and neighbors() performs 1-2 hop BFS filtered by namespace.
    """

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self.edges_path = self.data_root / "graph" / "edges.jsonl"
        self._lock = threading.RLock()

    def _read_all(self) -> List[Edge]:
        if not self.edges_path.exists():
            return []
        edges: List[Edge] = []
        for line in self.edges_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                edges.append(Edge.from_dict(json.loads(line)))
            except Exception:
                continue
        return edges

    def _write_all(self, edges: List[Edge]) -> None:
        lines = [json.dumps(e.to_dict(), ensure_ascii=False) for e in edges]
        content = "\n".join(lines) + ("\n" if lines else "")
        FileStore.atomic_write_text(self.edges_path, content)

    def add_edge(self, edge: Edge) -> None:
        with self._lock:
            edges = self._read_all()
            for e in edges:
                if (
                    e.source_id == edge.source_id
                    and e.target_id == edge.target_id
                    and e.edge_type == edge.edge_type
                    and e.user_id == edge.user_id
                    and e.owner_type == edge.owner_type
                    and e.agent_id == edge.agent_id
                ):
                    return
            edges.append(edge)
            self._write_all(edges)

    def get_edges(self, ns: Namespace, node_id: Optional[str] = None) -> List[Edge]:
        with self._lock:
            all_edges = self._read_all()
        scoped = [
            e
            for e in all_edges
            if e.user_id == ns.user_id and e.owner_type == ns.owner_type and e.agent_id == ns.agent_id
        ]
        if node_id is None:
            return scoped
        return [e for e in scoped if e.source_id == node_id or e.target_id == node_id]

    def neighbors(self, ns: Namespace, start_id: str, max_hops: int = 1) -> List[str]:
        """1-2 hop BFS over namespace-scoped edges."""
        if max_hops <= 0:
            return []
        scoped = self.get_edges(ns)
        adj: Dict[str, Set[str]] = {}
        for e in scoped:
            adj.setdefault(e.source_id, set()).add(e.target_id)
            adj.setdefault(e.target_id, set()).add(e.source_id)

        visited: Set[str] = {start_id}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        found: List[str] = []

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for nxt in sorted(adj.get(curr, set())):
                if nxt not in visited:
                    visited.add(nxt)
                    found.append(nxt)
                    queue.append((nxt, depth + 1))
        return found
