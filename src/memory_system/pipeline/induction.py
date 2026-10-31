from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from ..core.memory_key import build_memory_key
from ..core.models import Event, EventType, MemoryNode, MemoryStatus
from ..core.namespace import Namespace
from ..core.registry import MemoryRegistry
from ..prompts.induction_prompts import INDUCTION_SYSTEM_PROMPT
from ..storage.filestore import FileStore
from .updater import MemoryUpdater

__all__ = [
    "INDUCTION_SYSTEM_PROMPT",
    "InductionEngine",
]


class InductionEngine:
    """Deterministic cross-session induction without LLM (§2.8):
    1. induce_preferred_values: same (app, biz) reaches vote_threshold -> preferred_value
    2. induce_service_to_app: business mapped to a single app with >= 2 observations -> service_to_app
    3. induce_periodic_preferences: same (app, biz) repeatedly in same time_window -> periodic_preference
    """

    def __init__(
        self,
        registry: MemoryRegistry,
        filestore: FileStore,
        updater: MemoryUpdater,
    ) -> None:
        self.registry = registry
        self.filestore = filestore
        self.updater = updater

    def run_all(self, ns: Namespace) -> List[MemoryNode]:
        """Run cross-session service_to_app pointer induction across the three major
        preference categories (periodic_preference, scene_preference, application_business).
        """
        created: List[MemoryNode] = []
        created.extend(self.induce_service_to_app(ns))
        return created

    def induce_preferred_values(self, ns: Namespace) -> List[MemoryNode]:
        """When an application_business node is active/reinforced (or meets vote_threshold)
        and carries parameter preferences, induce a preferred_value node idempotently.
        """
        spec_pv = self.registry.get("preferred_value")
        spec_ab = self.registry.get("application_business")
        if spec_pv is None or spec_ab is None:
            return []

        induced: List[MemoryNode] = []
        ab_nodes = list(self.filestore.iter_nodes(ns, memory_type="preference", subtype="application_business"))
        for ab in ab_nodes:
            if ab.validity.type == "temporary":
                continue
            if ab.status not in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value):
                if ab.stats.positive_votes < max(1, spec_ab.vote_threshold):
                    continue
            app = ab.content.get("application")
            biz = ab.content.get("business")
            if not app or not biz:
                continue
            # Check if node has concrete preference parameters
            param_keys = ("resolution", "priority", "latency", "bitrate")
            has_params = any(ab.content.get(pk) not in (None, "") for pk in param_keys)
            if not has_params:
                continue

            pv_key = build_memory_key(spec_pv, {"application": app, "business": biz})
            existing_pv = self.filestore.find_by_key(ns, spec_pv.memory_type, spec_pv.subtype, pv_key)
            if existing_pv is not None:
                # Skip if already up to date with same parameters (idempotent)
                if all(existing_pv.content.get(pk) == ab.content.get(pk) for pk in param_keys if ab.content.get(pk)):
                    continue

            pv_content: Dict[str, Any] = {
                "application": app,
                "business": biz,
                "validity": "long_term",
                "note": f"跨会话归纳首选参数：{app} ({biz})",
            }
            for pk in param_keys:
                if ab.content.get(pk) not in (None, ""):
                    pv_content[pk] = ab.content[pk]

            ev = Event(
                event_type=EventType.PREFERENCE_SET.value,
                subtype="preferred_value",
                memory_key_fields={"application": app, "business": biz},
                content=pv_content,
                validity="long_term",
                confidence=ab.confidence,
                user_id=ns.user_id,
                tenant_id=ns.tenant_id,
            )
            induced.append(self.updater.apply_event(ns, ev))
        return induced

    def induce_service_to_app(self, ns: Namespace) -> List[MemoryNode]:
        """When a business is associated with only 1 application and has >= 2 observations,
        induce a service_to_app node idempotently.
        """
        spec_s2a = self.registry.get("service_to_app")
        if spec_s2a is None:
            return []

        major_nodes: List[MemoryNode] = []
        for st in ("preferred_value", "scene_preference", "application_business"):
            major_nodes.extend(self.filestore.iter_nodes(ns, memory_type="preference", subtype=st))

        biz_to_apps: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        biz_latest_ts: Dict[str, str] = {}

        for ab in major_nodes:
            if ab.validity.type == "temporary":
                continue
            if ab.status not in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value):
                continue
            app = ab.content.get("application")
            biz = ab.content.get("business")
            if app and biz:
                obs = max(1, ab.stats.positive_votes)
                biz_to_apps[str(biz)][str(app)] += obs
                if ab.updated_at and ab.updated_at > biz_latest_ts.get(str(biz), ""):
                    biz_latest_ts[str(biz)] = ab.updated_at

        induced: List[MemoryNode] = []
        for biz, app_counts in biz_to_apps.items():
            if len(app_counts) != 1:
                continue
            sole_app, total_obs = next(iter(app_counts.items()))
            if total_obs < 2:
                continue

            s2a_key = build_memory_key(spec_s2a, {"business": biz})
            existing = self.filestore.find_by_key(ns, spec_s2a.memory_type, spec_s2a.subtype, s2a_key)
            if existing is not None and existing.content.get("application") == sole_app:
                continue

            ev = Event(
                event_type=EventType.PREFERENCE_SET.value,
                subtype="service_to_app",
                memory_key_fields={"business": biz},
                content={
                    "business": biz,
                    "application": sole_app,
                    "validity": "long_term",
                    "note": f"跨会话归纳路由指针：{biz} 默认使用 {sole_app}（累计 {total_obs} 次观测）",
                },
                validity="long_term",
                confidence=0.9,
                user_id=ns.user_id,
                tenant_id=ns.tenant_id,
                event_time=biz_latest_ts.get(str(biz)) or "",
            )
            induced.append(self.updater.apply_event(ns, ev))
        return induced

    def induce_periodic_preferences(self, ns: Namespace) -> List[MemoryNode]:
        """When the same (app, biz) repeatedly appears in the same time_window,
        induce a lightweight periodic_preference pointer node where condition=time_window and value=app:biz.
        Does NOT duplicate QoS parameters (resolution/latency/priority), preserving Single Source of Truth.
        """
        spec_pp = self.registry.get("periodic_preference")
        if spec_pp is None:
            return []

        ab_nodes = list(self.filestore.iter_nodes(ns, memory_type="preference", subtype="application_business"))
        induced: List[MemoryNode] = []

        for ab in ab_nodes:
            if ab.validity.type == "temporary":
                continue
            if ab.status not in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value):
                continue
            app = ab.content.get("application")
            biz = ab.content.get("business")
            tw = ab.content.get("time_window")
            if not app or not biz or not tw:
                continue

            val_str = f"{app}:{biz}"
            pp_key = build_memory_key(spec_pp, {"condition": str(tw), "value": val_str})
            existing = self.filestore.find_by_key(ns, spec_pp.memory_type, spec_pp.subtype, pp_key)

            # Determine how many observations the source node has
            target_votes = max(1, ab.stats.positive_votes)
            current_votes = existing.stats.positive_votes if existing is not None else 0

            if existing is not None and current_votes >= target_votes:
                continue

            pp_content: Dict[str, Any] = {
                "condition": str(tw),
                "value": val_str,
                "application": app,
                "business": biz,
                "time_window": str(tw),
                "validity": "long_term",
                "note": f"时段路由指针：{tw} -> {app} ({biz})",
            }

            steps_needed = max(1, target_votes - current_votes)
            last_node = existing
            for _ in range(steps_needed):
                ev = Event(
                    event_type=EventType.PREFERENCE_SET.value,
                    subtype="periodic_preference",
                    memory_key_fields={"condition": str(tw), "value": val_str},
                    content=pp_content,
                    validity="long_term",
                    confidence=ab.confidence,
                    user_id=ns.user_id,
                    tenant_id=ns.tenant_id,
                    event_time=ab.updated_at or "",
                )
                last_node = self.updater.apply_event(ns, ev)
            if last_node is not None:
                induced.append(last_node)

        return induced
