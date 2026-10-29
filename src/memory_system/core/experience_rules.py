"""Data-driven experience trigger matching and statutory-holiday calendar helpers.

These helpers replace hardcoded sample phrases / dates in api.py and downstream bridges:
  - Experience cards declare their own ``crystallize_trigger`` / ``apply_trigger`` keyword groups
    (AND across groups, OR within a group), so the code never embeds per-sample literals.
  - Holiday shifting is computed from a calendar table (configurable data) instead of
    hardcoding a specific "from date -> to date" pair.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

KeywordGroups = Sequence[Sequence[str]]


HOLIDAY_DATE_TOKEN = "@HOLIDAY_DATE"


def _kw_hit(kw: str, text: str, ref_year: Optional[int]) -> bool:
    if not kw:
        return False
    if kw == HOLIDAY_DATE_TOKEN:
        years = [ref_year] if ref_year else sorted(CN_STATUTORY_HOLIDAYS.keys())
        return any(find_holiday_dates(text, y) for y in years)
    return kw in text


def match_keyword_groups(groups: Optional[KeywordGroups], text: str, ref_year: Optional[int] = None) -> bool:
    """Return True iff every group has at least one keyword contained in ``text``.

    Special token ``@HOLIDAY_DATE`` matches an explicit 'M月D日' date that falls on a statutory holiday.
    An empty / missing group list never matches (prevents accidental always-on triggers).
    """
    if not groups or not text:
        return False
    for grp in groups:
        if not grp:
            continue
        if not any(_kw_hit(kw, text, ref_year) for kw in grp):
            return False
    return True


def card_trigger(card: Optional[Dict[str, Any]], kind: str) -> Dict[str, KeywordGroups]:
    """Fetch a trigger spec ({'first_turn': groups, 'later_turns': groups}) from an experience card."""
    if not isinstance(card, dict):
        return {}
    spec = card.get(kind)
    if isinstance(spec, dict):
        return {k: v for k, v in spec.items() if isinstance(v, (list, tuple))}
    if isinstance(spec, (list, tuple)):
        return {"first_turn": spec}
    return {}


def trigger_matches(
    card: Optional[Dict[str, Any]],
    kind: str,
    first_turn_text: str,
    later_turns_text: str = "",
    ref_year: Optional[int] = None,
) -> bool:
    """Evaluate a card trigger spec. Every declared section (first_turn / later_turns) must match."""
    spec = card_trigger(card, kind)
    if not spec:
        return False
    if "first_turn" in spec and not match_keyword_groups(spec["first_turn"], first_turn_text, ref_year):
        return False
    if "later_turns" in spec and not match_keyword_groups(spec["later_turns"], later_turns_text, ref_year):
        return False
    return True


# ---------------------------------------------------------------------------
# Statutory holiday calendar (PRC). Pure data — extend per year as announced by the State Council.
# Each entry: (first_off_day, last_off_day); adjusted working days (调休上班) listed separately.
# ---------------------------------------------------------------------------
CN_STATUTORY_HOLIDAYS: Dict[int, List[tuple[str, str, str]]] = {
    2026: [
        ("元旦", "2026-01-01", "2026-01-03"),
        ("春节", "2026-02-15", "2026-02-23"),
        ("清明节", "2026-04-04", "2026-04-06"),
        ("劳动节", "2026-05-01", "2026-05-05"),
        ("端午节", "2026-06-19", "2026-06-21"),
        ("中秋节", "2026-09-25", "2026-09-27"),
        ("国庆节", "2026-10-01", "2026-10-07"),
    ],
}
CN_ADJUSTED_WORKDAYS: Dict[int, List[str]] = {
    2026: ["2026-01-04", "2026-02-14", "2026-02-28", "2026-05-09", "2026-09-20", "2026-10-10"],
}


def _d(s: str) -> date:
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


def holiday_name(d: date) -> Optional[str]:
    for name, start, end in CN_STATUTORY_HOLIDAYS.get(d.year, []):
        if _d(start) <= d <= _d(end):
            return name
    return None


def is_cn_workday(d: date) -> bool:
    if d.isoformat() in CN_ADJUSTED_WORKDAYS.get(d.year, []):
        return True
    if holiday_name(d):
        return False
    return d.weekday() < 5


def next_cn_workday_after(d: date, max_days: int = 30) -> Optional[date]:
    cur = d
    for _ in range(max_days):
        cur = cur + timedelta(days=1)
        if is_cn_workday(cur):
            return cur
    return None


_MD_RE = re.compile(r"(\d{1,2})月(\d{1,2})[日号]")


def find_holiday_dates(text: str, ref_year: int) -> List[date]:
    """Return explicit 'M月D日' dates in ``text`` that fall on a statutory holiday of ``ref_year``."""
    out: List[date] = []
    for m in _MD_RE.finditer(text or ""):
        try:
            d = date(ref_year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            continue
        if holiday_name(d) and d not in out:
            out.append(d)
    return out


def iter_triggerable_cards(cards: Iterable[Dict[str, Any]], kind: str) -> Iterable[Dict[str, Any]]:
    for c in cards:
        if card_trigger(c, kind):
            yield c
