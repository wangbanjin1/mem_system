"""Dialogue-act classification for working-memory (historical order ledger) references.

Replaces hardcoded sample sentences ("想按之前的意图保障单规格", "把昨天预约的…") with regex families
that cover paraphrases, plus a structural signal (whether the user already stated today's time).
"""
from __future__ import annotations

import re
from typing import Optional

# "把昨天预约的抖音开直播保障往后推迟一个小时" / "前天约的那个会议保障提前1小时"
_RESCHEDULE_RE = re.compile(
    r"(?:昨天|前天|大前天|上周[一二三四五六日天]?|前几天|上次|之前|刚才|早上|上午)(?:预约|预订|约|订|办|开|定)(?:好)?的.{0,30}?"
    r"(?:往后推迟|推迟|延后|往后挪|往后推|顺延|提前|往前挪|往前提)"
)
_SHIFT_RE = re.compile(
    r"(往后推迟|推迟|延后|往后挪|往后推|顺延|提前|往前挪|往前提)\s*"
    r"(半|一|两|二|三|四|\d+(?:\.\d+)?)?\s*个?\s*(半)?\s*(小时|钟头|分钟)?"
)
_CN_NUM = {"半": 0.5, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4}
_WEEKDAY_CN = "一二三四五六日"
# cross-period reference to a historical order's spec
_ORDER_REF_RE = re.compile(
    r"(?:上周|上个月初?|上个月|上回|上次|之前|以前|前几天|前阵子|那次)"
    r".{0,24}?(?:意图保障单|保障单|工单)(?:的)?(?:规格|那套|配置|参数|标准)?"
)
_ORDER_REF_RE_2 = re.compile(r"(?:按|照|调)(?:之前|上次|上周|上个月|以前)(?:的)?.{0,12}?(?:规格|那套|配置)")
_TIME_RE = re.compile(r"\d{1,2}\s*[时点:：]|[一二三四五六七八九十两]{1,3}点")


def has_explicit_time(text: str) -> bool:
    return bool(_TIME_RE.search(text or ""))


def classify_working_record_reference(first_turn_text: str, has_time: Optional[bool] = None) -> Optional[str]:
    """Return 'reschedule' | 'disambig' | 'reuse' | None for the first user turn.

    - reschedule: shifting a previously booked order earlier/later.
    - reuse:      referencing a historical order's spec without stating today's time
                  (spec is filled from the ledger, today's time is asked next).
    - disambig:   referencing a historical spec while already stating today's time
                  (time is filled from the utterance, the spec is confirmed next).
    """
    text = first_turn_text or ""
    if _RESCHEDULE_RE.search(text):
        return "reschedule"
    if _ORDER_REF_RE.search(text) or _ORDER_REF_RE_2.search(text):
        if has_time is None:
            has_time = has_explicit_time(text)
        return "disambig" if has_time else "reuse"
    return None


def reschedule_shift_minutes(text: str) -> Optional[int]:
    """Signed shift in minutes for a reschedule request ('往后推迟一个小时' -> +60, '提前半小时' -> -30)."""
    m = _SHIFT_RE.search(text or "")
    if not m:
        return None
    word, num, half, unit = m.group(1), m.group(2), m.group(3), m.group(4)
    sign = -1 if word.startswith(("提前", "往前")) else 1
    if num is None and unit is None:
        val_min = 60  # bare "推迟/提前" defaults to one hour only when no amount is given
    else:
        n = _CN_NUM.get(num, None) if num else 1
        if n is None:
            n = float(num)
        if half:
            n += 0.5
        val_min = int(round(n)) if unit == "分钟" else int(round(n * 60))
    return sign * val_min


def referenced_day_offset(text: str, ref_weekday: Optional[int] = None) -> Optional[tuple[int, int]]:
    """Map a relative day word to an inclusive (min_days_ago, max_days_ago) window.

    昨天 -> (1,1), 前天 -> (2,2), 大前天 -> (3,3), 上周X -> exact day of last week, 上周/前几天 -> wider window.
    """
    t = text or ""
    if "大前天" in t:
        return (3, 3)
    if "前天" in t:
        return (2, 2)
    if "昨天" in t:
        return (1, 1)
    m = re.search(r"上周([一二三四五六日天])", t)
    if m and ref_weekday is not None:
        wd = 6 if m.group(1) in "日天" else _WEEKDAY_CN.index(m.group(1))
        days = ref_weekday + 7 - wd
        return (days, days)
    if "上周" in t:
        return (ref_weekday + 1 if ref_weekday is not None else 1, (ref_weekday + 7) if ref_weekday is not None else 13)
    if "前几天" in t:
        return (2, 6)
    return None
