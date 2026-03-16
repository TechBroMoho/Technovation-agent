"""
tools.py — LangChain tools for the coaching scheduling agent.

Uses the @tool decorator so LangChain can pass these directly to the LLM
for real tool-calling (the LLM decides when and how to invoke them).
"""

from datetime import datetime, timedelta
from langchain.tools import tool

# ---------------------------------------------------------------------------
# Mock "database"
# ---------------------------------------------------------------------------

BOOKED_SLOTS = {"Tue 2PM", "Wed 10AM"}

COACH_SCHEDULE = {
    "Mon": ["10AM", "11AM", "2PM", "3PM"],
    "Tue": ["9AM", "2PM", "3PM", "4PM"],
    "Wed": ["10AM", "1PM", "3PM"],
    "Thu": ["9AM", "11AM", "2PM", "4PM"],
    "Fri": ["10AM", "12PM", "2PM"],
}

# Coach-specific scheduling preferences.
# Each coach can express a preferred time-of-day or specific hourly blocks.
COACH_PREFERENCES = {
    "default": {
        "time_of_day": "morning",          # "morning", "afternoon", or None
        "preferred_hours": ["10AM", "11AM"],  # specific slots the coach prefers
    },
}


def _day_label(offset: int) -> str:
    target = datetime.now() + timedelta(days=offset)
    return target.strftime("%a")


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------

@tool
def get_available_slots(day: str, time_of_day: str = "any", preferences: str = "") -> str:
    """
    Get available coaching slots for a given day.
    Use this when the user asks about availability or open times.

    IMPORTANT: If a coach has specific scheduling preferences (e.g. they prefer
    morning sessions or certain hourly blocks), pass those preferences so the
    returned list is sorted with the coach's preferred slots FIRST. Always
    prioritize any slots that match the coach's preference metadata — present
    those at the top of the list so the user naturally gravitates toward them.

    Args:
        day: The day to check. Use 'today', 'tomorrow', or a weekday name
             like 'Tuesday'.
        time_of_day: Filter by 'morning', 'afternoon', or 'any'.
        preferences: Optional — a coach name key (e.g. 'default') OR a
                     comma-separated list of preferred hours (e.g.
                     '10AM,11AM') OR a JSON dict string with keys
                     'time_of_day' and/or 'preferred_hours'. When provided,
                     matching slots are listed first in the results.

    Returns:
        A formatted string listing available slots (preferred slots listed
        first), or a message if none are found.
    """
    import json as _json

    day_clean = day.strip().lower()
    if day_clean == "today":
        label = _day_label(0)
    elif day_clean == "tomorrow":
        label = _day_label(1)
    else:
        label = day_clean[:3].capitalize()

    raw_slots = COACH_SCHEDULE.get(label, [])

    def is_morning(t):   return "AM" in t
    def is_afternoon(t): return "PM" in t

    if time_of_day == "morning":
        raw_slots = [t for t in raw_slots if is_morning(t)]
    elif time_of_day == "afternoon":
        raw_slots = [t for t in raw_slots if is_afternoon(t)]

    available = [f"{label} {t}" for t in raw_slots if f"{label} {t}" not in BOOKED_SLOTS]

    if not available:
        return f"No available slots found for {day} ({time_of_day})."

    # --- resolve preference metadata ---
    pref_hours: list[str] = []
    pref_tod: str | None = None

    if preferences:
        pref_str = preferences.strip()
        # 1) Check if it's a known coach key
        if pref_str in COACH_PREFERENCES:
            pref_data = COACH_PREFERENCES[pref_str]
            pref_hours = pref_data.get("preferred_hours", [])
            pref_tod = pref_data.get("time_of_day")
        else:
            # 2) Try JSON dict
            try:
                pref_data = _json.loads(pref_str)
                if isinstance(pref_data, dict):
                    pref_hours = pref_data.get("preferred_hours", [])
                    pref_tod = pref_data.get("time_of_day")
            except (_json.JSONDecodeError, TypeError):
                # 3) Treat as comma-separated hour list
                pref_hours = [h.strip() for h in pref_str.split(",") if h.strip()]

    # Build a set of preferred time strings for fast lookup
    preferred_set: set[str] = set()
    for slot in available:
        _, t = slot.split(" ", 1)
        if t in pref_hours:
            preferred_set.add(slot)
        elif pref_tod == "morning" and is_morning(t):
            preferred_set.add(slot)
        elif pref_tod == "afternoon" and is_afternoon(t):
            preferred_set.add(slot)

    # Sort: preferred slots first, then the rest
    preferred = [s for s in available if s in preferred_set]
    others = [s for s in available if s not in preferred_set]
    sorted_available = preferred + others

    if preferred:
        pref_label = ", ".join(preferred)
        other_label = ", ".join(others) if others else "none"
        return (
            f"Available slots for {day} (preferred first): {pref_label}"
            + (f" | Other slots: {other_label}" if others else "")
        )

    return f"Available slots for {day}: {', '.join(sorted_available)}"


@tool
def book_slot(slot: str) -> str:
    """
    Book a specific coaching slot.
    Use this when the user has chosen a slot and wants to confirm a booking.

    Args:
        slot: The slot to book, in the format 'Day Time' e.g. 'Tue 3PM'.

    Returns:
        A confirmation message or an error if the slot is unavailable.
    """
    if slot in BOOKED_SLOTS:
        return f"Sorry, '{slot}' is already booked. Please choose another slot."

    parts = slot.strip().split()
    if len(parts) != 2:
        return f"Invalid slot format: '{slot}'. Please use format like 'Tue 3PM'."

    day, time = parts
    if time not in COACH_SCHEDULE.get(day, []):
        return f"'{slot}' is not a valid coaching time. Please pick from available slots."

    BOOKED_SLOTS.add(slot)
    return f"✅ Booked! You're confirmed for {slot}. You'll receive a calendar invite shortly."
