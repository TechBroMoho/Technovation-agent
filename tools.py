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


def _day_label(offset: int) -> str:
    target = datetime.now() + timedelta(days=offset)
    return target.strftime("%a")


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------

@tool
def get_available_slots(day: str, time_of_day: str = "any") -> str:
    """
    Get available coaching slots for a given day.
    Use this when the user asks about availability or open times.

    Args:
        day: The day to check. Use 'today', 'tomorrow', or a weekday name like 'Tuesday'.
        time_of_day: Filter by 'morning', 'afternoon', or 'any'.

    Returns:
        A formatted string listing available slots, or a message if none are found.
    """
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

    return f"Available slots for {day}: {', '.join(available)}"


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
