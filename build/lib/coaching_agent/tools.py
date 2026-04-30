"""
tools.py — LangChain tools for the coaching scheduling agent.

Uses real Cal.com API for slot discovery/booking and PostgreSQL for coach info.
"""

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain.tools import tool
from zoneinfo import ZoneInfo

from coaching_agent.database import get_coach
from coaching_agent.cal_com import fetch_available_slots, create_booking

load_dotenv()

# Attendee info loaded once from environment
ATTENDEE_NAME = os.getenv("ATTENDEE_NAME", "Student")
ATTENDEE_EMAIL = os.getenv("ATTENDEE_EMAIL", "")
ATTENDEE_TIMEZONE = os.getenv("ATTENDEE_TIMEZONE", "America/New_York")

# Cal.com event defaults (not stored in DB yet)
DEFAULT_EVENT_SLUG = os.getenv("CAL_EVENT_SLUG", "coaching-session")
DEFAULT_EVENT_TYPE_ID = int(os.getenv("CAL_EVENT_TYPE_ID", "0"))


def _resolve_day(day: str) -> tuple[str, str]:
    """
    Turn a human day reference into an ISO date window (start, end).

    Accepts: 'today', 'tomorrow', a weekday name like 'Tuesday',
    or an ISO date like '2026-04-10'.

    Returns (start_iso, end_iso) — both in UTC with Z suffix.
    Uses the attendee timezone so that 'today'/'tomorrow' match the user's local date.
    """
    day_clean = day.strip().lower()
    tz = ZoneInfo(ATTENDEE_TIMEZONE)
    now = datetime.now(tz)

    if day_clean == "today":
        target = now
    elif day_clean == "tomorrow":
        target = now + timedelta(days=1)
    else:
        # Try ISO date first
        try:
            target = datetime.fromisoformat(day_clean)
        except ValueError:
            # Weekday name — find next occurrence
            weekdays = {
                "monday": 0, "tuesday": 1, "wednesday": 2,
                "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
            }
            target_wd = weekdays.get(day_clean)
            if target_wd is None:
                # Try 3-letter abbreviation
                abbrevs = {
                    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
                    "fri": 4, "sat": 5, "sun": 6,
                }
                target_wd = abbrevs.get(day_clean[:3])
            if target_wd is None:
                target = now
            else:
                days_ahead = (target_wd - now.weekday()) % 7
                target = now + timedelta(days=days_ahead)

    # Build start-of-day and end-of-day in the user's local timezone,
    # then convert to UTC so the Cal.com query window matches the user's actual day.
    local_start = target.replace(hour=0, minute=0, second=0, microsecond=0)
    if local_start.tzinfo is None:
        local_start = local_start.replace(tzinfo=tz)
    local_end = local_start + timedelta(days=1) - timedelta(seconds=1)

    utc_start = local_start.astimezone(ZoneInfo("UTC"))
    utc_end = local_end.astimezone(ZoneInfo("UTC"))

    start = utc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = utc_end.strftime("%Y-%m-%dT%H:%M:%SZ")
    return start, end


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------

@tool
def get_available_slots(day: str, coach_name: str = "") -> str:
    """
    Get available coaching slots for a given day from Cal.com.
    Use this when the user asks about availability or open times.

    Args:
        day: The day to check. Use 'today', 'tomorrow', a weekday name
             like 'Tuesday', or an ISO date like '2026-04-10'.
        coach_name: Optional coach name. If omitted, uses the default coach.

    Returns:
        A formatted string listing available ISO 8601 time slots, or an error message.
    """
    coach = get_coach(coach_name if coach_name else None)
    if not coach:
        return "Error: No coach found in the database. Please check your setup."

    start, end = _resolve_day(day)

    result = fetch_available_slots(
        event_type_id=DEFAULT_EVENT_TYPE_ID,
        start_time=start,
        end_time=end,
    )

    if "error" in result:
        return f"Error fetching availability: {result['error']}"

    # Flatten all slots, converting UTC to the attendee's local timezone.
    tz = ZoneInfo(ATTENDEE_TIMEZONE)
    tz_abbrev = datetime.now(tz).strftime("%Z")
    all_slots = []
    for date_key, times in result.get("slots", {}).items():
        if isinstance(times, list):
            for slot in times:
                utc_str = slot.get("start", "") if isinstance(slot, dict) else slot
                if not utc_str:
                    continue
                utc_dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                local_dt = utc_dt.astimezone(tz)
                local_label = local_dt.strftime("%I:%M %p").lstrip("0")
                # Include both local time and UTC so the LLM can display local
                # but still pass the exact UTC string to book_slot.
                all_slots.append(f"{local_label} {tz_abbrev} (UTC: {utc_str})")

    if not all_slots:
        return f"No available slots found for {day}."

    return f"Available slots for {day}:\n" + "\n".join(all_slots)


@tool
def book_slot(slot_time: str, coach_name: str = "") -> str:
    """
    Book a specific coaching slot via Cal.com.
    Use this when the user has chosen a slot and wants to confirm a booking.

    Args:
        slot_time: The ISO 8601 timestamp of the slot to book
                   (e.g. '2026-04-10T10:00:00Z'). Use the exact timestamp
                   from get_available_slots results.
        coach_name: Optional coach name. If omitted, uses the default coach.

    Returns:
        A confirmation message or an error if the booking fails.
    """
    coach = get_coach(coach_name if coach_name else None)
    if not coach:
        return "Error: No coach found in the database. Please check your setup."

    username = coach["calcom_username"]
    event_slug = DEFAULT_EVENT_SLUG
    event_type_id = DEFAULT_EVENT_TYPE_ID

    if not event_type_id:
        return "Error: CAL_EVENT_TYPE_ID is not set in .env. Please configure it."

    result = create_booking(
        start_time=slot_time,
        event_type_id=event_type_id,
        event_type_slug=event_slug,
        username=username,
        attendee_name=ATTENDEE_NAME,
        attendee_email=ATTENDEE_EMAIL,
        attendee_timezone=ATTENDEE_TIMEZONE,
    )

    if isinstance(result, dict) and "error" in result:
        return f"Booking failed: {result['error']}"

    booking_id = result.get("id", result.get("uid", "unknown"))
    return (
        f"Booked! Your coaching session at {slot_time} is confirmed. "
        f"Booking ID: {booking_id}. You'll receive a calendar invite at {ATTENDEE_EMAIL}."
    )
