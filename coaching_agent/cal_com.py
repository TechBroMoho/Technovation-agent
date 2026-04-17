"""
cal_com.py — Cal.com v2 API wrapper for slot discovery and booking.

All functions return plain dicts. On failure they return {"error": "..."}.
No exceptions are propagated to the caller.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

CAL_API_KEY = os.getenv("CAL_COM_API_KEY")
BASE_URL = "https://api.cal.com/v2"


def _headers(api_version: str = "2024-09-04") -> dict:
    return {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "cal-api-version": api_version,
        "Content-Type": "application/json",
    }


def fetch_available_slots(
    event_type_id: int,
    start_time: str,
    end_time: str,
) -> dict:
    """
    Fetch available time slots from Cal.com.

    Args:
        event_type_id: Numeric event type ID.
        start_time:    ISO 8601 start of the window (e.g. "2026-04-10T00:00:00Z").
        end_time:      ISO 8601 end of the window (e.g. "2026-04-10T23:59:59Z").

    Returns:
        On success: {"slots": {"2026-04-10": [{"start": "..."}, ...], ...}}
        On failure: {"error": "..."}
    """
    try:
        resp = requests.get(
            f"{BASE_URL}/slots",
            headers=_headers(),
            params={
                "eventTypeId": event_type_id,
                "start": start_time,
                "end": end_time,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return {"error": f"Cal.com API returned {resp.status_code}: {resp.text}"}
        data = resp.json()
        return {"slots": data.get("data", {})}
    except requests.RequestException as exc:
        return {"error": f"Cal.com request failed: {exc}"}


def create_booking(
    start_time: str,
    event_type_id: int,
    event_type_slug: str,
    username: str,
    attendee_name: str,
    attendee_email: str,
    attendee_timezone: str,
) -> dict:
    """
    Create a booking on Cal.com.

    Args:
        start_time:        ISO 8601 start time of the slot (e.g. "2026-04-10T10:00:00Z").
        event_type_id:     Numeric event type ID from the coaches table.
        event_type_slug:   Event type slug.
        username:          Cal.com username of the coach.
        attendee_name:     Name of the person booking.
        attendee_email:    Email of the person booking.
        attendee_timezone: IANA timezone of the attendee.

    Returns:
        On success: the booking data dict from Cal.com.
        On failure: {"error": "..."}
    """
    try:
        payload = {
            "start": start_time,
            "eventTypeId": event_type_id,
            "eventTypeSlug": event_type_slug,
            "username": username,
            "attendee": {
                "name": attendee_name,
                "email": attendee_email,
                "timeZone": attendee_timezone,
            },
        }
        resp = requests.post(
            f"{BASE_URL}/bookings",
            headers=_headers("2024-08-13"),
            json=payload,
            timeout=15,
        )
        if resp.status_code not in (200, 201):
            return {"error": f"Cal.com booking failed ({resp.status_code}): {resp.text}"}
        return resp.json().get("data", {})
    except requests.RequestException as exc:
        return {"error": f"Cal.com booking request failed: {exc}"}
