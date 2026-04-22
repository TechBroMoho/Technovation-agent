"""
cal_com.py — Cal.com v2 API wrapper for slot discovery and booking.

All functions return plain dicts. On failure they return {"error": "..."}.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Supports both naming conventions across agent and backend .env files
CAL_API_KEY = os.getenv("CAL_COM_API_KEY") or os.getenv("CALCOM_API_KEY")
BASE_URL = "https://api.cal.com/v2"


def _headers(api_version: str = "2024-08-13") -> dict:
    return {
        "Authorization":   f"Bearer {CAL_API_KEY}",
        "cal-api-version": api_version,
        "Content-Type":    "application/json",
    }


def fetch_available_slots(
    event_type_id: int,
    start_time: str,
    end_time: str,
) -> dict:
    """
    Fetch available slots from Cal.com v2.

    Args:
        event_type_id: Numeric event type ID.
        start_time:    ISO 8601 e.g. "2026-04-10T00:00:00Z"
        end_time:      ISO 8601 e.g. "2026-04-10T23:59:59Z"

    Returns:
        {"slots": {"2026-04-10": [{"start": "..."}, ...], ...}}
        or {"error": "..."}
    """
    if not CAL_API_KEY:
        return {"error": "CAL_COM_API_KEY / CALCOM_API_KEY not set in .env"}

    try:
        resp = requests.get(
            f"{BASE_URL}/slots/available",
            headers=_headers(),
            params={
                "eventTypeId": event_type_id,
                "startTime":   start_time,
                "endTime":     end_time,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return {"error": f"Cal.com API returned {resp.status_code}: {resp.text}"}
        data = resp.json()
        inner = data.get("data", data)
        return {"slots": inner.get("slots", inner)}
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
    Create a booking on Cal.com v2.

    Returns booking data dict or {"error": "..."}
    """
    if not CAL_API_KEY:
        return {"error": "CAL_COM_API_KEY / CALCOM_API_KEY not set in .env"}

    try:
        payload = {
            "start":       start_time,
            "eventTypeId": event_type_id,
            "attendee": {
                "name":     attendee_name,
                "email":    attendee_email,
                "timeZone": attendee_timezone,
                "language": "en",
            },
            "metadata": {},
        }
        resp = requests.post(
            f"{BASE_URL}/bookings",
            headers=_headers("2024-08-13"),
            json=payload,
            timeout=15,
        )
        if resp.status_code not in (200, 201):
            return {"error": f"Cal.com booking failed ({resp.status_code}): {resp.text}"}
        data = resp.json()
        return data.get("data", data)
    except requests.RequestException as exc:
        return {"error": f"Cal.com booking request failed: {exc}"}


def get_booking(uid: str) -> dict:
    """Fetch a single booking by UID. Useful for follow-up checks."""
    if not CAL_API_KEY:
        return {"error": "CAL_COM_API_KEY / CALCOM_API_KEY not set in .env"}
    try:
        resp = requests.get(f"{BASE_URL}/bookings/{uid}", headers=_headers(), timeout=15)
        if resp.status_code != 200:
            return {"error": f"Cal.com API returned {resp.status_code}: {resp.text}"}
        return resp.json().get("data", {})
    except requests.RequestException as exc:
        return {"error": f"Cal.com request failed: {exc}"}


def cancel_booking(uid: str, reason: str = "") -> dict:
    """Cancel a booking by UID."""
    if not CAL_API_KEY:
        return {"error": "CAL_COM_API_KEY / CALCOM_API_KEY not set in .env"}
    try:
        resp = requests.post(
            f"{BASE_URL}/bookings/{uid}/cancel",
            headers=_headers(),
            json={"cancellationReason": reason},
            timeout=15,
        )
        if resp.status_code not in (200, 201):
            return {"error": f"Cal.com cancel failed ({resp.status_code}): {resp.text}"}
        return resp.json().get("data", {})
    except requests.RequestException as exc:
        return {"error": f"Cal.com request failed: {exc}"}
