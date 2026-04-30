"""
database.py — Read-only PostgreSQL access for coach information.

Reads from the `coaches` table that Group 1 populates.
Uses short-lived connections (connect, query, close) to keep things simple.

Actual schema:
    id, name, email, timezone, expertise, calcom_user_id, calcom_username,
    google_access_token, google_refresh_token, google_token_expiry,
    is_active, created_at, updated_at
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def _connect():
    """Open a new connection to the coaching database."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def get_coach(coach_name: str | None = None) -> dict | None:
    """
    Return a single coach record as a dict.

    If coach_name is provided, looks up by name (case-insensitive).
    If omitted, returns the first active coach found.

    Returns dict with keys from the coaches table, or None if not found.
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            if coach_name:
                cur.execute(
                    "SELECT * FROM coaches WHERE LOWER(name) = LOWER(%s) AND is_active = true LIMIT 1",
                    (coach_name,),
                )
            else:
                cur.execute("SELECT * FROM coaches WHERE is_active = true LIMIT 1")
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def list_coaches() -> list[dict]:
    """Return all active coaches as a list of dicts."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM coaches WHERE is_active = true ORDER BY name")
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
