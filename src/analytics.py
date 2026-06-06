"""File-backed usage analytics for the admin dashboard.

Tracks per-session "users" with click counts, timestamps, location data, and
optional user names. No authentication, minimal personal data collection --
just a random session id generated in the browser session, click count, and
basic session metadata. Persisted as JSON under ``data/``.

Storage shape::

    {
      "users": {
        "<session_id>": {
          "clicks": <int>,
          "start_time": "<ISO timestamp>",
          "last_activity": "<ISO timestamp>",
          "location": {
            "country": "<country>",
            "city": "<city>",
            "ip": "<ip>"
          },
          "user_name": "<optional name>"
        },
        ...
      }
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import requests

from . import config


def _load() -> dict:
    path = config.ANALYTICS_PATH
    if not path.exists():
        return {"users": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("users", {})
        return data
    except (OSError, json.JSONDecodeError):
        return {"users": {}}


def _save(data: dict) -> None:
    path = config.ANALYTICS_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # analytics are best-effort; never break the app over them


def _get_geolocation(ip: str | None = None) -> dict:
    """Fetch geolocation data for an IP address (best-effort)."""
    if not ip:
        return {"country": "Unknown", "city": "Unknown", "ip": ip}
    
    try:
        # Using ipapi.co which has a free tier with no auth required
        response = requests.get(f"https://ipapi.co/{ip}/json/", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return {
                "country": data.get("country_name", "Unknown"),
                "city": data.get("city", "Unknown"),
                "ip": ip,
            }
    except Exception:
        pass
    
    return {"country": "Unknown", "city": "Unknown", "ip": ip}


def _normalize_session(session_data: dict | int) -> dict:
    """Normalize old and new session data formats to the new structure."""
    if isinstance(session_data, int):
        # Old format: just a click count
        return {
            "clicks": session_data,
            "start_time": None,
            "last_activity": None,
            "location": {"country": "Unknown", "city": "Unknown", "ip": None},
            "user_name": None,
        }
    # Already in new format or partial format
    return {
        "clicks": session_data.get("clicks", 0),
        "start_time": session_data.get("start_time"),
        "last_activity": session_data.get("last_activity"),
        "location": session_data.get("location", {"country": "Unknown", "city": "Unknown", "ip": None}),
        "user_name": session_data.get("user_name"),
    }



def record_visit(session_id: str, ip: str | None = None, user_name: str | None = None) -> None:
    """Register a session as a visitor with metadata (no-op if already seen).
    
    Args:
        session_id: Unique session identifier
        ip: Optional IP address for geolocation
        user_name: Optional user name/identifier
    """
    data = _load()
    if session_id not in data["users"]:
        now = datetime.now(timezone.utc).isoformat()
        location = _get_geolocation(ip)
        data["users"][session_id] = {
            "clicks": 0,
            "start_time": now,
            "last_activity": now,
            "location": location,
            "user_name": user_name,
        }
        _save(data)


def record_click(session_id: str, count: int = 1, ip: str | None = None) -> None:
    """Increment the click counter for a session (creating it if new).
    
    Args:
        session_id: Unique session identifier
        count: Number of clicks to add (default 1)
        ip: Optional IP address for geolocation
    """
    data = _load()
    if session_id not in data["users"]:
        record_visit(session_id, ip=ip)
        data = _load()  # Reload after visit registration
    
    session_data = data["users"][session_id]
    session_data = _normalize_session(session_data)
    session_data["clicks"] = session_data.get("clicks", 0) + count
    session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
    data["users"][session_id] = session_data
    _save(data)


def set_user_name(session_id: str, user_name: str) -> None:
    """Set or update the user name for a session."""
    data = _load()
    if session_id not in data["users"]:
        record_visit(session_id, user_name=user_name)
    else:
        session_data = _normalize_session(data["users"][session_id])
        session_data["user_name"] = user_name
        data["users"][session_id] = session_data
        _save(data)



def summary() -> dict:
    """Return aggregate stats for the admin dashboard.

    Keys: ``total_users``, ``total_clicks``, ``avg_clicks`` and ``per_user``
    (a list of detailed session data sorted by clicks desc).
    """
    users_data = _load()["users"]
    total_users = len(users_data)
    total_clicks = 0
    per_user = []
    
    for session_id, session_data in users_data.items():
        session_data = _normalize_session(session_data)
        clicks = session_data.get("clicks", 0)
        total_clicks += clicks
        
        per_user.append({
            "session": session_id,
            "clicks": int(clicks),
            "start_time": session_data.get("start_time"),
            "last_activity": session_data.get("last_activity"),
            "country": session_data.get("location", {}).get("country", "Unknown"),
            "city": session_data.get("location", {}).get("city", "Unknown"),
            "user_name": session_data.get("user_name"),
        })
    
    per_user = sorted(per_user, key=lambda r: r["clicks"], reverse=True)
    
    return {
        "total_users": total_users,
        "total_clicks": int(total_clicks),
        "avg_clicks": (total_clicks / total_users) if total_users else 0.0,
        "per_user": per_user,
    }
