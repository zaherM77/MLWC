"""File-backed usage analytics for the admin dashboard.

Tracks per-session "users" with click counts, timestamps, and optional user
details (name and message to admin). No authentication, no personal data collection.
Just a random session id generated in the browser session, click count, and
basic session metadata. Persisted as JSON under ``data/``.

Storage shape::

    {
      "users": {
        "<session_id>": {
          "clicks": <int>,
          "start_time": "<ISO timestamp>",
          "last_activity": "<ISO timestamp>",
          "user_name": "<optional name>",
          "message": "<optional message to admin>"
        },
        ...
      }
    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

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





def _normalize_session(session_data: dict | int) -> dict:
    """Normalize old and new session data formats to the new structure."""
    if isinstance(session_data, int):
        # Old format: just a click count
        return {
            "clicks": session_data,
            "start_time": None,
            "last_activity": None,
            "user_name": None,
            "message": None,
        }
    # Already in new format or partial format
    return {
        "clicks": session_data.get("clicks", 0),
        "start_time": session_data.get("start_time"),
        "last_activity": session_data.get("last_activity"),
        "user_name": session_data.get("user_name"),
        "message": session_data.get("message"),
    }



def record_visit(session_id: str, user_name: str | None = None) -> None:
    """Register a session as a visitor with metadata (no-op if already seen).
    
    Args:
        session_id: Unique session identifier
        user_name: Optional user name/identifier
    """
    data = _load()
    if session_id not in data["users"]:
        now = datetime.now(timezone.utc).isoformat()
        data["users"][session_id] = {
            "clicks": 0,
            "start_time": now,
            "last_activity": now,
            "user_name": user_name,
            "message": None,
        }
        _save(data)


def record_click(session_id: str, count: int = 1) -> None:
    """Increment the click counter for a session (creating it if new).
    
    Args:
        session_id: Unique session identifier
        count: Number of clicks to add (default 1)
    """
    data = _load()
    if session_id not in data["users"]:
        record_visit(session_id)
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


def set_user_message(session_id: str, message: str) -> None:
    """Set or update the user message for a session."""
    data = _load()
    if session_id not in data["users"]:
        record_visit(session_id)
        data = _load()  # Reload after visit registration
    
    session_data = _normalize_session(data["users"][session_id])
    session_data["message"] = message if message.strip() else None
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
            "user_name": session_data.get("user_name"),
            "message": session_data.get("message"),
        })
    
    per_user = sorted(per_user, key=lambda r: r["clicks"], reverse=True)
    
    return {
        "total_users": total_users,
        "total_clicks": int(total_clicks),
        "avg_clicks": (total_clicks / total_users) if total_users else 0.0,
        "per_user": per_user,
    }
