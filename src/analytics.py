"""Minimal, file-backed usage analytics for the admin dashboard.

Tracks anonymous per-session "users" and the number of clicks each makes, with
no authentication and no personal data -- just a random session id generated in
the browser session and a running click count. Persisted as a small JSON file
under ``data/`` (best-effort; on an ephemeral host it resets when the container
restarts, which is fine for a lightweight dashboard).

Storage shape::

    {"users": {"<session_id>": <click_count>, ...}}

so ``len(users)`` is the visitor count, ``sum(users.values())`` the total
clicks, and each entry is that visitor's click count.
"""

from __future__ import annotations

import json

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


def record_visit(session_id: str) -> None:
    """Register a session as a visitor (no-op if already seen)."""
    data = _load()
    if session_id not in data["users"]:
        data["users"][session_id] = 0
        _save(data)


def record_click(session_id: str, count: int = 1) -> None:
    """Increment the click counter for a session (creating it if new)."""
    data = _load()
    data["users"][session_id] = data["users"].get(session_id, 0) + count
    _save(data)


def summary() -> dict:
    """Return aggregate stats for the admin dashboard.

    Keys: ``total_users``, ``total_clicks``, ``avg_clicks`` and ``per_user``
    (a list of {session, clicks} sorted by clicks desc).
    """
    users = _load()["users"]
    total_users = len(users)
    total_clicks = int(sum(users.values()))
    per_user = sorted(
        ({"session": k, "clicks": int(v)} for k, v in users.items()),
        key=lambda r: r["clicks"], reverse=True,
    )
    return {
        "total_users": total_users,
        "total_clicks": total_clicks,
        "avg_clicks": (total_clicks / total_users) if total_users else 0.0,
        "per_user": per_user,
    }
