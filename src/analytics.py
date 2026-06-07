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
        pass 





def _normalize_session(session_data: dict | int) -> dict:
    if isinstance(session_data, int):
        return {
            "clicks": session_data,
            "start_time": None,
            "last_activity": None,
            "user_name": None,
            "message": None,
        }
    return {
        "clicks": session_data.get("clicks", 0),
        "start_time": session_data.get("start_time"),
        "last_activity": session_data.get("last_activity"),
        "user_name": session_data.get("user_name"),
        "message": session_data.get("message"),
    }



def record_visit(session_id: str, user_name: str | None = None) -> None:

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
    data = _load()
    if session_id not in data["users"]:
        record_visit(session_id, user_name=user_name)
    else:
        session_data = _normalize_session(data["users"][session_id])
        session_data["user_name"] = user_name
        data["users"][session_id] = session_data
        _save(data)


def set_user_message(session_id: str, message: str) -> None:
    data = _load()
    if session_id not in data["users"]:
        record_visit(session_id)
        data = _load()
    
    session_data = _normalize_session(data["users"][session_id])
    session_data["message"] = message if message.strip() else None
    data["users"][session_id] = session_data
    _save(data)




def summary() -> dict:

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