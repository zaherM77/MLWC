from src import analytics, config


def test_visit_and_click_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ANALYTICS_PATH", tmp_path / "analytics.json")

    analytics.record_visit("u1")
    analytics.record_visit("u1") 
    analytics.record_click("u1")
    analytics.record_click("u1")
    analytics.record_visit("u2")
    analytics.record_click("u2")

    s = analytics.summary()
    assert s["total_users"] == 2
    assert s["total_clicks"] == 3
    assert abs(s["avg_clicks"] - 1.5) < 1e-9
    top = s["per_user"][0]
    assert top["session"] == "u1" and top["clicks"] == 2


def test_summary_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ANALYTICS_PATH", tmp_path / "none.json")
    s = analytics.summary()
    assert s == {"total_users": 0, "total_clicks": 0, "avg_clicks": 0.0, "per_user": []}