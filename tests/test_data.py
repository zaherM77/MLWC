"""Tests for the data loader.

These hit the network on first run, then reuse the cached CSVs in ``data/``.
"""

from src import data

EXPECTED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}


def test_load_matches_returns_expected_shape():
    df = data.load_matches(verbose=False)
    assert len(df) > 40_000
    assert EXPECTED_COLUMNS.issubset(df.columns)
    assert df["home_score"].notna().all()
    assert df["away_score"].notna().all()
    # chronologically sorted
    assert df["date"].is_monotonic_increasing


def test_canonical_team_maps_known_variants():
    assert data.canonical_team("Korea Republic") == "South Korea"
    assert data.canonical_team("USA") == "United States"
    # unknown names pass through unchanged
    assert data.canonical_team("Brazil") == "Brazil"
