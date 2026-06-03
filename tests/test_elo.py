"""Tests for the Elo rating system."""

import pandas as pd

from src import config, elo


def test_expected_score_symmetry():
    assert elo.expected_score(1500, 1500) == 0.5


def test_expected_score_favours_higher_rating():
    assert elo.expected_score(1700, 1500) > 0.5


def test_update_ratings_conserves_points():
    home, away = elo.update_ratings(1500, 1500, home_score=1.0)
    assert round(home + away, 6) == 3000.0


def test_assign_k_tiers():
    assert elo.assign_k("Friendly") == config.ELO_K_WEIGHTS["friendly"]
    assert elo.assign_k("FIFA World Cup qualification") == config.ELO_K_WEIGHTS["qualifier"]
    assert elo.assign_k("FIFA World Cup") == config.ELO_K_WEIGHTS["world_cup"]
    assert elo.assign_k("UEFA Euro") == config.ELO_K_WEIGHTS["continental_final"]
    assert elo.assign_k("UEFA Nations League") == config.ELO_K_WEIGHTS["other_competitive"]


def test_mov_multiplier_log_dampened():
    # a one-goal margin (or draw) is unscaled; bigger margins inflate, dampened
    assert elo.mov_multiplier(1) == 1.0
    assert elo.mov_multiplier(0) == 1.0
    assert elo.mov_multiplier(3) > elo.mov_multiplier(2) > 1.0
    # diminishing returns: 2->3 adds more than 4->5
    assert (elo.mov_multiplier(3) - elo.mov_multiplier(2)) > (
        elo.mov_multiplier(5) - elo.mov_multiplier(4)
    )


def _toy_matches():
    return pd.DataFrame(
        [
            # A beats B at home (non-neutral)
            ("2000-01-01", "A", "B", 2, 0, "Friendly", False),
            # B beats C on neutral ground
            ("2000-02-01", "B", "C", 1, 0, "FIFA World Cup", True),
        ],
        columns=[
            "date", "home_team", "away_team",
            "home_score", "away_score", "tournament", "neutral",
        ],
    ).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_engine_zero_sum_per_match():
    engine = elo.EloRatings().run(_toy_matches())
    total = sum(engine.ratings.values())
    # three teams all started at base; updates are zero-sum
    assert round(total, 6) == round(3 * config.ELO_BASE_RATING, 6)


def test_ratings_as_of_has_no_lookahead():
    engine = elo.EloRatings().run(_toy_matches())
    # As of the first match, C has not played yet and is absent.
    early = engine.ratings_as_of("2000-01-15")
    assert "C" not in early.index
    assert "A" in early.index and "B" in early.index
    # A won its only match, so it should sit above base.
    assert early["A"] > config.ELO_BASE_RATING
