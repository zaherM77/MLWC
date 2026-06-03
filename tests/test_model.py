"""Tests for forecasting metrics, feature integrity, and the time split."""

import numpy as np
import pandas as pd

from src import features, model


def test_rps_perfect_forecast_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    outcomes = np.array([0, 1])
    assert model.ranked_probability_score(probs, outcomes) == 0.0


def test_rps_rewards_adjacent_over_distant():
    # true outcome = home win (0). A draw-heavy forecast (adjacent) should beat
    # an away-heavy one (distant), even with equal mass off the truth.
    adjacent = np.array([[0.0, 1.0, 0.0]])  # all on draw
    distant = np.array([[0.0, 0.0, 1.0]])   # all on away win
    truth = np.array([0])
    assert model.ranked_probability_score(adjacent, truth) < model.ranked_probability_score(
        distant, truth
    )


def test_outcome_probs_sum_to_one():
    p = model.outcome_probs_from_goals(1.5, 1.1, rho=-0.05)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p >= 0).all()


def test_time_split_has_no_overlap():
    feats = pd.DataFrame({"date": pd.to_datetime(["2021-12-31", "2022-01-01"])})
    train, test = model.time_split(feats, test_from_year=2022)
    assert train["date"].dt.year.max() < 2022
    assert test["date"].dt.year.min() >= 2022


def test_features_are_point_in_time():
    # Two matches; the second match's features must reflect ONLY the first.
    matches = pd.DataFrame(
        [
            ("2001-01-01", "A", "B", 3, 0, "Friendly", False),
            ("2002-01-01", "A", "B", 0, 0, "Friendly", False),
        ],
        columns=["date", "home_team", "away_team", "home_score",
                 "away_score", "tournament", "neutral"],
    ).assign(date=lambda d: pd.to_datetime(d["date"]))

    feats = features.build_features(matches, start_year=2002)
    assert len(feats) == 1  # only the 2002 match is emitted
    row = feats.iloc[0]
    # A's form going into 2002 reflects the 3-0 win in 2001, not the 0-0.
    assert row["home_gf_avg"] == 3.0
    assert row["home_ga_avg"] == 0.0
    # A won in 2001, so its pre-match Elo should exceed B's.
    assert row["elo_diff"] > 0
