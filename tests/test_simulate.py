"""Tests for match simulation.

Uses a synthetic two-team state (offline, fast) plus the saved model, so the
test exercises the real model without re-downloading or replaying history.
"""

import numpy as np

from src import simulate

# A clearly stronger team vs a clearly weaker one.
STATE = simulate.MatchupState(
    ratings={"Strong": 2000.0, "Weak": 1300.0},
    form_gf={"Strong": 2.5, "Weak": 0.6},
    form_ga={"Strong": 0.4, "Weak": 2.3},
)


def test_stronger_team_wins_distinctly_more_than_half():
    out = simulate.simulate_matches(
        "Strong", "Weak", neutral=True, n=5000, state=STATE, seed=1
    )
    win_rate = np.mean(out["winner"] == "Strong")
    assert win_rate > 0.6, f"expected stronger team to dominate, got {win_rate:.3f}"


def test_single_match_returns_ints_and_valid_winner():
    res = simulate.simulate_match("Strong", "Weak", neutral=True, state=STATE, seed=7)
    assert isinstance(res.goals_a, int) and isinstance(res.goals_b, int)
    assert res.winner in {"Strong", "Weak", "draw"}


def test_knockout_never_draws():
    out = simulate.simulate_matches(
        "Strong", "Weak", neutral=True, knockout=True, n=2000, state=STATE, seed=3
    )
    assert not np.any(out["winner"] == "draw")
    # every match must have a winner among the two teams
    assert set(np.unique(out["winner"])).issubset({"Strong", "Weak"})
