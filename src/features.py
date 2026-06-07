from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

from . import config, elo

FORM_WINDOW = 10
DEFAULT_START_YEAR = 2002

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_gf_avg",
    "home_ga_avg",
    "away_gf_avg",
    "away_ga_avg",
    "neutral",
    "tier",
]
TARGET_COLUMNS = ["home_score", "away_score"]


def importance_tier(tournament: str) -> int:

    k = elo.assign_k(tournament)
    w = config.ELO_K_WEIGHTS
    if k <= w["friendly"]:
        return 0
    if k >= w["continental_final"]:
        return 2
    return 1


def _form_avg(window: deque) -> tuple[float, float]:

    if not window:
        return float("nan"), float("nan")
    gf = sum(g for g, _ in window) / len(window)
    ga = sum(a for _, a in window) / len(window)
    return gf, ga


def build_features(
    matches: pd.DataFrame | None = None,
    start_year: int = DEFAULT_START_YEAR,
    form_window: int = FORM_WINDOW,
) -> pd.DataFrame:

    if matches is None:
        from . import data

        matches = data.load_matches(verbose=False)

    df = matches.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=form_window))

    rows = []
    for m in df.itertuples(index=False):
        home, away = m.home_team, m.away_team
        r_home = ratings.get(home, config.ELO_BASE_RATING)
        r_away = ratings.get(away, config.ELO_BASE_RATING)

        home_gf, home_ga = _form_avg(form[home])
        away_gf, away_ga = _form_avg(form[away])

        if m.date.year >= start_year:
            rows.append(
                {
                    "date": m.date,
                    "home_team": home,
                    "away_team": away,
                    "home_elo": r_home,
                    "away_elo": r_away,
                    "elo_diff": r_home - r_away,
                    "home_gf_avg": home_gf,
                    "home_ga_avg": home_ga,
                    "away_gf_avg": away_gf,
                    "away_ga_avg": away_ga,
                    "neutral": int(bool(m.neutral)),
                    "tier": importance_tier(m.tournament),
                    "home_score": int(m.home_score),
                    "away_score": int(m.away_score),
                }
            )

        # --- update state AFTER recording (point-in-time, no leakage) ---
        k = elo.assign_k(m.tournament)
        delta = elo.match_delta(
            r_home, r_away, m.home_score, m.away_score,
            k=k, neutral=bool(m.neutral),
        )
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
        form[home].append((m.home_score, m.away_score))
        form[away].append((m.away_score, m.home_score))

    return pd.DataFrame(rows)