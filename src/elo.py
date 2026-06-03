"""International football Elo system.

Ratings are computed by processing the full results history in chronological
order. Every team starts at :data:`config.ELO_BASE_RATING`. Each match applies a
standard Elo update, scaled by:

* a **margin-of-victory** multiplier (log-dampened in the goal difference), and
* a **match-importance** weight ``K`` derived from the ``tournament`` column.

Home advantage is added to the home team's *effective* rating for the
expectation calculation, except when the match is played on neutral ground.

The engine records the rating of every team after every match, which lets it
answer point-in-time queries (a team's Elo as of any date, with no look-ahead).
"""

from __future__ import annotations

import json
import math

import pandas as pd

from . import config

# --- Tournament classification ------------------------------------------------
#
# Continental championship *finals* tournaments (qualifiers are handled
# separately by the "qualification" keyword).
CONTINENTAL_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "Oceania Nations Cup",
    "Confederations Cup",
    "FIFA Confederations Cup",
}


def assign_k(tournament: str) -> float:
    """Map a ``tournament`` label to its match-importance weight ``K``."""
    weights = config.ELO_K_WEIGHTS
    if not isinstance(tournament, str):
        return weights["friendly"]

    t = tournament.strip()
    if t == "Friendly":
        return weights["friendly"]
    if "qualification" in t.lower():
        return weights["qualifier"]
    if t == "FIFA World Cup":
        return weights["world_cup"]
    if t in CONTINENTAL_FINALS:
        return weights["continental_final"]
    return weights["other_competitive"]


# --- Core Elo maths -----------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for A vs B given their ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(goal_diff: int) -> float:
    """Log-dampened margin-of-victory multiplier.

    A one-goal margin (or a draw) leaves the change unscaled; each additional
    goal inflates it by a diminishing amount: 1 -> 1.0, 2 -> 1.69, 3 -> 2.10,
    5 -> 2.61.
    """
    return 1.0 + math.log(max(abs(goal_diff), 1))


def result_score(home_score: int, away_score: int) -> float:
    """Realised result for the home team: 1.0 win, 0.5 draw, 0.0 loss."""
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def match_delta(
    rating_home: float,
    rating_away: float,
    home_score: int,
    away_score: int,
    k: float,
    neutral: bool,
    home_advantage: float = config.ELO_HOME_ADVANTAGE,
) -> float:
    """Rating change applied to the home team for a single match.

    Combines the importance weight ``k``, the log-dampened margin-of-victory
    multiplier, and home advantage (skipped on neutral ground). The away team
    receives the negative of this delta (zero-sum). This is the single source of
    truth for the Elo update, shared by the engine and the feature builder.
    """
    eff_home = rating_home + (0.0 if neutral else home_advantage)
    expected_home = expected_score(eff_home, rating_away)
    score = result_score(home_score, away_score)
    mult = mov_multiplier(home_score - away_score)
    return k * mult * (score - expected_home)


def update_ratings(
    rating_home: float,
    rating_away: float,
    home_score: float,
    k: float = config.ELO_K_FACTOR,
    home_advantage: float = config.ELO_HOME_ADVANTAGE,
) -> tuple[float, float]:
    """Update home/away ratings after a match (zero-sum, no margin scaling).

    ``home_score`` is the realised result for the home team (1=win, 0.5=draw,
    0=loss). Retained for simple/standalone use and unit tests.
    """
    expected_home = expected_score(rating_home + home_advantage, rating_away)
    delta = k * (home_score - expected_home)
    return rating_home + delta, rating_away - delta


# --- Chronological engine -----------------------------------------------------


class EloRatings:
    """Chronological Elo engine over the full match history."""

    def __init__(
        self,
        base: float = config.ELO_BASE_RATING,
        home_advantage: float = config.ELO_HOME_ADVANTAGE,
    ):
        self.base = base
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}
        self.matches_played: dict[str, int] = {}
        # One record per team per match: the rating *after* that match.
        self._history: list[dict] = []
        self._history_df: pd.DataFrame | None = None

    def _rating(self, team: str) -> float:
        """Current rating for ``team`` (base rating if unseen so far)."""
        return self.ratings.get(team, self.base)

    def run(self, matches: pd.DataFrame) -> "EloRatings":
        """Process all matches in chronological order.

        ``matches`` must contain: date, home_team, away_team, home_score,
        away_score, tournament, neutral.
        """
        df = matches.sort_values("date")
        records = []
        for row in df.itertuples(index=False):
            home, away = row.home_team, row.away_team
            r_home, r_away = self._rating(home), self._rating(away)

            k = assign_k(row.tournament)
            delta = match_delta(
                r_home, r_away, row.home_score, row.away_score,
                k=k, neutral=bool(row.neutral), home_advantage=self.home_advantage,
            )
            new_home, new_away = r_home + delta, r_away - delta
            self.ratings[home] = new_home
            self.ratings[away] = new_away
            self.matches_played[home] = self.matches_played.get(home, 0) + 1
            self.matches_played[away] = self.matches_played.get(away, 0) + 1

            records.append({"date": row.date, "team": home, "rating": new_home})
            records.append({"date": row.date, "team": away, "rating": new_away})

        self._history = records
        self._history_df = pd.DataFrame(records)
        return self

    def current_table(self) -> pd.DataFrame:
        """Final current ratings as a table, sorted by Elo descending."""
        rows = [
            {
                "team": team,
                "elo": rating,
                "matches_played": self.matches_played.get(team, 0),
            }
            for team, rating in self.ratings.items()
        ]
        table = pd.DataFrame(rows).sort_values("elo", ascending=False)
        return table.reset_index(drop=True)

    def ratings_as_of(self, date) -> pd.Series:
        """Each team's Elo as of ``date`` (inclusive), with no look-ahead.

        Returns a Series indexed by team, sorted descending. Teams that had not
        yet played by ``date`` are omitted (they sit at the base rating).
        """
        if self._history_df is None:
            raise RuntimeError("call run() before querying ratings_as_of()")

        cutoff = pd.Timestamp(date)
        past = self._history_df[self._history_df["date"] <= cutoff]
        if past.empty:
            return pd.Series(dtype=float)
        latest = past.sort_values("date").groupby("team")["rating"].last()
        return latest.sort_values(ascending=False)

    def save(self, path=config.ELO_CURRENT_PATH) -> None:
        """Persist the final current ratings to JSON."""
        table = self.current_table()
        as_of = (
            pd.to_datetime(self._history_df["date"]).max().date().isoformat()
            if self._history_df is not None and not self._history_df.empty
            else None
        )
        payload = {
            "as_of": as_of,
            "base_rating": self.base,
            "home_advantage": self.home_advantage,
            "n_teams": len(table),
            "ratings": {
                r.team: round(r.elo, 2) for r in table.itertuples(index=False)
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)


# --- Convenience entry point --------------------------------------------------


def compute_elo(
    matches: pd.DataFrame | None = None,
    persist: bool = True,
    top_n: int = 25,
    verbose: bool = True,
) -> EloRatings:
    """Build Elo ratings from match history, persist, and print the top teams."""
    if matches is None:
        from . import data

        matches = data.load_matches(verbose=False)

    engine = EloRatings().run(matches)

    if persist:
        engine.save()

    if verbose:
        table = engine.current_table().head(top_n)
        print(f"Top {top_n} teams by current Elo:\n")
        for i, r in enumerate(table.itertuples(index=False), start=1):
            print(f"{i:>2}. {r.team:<22} {r.elo:7.1f}  ({r.matches_played} matches)")

    return engine


if __name__ == "__main__":
    compute_elo()
