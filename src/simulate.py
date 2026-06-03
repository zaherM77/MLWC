"""Monte Carlo simulation of matches, seasons, and tournaments.

A matchup is turned into a scoreline distribution by:

1. building the point-in-time feature vector (current Elo + recent form),
2. asking the saved model for expected goals (lambda_a, lambda_b),
3. drawing a scoreline -- independent Poisson per side, or, if the saved model
   is the Dixon-Coles bivariate Poisson, sampling from its low-score-corrected
   joint distribution.

Knockout ties are resolved by extra time (lambdas scaled to 30 minutes) and, if
still level, a penalty shootout weighted toward the stronger side.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import config, elo, features as features_mod, model as model_mod

DEFAULT_TIER = 2  # simulations are usually for tournaments (major)


class MatchResult(NamedTuple):
    """Outcome of a single simulated match."""

    goals_a: int
    goals_b: int
    winner: str  # team name, or "draw" (only possible when knockout=False)


# --- Current-state provider ---------------------------------------------------


@dataclass
class MatchupState:
    """Latest per-team Elo rating and recent-form averages from history."""

    ratings: dict[str, float] = field(default_factory=dict)
    form_gf: dict[str, float] = field(default_factory=dict)
    form_ga: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_history(
        cls, matches: pd.DataFrame, form_window: int = features_mod.FORM_WINDOW
    ) -> "MatchupState":
        """Replay all matches chronologically and keep each team's final state."""
        df = matches.sort_values("date")
        ratings: dict[str, float] = {}
        form: dict[str, deque] = defaultdict(lambda: deque(maxlen=form_window))

        for m in df.itertuples(index=False):
            home, away = m.home_team, m.away_team
            r_home = ratings.get(home, config.ELO_BASE_RATING)
            r_away = ratings.get(away, config.ELO_BASE_RATING)
            k = elo.assign_k(m.tournament)
            delta = elo.match_delta(
                r_home, r_away, m.home_score, m.away_score,
                k=k, neutral=bool(m.neutral),
            )
            ratings[home] = r_home + delta
            ratings[away] = r_away - delta
            form[home].append((m.home_score, m.away_score))
            form[away].append((m.away_score, m.home_score))

        form_gf = {t: float(np.mean([g for g, _ in w])) for t, w in form.items()}
        form_ga = {t: float(np.mean([a for _, a in w])) for t, w in form.items()}
        return cls(ratings=ratings, form_gf=form_gf, form_ga=form_ga)

    def elo(self, team: str) -> float:
        """Current Elo for ``team`` (base rating if it has no history)."""
        return self.ratings.get(team, config.ELO_BASE_RATING)

    def feature_row(
        self, team_a: str, team_b: str, neutral: bool, tier: int
    ) -> pd.DataFrame:
        """One-row feature frame for ``team_a`` (home) vs ``team_b`` (away)."""
        ea, eb = self.elo(team_a), self.elo(team_b)
        nan = float("nan")
        row = {
            "home_elo": ea,
            "away_elo": eb,
            "elo_diff": ea - eb,
            "home_gf_avg": self.form_gf.get(team_a, nan),
            "home_ga_avg": self.form_ga.get(team_a, nan),
            "away_gf_avg": self.form_gf.get(team_b, nan),
            "away_ga_avg": self.form_ga.get(team_b, nan),
            "neutral": int(bool(neutral)),
            "tier": int(tier),
        }
        return pd.DataFrame([row], columns=features_mod.FEATURE_COLUMNS)


# Lazily-built, cached singletons so repeated sims don't reload everything.
_STATE: MatchupState | None = None
_MODEL = None


def get_state() -> MatchupState:
    """Return the cached current-state, building it from history on first use."""
    global _STATE
    if _STATE is None:
        from . import data

        _STATE = MatchupState.from_history(data.load_matches(verbose=False))
    return _STATE


def get_model():
    """Return the cached saved match model, loading it on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = model_mod.load("match_model")
    return _MODEL


# --- Scoreline sampling -------------------------------------------------------


def _sample_scores(
    lam: float, mu: float, rho: float | None, n: int, rng, max_goals: int
):
    """Draw ``n`` (home, away) scorelines."""
    if rho is None:
        return rng.poisson(lam, n), rng.poisson(mu, n)
    # Dixon-Coles: sample from the corrected joint distribution.
    grid = model_mod.score_matrix(lam, mu, rho=rho, max_goals=max_goals)
    flat = grid.ravel()
    idx = rng.choice(flat.size, size=n, p=flat)
    ga, gb = np.divmod(idx, max_goals + 1)
    return ga, gb


# --- Batch (vectorised) simulation --------------------------------------------


def simulate_matches(
    team_a: str,
    team_b: str,
    neutral: bool = False,
    knockout: bool = False,
    n: int = config.N_SIMULATIONS,
    tier: int = DEFAULT_TIER,
    state: MatchupState | None = None,
    model=None,
    seed: int | None = config.RANDOM_SEED,
    max_goals: int = model_mod.MAX_GOALS,
) -> dict:
    """Vectorised simulation of ``n`` matchups between the two teams.

    Returns a dict of arrays: ``goals_a``, ``goals_b`` (post extra-time for
    knockouts) and ``winner`` (team name, or "draw" for level league games).
    """
    state = state or get_state()
    model = model or get_model()
    rng = np.random.default_rng(seed)

    feats = state.feature_row(team_a, team_b, neutral, tier)
    lam_arr, mu_arr = model.predict_expected_goals(feats)
    lam, mu = float(lam_arr[0]), float(mu_arr[0])
    rho = getattr(model, "rho", None) if isinstance(model, model_mod.DixonColesModel) else None

    ga, gb = _sample_scores(lam, mu, rho, n, rng, max_goals)
    ga = ga.astype(int)
    gb = gb.astype(int)

    if knockout:
        level = ga == gb
        if level.any():
            # Extra time: independent Poisson with lambdas scaled to 30 minutes.
            et_a = rng.poisson(lam * config.EXTRA_TIME_SCALE, n)
            et_b = rng.poisson(mu * config.EXTRA_TIME_SCALE, n)
            ga = np.where(level, ga + et_a, ga)
            gb = np.where(level, gb + et_b, gb)

        still_level = ga == gb
        # Penalty shootout: coin flip nudged by relative strength.
        edge = elo.expected_score(state.elo(team_a), state.elo(team_b)) - 0.5
        p_a = 0.5 + edge * config.SHOOTOUT_STRENGTH_WEIGHT
        a_wins_so = rng.random(n) < p_a
    else:
        still_level = np.zeros(n, dtype=bool)
        a_wins_so = None

    winner = np.where(ga > gb, team_a, np.where(gb > ga, team_b, "draw"))
    if knockout:
        winner = np.where(still_level, np.where(a_wins_so, team_a, team_b), winner)

    return {"goals_a": ga, "goals_b": gb, "winner": winner}


def simulate_match(
    team_a: str,
    team_b: str,
    neutral: bool = False,
    knockout: bool = False,
    tier: int = DEFAULT_TIER,
    state: MatchupState | None = None,
    model=None,
    seed: int | None = None,
    max_goals: int = model_mod.MAX_GOALS,
) -> MatchResult:
    """Simulate a single matchup. Returns goals for each side and the winner."""
    out = simulate_matches(
        team_a, team_b, neutral=neutral, knockout=knockout, n=1, tier=tier,
        state=state, model=model, seed=seed, max_goals=max_goals,
    )
    return MatchResult(int(out["goals_a"][0]), int(out["goals_b"][0]), str(out["winner"][0]))


# --- Tournament simulation ----------------------------------------------------

# Furthest round reached, as an ordered level. Higher = further.
GROUP, R32, R16, QF, SF, FINAL, CHAMPION = range(7)
ROUND_NAMES = {
    GROUP: "Group",
    R32: "Round of 32",
    R16: "Round of 16",
    QF: "Quarter-final",
    SF: "Semi-final",
    FINAL: "Final",
    CHAMPION: "Champion",
}
# Labels used for the per-team probability summary.
PROGRESS_LABELS = [
    ("escape_group", R32),
    ("reach_r16", R16),
    ("reach_qf", QF),
    ("reach_sf", SF),
    ("reach_final", FINAL),
    ("win", CHAMPION),
]


@dataclass
class _GroupStats:
    """Running points, goals-for and goals-against for one team in a group."""

    pts: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        """Goal difference (goals for minus against)."""
        return self.gf - self.ga


class TournamentEngine:
    """Pre-computes matchup expected goals once, then simulates many tournaments.

    Goals are sampled as independent Poisson per side from the model's expected
    goals (the Dixon-Coles low-score correction, if any, is omitted here for
    speed -- it barely moves tournament-level probabilities). All matches are
    treated as neutral-venue.
    """

    def __init__(
        self,
        teams: list[str],
        groups: dict[str, list[str]],
        hg: np.ndarray,
        ag: np.ndarray,
        team_elo: np.ndarray,
    ):
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        self.groups = groups
        self.HG = hg          # HG[i, j] = expected goals for i (nominal home) vs j
        self.AG = ag          # AG[i, j] = expected goals for j (nominal away)
        self.elo = team_elo

        # Fixed group fixtures (6 per group); nominal home is the lower index.
        self._group_fixtures = {
            g: list(itertools.combinations([self.idx[t] for t in members], 2))
            for g, members in groups.items()
        }

    # -- construction from the saved model --
    @classmethod
    def from_world_cup(cls, state: MatchupState | None = None, model=None) -> "TournamentEngine":
        groups = config.WORLD_CUP_GROUPS
        bad = [g for g, m in groups.items() if len(m) != 4]
        if bad or len(groups) != 12:
            raise ValueError(
                "config.WORLD_CUP_GROUPS must hold 12 groups of 4 teams each "
                f"(unfilled/invalid groups: {bad or 'missing groups'}). "
                "Fill it with the official confirmed groups first."
            )
        state = state or get_state()
        model = model or get_model()
        teams = [t for g in config.WORLD_CUP_GROUP_NAMES for t in groups[g]]
        hosts = set(config.WORLD_CUP_HOSTS) & set(teams)

        # Expected goals for every ordered pair, with venue applied: a co-host
        # plays at home (home advantage) and its opponent is away; every other
        # tie (including host-vs-host) is neutral. For each pair we record the
        # query to run and whether the model's (home, away) outputs must be
        # swapped so HG[a, b] is always team a's goals and AG[a, b] team b's.
        pairs = [(a, b) for a in teams for b in teams if a != b]
        rows, swap = [], []
        for a, b in pairs:
            a_host, b_host = a in hosts, b in hosts
            if a_host and not b_host:            # a at home
                rows.append(state.feature_row(a, b, neutral=False, tier=DEFAULT_TIER))
                swap.append(False)
            elif b_host and not a_host:          # b at home -> query (b, a), then swap
                rows.append(state.feature_row(b, a, neutral=False, tier=DEFAULT_TIER))
                swap.append(True)
            else:                                # neutral
                rows.append(state.feature_row(a, b, neutral=True, tier=DEFAULT_TIER))
                swap.append(False)
        lam, mu = model.predict_expected_goals(pd.concat(rows, ignore_index=True))

        n = len(teams)
        HG = np.zeros((n, n))
        AG = np.zeros((n, n))
        idx = {t: i for i, t in enumerate(teams)}
        for (a, b), la, m, sw in zip(pairs, lam, mu, swap):
            ga, gb = (m, la) if sw else (la, m)  # team a's goals, team b's goals
            HG[idx[a], idx[b]] = ga
            AG[idx[a], idx[b]] = gb

        team_elo = np.array([state.elo(t) for t in teams])
        return cls(teams, groups, HG, AG, team_elo)

    # -- group stage --
    def _rank_group(self, group_letter: str, rng) -> tuple[list[int], dict[int, _GroupStats]]:
        members = [self.idx[t] for t in self.groups[group_letter]]
        results = []  # (home_idx, away_idx, gh, ga)
        for i, j in self._group_fixtures[group_letter]:
            gh = rng.poisson(self.HG[i, j])
            ga = rng.poisson(self.AG[i, j])
            results.append((i, j, gh, ga))

        stats = {t: _GroupStats() for t in members}
        for i, j, gh, ga in results:
            stats[i].gf += gh; stats[i].ga += ga
            stats[j].gf += ga; stats[j].ga += gh
            if gh > ga:
                stats[i].pts += 3
            elif gh < ga:
                stats[j].pts += 3
            else:
                stats[i].pts += 1; stats[j].pts += 1

        ordered = self._apply_tiebreakers(members, results, stats, rng)
        return ordered, stats

    def _apply_tiebreakers(self, teams, results, stats, rng) -> list[int]:
        """FIFA 2026 order: points, GD, goals, then head-to-head, then lots.

        Fair-play points are not modelled (no cards in the data), so a residual
        tie after head-to-head is resolved by drawing of lots (random).
        """
        # Overall: points, GD, goals scored.
        ordered = sorted(
            teams, key=lambda t: (stats[t].pts, stats[t].gd, stats[t].gf), reverse=True
        )
        # Resolve blocks still equal on all three via head-to-head, then lots.
        out: list[int] = []
        i = 0
        while i < len(ordered):
            j = i + 1
            key = (stats[ordered[i]].pts, stats[ordered[i]].gd, stats[ordered[i]].gf)
            while j < len(ordered) and (
                stats[ordered[j]].pts, stats[ordered[j]].gd, stats[ordered[j]].gf
            ) == key:
                j += 1
            block = ordered[i:j]
            out.extend(block if len(block) == 1 else self._break_h2h(block, results, rng))
            i = j
        return out

    def _break_h2h(self, block, results, rng) -> list[int]:
        bset = set(block)
        h2h = {t: _GroupStats() for t in block}
        for i, j, gh, ga in results:
            if i in bset and j in bset:
                h2h[i].gf += gh; h2h[i].ga += ga
                h2h[j].gf += ga; h2h[j].ga += gh
                if gh > ga:
                    h2h[i].pts += 3
                elif gh < ga:
                    h2h[j].pts += 3
                else:
                    h2h[i].pts += 1; h2h[j].pts += 1
        return sorted(
            block,
            key=lambda t: (h2h[t].pts, h2h[t].gd, h2h[t].gf, rng.random()),
            reverse=True,
        )

    # -- best thirds + bracket slotting --
    def _assign_thirds(self, thirds: list[tuple[int, str]], rng) -> dict[str, int]:
        """Map the 8 best thirds to slots 3T1..3T8, avoiding same-group meetings.

        ``thirds`` is the chosen eight as (team_idx, group_letter). Uses the
        official allocation table if present in config, else a constraint search.
        """
        slots = list(config.THIRD_SLOT_OPPONENT_GROUP)  # 3T1..3T8
        qualifying = frozenset(g for _, g in thirds)
        by_group = {g: t for t, g in thirds}

        official = config.THIRD_PLACE_ALLOCATION.get(qualifying)
        if official is not None:
            return {slot: by_group[official[slot]] for slot in slots}

        # Constraint assignment: slot's opponent group != third's own group.
        forbidden = config.THIRD_SLOT_OPPONENT_GROUP
        order = sorted(thirds, key=lambda x: rng.random())  # shuffle for variety

        assignment: dict[str, int] = {}
        used_slots: set[str] = set()

        def place(k: int) -> bool:
            """Backtracking: try to seat the k-th third in a valid open slot."""
            if k == len(order):
                return True
            team, grp = order[k]
            for slot in slots:
                if slot in used_slots or forbidden[slot] == grp:
                    continue
                assignment[slot] = team
                used_slots.add(slot)
                if place(k + 1):
                    return True
                used_slots.remove(slot)
                del assignment[slot]
            return False

        if not place(0):  # extremely unlikely; fall back to any bijection
            for slot, (team, _) in zip(slots, order):
                assignment[slot] = team
        return assignment

    # -- knockout rounds --
    def _play_round(self, home, away, rng) -> np.ndarray:
        """Vectorised knockout round; returns the array of winner indices."""
        home = np.asarray(home); away = np.asarray(away)
        hg = self.HG[home, away]; ag = self.AG[home, away]
        gh = rng.poisson(hg); ga = rng.poisson(ag)

        level = gh == ga
        if level.any():
            gh = gh + np.where(level, rng.poisson(hg * config.EXTRA_TIME_SCALE), 0)
            ga = ga + np.where(level, rng.poisson(ag * config.EXTRA_TIME_SCALE), 0)

        still = gh == ga
        # Penalty shootout: Elo-weighted coin flip for any still-level ties.
        eh, ea = self.elo[home], self.elo[away]
        exp_h = 1.0 / (1.0 + 10.0 ** ((ea - eh) / 400.0))
        p_home = 0.5 + (exp_h - 0.5) * config.SHOOTOUT_STRENGTH_WEIGHT
        home_wins = np.where(still, rng.random(len(home)) < p_home, gh > ga)
        return np.where(home_wins, home, away)

    # -- one full tournament --
    def simulate_once(self, rng) -> dict:
        winners, runners = {}, {}
        thirds = []  # (team_idx, group_letter)
        third_stats = {}

        for g in config.WORLD_CUP_GROUP_NAMES:
            ordered, stats = self._rank_group(g, rng)
            winners[g], runners[g] = ordered[0], ordered[1]
            thirds.append((ordered[2], g))
            third_stats[ordered[2]] = stats[ordered[2]]

        # Eight best thirds: points, GD, goals scored, then lots.
        best_thirds = sorted(
            thirds,
            key=lambda x: (
                third_stats[x[0]].pts, third_stats[x[0]].gd,
                third_stats[x[0]].gf, rng.random(),
            ),
            reverse=True,
        )[:8]
        third_slot = self._assign_thirds(best_thirds, rng)

        # Resolve R32 slots to team indices.
        def resolve(slot: str) -> int:
            """Map a bracket slot label (e.g. '1A', '2B', '3T1') to a team index."""
            if slot.startswith("3T"):
                return third_slot[slot]
            pos, grp = slot[0], slot[1:]
            return winners[grp] if pos == "1" else runners[grp]

        levels = {self.idx[t]: GROUP for t in self.teams}
        bracket = [(resolve(a), resolve(b)) for a, b in config.ROUND_OF_32]
        for a, b in bracket:
            levels[a] = R32; levels[b] = R32  # all 32 escaped the group

        # Knockout rounds: R32 -> R16 -> QF -> SF -> Final.
        for reached in (R16, QF, SF, FINAL):
            home = [m[0] for m in bracket]
            away = [m[1] for m in bracket]
            round_winners = self._play_round(home, away, rng)
            for w in round_winners:
                levels[int(w)] = reached
            bracket = [
                (int(round_winners[k]), int(round_winners[k + 1]))
                for k in range(0, len(round_winners), 2)
            ]

        champion_idx = self._play_round([bracket[0][0]], [bracket[0][1]], rng)[0]
        levels[int(champion_idx)] = CHAMPION

        return {
            "exit_round": {self.teams[i]: ROUND_NAMES[lvl] for i, lvl in levels.items()},
            "champion": self.teams[int(champion_idx)],
            "_levels": levels,
        }

    # -- one full tournament WITH scorelines (for visualisation) --
    def _knockout_match_detailed(self, i: int, j: int, rng) -> tuple[int, dict]:
        """Play one knockout tie and return (winner_idx, detail dict).

        Detail includes the regulation score, extra-time total if level, a
        flavour penalty score if still level, and how it was decided.
        """
        hg = int(rng.poisson(self.HG[i, j]))
        ag = int(rng.poisson(self.AG[i, j]))
        detail = {
            "home": self.teams[i], "away": self.teams[j],
            "hg": hg, "ag": ag, "et": None, "pens": None, "decided": "regulation",
        }
        if hg == ag:
            th = hg + int(rng.poisson(self.HG[i, j] * config.EXTRA_TIME_SCALE))
            ta = ag + int(rng.poisson(self.AG[i, j] * config.EXTRA_TIME_SCALE))
            detail["et"] = (th, ta)
            detail["decided"] = "extra time"
            if th == ta:
                exp_h = 1.0 / (1.0 + 10.0 ** ((self.elo[j] - self.elo[i]) / 400.0))
                p_home = 0.5 + (exp_h - 0.5) * config.SHOOTOUT_STRENGTH_WEIGHT
                home_wins = rng.random() < p_home
                base = int(rng.integers(2, 5))  # flavour shootout score
                detail["pens"] = (base + 1, base) if home_wins else (base, base + 1)
                detail["decided"] = "penalties"
                win = i if home_wins else j
            else:
                win = i if th > ta else j
        else:
            win = i if hg > ag else j
        detail["winner"] = self.teams[win]
        return win, detail

    def simulate_detailed(self, rng) -> dict:
        """Play ONE full World Cup and return every scoreline, groups to final.

        Returns a dict with: ``groups`` (per-group match scores + final table
        with W/D/L and an advance/out status), ``thirds`` (all 12 third-placed
        teams ranked, flagged if among the best 8), ``knockout`` (each round's
        ties with scores and how they were decided), ``champion`` and
        ``runner_up``. This is the single-tournament view for the UI; every play
        is independent and random, so the winner varies run to run.
        """
        group_results: dict[str, dict] = {}
        winners, runners, thirds, third_stats = {}, {}, [], {}

        for g in config.WORLD_CUP_GROUP_NAMES:
            members = [self.idx[t] for t in self.groups[g]]
            results = []
            for i, j in self._group_fixtures[g]:
                gh = int(rng.poisson(self.HG[i, j]))
                ga = int(rng.poisson(self.AG[i, j]))
                results.append((i, j, gh, ga))

            stats = {t: _GroupStats() for t in members}
            wdl = {t: [0, 0, 0] for t in members}  # wins, draws, losses
            for i, j, gh, ga in results:
                stats[i].gf += gh; stats[i].ga += ga
                stats[j].gf += ga; stats[j].ga += gh
                if gh > ga:
                    stats[i].pts += 3; wdl[i][0] += 1; wdl[j][2] += 1
                elif gh < ga:
                    stats[j].pts += 3; wdl[j][0] += 1; wdl[i][2] += 1
                else:
                    stats[i].pts += 1; stats[j].pts += 1; wdl[i][1] += 1; wdl[j][1] += 1

            ordered = self._apply_tiebreakers(members, results, stats, rng)
            winners[g], runners[g] = ordered[0], ordered[1]
            thirds.append((ordered[2], g))
            third_stats[ordered[2]] = stats[ordered[2]]

            group_results[g] = {
                "matches": [
                    {"home": self.teams[i], "away": self.teams[j], "hg": gh, "ag": ga}
                    for i, j, gh, ga in results
                ],
                "table": [
                    {
                        "pos": pos + 1, "team": self.teams[t],
                        "P": 3, "W": wdl[t][0], "D": wdl[t][1], "L": wdl[t][2],
                        "GF": stats[t].gf, "GA": stats[t].ga, "GD": stats[t].gd,
                        "Pts": stats[t].pts,
                    }
                    for pos, t in enumerate(ordered)
                ],
            }

        best_thirds = sorted(
            thirds,
            key=lambda x: (third_stats[x[0]].pts, third_stats[x[0]].gd,
                           third_stats[x[0]].gf, rng.random()),
            reverse=True,
        )[:8]
        advancing = {t for t, _ in best_thirds}
        third_slot = self._assign_thirds(best_thirds, rng)

        # Annotate each table row with whether that team advanced.
        for g in config.WORLD_CUP_GROUP_NAMES:
            for row in group_results[g]["table"]:
                tid = self.idx[row["team"]]
                if row["pos"] <= 2:
                    row["status"] = "advanced"
                elif row["pos"] == 3:
                    row["status"] = "advanced" if tid in advancing else "out"
                else:
                    row["status"] = "out"

        thirds_ranked = sorted(
            thirds,
            key=lambda x: (third_stats[x[0]].pts, third_stats[x[0]].gd, third_stats[x[0]].gf),
            reverse=True,
        )
        thirds_summary = [
            {"group": g, "team": self.teams[t], "Pts": third_stats[t].pts,
             "GD": third_stats[t].gd, "GF": third_stats[t].gf, "advanced": t in advancing}
            for t, g in thirds_ranked
        ]

        def resolve(slot: str) -> int:
            if slot.startswith("3T"):
                return third_slot[slot]
            pos, grp = slot[0], slot[1:]
            return winners[grp] if pos == "1" else runners[grp]

        bracket = [(resolve(a), resolve(b)) for a, b in config.ROUND_OF_32]
        round_labels = ["Round of 32", "Round of 16", "Quarter-finals",
                        "Semi-finals", "Final"]
        rounds = []
        for label in round_labels:
            matches, advancing_idx = [], []
            for i, j in bracket:
                win, detail = self._knockout_match_detailed(i, j, rng)
                matches.append(detail)
                advancing_idx.append(win)
            rounds.append({"round": label, "matches": matches})
            if len(advancing_idx) > 1:
                bracket = [
                    (advancing_idx[k], advancing_idx[k + 1])
                    for k in range(0, len(advancing_idx), 2)
                ]

        final = rounds[-1]["matches"][0]
        champion = final["winner"]
        runner_up = final["away"] if champion == final["home"] else final["home"]

        return {
            "groups": group_results,
            "thirds": thirds_summary,
            "knockout": rounds,
            "champion": champion,
            "runner_up": runner_up,
        }


# Cached engine so repeated tournament sims don't re-predict every matchup.
_ENGINE: TournamentEngine | None = None


def get_engine() -> TournamentEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TournamentEngine.from_world_cup()
    return _ENGINE


def simulate_tournament(seed: int | None = None, engine: TournamentEngine | None = None) -> dict:
    """Simulate one full 2026 World Cup.

    Returns ``{"exit_round": {team: round}, "champion": team}``.
    """
    engine = engine or get_engine()
    rng = np.random.default_rng(seed)
    out = engine.simulate_once(rng)
    return {"exit_round": out["exit_round"], "champion": out["champion"]}


def play_tournament(seed: int | None = None, engine: TournamentEngine | None = None) -> dict:
    """Play one full World Cup WITH scorelines (groups -> final) for the UI.

    See :meth:`TournamentEngine.simulate_detailed`. Every call is independent
    and random unless a fixed ``seed`` is given.
    """
    engine = engine or get_engine()
    rng = np.random.default_rng(seed)
    return engine.simulate_detailed(rng)


def _progression_df(counts: dict, n: int) -> pd.DataFrame:
    """Build the per-team progression-probability table from raw counts."""
    denom = max(n, 1)
    rows = []
    for team, c in counts.items():
        row = {"team": team}
        row.update({label: c[k] / denom for k, (label, _) in enumerate(PROGRESS_LABELS)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("win", ascending=False).reset_index(drop=True)


def monte_carlo_stream(
    engine: TournamentEngine, n: int, seed: int | None = None, batches: int = 24
):
    """Yield ``(completed, current_progression_df)`` as simulations accumulate.

    Lets the UI animate the odds converging live instead of blocking on a
    single opaque call. The final yield equals what :func:`run_monte_carlo`
    returns for the same ``n``.
    """
    rng = np.random.default_rng(seed)
    counts = {t: np.zeros(len(PROGRESS_LABELS)) for t in engine.teams}
    done, per = 0, max(1, n // max(batches, 1))
    while done < n:
        step = min(per, n - done)
        for _ in range(step):
            levels = engine.simulate_once(rng)["_levels"]
            for i, lvl in levels.items():
                team = engine.teams[i]
                for k, (_, threshold) in enumerate(PROGRESS_LABELS):
                    if lvl >= threshold:
                        counts[team][k] += 1
            done += 1
        yield done, _progression_df(counts, done)


def run_monte_carlo(
    n: int = config.N_SIMULATIONS,
    seed: int = config.RANDOM_SEED,
    engine: TournamentEngine | None = None,
    cache_path=config.TOURNAMENT_MC_PATH,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate ``n`` tournament simulations into per-team progression odds.

    Caches the resulting table to disk (keyed by n) and prints the title-odds
    table sorted by win probability.
    """
    if cache_path is not None and cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("n") == n:
            table = pd.DataFrame(cached["teams"])
            if verbose:
                _print_title_odds(table, n, cached=True)
            return table

    engine = engine or get_engine()
    rng = np.random.default_rng(seed)

    counts = {t: np.zeros(len(PROGRESS_LABELS)) for t in engine.teams}
    for _ in range(n):
        levels = engine.simulate_once(rng)["_levels"]
        for i, lvl in levels.items():
            team = engine.teams[i]
            for k, (_, threshold) in enumerate(PROGRESS_LABELS):
                if lvl >= threshold:
                    counts[team][k] += 1

    table = _progression_df(counts, n)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"n": n, "seed": seed, "teams": table.to_dict(orient="records")},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    if verbose:
        _print_title_odds(table, n)
    return table


def _print_title_odds(table: pd.DataFrame, n: int, cached: bool = False) -> None:
    tag = " (cached)" if cached else ""
    print(f"2026 World Cup title odds  --  {n:,} simulations{tag}\n")
    print(f"{'Team':<22}{'Win%':>8}{'Final%':>9}{'SF%':>8}{'QF%':>8}{'R16%':>8}{'Grp+%':>8}")
    print("-" * 71)
    for r in table.itertuples(index=False):
        print(
            f"{r.team:<22}{100 * r.win:>8.1f}{100 * r.reach_final:>9.1f}"
            f"{100 * r.reach_sf:>8.1f}{100 * r.reach_qf:>8.1f}"
            f"{100 * r.reach_r16:>8.1f}{100 * r.escape_group:>8.1f}"
        )


# --- Season simulation (probabilities -> table) -------------------------------


def simulate_season(
    match_probabilities: pd.DataFrame,
    n_simulations: int = config.N_SIMULATIONS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Run Monte Carlo simulations over a fixture list of outcome probabilities.

    Returns aggregated outcome statistics (e.g. final-table positions).
    """
    rng = np.random.default_rng(seed)
    del rng  # placeholder until implemented
    raise NotImplementedError
