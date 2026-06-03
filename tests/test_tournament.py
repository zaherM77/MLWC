"""Tests for tournament simulation (offline, synthetic engine)."""

from collections import Counter

import numpy as np
import pytest

from src import config, simulate


def _synthetic_engine(elo_boost: dict[int, float] | None = None) -> simulate.TournamentEngine:
    teams = [f"T{i:02d}" for i in range(48)]
    groups = {g: teams[k * 4:(k + 1) * 4] for k, g in enumerate(config.WORLD_CUP_GROUP_NAMES)}
    hg = np.full((48, 48), 1.3)
    ag = np.full((48, 48), 1.3)
    team_elo = np.full(48, 1500.0)
    return simulate.TournamentEngine(teams, groups, hg, ag, team_elo)


def test_simulate_detailed_structure():
    # The single-tournament view must expose every stage with the right shape.
    eng = _synthetic_engine()
    cup = eng.simulate_detailed(np.random.default_rng(0))

    assert set(cup["groups"]) == set(config.WORLD_CUP_GROUP_NAMES)
    for g in config.WORLD_CUP_GROUP_NAMES:
        assert len(cup["groups"][g]["table"]) == 4
        assert len(cup["groups"][g]["matches"]) == 6

    rounds = cup["knockout"]
    assert [r["round"] for r in rounds] == [
        "Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"
    ]
    assert [len(r["matches"]) for r in rounds] == [16, 8, 4, 2, 1]

    # Exactly 8 of the 12 third-placed teams advance.
    assert len(cup["thirds"]) == 12
    assert sum(1 for t in cup["thirds"] if t["advanced"]) == 8

    # Champion is the final's winner and a real team.
    final = rounds[-1]["matches"][0]
    assert cup["champion"] in (final["home"], final["away"])
    assert cup["champion"] != cup["runner_up"]
    all_teams = {t for members in eng.groups.values() for t in members}
    assert cup["champion"] in all_teams


def test_monte_carlo_stream_matches_full_run():
    # The streaming MC must end at n and produce a valid probability table.
    eng = _synthetic_engine()
    last = None
    for done, df in simulate.monte_carlo_stream(eng, n=120, seed=1, batches=6):
        last = (done, df)
    done, df = last
    assert done == 120
    assert len(df) == 48
    assert abs(df["win"].sum() - 1.0) < 1e-9
    # nested progression holds at the end too
    r = df.iloc[0]
    assert r.escape_group >= r.reach_final >= r.win


def test_bracket_advances_exact_counts_each_round():
    eng = _synthetic_engine()
    out = eng.simulate_once(np.random.default_rng(0))
    c = Counter(out["_levels"].values())
    at_least = lambda thr: sum(v for k, v in c.items() if k >= thr)
    assert at_least(simulate.R32) == 32      # 24 group qualifiers + 8 best thirds
    assert at_least(simulate.R16) == 16
    assert at_least(simulate.QF) == 8
    assert at_least(simulate.SF) == 4
    assert at_least(simulate.FINAL) == 2
    assert at_least(simulate.CHAMPION) == 1


def test_monte_carlo_probabilities_are_consistent():
    eng = _synthetic_engine()
    table = simulate.run_monte_carlo(n=500, engine=eng, cache_path=None, verbose=False)
    assert len(table) == 48
    # exactly one champion per tournament -> win probs sum to 1
    assert abs(table["win"].sum() - 1.0) < 1e-9
    # progression is nested: escaping >= reaching R16 >= ... >= winning
    for r in table.itertuples(index=False):
        assert r.escape_group >= r.reach_r16 >= r.reach_qf >= r.reach_sf >= r.reach_final >= r.win
        assert 0.0 <= r.win <= 1.0
    # equal-strength field: ~32/48 escape the group on average
    assert abs(table["escape_group"].mean() - 32 / 48) < 0.05


def test_thirds_never_meet_own_group_winner():
    # With the constraint allocation, no advancing third is slotted against the
    # winner of its own group.
    eng = _synthetic_engine()
    rng = np.random.default_rng(2)
    for _ in range(50):
        # reconstruct a group ranking to get thirds, then check the assignment
        winners, thirds, tstats = {}, [], {}
        for g in config.WORLD_CUP_GROUP_NAMES:
            ordered, stats = eng._rank_group(g, rng)
            winners[g] = ordered[0]
            thirds.append((ordered[2], g))
            tstats[ordered[2]] = stats[ordered[2]]
        best = sorted(thirds, key=lambda x: rng.random())[:8]
        assignment = eng._assign_thirds(best, rng)
        for slot, team in assignment.items():
            third_group = next(g for t, g in best if t == team)
            assert config.THIRD_SLOT_OPPONENT_GROUP[slot] != third_group


def test_round_of_32_bracket_structure():
    # The first knockout round must be a Round of 32: 16 matches over 32 unique
    # slots = 12 group winners + 12 runners-up + 8 best-third slots.
    r32 = config.ROUND_OF_32
    assert len(r32) == 16
    slots = [s for pair in r32 for s in pair]
    assert len(slots) == 32 and len(set(slots)) == 32
    winners = {s for s in slots if s.startswith("1")}
    runners = {s for s in slots if s.startswith("2")}
    thirds = {s for s in slots if s.startswith("3T")}
    assert winners == {f"1{g}" for g in config.WORLD_CUP_GROUP_NAMES}
    assert runners == {f"2{g}" for g in config.WORLD_CUP_GROUP_NAMES}
    assert thirds == {f"3T{i}" for i in range(1, 9)}


def test_full_bracket_never_pits_third_against_own_group():
    # End-to-end: across many simulated tournaments, no advancing third-placed
    # team is ever drawn against a side from its own group in the Round of 32.
    eng = _synthetic_engine()
    rng = np.random.default_rng(7)
    group_of = {t: g for g in config.WORLD_CUP_GROUP_NAMES for t in eng.groups[g]}
    pair_of = {}
    for a, b in config.ROUND_OF_32:
        pair_of[a] = b
        pair_of[b] = a
    for _ in range(100):
        winners, runners, thirds, tstats = {}, {}, [], {}
        for g in config.WORLD_CUP_GROUP_NAMES:
            ordered, stats = eng._rank_group(g, rng)
            winners[g], runners[g] = ordered[0], ordered[1]
            thirds.append((ordered[2], g))
            tstats[ordered[2]] = stats[ordered[2]]
        best = sorted(
            thirds,
            key=lambda x: (tstats[x[0]].pts, tstats[x[0]].gd, tstats[x[0]].gf, rng.random()),
            reverse=True,
        )[:8]
        assert len(best) == 8
        third_slot = eng._assign_thirds(best, rng)

        def resolve(slot):
            if slot.startswith("3T"):
                return third_slot[slot]
            pos, grp = slot[0], slot[1:]
            return winners[grp] if pos == "1" else runners[grp]

        for slot, team in third_slot.items():
            opp_team = resolve(pair_of[slot])
            third_team = eng.teams[team]
            opp = eng.teams[opp_team]
            assert group_of[third_team] != group_of[opp]


def test_from_world_cup_requires_filled_groups(monkeypatch):
    # from_world_cup must refuse to build unless config holds 12 groups of 4.
    monkeypatch.setattr(config, "WORLD_CUP_GROUPS", {"A": [], "B": ["x", "y", "z", "w"]})
    with pytest.raises(ValueError, match="WORLD_CUP_GROUPS"):
        simulate.TournamentEngine.from_world_cup()
