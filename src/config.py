"""Central configuration for the football forecasting app."""

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Data source: martj42/international_results
DATA_SOURCE_BASE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/"
)
DATASETS = ("results", "goalscorers", "shootouts")

# Elo parameters
ELO_BASE_RATING = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 65.0

# Match-importance weights (K) by tournament tier, following the
# World Football Elo convention.
ELO_K_WEIGHTS = {
    "world_cup": 60.0,          # FIFA World Cup finals
    "continental_final": 50.0,  # Euro / Copa América / AFCON / Asian Cup ... finals
    "qualifier": 40.0,          # World Cup & continental qualifiers
    "other_competitive": 30.0,  # Nations League, minor cups, etc.
    "friendly": 20.0,           # friendlies
}

# Persisted current-ratings file
ELO_CURRENT_PATH = MODELS_DIR / "elo_current.json"

# Simulation parameters
N_SIMULATIONS = 10_000
RANDOM_SEED = 42

# Knockout resolution
EXTRA_TIME_SCALE = 30.0 / 90.0  # extra time is 30 min vs 90 of regulation
# Shootouts are mostly a coin flip; dampen the strength edge toward 0.5.
# p(stronger side wins) = 0.5 + (elo_expectation - 0.5) * SHOOTOUT_STRENGTH_WEIGHT
SHOOTOUT_STRENGTH_WEIGHT = 0.5

# =============================================================================
# 2026 FIFA World Cup structure
# =============================================================================
# 48 teams in 12 groups (A-L) of 4. The top two of each group plus the eight
# best third-placed teams (32 total) advance to the Round of 32.

WORLD_CUP_GROUP_NAMES = list("ABCDEFGHIJKL")  # A .. L  (12 groups)

# The three co-hosts play their matches at home; everyone else is neutral.
# (When two hosts meet, the tie is treated as neutral — no designated venue.)
WORLD_CUP_HOSTS = ["United States", "Canada", "Mexico"]

# -----------------------------------------------------------------------------
# Official 2026 final-draw groups (drawn 5 Dec 2025, Washington D.C.; completed
# after the March 2026 play-offs). Source: Wikipedia "2026 FIFA World Cup draw",
# cross-checked against fifa.com. Names use the canonical martj42 spellings
# (e.g. South Korea, United States, Turkey, Czech Republic, Ivory Coast,
# Congo DR, Cape Verde, Curaçao) and were verified to match the dataset exactly.
# DOUBLE-CHECK against the official source before trusting downstream numbers.
# -----------------------------------------------------------------------------
WORLD_CUP_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# -----------------------------------------------------------------------------
# Round-of-32 bracket.
#
# Slots are encoded as:
#   "1<G>" = winner of group G          (e.g. "1A")
#   "2<G>" = runner-up of group G       (e.g. "2B")
#   "3T<i>" = i-th third-place slot     (filled by the best-thirds allocation)
#
# The 16 matches are listed in BRACKET ORDER: the Round of 16 pairs adjacent
# matches (0v1, 2v3, ...), the quarter-finals pair those, and so on up to the
# final. Group-winner / runner-up positions are fixed by FIFA's official
# bracket.
#
# NOTE: the exact official slot pairings should be VERIFIED against FIFA's
# published 2026 bracket. The layout below is a structurally valid default
# (8 winners face the 8 advancing thirds; the rest are winner-v-runner and
# runner-v-runner) so the simulator runs out of the box; replace it with the
# official pairings when confirmed.
# -----------------------------------------------------------------------------
ROUND_OF_32 = [
    ("1A", "3T1"),
    ("1B", "3T2"),
    ("1C", "3T3"),
    ("1D", "3T4"),
    ("1E", "3T5"),
    ("1F", "3T6"),
    ("1G", "3T7"),
    ("1H", "3T8"),
    ("1I", "2A"),
    ("1J", "2B"),
    ("1K", "2C"),
    ("1L", "2D"),
    ("2E", "2F"),
    ("2G", "2H"),
    ("2I", "2J"),
    ("2K", "2L"),
]

# Which group winner each third-place slot (3T1..3T8) faces. Used to forbid a
# third-placed team from being drawn against a side from its own group.
THIRD_SLOT_OPPONENT_GROUP = {
    "3T1": "A", "3T2": "B", "3T3": "C", "3T4": "D",
    "3T5": "E", "3T6": "F", "3T7": "G", "3T8": "H",
}

# Optional OFFICIAL third-place allocation lookup. FIFA publishes a table that
# maps the set of eight groups whose third-placed team advances to specific
# bracket slots. Fill this with the official mapping if you want exact paths:
#   key   = frozenset of 8 group letters that produced an advancing third
#   value = dict {third_slot_label: group_letter}
# If a combination is absent, the simulator falls back to a constraint-based
# assignment (no team meets a same-group side in the Round of 32).
THIRD_PLACE_ALLOCATION: dict[frozenset, dict[str, str]] = {}

# Cached Monte Carlo tournament results
TOURNAMENT_MC_PATH = DATA_DIR / "tournament_mc.json"

# Usage analytics (admin dashboard). Anonymous per-session click counts only.
ANALYTICS_PATH = DATA_DIR / "analytics.json"

# Admin dashboard gate. The dashboard is reached via the URL query parameter
# ?admin=<token>. There is no login; the token IS the secret. Override it in
# production via Streamlit secrets (`admin_token` in .streamlit/secrets.toml or
# the Streamlit Cloud "Secrets" UI) or the ADMIN_TOKEN environment variable.
DEFAULT_ADMIN_TOKEN = "change-me-admin"
