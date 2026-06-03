"""Data loading, caching, cleaning and team-name normalisation.

Source: https://github.com/martj42/international_results
Three CSVs are pulled from the repo's ``master`` branch:

* ``results``    — one row per international match
* ``goalscorers`` — one row per goal
* ``shootouts``  — one row per penalty shootout

Files are cached under ``data/`` and reloaded from there unless
``force_refresh=True`` is passed.
"""

from __future__ import annotations

import requests
import pandas as pd

from . import config

# --- Team-name normalisation --------------------------------------------------
#
# Maps known variant spellings (as they appear in other feeds) to the canonical
# names used by the martj42 dataset. Keys are variants; values are canonical.
TEAM_NAME_MAP: dict[str, str] = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "United States",
    "United States of America": "United States",
    "Czechia": "Czech Republic",
    "China PR": "China",
    "IR Iran": "Iran",
    "Republic of Ireland": "Ireland",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "Congo DR",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Kyrgyz Republic": "Kyrgyzstan",
}


def canonical_team(name: str) -> str:
    """Return the canonical spelling for a single team name."""
    if not isinstance(name, str):
        return name
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


def normalise_team_names(
    df: pd.DataFrame, columns: tuple[str, ...] = ("home_team", "away_team")
) -> pd.DataFrame:
    """Return a copy of ``df`` with team-name columns mapped to canonical names."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(canonical_team)
    return out


# --- Download & cache ---------------------------------------------------------


def cache_path(name: str):
    """Return the cached CSV path for a named dataset.

    ``name`` must be one of the fixed datasets in ``config.DATASETS``. This
    whitelist prevents a name with path separators or ``..`` from escaping
    ``DATA_DIR`` (path traversal) and prevents the download URL below from
    being pointed at an arbitrary path on the data host (SSRF hardening).
    """
    if name not in config.DATASETS:
        raise ValueError(f"unknown dataset name: {name!r}")
    return config.DATA_DIR / f"{name}.csv"


def download_dataset(name: str, force_refresh: bool = False) -> pd.DataFrame:
    """Download one dataset CSV, caching it under ``data/``.

    If a cached copy exists and ``force_refresh`` is False, it is reused.
    """
    path = cache_path(name)  # validates name against the dataset whitelist
    if path.exists() and not force_refresh:
        return pd.read_csv(path)

    # URL is built only from fixed config constants + a whitelisted name, over
    # HTTPS, with a timeout and an HTTP-error check. No user input reaches it.
    url = f"{config.DATA_SOURCE_BASE_URL}{name}.csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return pd.read_csv(path)


def load_datasets(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Download/load all configured datasets, keyed by name."""
    return {
        name: download_dataset(name, force_refresh=force_refresh)
        for name in config.DATASETS
    }


# --- Cleaning -----------------------------------------------------------------


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw ``results`` frame.

    Parses dates, drops rows with null scores, and sorts chronologically.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "home_score", "away_score"])
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    out = out.sort_values("date").reset_index(drop=True)
    return out


# --- Validation ---------------------------------------------------------------


def validate(df: pd.DataFrame) -> None:
    """Print a quick summary of a cleaned ``results`` frame.

    Reports row count, date range, unique-team count, and any names our
    normalisation map points at that don't actually appear in the data
    (i.e. broken/unmapped mapping targets worth fixing or pruning).
    """
    print(f"rows: {len(df):,}")
    print(f"date range: {df['date'].min().date()} to {df['date'].max().date()}")

    teams = set(df["home_team"]).union(df["away_team"])
    print(f"unique teams: {len(teams):,}")

    # Mapping targets that never appear in the data point to a typo or a name
    # that doesn't exist in this source.
    bad_targets = sorted({v for v in TEAM_NAME_MAP.values() if v not in teams})
    if bad_targets:
        print(f"unmapped names (map targets absent from data): {bad_targets}")
    else:
        print("unmapped names: none")


# --- Top-level loader ---------------------------------------------------------


def load_matches(force_refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Load, clean, and normalise the international match results.

    Downloads (or reuses cached) ``results.csv``, parses/cleans it, normalises
    team names to the canonical set, and optionally prints a validation summary.

    Returns columns: date, home_team, away_team, home_score, away_score,
    tournament, city, country, neutral.
    """
    datasets = load_datasets(force_refresh=force_refresh)
    df = clean_results(datasets["results"])
    df = normalise_team_names(df)
    if verbose:
        validate(df)
    return df
