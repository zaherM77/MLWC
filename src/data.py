from __future__ import annotations

import requests
import pandas as pd

from . import config


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
    if not isinstance(name, str):
        return name
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


def normalise_team_names(
    df: pd.DataFrame, columns: tuple[str, ...] = ("home_team", "away_team")
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(canonical_team)
    return out


def cache_path(name: str):

    if name not in config.DATASETS:
        raise ValueError(f"unknown dataset name: {name!r}")
    return config.DATA_DIR / f"{name}.csv"


def download_dataset(name: str, force_refresh: bool = False) -> pd.DataFrame:

    path = cache_path(name) 
    if path.exists() and not force_refresh:
        return pd.read_csv(path)

    url = f"{config.DATA_SOURCE_BASE_URL}{name}.csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return pd.read_csv(path)


def load_datasets(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {
        name: download_dataset(name, force_refresh=force_refresh)
        for name in config.DATASETS
    }




def clean_results(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "home_score", "away_score"])
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    out = out.sort_values("date").reset_index(drop=True)
    return out




def validate(df: pd.DataFrame) -> None:
    print(f"rows: {len(df):,}")
    print(f"date range: {df['date'].min().date()} to {df['date'].max().date()}")

    teams = set(df["home_team"]).union(df["away_team"])
    print(f"unique teams: {len(teams):,}")

    bad_targets = sorted({v for v in TEAM_NAME_MAP.values() if v not in teams})
    if bad_targets:
        print(f"unmapped names (map targets absent from data): {bad_targets}")
    else:
        print("unmapped names: none")



def load_matches(force_refresh: bool = False, verbose: bool = True) -> pd.DataFrame:

    datasets = load_datasets(force_refresh=force_refresh)
    df = clean_results(datasets["results"])
    df = normalise_team_names(df)
    if verbose:
        validate(df)
    return df
