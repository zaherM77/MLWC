"""Training, evaluation and persistence of match-forecasting models.

Two models are trained and compared on an *out-of-time* holdout:

1. ``DixonColesModel`` — a Dixon-Coles style bivariate Poisson. Each team has an
   attack and a defence strength; home advantage is a single shared term
   (suppressed on neutral ground); a low-score dependence parameter ``rho``
   corrects the independence assumption for 0-0/1-0/0-1/1-1 scorelines. It is
   the *interpretable* baseline: every coefficient is a team strength you can
   read off directly.

2. ``PoissonGBModel`` — two ``HistGradientBoostingRegressor(loss="poisson")``
   models (one per side) that predict expected goals from the engineered
   features, then convert those to a scoreline distribution.

Both turn expected goals into a 3-way outcome distribution
``[P(home win), P(draw), P(away win)]``.

------------------------------------------------------------------------------
Why a TIME-BASED split (train < 2022, test 2022+), never a random one
------------------------------------------------------------------------------
Team strength drifts over time (squads, managers, generations of players). A
random train/test split would put matches from 2024 in the training set and
matches from 2019 in the test set — i.e. the model would "know the future"
relative to what it is asked to predict. That leaks information that would not
exist at real forecast time and produces optimistic, dishonest metrics. A
chronological cut mirrors deployment exactly: fit on the past, forecast the
unseen future. (The features themselves are likewise point-in-time; see
``features.py``.)

------------------------------------------------------------------------------
Why RANKED PROBABILITY SCORE (RPS) is the headline metric
------------------------------------------------------------------------------
A football result is *ordinal*: home win > draw > away win. If the true result
is a home win, a forecast that put its mass on "draw" was less wrong than one
that put it on "away win". Log-loss and the (multiclass) Brier score treat the
three outcomes as unordered and penalise both errors equally. RPS works on the
*cumulative* distribution, so it rewards probability mass placed *near* the
true outcome. That makes it the standard proper scoring rule for ordered
forecasts like match results. We still report log-loss and Brier alongside it,
plus a reliability check, but selection is on RPS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import log_loss

from . import config, features as features_mod

MAX_GOALS = 10
TEST_FROM_YEAR = 2022
OUTCOME_LABELS = ("home_win", "draw", "away_win")  # ordered for RPS

# Security: model artifacts are joblib/pickle, so loading one EXECUTES the code
# embedded in it. We therefore treat the model file as a trusted, project-owned
# artifact and never let a caller-supplied ``name`` widen that trust boundary.
# ``name`` is a fixed constant everywhere in this codebase ("match_model"); the
# guard below is defense-in-depth so it can never become a path-traversal /
# arbitrary-file-load primitive even if a future caller passes it through.
ALLOWED_MODEL_NAMES = frozenset({"match_model"})


def _model_path(name: str):
    """Resolve ``<MODELS_DIR>/<name>.joblib``, rejecting anything untrusted.

    The name must be on the whitelist (which also forbids separators, ``..``,
    and absolute paths), and the resolved file must stay inside ``MODELS_DIR``.
    """
    if name not in ALLOWED_MODEL_NAMES:
        raise ValueError(f"unknown / untrusted model name: {name!r}")
    models_dir = config.MODELS_DIR.resolve()
    path = (models_dir / f"{name}.joblib").resolve()
    # Belt-and-braces: ensure the resolved path did not escape MODELS_DIR.
    if path.parent != models_dir:
        raise ValueError(f"resolved model path escapes models dir: {path}")
    return path


# --- Outcome helpers ----------------------------------------------------------


def outcome_index(home_score: int, away_score: int) -> int:
    """0 = home win, 1 = draw, 2 = away win (ordered)."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def _dc_tau(h: int, a: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score dependence adjustment."""
    if h == 0 and a == 0:
        return 1.0 - lam * mu * rho
    if h == 0 and a == 1:
        return 1.0 + lam * rho
    if h == 1 and a == 0:
        return 1.0 + mu * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(
    lam: float, mu: float, rho: float | None = None, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """Normalised scoreline distribution; ``grid[i, j] = P(home=i, away=j)``.

    With ``rho`` set, applies the Dixon-Coles correction to the four low-score
    cells; otherwise assumes independent Poisson margins.
    """
    # Guard against pathological expected goals (e.g. extreme coefficients from
    # data-sparse teams), which would otherwise underflow the truncated grid.
    lam = float(np.clip(lam, 1e-3, 15.0))
    mu = float(np.clip(mu, 1e-3, 15.0))
    h = poisson.pmf(np.arange(max_goals + 1), lam)
    a = poisson.pmf(np.arange(max_goals + 1), mu)
    grid = np.outer(h, a)  # grid[i, j] = P(home=i, away=j)

    if rho is not None:
        for i in (0, 1):
            for j in (0, 1):
                grid[i, j] *= _dc_tau(i, j, lam, mu, rho)

    grid /= grid.sum()
    return grid


def outcome_probs_from_goals(
    lam: float, mu: float, rho: float | None = None, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """Convert expected goals (lam home, mu away) to [P(home), P(draw), P(away)]."""
    grid = score_matrix(lam, mu, rho=rho, max_goals=max_goals)
    p_home = np.tril(grid, -1).sum()  # i > j
    p_draw = np.trace(grid)           # i == j
    p_away = np.triu(grid, 1).sum()   # i < j
    return np.array([p_home, p_draw, p_away])


# --- Metrics ------------------------------------------------------------------


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean RPS over samples. ``probs`` is (n, 3) ordered; lower is better."""
    onehot = np.eye(3)[outcomes]
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    # sum over the first K-1 cumulative components (the K-th is always 1-1=0)
    return float((((cum_p - cum_o) ** 2)[:, :2].sum(axis=1) / 2).mean())


def multiclass_brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error across the three class probabilities."""
    onehot = np.eye(3)[outcomes]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def reliability_table(
    p_home: np.ndarray, home_win: np.ndarray, n_bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """Reliability (calibration) of the home-win probability.

    Bins predicted P(home win) and compares it to the observed home-win rate.
    Returns the table and the expected calibration error (ECE).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_home, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    n = len(p_home)
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(p_home[mask].mean())
        obs = float(home_win[mask].mean())
        ece += count / n * abs(mean_pred - obs)
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": count,
                "mean_pred": round(mean_pred, 3),
                "obs_freq": round(obs, 3),
            }
        )
    return pd.DataFrame(rows), ece


# --- Model 1: Dixon-Coles bivariate Poisson -----------------------------------


class DixonColesModel:
    """Interpretable bivariate-Poisson baseline (attack/defence + home + rho)."""

    def __init__(self, max_goals: int = MAX_GOALS):
        self.max_goals = max_goals
        self.intercept = 0.0
        self.home_adv = 0.0
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.rho = 0.0

    def _lambdas(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        """Expected goals (home, away) from the fitted attack/defence terms."""
        ha = 0.0 if neutral else self.home_adv
        lam = np.exp(
            self.intercept + ha
            + self.attack.get(home, 0.0) + self.defence.get(away, 0.0)
        )
        mu = np.exp(
            self.intercept
            + self.attack.get(away, 0.0) + self.defence.get(home, 0.0)
        )
        return float(lam), float(mu)

    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        # Long format: one row per (team, goals-scored) observation.
        long = pd.DataFrame(
            {
                "goals": np.concatenate([df["home_score"], df["away_score"]]),
                "attack": np.concatenate([df["home_team"], df["away_team"]]),
                "defence": np.concatenate([df["away_team"], df["home_team"]]),
                # home advantage applies only to the home side and only when the
                # match is not on neutral ground.
                "home_field": np.concatenate(
                    [1 - df["neutral"].to_numpy(), np.zeros(len(df))]
                ),
            }
        )

        att_d = pd.get_dummies(long["attack"], prefix="att", drop_first=True)
        def_d = pd.get_dummies(long["defence"], prefix="def", drop_first=True)
        X = pd.concat([long[["home_field"]], att_d, def_d], axis=1).astype(float)
        X = sm.add_constant(X)
        glm = sm.GLM(long["goals"], X, family=sm.families.Poisson()).fit()
        params = glm.params

        teams = sorted(set(df["home_team"]).union(df["away_team"]))
        attack = {t: float(params.get(f"att_{t}", 0.0)) for t in teams}
        defence = {t: float(params.get(f"def_{t}", 0.0)) for t in teams}

        # Re-centre attack/defence to sum-to-zero and fold the means into the
        # intercept (leaves all fitted values unchanged). This makes a coefficient
        # of 0 mean "league-average", so unseen teams default sensibly.
        mean_att = np.mean(list(attack.values()))
        mean_def = np.mean(list(defence.values()))
        self.attack = {t: v - mean_att for t, v in attack.items()}
        self.defence = {t: v - mean_def for t, v in defence.items()}
        self.intercept = float(params["const"]) + mean_att + mean_def
        self.home_adv = float(params["home_field"])

        self._fit_rho(df)
        return self

    def _fit_rho(self, df: pd.DataFrame) -> None:
        """Estimate rho by MLE with the attack/defence/home terms held fixed."""
        lam = np.empty(len(df))
        mu = np.empty(len(df))
        for i, m in enumerate(df.itertuples(index=False)):
            lam[i], mu[i] = self._lambdas(m.home_team, m.away_team, bool(m.neutral))
        h = df["home_score"].to_numpy()
        a = df["away_score"].to_numpy()

        def neg_ll(rho: float) -> float:
            """Negative log-likelihood of the low-score correction at ``rho``."""
            tau = np.ones(len(df))
            m00 = (h == 0) & (a == 0)
            m01 = (h == 0) & (a == 1)
            m10 = (h == 1) & (a == 0)
            m11 = (h == 1) & (a == 1)
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m11] = 1.0 - rho
            if np.any(tau <= 0):
                return 1e10
            return -np.log(tau).sum()

        res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(res.x)

    def predict_expected_goals(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lam = np.empty(len(df))
        mu = np.empty(len(df))
        for i, m in enumerate(df.itertuples(index=False)):
            lam[i], mu[i] = self._lambdas(m.home_team, m.away_team, bool(m.neutral))
        return lam, mu

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        lam, mu = self.predict_expected_goals(df)
        out = np.empty((len(df), 3))
        for i in range(len(df)):
            out[i] = outcome_probs_from_goals(lam[i], mu[i], rho=self.rho, max_goals=self.max_goals)
        return out


# --- Model 2: HistGradientBoosting Poisson regressors -------------------------


class PoissonGBModel:
    """Two Poisson-loss gradient-boosted regressors -> outcome probabilities."""

    def __init__(self, max_goals: int = MAX_GOALS, random_state: int = config.RANDOM_SEED):
        self.max_goals = max_goals
        self.random_state = random_state
        self.home_model: HistGradientBoostingRegressor | None = None
        self.away_model: HistGradientBoostingRegressor | None = None

    def fit(self, df: pd.DataFrame) -> "PoissonGBModel":
        X = df[features_mod.FEATURE_COLUMNS]  # NaNs handled natively by HGB
        self.home_model = HistGradientBoostingRegressor(
            loss="poisson", random_state=self.random_state
        ).fit(X, df["home_score"])
        self.away_model = HistGradientBoostingRegressor(
            loss="poisson", random_state=self.random_state
        ).fit(X, df["away_score"])
        return self

    def predict_expected_goals(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[features_mod.FEATURE_COLUMNS]
        lam = np.clip(self.home_model.predict(X), 1e-6, None)
        mu = np.clip(self.away_model.predict(X), 1e-6, None)
        return lam, mu

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        lam, mu = self.predict_expected_goals(df)
        out = np.empty((len(df), 3))
        for i in range(len(df)):
            out[i] = outcome_probs_from_goals(lam[i], mu[i], rho=None, max_goals=self.max_goals)
        return out


# --- Train / evaluate / compare -----------------------------------------------


def time_split(df: pd.DataFrame, test_from_year: int = TEST_FROM_YEAR):
    """Chronological split: train strictly before ``test_from_year``."""
    years = df["date"].dt.year
    return df[years < test_from_year].copy(), df[years >= test_from_year].copy()


def _score(probs: np.ndarray, outcomes: np.ndarray) -> dict:
    """All three holdout metrics (RPS, log-loss, Brier) for a probability set."""
    return {
        "RPS": ranked_probability_score(probs, outcomes),
        "LogLoss": log_loss(outcomes, probs, labels=[0, 1, 2]),
        "Brier": multiclass_brier(probs, outcomes),
    }


def evaluate_and_compare(
    feats: pd.DataFrame | None = None, persist: bool = True, verbose: bool = True
) -> dict:
    """Fit both models on the time-based train split, score on the holdout."""
    if feats is None:
        feats = features_mod.build_features()

    train, test = time_split(feats)
    y_test = np.array(
        [outcome_index(h, a) for h, a in zip(test["home_score"], test["away_score"])]
    )

    models = {
        "DixonColes": DixonColesModel().fit(train),
        "PoissonGB": PoissonGBModel().fit(train),
    }

    results = {}
    for name, mdl in models.items():
        probs = mdl.predict_proba(test)
        results[name] = {"model": mdl, "probs": probs, "metrics": _score(probs, y_test)}

    # winner = lowest RPS
    best = min(results, key=lambda n: results[n]["metrics"]["RPS"])

    if verbose:
        print(
            f"Time-based holdout: train {train['date'].dt.year.min()}-"
            f"{TEST_FROM_YEAR - 1} ({len(train):,} matches), "
            f"test {TEST_FROM_YEAR}+ ({len(test):,} matches)\n"
        )
        print(f"{'Model':<12}{'RPS':>10}{'LogLoss':>10}{'Brier':>10}")
        print("-" * 42)
        for name in results:
            m = results[name]["metrics"]
            flag = "  <- best (RPS)" if name == best else ""
            print(f"{name:<12}{m['RPS']:>10.4f}{m['LogLoss']:>10.4f}{m['Brier']:>10.4f}{flag}")

        home_win = (y_test == 0).astype(int)
        table, ece = reliability_table(results[best]["probs"][:, 0], home_win)
        print(f"\nCalibration of {best} P(home win)  --  ECE = {ece:.4f}")
        print(table.to_string(index=False))

    if persist:
        save(results[best]["model"], name="match_model")
        if verbose:
            print(f"\nSaved best model ({best}) -> {config.MODELS_DIR / 'match_model.joblib'}")

    return {"results": results, "best": best, "train": train, "test": test}


def save(model, name: str = "match_model") -> None:
    """Persist a trained model to the models directory."""
    path = _model_path(name)  # validates name / containment before any write
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(name: str = "match_model"):
    """Load a persisted model from the models directory.

    NOTE: joblib uses pickle, so this executes code embedded in the file. Only
    project-owned artifacts in ``MODELS_DIR`` are ever loaded; ``_model_path``
    enforces that ``name`` is whitelisted and cannot traverse out of that dir.
    """
    return joblib.load(_model_path(name))


def ensure_model(name: str = "match_model"):
    """Return the saved model, training and caching it first if it is absent.

    This lets the app bootstrap itself on a clean machine -- e.g. a fresh
    Streamlit Cloud instance with no committed artifacts. If no model file
    exists, features are built from the (auto-downloaded) public dataset and a
    Poisson gradient-boosting model is trained on all available matches, then
    saved to disk so later loads are instant. The time-based backtest in
    :func:`evaluate_and_compare` is separate and still trains on the pre-2022
    split only.
    """
    path = _model_path(name)  # validates name / containment
    if path.exists():
        return load(name)

    from . import features as features_mod

    feats = features_mod.build_features()
    model = PoissonGBModel().fit(feats)
    save(model, name)
    return model


if __name__ == "__main__":
    evaluate_and_compare()
