"""Streamlit front-end for the probabilistic 2026 World Cup forecaster.

Run with:  streamlit run app.py

Nothing is shown pre-computed: the visitor chooses to *play out a single
tournament* (full scorelines, groups to final) or *run a live Monte Carlo*
(odds converging on screen). A hidden admin dashboard (usage analytics) is
reached via the ``?admin=<token>`` URL parameter.
"""

from __future__ import annotations

import html
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analytics, config, model as model_mod, simulate

st.set_page_config(page_title="2026 World Cup Forecasting", page_icon="⚽", layout="wide")

# --- theme / palette ----------------------------------------------------------
# "Aurora" palette, shared between the CSS and the Plotly charts.
PALETTE = {
    "bg": "#0b0a18",
    "panel": "#160f2e",
    "cyan": "#22d3ee",
    "violet": "#a78bfa",
    "magenta": "#f472b6",
    "gold": "#fde047",
    "text": "#ece9f7",
    "muted": "#9b93c0",
    "grid": "rgba(167,139,250,0.14)",
}
# Electric cyan -> violet -> magenta scale used across the charts.
WC_SCALE = [[0.0, "#0e7490"], [0.35, "#22d3ee"], [0.7, "#a78bfa"], [1.0, "#f472b6"]]

_STYLES_PATH = Path(__file__).resolve().parent / "assets" / "styles.css"


def _esc(value) -> str:
    """HTML-escape any dynamic value before it goes into injected markup."""
    return html.escape(str(value))


def _style_fig(fig, *, height=None):
    """Apply the dark Aurora look to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=PALETTE["text"], size=13),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    return fig


@st.cache_data(show_spinner=False)
def _load_css() -> str:
    """Read the static stylesheet (cached)."""
    try:
        return _STYLES_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def inject_theme():
    """Inject the static stylesheet (fixed markup, no dynamic data)."""
    css = _load_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero():
    """Static aurora hero header (no dynamic content)."""
    st.markdown(
        """
        <div class="wc-hero">
          <span class="wc-eyebrow">FIFA World Cup 2026 &middot; Forecast</span>
          <div class="wc-title">Simulate the Road to the Trophy</div>
          <p class="wc-sub">Play out the 48-team tournament match by match, or run
          thousands of simulations and watch the title odds take shape live. Built
          on 150&nbsp;years of international results.</p>
          <div class="wc-chips">
            <span class="wc-chip"><b>48</b> teams</span>
            <span class="wc-chip"><b>12</b> groups</span>
            <span class="wc-chip"><b>104</b> matches</span>
            <span class="wc-chip">Elo&nbsp;+&nbsp;boosted Poisson</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- cached resources ---------------------------------------------------------


def _groups_filled() -> bool:
    """True once all 12 World Cup groups hold four teams."""
    g = config.WORLD_CUP_GROUPS
    return len(g) == 12 and all(len(v) == 4 for v in g.values())


@st.cache_resource(show_spinner="Preparing the forecasting model (first run only)…")
def load_model():
    """Load the saved match model, training it on first run if absent (cached)."""
    return model_mod.ensure_model("match_model")


@st.cache_resource(show_spinner="Loading match history…")
def load_state():
    """Load current Elo + recent-form state from history (cached)."""
    return simulate.get_state()


@st.cache_resource(show_spinner="Preparing tournament engine…")
def load_engine():
    """Build the tournament engine for the configured groups (cached)."""
    return simulate.get_engine()


@st.cache_data(show_spinner=False)
def team_list() -> list[str]:
    """Sorted list of every team with a current Elo rating."""
    return sorted(load_state().ratings.keys())


@st.cache_data(show_spinner="Predicting…")
def predict_match(team_a: str, team_b: str, neutral: bool):
    """Expected goals, scoreline grid, and W/D/W probabilities for a matchup."""
    state, mdl = load_state(), load_model()
    row = state.feature_row(team_a, team_b, neutral=neutral, tier=simulate.DEFAULT_TIER)
    lam, mu = mdl.predict_expected_goals(row)
    lam, mu = float(lam[0]), float(mu[0])
    rho = getattr(mdl, "rho", None) if isinstance(mdl, model_mod.DixonColesModel) else None
    grid = model_mod.score_matrix(lam, mu, rho=rho)
    probs = model_mod.outcome_probs_from_goals(lam, mu, rho=rho)
    return lam, mu, grid, probs


# Out-of-time backtest results (train < 2022, holdout 2022+); shown statically so
# the methodology page never refits the model on a constrained cloud instance.
BACKTEST = pd.DataFrame(
    [
        {"Model": "Gradient-boosted Poisson (selected)", "RPS": 0.1721, "LogLoss": 0.8804, "Brier": 0.5161},
        {"Model": "Dixon-Coles bivariate Poisson (baseline)", "RPS": 0.1764, "LogLoss": 0.8962, "Brier": 0.5236},
    ]
)
BACKTEST_TRAIN_N, BACKTEST_TEST_N = 18_703, 4_421


# --- analytics / session ------------------------------------------------------


def _ensure_session() -> str:
    """Assign this browser session an anonymous id and count the visit once."""
    if "uid" not in st.session_state:
        st.session_state["uid"] = uuid.uuid4().hex[:12]
        analytics.record_visit(st.session_state["uid"])
    return st.session_state["uid"]


def _track_click() -> None:
    """Count a simulation action for the current session."""
    analytics.record_click(st.session_state.get("uid", "anon"))


def _admin_token() -> str:
    """Resolve the admin token: env var > Streamlit secret > config default."""
    tok = os.environ.get("ADMIN_TOKEN")
    if tok:
        return tok
    try:
        return st.secrets["admin_token"]
    except Exception:
        return config.DEFAULT_ADMIN_TOKEN


# --- shared chart helpers -----------------------------------------------------


def _odds_bar(df: pd.DataFrame, height: int = 520):
    """Horizontal championship-probability bar chart for a slice of the table."""
    d = df.copy()
    d["Win %"] = d["win"] * 100
    top_max = max(8.0, float(d["Win %"].max()) * 1.18) if len(d) else 8.0
    fig = px.bar(
        d, x="Win %", y="team", orientation="h",
        text=d["Win %"].map("{:.1f}".format), color="Win %",
        color_continuous_scale=WC_SCALE,
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(range=[0, top_max], title="Win probability (%)"),
        coloraxis_showscale=False,
    )
    _style_fig(fig, height=height)
    fig.update_traces(
        textposition="outside", cliponaxis=False,
        textfont=dict(color=PALETTE["text"]),
        hovertemplate="<b>%{y}</b><br>Win: %{x:.2f}%<extra></extra>",
    )
    return fig


def _bar3d_mesh(x, y, dz, color, name):
    """A single 3D bar (unit-footprint cuboid) as a Mesh3d, height ``dz``."""
    w = 0.4
    xs = [x - w, x + w, x + w, x - w, x - w, x + w, x + w, x - w]
    ys = [y - w, y - w, y + w, y + w, y - w, y - w, y + w, y + w]
    zs = [0, 0, 0, 0, dz, dz, dz, dz]
    i = [0, 0, 0, 0, 4, 4, 6, 6, 1, 1, 2, 2]
    j = [1, 2, 4, 7, 5, 6, 5, 2, 5, 6, 6, 3]
    k = [2, 3, 7, 3, 6, 7, 2, 3, 6, 2, 7, 7]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color, opacity=0.97,
        flatshading=True, hovertext=name, hoverinfo="text+z", name=name,
        lighting=dict(ambient=0.6, diffuse=0.85, specular=0.3, roughness=0.4),
    )


def _render_3d_podium(top: pd.DataFrame):
    """Rotatable 3D bar chart of the top contenders (an optional alt view)."""
    st.caption("Drag to rotate · scroll to zoom — taller, pinker bars are the bigger title shouts.")
    chart = top.head(12).reset_index(drop=True)
    wins = (chart["win"] * 100).tolist()
    teams = chart["team"].tolist()
    cols, hi = 4, (max(wins) or 1.0)
    fig = go.Figure()
    xs, ys, texts = [], [], []
    for idx, (team, w) in enumerate(zip(teams, wins)):
        gx, gy = idx % cols, idx // cols
        color = px.colors.sample_colorscale(WC_SCALE, [min(w / hi, 1.0)])[0]
        fig.add_trace(_bar3d_mesh(gx, -gy, w, color, team))
        xs.append(gx); ys.append(-gy); texts.append(f"{team}<br>{w:.1f}%")
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=[w + hi * 0.06 for w in wins], mode="text", text=texts,
        textposition="top center", textfont=dict(color=PALETTE["text"], size=11),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        height=560, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(title="Win %", color=PALETTE["muted"],
                       gridcolor=PALETTE["grid"], backgroundcolor="rgba(0,0,0,0)"),
            aspectmode="manual", aspectratio=dict(x=1.4, y=1.1, z=0.9),
            camera=dict(eye=dict(x=1.7, y=1.7, z=0.85)),
        ),
    )
    st.plotly_chart(fig, width="stretch")


def _render_groups_overview():
    """A scannable grid of all 12 groups (the 2026 draw) for context."""
    st.caption(
        "2026 final draw (drawn 5 Dec 2025, completed after the March 2026 "
        "play-offs). Numbers are seeded draw positions."
    )
    names = config.WORLD_CUP_GROUP_NAMES
    per_row = 3
    for start in range(0, len(names), per_row):
        cols = st.columns(per_row)
        for col, g in zip(cols, names[start:start + per_row]):
            with col.container(border=True):
                st.markdown(f"**Group {g}**")
                st.markdown(
                    "\n".join(
                        f"{i}. {t}" for i, t in enumerate(config.WORLD_CUP_GROUPS[g], 1)
                    )
                )


# --- single-tournament rendering ----------------------------------------------


def _match_card_html(m: dict) -> str:
    """Build one knockout match card. All dynamic values are escaped."""
    a, b = m["home"], m["away"]
    hg, ag, tag = m["hg"], m["ag"], ""
    if m["decided"] == "extra time":
        hg, ag = m["et"]
        tag = "a.e.t."
    elif m["decided"] == "penalties":
        hg, ag = m["et"]
        ph, pa = m["pens"]
        tag = f"pens {ph}–{pa}"
    a_win = m["winner"] == a
    cls_a, cls_b = ("win" if a_win else ""), ("" if a_win else "win")
    decider = f'<div class="wc-decider">{_esc(tag)}</div>' if tag else ""
    return (
        '<div class="wc-match">'
        f'<div class="wc-team {cls_a}"><span class="nm">{_esc(a)}</span>'
        f'<span class="wc-score">{int(hg)}</span></div>'
        f'<div class="wc-team {cls_b}"><span class="nm">{_esc(b)}</span>'
        f'<span class="wc-score">{int(ag)}</span></div>'
        f'{decider}</div>'
    )


def _render_round(rnd: dict):
    """Render one knockout round as a row of match cards."""
    cards = "".join(_match_card_html(m) for m in rnd["matches"])
    st.markdown(
        f'<div class="wc-round"><div class="wc-round-title">{_esc(rnd["round"])}</div>'
        f'<div class="wc-bracket">{cards}</div></div>',
        unsafe_allow_html=True,
    )


def _render_champion(cup: dict):
    """Render the celebratory champion banner (escaped names)."""
    st.markdown(
        f'<div class="wc-champion"><div class="wc-champ-label">Champion</div>'
        f'<div class="wc-champ-name">{_esc(cup["champion"])}</div>'
        f'<div class="wc-champ-sub">Runner-up: {_esc(cup["runner_up"])}</div></div>',
        unsafe_allow_html=True,
    )


def _render_groups(cup: dict):
    """Group standings (with advance markers) plus the scorelines per group."""
    names = config.WORLD_CUP_GROUP_NAMES
    per_row = 2
    for start in range(0, len(names), per_row):
        cols = st.columns(per_row)
        for col, g in zip(cols, names[start:start + per_row]):
            with col:
                st.markdown(f"**Group {g}**")
                rows = cup["groups"][g]["table"]
                disp = pd.DataFrame([
                    {"#": r["pos"], "Team": r["team"], "W": r["W"], "D": r["D"],
                     "L": r["L"], "GF": r["GF"], "GA": r["GA"], "GD": r["GD"],
                     "Pts": r["Pts"], "▶": "✅" if r["status"] == "advanced" else ""}
                    for r in rows
                ])
                st.dataframe(disp, hide_index=True, width="stretch")
                scores = " · ".join(
                    f'{m["home"]} {m["hg"]}-{m["ag"]} {m["away"]}'
                    for m in cup["groups"][g]["matches"]
                )
                st.caption(scores)


def _render_thirds(cup: dict):
    """The race for the eight best third-placed teams."""
    tdf = pd.DataFrame(cup["thirds"])
    tdf = tdf.rename(columns={"group": "Group", "team": "Team"})
    tdf["Through"] = tdf.pop("advanced").map({True: "✅", False: "—"})
    st.dataframe(tdf, hide_index=True, width="stretch")


def _animate_cup(cup: dict):
    """Stream the 'simulation in progress' steps for an interactive reveal."""
    with st.status("Kicking off the tournament…", expanded=True) as status:
        st.write("⚽ **Group stage** — 12 groups, 72 matches played")
        time.sleep(0.5)
        for rnd in cup["knockout"]:
            n = len(rnd["matches"])
            st.write(f"🥅 **{rnd['round']}** — {n} tie{'s' if n > 1 else ''} decided")
            time.sleep(0.42)
        status.update(label="Full-time — we have a champion! 🏆",
                      state="complete", expanded=False)


def page_play():
    """Play out one full tournament with an animated reveal and a full bracket."""
    if not _groups_filled():
        st.info("Fill `config.WORLD_CUP_GROUPS` with the official groups to simulate.")
        return

    st.markdown("## ⚽ Play out a World Cup")
    st.write(
        "Simulate one full tournament — every group, every knockout tie, one "
        "champion. Each run is **independent and random**, so you'll usually get "
        "a different winner. (Spain are favourites, but they win only ~1 in 4.)"
    )
    with st.expander("See the 12 groups (2026 final draw)"):
        _render_groups_overview()

    c1, c2 = st.columns([1, 1])
    play = c1.button("⚽ Kick off a tournament", type="primary", width="stretch")
    again = c2.button("🔄 Play another", width="stretch",
                      disabled="cup" not in st.session_state)

    if play or again:
        _track_click()
        st.session_state["cup"] = simulate.play_tournament(engine=load_engine())
        st.session_state["cup_animate"] = True

    cup = st.session_state.get("cup")
    if not cup:
        st.info("Press **Kick off a tournament** to play one out.")
        return

    if st.session_state.pop("cup_animate", False):
        _animate_cup(cup)
        st.balloons()

    _render_champion(cup)
    st.markdown("### 🏟️ Knockout bracket")
    for rnd in cup["knockout"]:
        _render_round(rnd)
    st.markdown("### 📋 Group stage")
    _render_groups(cup)
    st.markdown("### 🥉 Race for the best third-placed teams")
    st.caption("Eight of the twelve third-placed teams advance to the Round of 32.")
    _render_thirds(cup)


def page_odds():
    """Run a live Monte Carlo and show the converging title odds."""
    if not _groups_filled():
        st.info("Fill `config.WORLD_CUP_GROUPS` with the official groups to simulate.")
        return

    st.markdown("## 🏆 Title odds")
    st.write(
        "Run thousands of independent tournaments and watch the championship "
        "probabilities **converge live**. The averaged picture favours the "
        "strongest teams — but even the top side wins only a minority of the time."
    )
    n = st.slider("Tournaments to simulate", 1_000, 20_000, 5_000, 1_000)
    run = st.button("📊 Run the simulation", type="primary")

    if run:
        _track_click()
        engine = load_engine()
        chart_ph = st.empty()
        prog = st.progress(0.0, text="Simulating…")
        final = None
        for done, df in simulate.monte_carlo_stream(engine, n, batches=24):
            final = df
            chart_ph.plotly_chart(_odds_bar(df.head(14), height=520), width="stretch")
            prog.progress(done / n, text=f"{done:,} / {n:,} tournaments")
        prog.empty()
        chart_ph.empty()
        st.session_state["mc"] = final
        st.session_state["mc_n"] = n

    mc = st.session_state.get("mc")
    if mc is None:
        st.info("Press **Run the simulation** to compute title odds live.")
        return

    st.caption(f"Based on {st.session_state.get('mc_n', 0):,} simulated tournaments (this session).")
    view = st.radio("View", ["Leaderboard", "3D podium"], horizontal=True,
                    label_visibility="collapsed")
    if view == "Leaderboard":
        st.plotly_chart(_odds_bar(mc.head(20), height=640), width="stretch")
    else:
        _render_3d_podium(mc)

    show = mc.head(24)[["team", "escape_group", "reach_final", "win"]].copy()
    for c in ("escape_group", "reach_final", "win"):
        show[c] = show[c] * 100
    show = show.rename(columns={
        "team": "Team", "escape_group": "Advance %",
        "reach_final": "Reach final %", "win": "Win %",
    })
    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        show, hide_index=True, width="stretch", height=440,
        column_config={"Advance %": pct, "Reach final %": pct, "Win %": pct},
    )


def page_match():
    """Head-to-head forecast: W/D/W, expected goals, and a scoreline heatmap."""
    st.markdown("## 🔮 Match predictor")
    teams = team_list()

    def _default(name, fallback=0):
        return teams.index(name) if name in teams else fallback

    c1, c2, c3 = st.columns([5, 5, 3])
    team_a = c1.selectbox("Team A (home)", teams, index=_default("Brazil"))
    team_b = c2.selectbox("Team B (away)", teams, index=_default("Argentina", 1))
    neutral = c3.toggle("Neutral venue", value=True)

    if team_a == team_b:
        st.warning("Pick two different teams.")
        return

    lam, mu, grid, probs = predict_match(team_a, team_b, neutral)
    p_a, p_draw, p_b = (float(x) for x in probs)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{team_a} win", f"{p_a * 100:.1f}%")
    m2.metric("Draw", f"{p_draw * 100:.1f}%")
    m3.metric(f"{team_b} win", f"{p_b * 100:.1f}%")
    st.caption(
        f"Expected goals — {team_a}: **{lam:.2f}**, {team_b}: **{mu:.2f}**"
        + ("  ·  neutral venue" if neutral else f"  ·  {team_a} at home")
    )

    ga, gb = divmod(int(grid[:11, :11].argmax()), grid.shape[1])
    st.markdown(f"**Most likely scoreline:** {team_a} {ga}–{gb} {team_b}")

    k = 7
    fig = px.imshow(
        grid[:k, :k], x=list(range(k)), y=list(range(k)),
        labels=dict(x=f"{team_b} goals", y=f"{team_a} goals", color="P"),
        color_continuous_scale=WC_SCALE, text_auto=".2f", aspect="auto",
    )
    _style_fig(fig, height=430)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")


def page_how():
    """Methodology, the why-always-Spain explainer, backtest, and caveats."""
    st.markdown("## 📖 How it works")
    st.markdown(
        """
**Data.** Every men's international result since 1872 (the public
[martj42/international_results](https://github.com/martj42/international_results)
dataset), cleaned and normalised.

**Elo ratings.** Each team carries a strength rating updated chronologically
after every match — scaled by margin of victory (log-dampened), weighted by
match importance, and given a home bonus off neutral ground. Ratings are strictly
point-in-time (no look-ahead).

**Goals model.** Both teams' Elo, recent form, the venue and the competition tier
feed a gradient-boosted model with a Poisson objective that predicts each side's
expected goals; those define a scoreline distribution. An interpretable
Dixon-Coles bivariate Poisson is kept as a baseline.

**Simulation.** A tournament is played by sampling every scoreline, applying the
FIFA 2026 tiebreakers, taking the top two plus the eight best third-placed teams,
and running the knockout rounds with extra time and shootouts.
        """
    )

    st.markdown("### 🤔 Why do the odds always show Spain on top?")
    st.info(
        "The **Title odds** page averages over thousands of tournaments, and Spain "
        "are the strongest team — so they top the *average*. But they still win only "
        "about a quarter of the time. Any **single** tournament (the *Play out a "
        "World Cup* page) is random and frequently crowns someone else. Aggregate "
        "odds answer 'who's most likely overall'; a single play answers 'what could "
        "happen this time'."
    )

    st.markdown("### 📏 Time-based backtest")
    st.markdown(
        "Models are trained on matches **before 2022** and scored on the unseen "
        "**2022+** holdout — never a random split. Lower is better; **RPS** is the "
        "headline because match results are *ordered* (a win is closer to a draw "
        "than to a loss)."
    )
    st.caption(
        f"Train: {BACKTEST_TRAIN_N:,} matches  ·  Holdout: {BACKTEST_TEST_N:,} matches."
    )
    st.dataframe(
        BACKTEST.style.format({"RPS": "{:.4f}", "LogLoss": "{:.4f}", "Brier": "{:.4f}"}),
        hide_index=True, width="stretch",
    )

    st.markdown("### ⚖️ An honest note on uncertainty")
    st.markdown(
        """
- These are **probabilities, not predictions** — a 25% favourite loses three times
  in four.
- Squad changes, injuries, form and tactics are **not** modelled; only results are.
- Tournament games are treated as **neutral-venue**, and bracket pairings follow a
  documented default structure.
        """
    )


# --- admin dashboard ----------------------------------------------------------


def render_admin():
    """Minimal usage dashboard, reached only via ?admin=<token>."""
    inject_theme()
    st.markdown(
        '<div class="wc-hero"><span class="wc-eyebrow">Admin</span>'
        '<div class="wc-title">Usage dashboard</div>'
        '<p class="wc-sub">Anonymous, per-session usage. No login, no personal '
        'data — access is gated by the secret URL token.</p></div>',
        unsafe_allow_html=True,
    )
    s = analytics.summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Visitors", f"{s['total_users']:,}")
    c2.metric("Total clicks", f"{s['total_clicks']:,}")
    c3.metric("Avg clicks / visitor", f"{s['avg_clicks']:.1f}")

    if s["per_user"]:
        df = pd.DataFrame(s["per_user"])
        top = df.head(30)
        fig = px.bar(top, x="session", y="clicks", color="clicks",
                     color_continuous_scale=WC_SCALE)
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="clicks")
        _style_fig(fig, height=360)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df.rename(columns={"session": "Session", "clicks": "Clicks"}),
                     hide_index=True, width="stretch")
    else:
        st.info("No usage recorded yet.")
    st.caption(
        "A 'click' is a simulation action (kick-off / run). Counts are anonymous "
        "per browser session and reset if the server restarts."
    )


# --- layout -------------------------------------------------------------------

PAGES = {
    "⚽ Play a Cup": page_play,
    "🏆 Title odds": page_odds,
    "🔮 Match predictor": page_match,
    "📖 How it works": page_how,
}


def main():
    """Render the page: admin gate, then hero + sidebar nav + the chosen page."""
    inject_theme()

    # Hidden admin view, gated by a secret URL token (no login).
    admin_val = st.query_params.get("admin")
    if admin_val is not None and admin_val == _admin_token():
        render_admin()
        return

    _ensure_session()
    hero()

    st.sidebar.markdown("### Navigate")
    choice = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption("Pick a mode, then hit simulate. Every run is a fresh draw of fate. ⚽")

    PAGES[choice]()


if __name__ == "__main__":
    main()
