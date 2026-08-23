"""TradeLedger's Phase 5 decision tool: a local Streamlit app over the
player_universe/decision_views layer (src/). No logic lives here - every
number on screen comes straight from a src/ function; this file is only
responsible for laying it out and wiring up navigation (pick a view in
the sidebar, click a row to open a player's detail page).

Run with: streamlit run app.py"""

import pandas as pd
import streamlit as st
from nba_api.stats.static import teams as nba_teams

from src.decision_views import (
    get_player_detail,
    rank_by_current_bpm,
    rank_by_fit,
    rank_by_projected_bpm,
    rank_by_rapm,
    rank_by_surplus,
)
from src.player_universe import build_player_universe

st.set_page_config(page_title="TradeLedger", layout="wide")

TEAM_ABBREVIATIONS = sorted(t["abbreviation"] for t in nba_teams.get_teams())


@st.cache_data(show_spinner="Loading player universe (RAPM, BPM, projections, surplus value)...")
def load_universe() -> pd.DataFrame:
    return build_player_universe()


@st.cache_data(show_spinner="Computing fit against this team's roster...")
def load_fit_view(team: str) -> pd.DataFrame:
    return rank_by_fit(load_universe(), team)


@st.cache_data(show_spinner="Loading player detail...")
def load_player_detail(nba_player_id: int) -> dict:
    return get_player_detail(load_universe(), nba_player_id)


def open_player_detail(nba_player_id) -> None:
    st.session_state.selected_player_id = nba_player_id
    st.session_state.nav = "Player detail"


def show_selectable_table(table: pd.DataFrame, display_cols: list[str], id_col: str) -> None:
    """A sortable table where clicking a row opens that player's detail
    page - Streamlit's built-in row-selection event on st.dataframe."""
    table = table.reset_index(drop=True)
    event = st.dataframe(
        table[display_cols],
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = event.selection.rows
    if selected_rows:
        open_player_detail(table.iloc[selected_rows[0]][id_col])
        st.rerun()


def render_browse() -> None:
    universe = load_universe()
    st.caption(f"{len(universe)} players")

    view = st.sidebar.selectbox(
        "Rank players by",
        ["Surplus score", "RAPM", "Current BPM", "Projected BPM", "Fit for a team"],
    )

    if view == "Surplus score":
        st.subheader("Ranked by surplus score — best contract value first")
        st.caption("'minutes' is this season's real minutes played — surplus score and RAPM have no "
                   "minutes floor, so a small sample can rank high; use it to judge how much to trust the number.")
        table = rank_by_surplus(universe)
        show_selectable_table(
            table,
            ["player", "team", "age", "minutes", "surplus_score", "avg_value", "avg_cap_pct", "years_left", "reasoning"],
            id_col="nba_player_id",
        )

    elif view == "RAPM":
        st.subheader("Ranked by last season's real RAPM")
        st.caption("'minutes' is this season's real minutes played — RAPM has no minutes floor, "
                   "so a small sample can still rank high; use it to judge how much to trust the number.")
        table = rank_by_rapm(universe)
        show_selectable_table(
            table, ["player", "team", "age", "minutes", "rapm", "reasoning"], id_col="nba_player_id"
        )

    elif view == "Current BPM":
        st.subheader("Ranked by this season's real BPM")
        table = rank_by_current_bpm(universe)
        show_selectable_table(
            table, ["player", "team", "age", "minutes", "current_bpm", "reasoning"], id_col="nba_player_id"
        )

    elif view == "Projected BPM":
        years = st.sidebar.selectbox("Years forward", [1, 2, 3], index=2)
        st.subheader(f"Ranked by BPM projected {years} year{'s' if years > 1 else ''} forward")
        table = rank_by_projected_bpm(universe, years=years)
        show_selectable_table(
            table, ["player", "team", "age", "minutes", f"bpm_+{years}y", "reasoning"], id_col="nba_player_id"
        )

    elif view == "Fit for a team":
        team = st.sidebar.selectbox("Team", TEAM_ABBREVIATIONS, index=TEAM_ABBREVIATIONS.index("OKC"))
        st.subheader(f"Candidates ranked by fit against {team}'s current roster")
        table = load_fit_view(team)
        show_selectable_table(
            table,
            ["player_name", "team", "fit_grade", "raw_fit_pct", "minutes", "surplus_score", "reasoning"],
            id_col="player_id",
        )


def render_player_detail() -> None:
    universe = load_universe()

    st.sidebar.button("← Back to Browse", on_click=lambda: st.session_state.update(nav="Browse"))

    options = universe.sort_values("player")
    labels = [f"{row['player']} ({row['team'] if pd.notna(row['team']) else 'no current team'})"
              for _, row in options.iterrows()]
    default_idx = 0
    if st.session_state.get("selected_player_id") in options["nba_player_id"].to_numpy():
        default_idx = list(options["nba_player_id"]).index(st.session_state.selected_player_id)
    picked = st.selectbox("Player", labels, index=default_idx)
    nba_player_id = options.iloc[labels.index(picked)]["nba_player_id"]

    detail = load_player_detail(nba_player_id)

    st.subheader(f"{detail['player']}" + (f" — {detail['team']}" if pd.notna(detail["team"]) else ""))

    cols = st.columns(7)
    cols[0].metric("Age", f"{detail['age']:.0f}")
    cols[1].metric("Minutes", f"{detail['minutes']:.0f}" if pd.notna(detail["minutes"]) else "—")
    cols[2].metric("RAPM", f"{detail['rapm']:+.1f}")
    cols[3].metric("Current BPM", f"{detail['current_bpm']:+.1f}" if pd.notna(detail["current_bpm"]) else "—")
    cols[4].metric("Surplus score", f"{detail['surplus_score']:+.0f}")
    cols[5].metric("Avg cap %", f"{detail['avg_cap_pct'] * 100:.1f}%")
    cols[6].metric("Years left", f"{detail['years_left']:.0f}")

    st.markdown("**Aging-curve projection** (this player's own BPM walked forward)")
    if detail["projection"]:
        for step in detail["projection"]:
            st.write(f"+{step['years']}y: **{step['bpm']:.1f}** {step['band']}")
    else:
        st.caption("No projection available (below the current-season minutes floor).")

    diet_cols = st.columns(2)
    with diet_cols[0]:
        st.markdown("**Shot-zone diet** (% of field goal attempts)")
        st.bar_chart(pd.Series(detail["shot_diet"]))
    with diet_cols[1]:
        st.markdown("**Play-type diet** (% of offensive possessions)")
        st.bar_chart(pd.Series(detail["playtype_diet"]))


st.title("TradeLedger")

if "nav" not in st.session_state:
    st.session_state.nav = "Browse"
if "selected_player_id" not in st.session_state:
    st.session_state.selected_player_id = None

if st.session_state.nav == "Browse":
    render_browse()
else:
    render_player_detail()
