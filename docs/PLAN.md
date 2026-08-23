# TradeLedger — Project Plan

## Overview

An end-to-end NBA player valuation and roster decision pipeline built on
public data. Rather than a single isolated metric, the project chains
several layers together: player impact, forward projection, contract cost,
and roster fit, culminating in a decision-support tool.

## Architecture: five layers

1. **Value layer — RAPM (Regularized Adjusted Plus-Minus)**
   Possession-level ridge regression on play-by-play data estimates each
   player's offensive/defensive impact per 100 possessions, isolated from
   who they shared the floor with. This is the statistical foundation
   everything else builds on.

2. **Projection layer — aging curves**
   Fit historical aging curves (Basketball-Reference season data) to project
   each player's value 1-3 years forward, with uncertainty bands rather than
   point estimates.

3. **Cost layer — contracts & surplus value**
   Convert projected wins to $/WAR using cap and salary data (current
   contracts). Surplus value = projected production value − contract cost.

4. **Fit layer — roster complementarity**
   Use shot chart coordinates and play-type frequency (both available via
   `nba_api`) to quantify whether a player's profile complements or
   duplicates what a specific roster already has (shot diet overlap,
   spacing gaps).

5. **Decision layer — tool + writeup**
   An interactive tool: pick a team, get ranked trade/free-agent targets by
   surplus value + fit, with reasoning shown. Packaged with a methodology
   writeup including honest validation (holdout performance) and stated
   limitations.

## Data sources (all public)

- `nba_api` — play-by-play, lineups, shot chart coordinates, play-type data
- Basketball-Reference — historical season stats (for aging curves),
  contract salaries, and salary cap history (Spotrac and HoopsHype were
  considered for contracts/cap data but ruled out: Spotrac blocks
  automated requests outright, HoopsHype's salary table is JS-rendered and
  paginated - Basketball-Reference's own contracts pages turned out to be
  a clean, scrapable source using the same approach already proven for
  season stats)

## Phases / status

- [x] Repo scaffolded: `uv`-managed venv (Python 3.14), `pyproject.toml`,
      folder structure, git branch workflow established
- [x] **Phase 1 — Value layer**: complete. Possession-level dataset built
      for the full 2025-26 season (1221/1230 games; 9 excluded for
      genuinely corrupted `GameRotation` data, all documented), joined to
      exact lineups, and a ridge-regression RAPM model fit and validated.
      Every game's reconstructed score matches the NBA's actual final
      score exactly except one, which is a confirmed data error in the
      NBA's own feed, not ours. See `src/` module breakdown below.
- [x] **Phase 2 — Projection layer**: complete. Aging curve fit on 15
      seasons of Basketball-Reference BPM (2011-12 through 2025-26),
      since RAPM only exists for the one season Phase 1 covers and
      isn't feasible to reprocess historically. Curve stays entirely in
      BPM's own units — it projects a player's own BPM forward as a
      separate signal, never converted into or blended with RAPM.
      Confidence bands are measured from the real spread of how players
      actually varied at each age, not an assumed round number. See
      `src/` module breakdown below.
- [x] **Phase 3 — Cost layer**: complete. Ranks every player's contract by
      value relative to cost, league-wide. Value is this season's real RAPM
      (Phase 1) walked forward using only the aging curve's plain average
      delta per age (Phase 2's `project_bpm`, not the simulated-band
      version - that's reserved for a future detail view), averaged across
      however many years are left on the player's deal. Cost is each of
      those years' salary as a percentage of that season's real salary cap
      (extrapolated from real recent cap growth for years the league
      hasn't published yet), also averaged across the same years, so a
      1-year deal and a 5-year deal are judged on equal footing. The two
      are combined by converting each to a percentile rank across the
      league and subtracting (`value_percentile - cost_percentile`) rather
      than dividing one by the other, since a ratio becomes undefined/
      unstable for any player whose value sits near replacement level -
      percentile subtraction stays bounded for everyone, so no player is
      excluded from the ranking. See `src/` module breakdown below.
- [x] **Phase 4 — Fit layer**: complete. Reduces every currently-rostered
      player to two "diets" - % of shots by court zone, % of offensive
      possessions by play type - and blends a team's current roster into a
      team-level diet weighted by playing time. A candidate's fit against
      that team is a distributional overlap score (shared ground between
      the two diets, zone/play-type by zone/play-type) subtracted from 1.
      That raw number is structurally compressed (no real player is ever
      100% unlike a team, since everyone shares some baseline habits with
      every roster), so a genuinely great complement still only scores
      ~60-70% raw - reading it at face value undersells real fits. The
      headline conclusion is instead a letter grade calibrated against the
      real spread of fit scores within the actual candidate pool (top 10%
      -> A, bottom 10% -> F), not fixed thresholds against the raw number's
      own scale. Validated against a real team (OKC): top-graded fits were
      almost entirely traditional rim-running bigs, bottom-graded were
      almost entirely perimeter wings - correctly identifying the actual
      gap in OKC's real (guard/wing-heavy) roster construction rather than
      producing plausible-looking noise. See `src/` module breakdown below.
- [ ] Phase 5 — Decision layer (tool + memo)

## Phase 1 module breakdown (`src/`)

- `nba_data.py` — fetches and caches play-by-play, boxscores, rotations,
  and game lists from `nba_api`, with retry/backoff. Everything downstream
  reads from this cache, never hits the API directly.
- `possessions.py` — the core logic: walks play-by-play event by event,
  tracks whose possession it is, and tallies points live as it goes
  (never via a timestamp lookup, since events sharing a clock reading —
  very common with fouls/free throws/turnovers — are otherwise
  indistinguishable). Handles and-1s, flagrant/technical/clear-path
  fouls, and team-attributed events with no team ID in the raw data.
- `possession_rows.py` — joins possessions to `GameRotation` lineups,
  splitting a possession further if a substitution falls inside it.
  Also where the two known data-corruption checks live (overlapping
  rotation stints; missing stints that still leave <5 players on the
  floor at some moments) — a game failing either check gets excluded
  outright rather than trusting partial/bad lineup data.
- `build_season_dataset.py` — runs the above across every game in a
  season, saving each game's result immediately (resumable — a crash
  partway through doesn't lose earlier work), then combines everything
  into `data/processed/possessions_2025-26.parquet`.
- `rapm.py` — builds the sparse design matrix (one column per player),
  fits ridge regression, picks the regularization strength via
  cross-validation split by game (not by possession, to avoid leaking
  lineups from the same game across train/test), outputs the RAPM
  leaderboard.
- `validate_clock_alignment.py`, `validate_rotation.py` — standalone
  scripts used to validate the approach before trusting it at scale; not
  part of the production pipeline but kept for reference.

## Phase 2 module breakdown (`src/`)

- `bref_data.py` — fetches and caches Basketball-Reference's season
  advanced-stats tables (BPM, VORP, Age), keyed on BR's own per-player ID
  rather than name, since multiple different NBA players have shared an
  exact printed name within this window. Handles traded players' rows
  (BR labels the combined-season row "2TM"/"3TM"/"4TM", not "TOT") and
  the page's encoding (UTF-8, though the server doesn't declare it).
- `aging_curve.py` — pools every player's real year-over-year BPM change
  by age, minutes-weighted, into one curve. Ages with too few
  player-transitions behind them share one pooled late-decline rate
  instead of each claiming its own unreliable number. Confidence bands
  come from the actual measured spread of real players' deltas, not an
  assumed distribution; multi-year projections simulate forward by
  resampling real per-age deltas rather than gluing single-year bands
  together.
- `name_matching.py` — joins nba_api names (used by `rapm.py`) to
  Basketball-Reference names (used by `bref_data.py`) — the two sources
  share no common player ID. Normalizes accents/punctuation/suffixes,
  then falls back to a small manual map for the handful of real
  nickname/name-order mismatches normalization can't resolve.
- `player_report.py` — combines Phase 1's RAPM, this season's real BPM,
  and the aging-curve-projected BPM trend per player into one table —
  three separate signals shown side by side, never merged into a single
  score.

## Phase 3 module breakdown (`src/`)

- `contracts_data.py` — fetches and caches Basketball-Reference's contract
  salaries and salary-cap-history tables. Parsing is split from the network
  fetch (`parse_*` vs `fetch_*`) so it's testable against saved HTML
  fixtures with no network calls. Handles two real data quirks found while
  validating against the live page: a mid-table colspan divider row that
  (if not filtered identically on both the pandas and the raw-HTML side)
  silently misaligns every player_id after it, and a handful of
  recently-traded players BR lists twice under two different teams with
  identical dollar figures - deduped rather than summed, since it's the
  same real contract, not two obligations. The cap-history table is
  wrapped in an HTML comment (a trick BR uses on some secondary tables to
  block naive scraping), stripped before parsing.
- `surplus_value.py` — builds the ranked surplus-value table: joins
  Phase 1's RAPM (via `name_matching`), each player's current age (via BR's
  own player ID, shared directly with the contracts table - no fuzzy
  matching needed there), and contract data; projects value forward per
  the Phase 3 methodology above; computes cost as average cap percentage
  over the same years (estimating unpublished future caps from real recent
  cap growth); and combines both into `surplus_score` via percentile-rank
  subtraction.

## Phase 4 module breakdown (`src/`)

- `roster_data.py` — fetches and caches current team rosters via
  `nba_api`'s `commonteamroster`, one call per team. Basketball-Reference's
  season file (`bref_data.py`) deliberately keeps only the combined-season
  row for traded players, so it has no reliable "who's on this roster
  right now" signal - this module is the actual source of truth for that,
  since fit is inherently team-scoped.
- `shot_profile_data.py` — fetches and caches the two raw playing-style
  inputs: shot location (`shotchartdetail`, one call per player - no bulk
  mode exists for that endpoint) and play-type frequency
  (`synergyplaytypes`, one call per play type - that endpoint returns
  every player at once, so 11 calls covers the whole league instead of
  one per player).
- `roster_fit.py` — the fit math, and the module that ties everything
  together. Reduces each player to two "diets" (% of shots by court zone,
  % of possessions by play type - with zones/play-types a player never
  touched filled in as 0, not dropped, since a player is simply absent
  from a category's table rather than present with a 0). Blends a team's
  current roster into one team-level diet weighted by playing time (total
  synergy possessions, used as a proxy since it's already part of this
  phase's own data rather than pulling in Phase 2/3's cross-source name
  matching just for a weight). A candidate's fit is `1 - overlap`, where
  overlap is the summed shared ground between the candidate's and team's
  diets category by category; players traded mid-season have one row per
  team stint in this data (no combined row like BR's contracts table), so
  stints are combined via a games-played-weighted average, not summed or
  arbitrarily picked. The headline `fit_grade` (A-F) is calibrated against
  the real spread of raw fit scores within the actual candidate pool
  rather than fixed thresholds, since the raw score is structurally
  compressed - no real player is ever a 100% unlike a real team, so even a
  great complement only scores ~60-70% raw, which reads as mediocre taken
  at face value but is actually the top of the real distribution.

## Working conventions

- Branch off `main` for each isolated piece of work; never commit directly
  to `main`. Commit often within a branch. Merge back only after checking in.
- Raw data pulls live in `data/raw/` (gitignored — reproducible via scripts
  in `src/`, not committed as files). Processed datasets live in
  `data/processed/` (also gitignored — regenerate via
  `build_season_dataset.py`, not committed as files).
- License: all rights reserved (see `LICENSE`) — public repo for viewing,
  not open source.
