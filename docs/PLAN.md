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
- Basketball-Reference — historical season stats (for aging curves), salary
  data
- Spotrac / HoopsHype — current contracts and cap figures

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
- [ ] Phase 2 — Projection layer (aging curves) — up next
- [ ] Phase 3 — Cost layer (contracts, surplus value)
- [ ] Phase 4 — Fit layer (shot/play-type complementarity)
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

## Working conventions

- Branch off `main` for each isolated piece of work; never commit directly
  to `main`. Commit often within a branch. Merge back only after checking in.
- Raw data pulls live in `data/raw/` (gitignored — reproducible via scripts
  in `src/`, not committed as files). Processed datasets live in
  `data/processed/` (also gitignored — regenerate via
  `build_season_dataset.py`, not committed as files).
- License: all rights reserved (see `LICENSE`) — public repo for viewing,
  not open source.
