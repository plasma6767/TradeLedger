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
- [ ] **Phase 1 — Value layer**: pull + validate one season of play-by-play
      data, build possession-level dataset, fit ridge-regression RAPM
- [ ] Phase 2 — Projection layer (aging curves)
- [ ] Phase 3 — Cost layer (contracts, surplus value)
- [ ] Phase 4 — Fit layer (shot/play-type complementarity)
- [ ] Phase 5 — Decision layer (tool + memo)

## Working conventions

- Branch off `main` for each isolated piece of work; never commit directly
  to `main`. Commit often within a branch. Merge back only after checking in.
- Raw data pulls live in `data/raw/` (gitignored — reproducible via scripts
  in `src/`, not committed as files).
- License: MIT.
