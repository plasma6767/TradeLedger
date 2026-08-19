# TradeLedger

An end-to-end NBA player valuation and trade decision pipeline: RAPM value
model → aging-curve projections → contract surplus value → roster fit model
→ decision tool, built on public data.

Full architecture, phase breakdown, and current status: see
[docs/PLAN.md](docs/PLAN.md). Treat that file as the source of truth on
scope and progress — update it as phases complete.

## Stack

- Python 3.14, managed with `uv` (`.venv/`, `pyproject.toml`, `uv.lock`)
- Core deps: `nba_api`, `pandas`, `numpy`, `scikit-learn`, `requests`
- `[tool.uv] package = false` — this is an application/analysis project, not
  a distributable library, so no `src/<pkgname>/` build layout needed

## Repo layout

- `src/` — real, reusable code (data fetch, possession building, models)
- `data/raw/` — untouched API pulls, gitignored (reproducible via `src/`
  scripts, not committed as files)
- `data/processed/` — derived/cleaned datasets, gitignored
- `notebooks/` — exploration, not production code
- `docs/` — plan and methodology writeup

## Git workflow (always follow)

- Never commit directly to `main`. Branch off `main` for each isolated
  piece of work (a phase, a feature, a fix).
- Commit often within a branch — small, logical commits, not one giant
  commit at the end.
- Tell the user when a piece of work is done before moving to the next
  thing; check in before merging a branch back to `main`.
