# TradeLedger

An end-to-end NBA player valuation and trade decision pipeline: RAPM value
model → aging-curve projections → contract surplus value → roster fit model
→ decision tool, built on public data.

Full architecture, phase breakdown, and current status: see
[docs/PLAN.md](docs/PLAN.md). Treat that file as the source of truth on
scope and progress — update it as phases complete.

## Stack

- Python 3.14, managed with `uv` (`.venv/`, `pyproject.toml`, `uv.lock`)
- Core deps: `nba_api`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `requests`, `pyarrow`, `streamlit`
- `[tool.uv] package = false` — this is an application/analysis project, not
  a distributable library, so no `src/<pkgname>/` build layout needed

## Repo layout

- `app.py` — the Phase 5 decision tool: a Streamlit UI (`streamlit run
  app.py`) with no logic of its own, wiring `src/decision_views.py`'s
  functions to a sidebar view picker and a player detail page
- `src/` — real, reusable code (data fetch, possession building, models)
- `tests/` — pytest suite, mirrors `src/` (`tests/test_foo.py` for
  `src/foo.py`); `tests/fixtures/` holds saved HTML/data samples so parsing
  logic is testable with no network calls
- `data/raw/` — untouched API pulls, gitignored (reproducible via `src/`
  scripts, not committed as files)
- `data/processed/` — derived/cleaned datasets, gitignored
- `notebooks/` — exploration, not production code
- `docs/` — plan and methodology writeup

## Testing

- `uv run pytest` runs the full suite.
- Split parsing/computation logic from network/file I/O (`parse_*` /
  `fetch_*`, or similar) so the interesting logic is unit-testable without
  hitting a live site or needing the full data cache built.
- New modules should ship with tests alongside them, not as a follow-up.

## Git workflow (always follow)

- Never commit directly to `main`. Branch off `main` for each isolated
  piece of work (a phase, a feature, a fix).
- Commit often within a branch — small, logical commits, not one giant
  commit at the end.
- Tell the user when a piece of work is done before moving to the next
  thing; check in before merging a branch back to `main`.
