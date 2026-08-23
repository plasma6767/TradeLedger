# TradeLedger

An end-to-end NBA player valuation and roster decision pipeline built on
public data.

Rather than a single isolated metric, TradeLedger chains together several
layers of analysis:

1. **Value** — a possession-level RAPM (Regularized Adjusted Plus-Minus)
   model estimates each player's offensive/defensive impact per 100
   possessions, isolated from teammates and opponents on the floor.
2. **Projection** — aging curves fit on historical data project a player's
   value forward, with uncertainty rather than a single point estimate.
3. **Cost** — contract and salary cap data convert projected value into
   surplus value (production relative to cost).
4. **Fit** — shot chart and play-type data quantify how well a player's
   profile complements a specific roster, rather than assuming value is
   context-independent.
5. **Decision** — an interactive tool surfaces ranked trade/free-agent
   targets for a given team, combining value, projection, cost, and fit,
   with the reasoning shown.

See [docs/PLAN.md](docs/PLAN.md) for the full architecture and current
status.

## Data sources

All data is public: [`nba_api`](https://github.com/swar/nba_api) for
play-by-play, lineups, and shot chart data; Basketball-Reference for
historical season stats, contract salaries, and salary cap history.

## Setup

```bash
uv sync
```

This creates a `.venv/` and installs all dependencies from
`pyproject.toml`.

## Run the decision tool

```bash
uv run streamlit run app.py
```

Opens the Phase 5 decision tool in your browser: rank every player by
surplus score (the default), RAPM, current BPM, projected BPM, or fit
against a specific team's roster, and click into any player for their
full detail page (aging-curve projection, shot-zone and play-type diet).

## License

MIT — see [LICENSE](LICENSE).
