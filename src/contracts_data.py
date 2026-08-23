"""Fetch and cache Basketball-Reference contract salaries and salary-cap
history. Same cache-first, browser-UA pattern as bref_data.py/nba_data.py.
Parsing is split from the network fetch (parse_* vs fetch_*) so it can be
tested against saved HTML fixtures with no network calls."""

import io
import re
import time
from pathlib import Path

import pandas as pd
from lxml import html as lxml_html
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "bref_contracts"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

CONTRACTS_URL = "https://www.basketball-reference.com/contracts/players.html"
CAP_HISTORY_URL = "https://www.basketball-reference.com/contracts/salary-cap-history.html"


def _dollars_to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )


def parse_contracts_html(html_text: str) -> pd.DataFrame:
    """One row per player: player_id (BR's own id, for joining to
    bref_advanced data), player name, team, one column per contract year
    (named after the actual season, e.g. '2026-27'), and total guaranteed
    money. Years with no salary listed (contract doesn't reach that far)
    come through as NaN, not zero - a real distinction, since zero would
    imply a $0 salary rather than "not under contract that year"."""
    tables = pd.read_html(io.StringIO(html_text), attrs={"id": "player-contracts"})
    df = tables[0]
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    # BR repeats the header every ~20 rows, sometimes as a literal "Player"
    # row and sometimes as a colspan-only divider row that pandas reads as
    # all-NaN - drop both, or the player_id list below (built from only the
    # real player rows) ends up misaligned with this dataframe
    df = df[(df["Player"] != "Player") & df["Player"].notna()].copy()

    year_cols = [c for c in df.columns if re.match(r"^\d{4}-\d{2}$", str(c))]
    for col in year_cols + ["Guaranteed"]:
        df[col] = _dollars_to_number(df[col])

    tree = lxml_html.fromstring(html_text)
    # select only rows with an actual player cell (data-append-csv present)
    # so header/divider rows never enter this list in the first place,
    # instead of relying on positional alignment with the dataframe above
    rows = tree.xpath(
        '//table[@id="player-contracts"]/tbody/tr[td[@data-stat="player"]/@data-append-csv]'
    )
    player_ids = [row.xpath('./td[@data-stat="player"]/@data-append-csv')[0] for row in rows]
    if len(player_ids) != len(df):
        raise ValueError(
            f"player row count mismatch: {len(df)} rows in the parsed table vs "
            f"{len(player_ids)} rows with a player id - the header/divider filtering "
            "above is no longer catching everything pandas is including"
        )
    df["player_id"] = player_ids

    df = df.rename(columns={"Player": "player", "Tm": "team", "Guaranteed": "guaranteed"})
    keep = ["player_id", "player", "team", *year_cols, "guaranteed"]
    df = df[keep]

    # a handful of players (recent trades/sign-and-trades where the page
    # hasn't settled on one team) show up as two rows with identical dollar
    # figures under two different teams - same real contract counted twice,
    # not two obligations to sum, so keep just one row per player
    df = df.drop_duplicates(subset="player_id", keep="first")
    return df.reset_index(drop=True)


def parse_cap_history_html(html_text: str) -> pd.DataFrame:
    """One row per season: season label ('2026-27') and that season's real
    salary cap. Basketball-Reference wraps this particular table in an HTML
    comment (a trick they use on some secondary tables to block naive
    scraping) - strip the comment markers before handing it to pandas, or
    the table is invisible to a normal parse."""
    uncommented = html_text.replace("<!--", "").replace("-->", "")
    tables = pd.read_html(io.StringIO(uncommented), attrs={"id": "salary_cap_history"})
    df = tables[0]
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    df = df[df["Year"] != "Year"].copy()
    df["cap"] = _dollars_to_number(df["Salary Cap"])
    df = df.rename(columns={"Year": "season"})
    return df[["season", "cap"]].reset_index(drop=True)


def fetch_contracts(sleep: float = 3.0) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "players.json"
    if path.exists():
        return pd.read_json(path)

    resp = requests.get(CONTRACTS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    df = parse_contracts_html(resp.text)
    df.to_json(path, orient="records")
    time.sleep(sleep)
    return df


def fetch_cap_history(sleep: float = 3.0) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "cap_history.json"
    if path.exists():
        return pd.read_json(path)

    resp = requests.get(CAP_HISTORY_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    df = parse_cap_history_html(resp.text)
    df.to_json(path, orient="records")
    time.sleep(sleep)
    return df


if __name__ == "__main__":
    contracts = fetch_contracts()
    print(f"{len(contracts)} player contracts")
    print(contracts.head(10).to_string())

    cap_history = fetch_cap_history()
    print(f"\n{len(cap_history)} seasons of cap history")
    print(cap_history.tail(10).to_string())
