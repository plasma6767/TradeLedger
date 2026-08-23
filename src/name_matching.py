"""Match player names between nba_api (used by rapm.py) and
Basketball-Reference (used by bref_data.py / aging_curve.py) - the two
sources don't share a common player ID, so this is a best-effort name
join: strip accents/punctuation/suffixes, then fall back to a small
manual map for the handful of real nickname/name-order mismatches that
normalization can't resolve. Anything still unmatched is reported, never
silently dropped."""

import re
import unicodedata

# genuine mismatches normalization can't fix (nickname vs legal name,
# or family-name-first vs given-name-first) - maps normalized BR name ->
# normalized nba_api name. Extend as new seasons surface new cases.
MANUAL_OVERRIDES = {
    "adama-alpha bal": "adama bal",
    "ron holland": "ronald holland",
    "tre scott": "trevon scott",
    "yang hansen": "hansen yang",
}


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"\b(Jr|Sr|II|III|IV|V)\.?$", "", name).strip()
    name = re.sub(r"\.", "", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return MANUAL_OVERRIDES.get(name, name)


def match_names(bref_names: list[str], other_names: list[str]) -> tuple[dict, list[str]]:
    """Returns (bref_name -> other_name for every match, list of bref
    names that still couldn't be matched)."""
    other_by_norm = {normalize_name(n): n for n in other_names}

    matched = {}
    unmatched = []
    for name in bref_names:
        key = normalize_name(name)
        if key in other_by_norm:
            matched[name] = other_by_norm[key]
        else:
            unmatched.append(name)
    return matched, unmatched
