import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
FAME_DB_FILE = "artist_fame.json"
LASTFM_API_URL = "https://ws.audioscrobbler.com/2.0/"
DEFAULT_TIMEOUT_SECONDS = 8


def normalize_artist_name(value) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold().strip()
    return re.sub(r"\s+", " ", text)


def _parse_int(value):
    if value is None or value == "":
        return None
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


@dataclass
class FameResult:
    listeners: int
    source: str            # "database" | "lastfm" | "prior"
    fame_estimated: bool   # True only for the low-prior fallback
    normalized_name: str
    matched_name: str | None = None   # what Last.fm autocorrected to, if anything
    note: str | None = None

    def to_dict(self):
        return {
            "listeners": self.listeners,
            "source": self.source,
            "fame_estimated": self.fame_estimated,
            "normalized_name": self.normalized_name,
            "matched_name": self.matched_name,
            "note": self.note,
        }


class FameResolver:
    def __init__(self, db_path=None, api_key=None, timeout=DEFAULT_TIMEOUT_SECONDS):
        self.db_path = Path(db_path) if db_path else ARTIFACT_DIR / FAME_DB_FILE
        self.api_key = api_key or os.getenv("LASTFM_API_KEY")
        self.timeout = timeout
        self._db = None
        self._prior = None
        self._lastfm_cache = {}

    def _load(self):
        if self._db is not None:
            return
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Missing {self.db_path}. Run: python -m model.fame  (builds the DB)"
            )
        payload = json.loads(self.db_path.read_text())
        self._db = payload["listeners"]
        self._prior = int(payload["low_prior_listeners"])

    def resolve(self, artist_name) -> FameResult:
        self._load()
        name = normalize_artist_name(artist_name)

        if not name:
            return FameResult(self._prior, "prior", True, name,
                              note="empty artist name; using low prior")

        # 1) local DB
        hit = self._db.get(name)
        if hit is not None:
            return FameResult(int(hit), "database", False, name)

        # 2) Last.fm
        lastfm = self._lastfm_lookup(name)
        if lastfm is not None:
            listeners, matched = lastfm
            return FameResult(listeners, "lastfm", False, name, matched_name=matched)

        # 3) low prior
        note = "artist not found in DB or Last.fm; using low prior (likely obscure)"
        if not self.api_key:
            note = "artist not in DB and LASTFM_API_KEY unset; using low prior"
        return FameResult(self._prior, "prior", True, name, note=note)

    def _lastfm_lookup(self, normalized_name):
        """Return (listeners, corrected_name) or None. Never raises."""
        if not self.api_key:
            return None
        if normalized_name in self._lastfm_cache:
            return self._lastfm_cache[normalized_name]

        result = None
        try:
            import requests

            response = requests.get(
                LASTFM_API_URL,
                params={
                    "method": "artist.getinfo",
                    "artist": normalized_name,
                    "api_key": self.api_key,
                    "format": "json",
                    "autocorrect": 1,   # matches the enrichment's name-fallback behavior
                },
                timeout=self.timeout,
            )
            data = response.json()
            # Last.fm signals failure with an "error" key and HTTP 200, so check
            # the body, not just the status code.
            if response.status_code == 200 and "error" not in data:
                stats = data.get("artist", {}).get("stats", {})
                listeners = _parse_int(stats.get("listeners"))
                if listeners is not None:
                    result = (listeners, data["artist"].get("name"))
        except Exception:
            result = None   # any network/parse failure -> fall through to the prior

        self._lastfm_cache[normalized_name] = result
        return result


# --------------------------------------------------------------------------- build
def build_fame_db(source=None, out_path=None, prior_quantile=0.25):
    """Build the serving DB from the enrichment output. Run once (or on refresh).

        python -m model.fame
    """
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    source = Path(source) if source else root / "data" / "interim" / "artist_popularity.parquet"
    out_path = Path(out_path) if out_path else ARTIFACT_DIR / FAME_DB_FILE

    df = pd.read_parquet(source)
    df = df.dropna(subset=["normalized_artist_name"]).drop_duplicates(
        "normalized_artist_name", keep="last"
    )
    listeners = pd.to_numeric(df["lastfm_listeners"], errors="coerce")
    df = df.assign(listeners=listeners).dropna(subset=["listeners"])

    mapping = dict(zip(df["normalized_artist_name"], df["listeners"].astype(int)))
    prior = int(df["listeners"].quantile(prior_quantile))

    payload = {
        "low_prior_listeners": prior,
        "prior_quantile": prior_quantile,
        "n_artists": len(mapping),
        "listeners": mapping,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"Wrote {out_path}: {len(mapping)} artists, low prior = {prior} "
          f"(p{int(prior_quantile * 100)} of known artists)")
    return payload


if __name__ == "__main__":
    build_fame_db()
