# src/make_match_flags.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from dateutil import parser as dtparser, tz

IN_CSV  = "data/processed/matches_with_weather.csv"
OUT_CSV = "data/processed/matches_with_weather_features.csv"

# Map RugbyPass timezone abbreviations to canonical tz names
TZ_ABBR = {
    "SAST": "Africa/Johannesburg",
    "BST":  "Europe/London",
    "GMT":  "Etc/GMT",
    "UTC":  "UTC",
    "IST":  "Europe/Dublin",
}

# --- South African venues (exact display names as in your CSV) ---
SA_VENUES = {
    "Loftus Versfeld",
    "Ellis Park",
    "DHL Stadium",                 # leave here only if you intend SA logic for Cape Town
    "Hollywoodbets Kings Park",
    "Danie Craven Stadium",
}

# --- Italian venues ---
ITA_VENUES = {
    "Stadio Comunale di Monigo",
    "Stadio Sergio Lanfranchi",
}

# --- Main home stadiums by home team (strict, as you set) ---
MAIN_HOME_BY_TEAM = {
    # Ireland
    "Leinster": {"Aviva Stadium"},
    "Munster": {"Thomond Park"},
    "Ulster": {"Affidea Stadium"},     # Ravenhill/Kingspan (displayed as Affidea Stadium)
    "Connacht": {"Dexcom Stadium"},

    # Scotland
    "Edinburgh": {"Hive Stadium"},
    "Glasgow Warriors": {"Scotstoun Stadium"},

    # Wales
    "Cardiff": {"Cardiff Arms Park"},
    "Dragons RFC": {"Rodney Parade"},
    "Ospreys": {"Swansea.com Stadium"},
    "Scarlets": {"Parc y Scarlets"},

    # Italy
    "Benetton": {"Stadio Comunale di Monigo"},
    "Zebre": {"Stadio Sergio Lanfranchi"},

    # South Africa
    "Bulls": {"Loftus Versfeld"},
    "Lions": {"Ellis Park"},
    "Stormers": {"DHL Stadium"},
    "Sharks": {"Hollywoodbets Kings Park"},
}

def parse_kickoff(dt_str: str) -> pd.Timestamp | None:
    """Parse 'Fri 20th September 2024, 08:35pm SAST' into tz-aware Timestamp."""
    if not isinstance(dt_str, str) or not dt_str.strip():
        return None
    parts = dt_str.strip().split()
    tz_token = parts[-1] if parts else ""
    tz_name = TZ_ABBR.get(tz_token.upper())
    if tz_name:
        dt_wo_tz = " ".join(parts[:-1])
        naive = dtparser.parse(dt_wo_tz, dayfirst=True)
        return pd.Timestamp(naive).tz_localize(tz.gettz(tz_name))
    # fallback
    t = dtparser.parse(dt_str, dayfirst=True)
    ts = pd.Timestamp(t)
    return ts if ts.tzinfo else ts.tz_localize("UTC")

def is_in_south_africa(venue: str) -> bool:
    return (venue or "").strip() in SA_VENUES

def is_in_italy(venue: str) -> bool:
    return (venue or "").strip() in ITA_VENUES

def is_main_home_stadium(home_team: str, venue: str) -> bool:
    homes = MAIN_HOME_BY_TEAM.get((home_team or "").strip())
    if not homes:
        return False
    return (venue or "").strip() in homes

def bucket_hour_three(h: int) -> str:
    """
    Three-category bucket:
      - early_afternoon: 12 <= h < 15
      - late_afternoon:  15 <= h < 18
      - evening:         everything else (including night/morning)
    """
    if 12 <= h < 15:
        return "early_afternoon"
    if 15 <= h < 18:
        return "late_afternoon"
    return "evening"

def main():
    df = pd.read_csv(IN_CSV)

    # --- Compute SAST kickoff hour from Date_time ---
    kickoff_ts = df["Date_time"].apply(parse_kickoff)
    # Hour in SAST string (already localized via SAST in parse function when token present)
    kickoff_hour_sast = kickoff_ts.dt.hour

    # --- Adjust hour for simplified local-time heuristic ---
    # If venue in SA or ITA: keep SAST hour
    # Else (UK/IE/Scotland/Wales etc.): subtract 2 hours (wrap around 0-23)
    def adjusted_hour(row) -> int | None:
        v = row.get("Venue")
        h = row.get("_hour_sast")
        if pd.isna(h):
            return None
        h = int(h)
        if is_in_south_africa(v) or is_in_italy(v):
            return h
        # approximate local hour for UK/IE by subtracting 2
        return (h - 2) % 24

    df["_hour_sast"] = kickoff_hour_sast
    df["time_bucket"] = df.apply(adjusted_hour, axis=1).apply(
        lambda x: bucket_hour_three(int(x)) if pd.notna(x) else None
    )

    # --- Flags ---
    df["is_in_south_africa"] = df["Venue"].apply(is_in_south_africa)
    df["is_main_home_stadium"] = df.apply(
        lambda r: is_main_home_stadium(r.get("Home_team"), r.get("Venue")), axis=1
    )

    # Keep all original columns + add the three new ones (and drop helper column)
    extras = ["is_in_south_africa", "is_main_home_stadium", "time_bucket"]
    for c in extras:
        if c not in df.columns:
            df[c] = None

    if "_hour_sast" in df.columns:
        df = df.drop(columns=["_hour_sast"])

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV} (rows={len(df)})")

if __name__ == "__main__":
    main()
