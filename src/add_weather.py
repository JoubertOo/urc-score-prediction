# src/add_weather.py
from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import requests
from dateutil import parser as dtparser, tz

# ---------- config ----------
IN_CSV  = "data/processed/matches_fixed_manually.csv"
OUT_CSV = "data/processed/matches_with_weather.csv"

CACHE_DIR = Path("cache/weather"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
GEOCODE_CACHE = CACHE_DIR / "geocode_cache.json"
WEATHER_CACHE_DIR = CACHE_DIR / "by_match"; WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Map common RugbyPass timezone abbreviations to canonical tz names
TZ_ABBR = {
    "SAST": "Africa/Johannesburg",
    "BST":  "Europe/London",   # British Summer Time
    "GMT":  "Etc/GMT",
    "UTC":  "UTC",
    "IST":  "Europe/Dublin",   # Ireland Summer Time
}

# Weathercode → short summary (Open-Meteo convention)
WX_MAP = {
    0:"Clear",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
    45:"Fog",48:"Depositing rime fog",
    51:"Drizzle: light",53:"Drizzle: moderate",55:"Drizzle: dense",
    56:"Freezing drizzle: light",57:"Freezing drizzle: dense",
    61:"Rain: slight",63:"Rain: moderate",65:"Rain: heavy",
    66:"Freezing rain: light",67:"Freezing rain: heavy",
    71:"Snow: slight",73:"Snow: moderate",75:"Snow: heavy",
    77:"Snow grains",
    80:"Rain showers: slight",81:"Rain showers: moderate",82:"Rain showers: violent",
    85:"Snow showers: slight",86:"Snow showers: heavy",
    95:"Thunderstorm: slight/moderate",96:"Thunderstorm with slight hail",99:"Thunderstorm with heavy hail"
}

# ---------- venue coordinate overrides ----------
# Add/adjust as needed (these are the *display* names you have in your CSV)
VENUE_COORDS: Dict[str, Tuple[float, float]] = {
    # Wales
    "Cardiff Arms Park": (51.4816, -3.1826),
    "Rodney Parade": (51.5896, -2.9917),
    "Parc y Scarlets": (51.6825, -4.1410),
    "Swansea.com Stadium": (51.6420, -3.9350),
    "Brewery Field": (51.5067, -3.5757),

    # Ireland
    "Aviva Stadium": (53.3351, -6.2283),
    "RDS Arena": (53.3256, -6.2317),
    "Thomond Park": (52.6740, -8.6426),
    "Virgin Media Park": (51.8849, -8.4803),  # Musgrave Park (Cork)
    "Dexcom Stadium": (53.2766, -9.0651),     # The Sportsground, Galway

    # Scotland
    "Hive Stadium": (55.9423, -3.2079),
    "Scottish Gas Murrayfield": (55.9426, -3.2409),
    "Scotstoun Stadium": (55.8825, -4.3391),
    "Hampden Park": (55.8258, -4.2521),

    # Italy
    "Stadio Comunale di Monigo": (45.6842, 12.2290),
    "Stadio Sergio Lanfranchi": (44.8209, 10.3296),  # Parma (Zebre)

    # South Africa
    "Ellis Park": (-26.1972, 28.0606),
    "Hollywoodbets Kings Park": (-29.8284, 31.0296),
    "DHL Stadium": (-33.9036, 18.4114),
    "Danie Craven Stadium": (-33.9402, 18.8683),
    "Loftus Versfeld": (-25.7536, 28.2225),          # Pretoria (Bulls)

    # Northern Ireland (Ulster)
    "Affidea Stadium": (54.5799, -5.9042),           # Ravenhill/Kingspan

    # One-offs & occasional venues
    "Croke Park": (53.3607, -6.2510),
    "Hastings Insurance MacHale Park": (53.8573, -9.3015),
    "Principality Stadium": (51.4816, -3.1827),      # Cardiff (national stadium)
}

# ---------- tiny disk cache helpers ----------
def _read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None

def _write_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

# ---------- geocoding ----------
def load_geocode_cache() -> Dict[str, dict]:
    return _read_json(GEOCODE_CACHE) or {}

def save_geocode_cache(cache: Dict[str, dict]):
    _write_json(GEOCODE_CACHE, cache)

def geocode_venue(venue: str, cache: Dict[str, dict]) -> Optional[Tuple[float,float,str]]:
    key = venue.strip().lower()

    # 0) Hard override first
    if venue in VENUE_COORDS:
        lat, lon = VENUE_COORDS[venue]
        cache[key] = {"lat": lat, "lon": lon, "name": venue}
        save_geocode_cache(cache)
        return lat, lon, venue

    # 1) Cache
    if key in cache:
        g = cache[key]
        return g["lat"], g["lon"], g.get("name") or venue

    # 2) Open-Meteo geocoding (no key)
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": venue, "count": 1, "language": "en", "format": "json"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = (data or {}).get("results") or []
    if not results:
        return None
    first = results[0]
    lat, lon = float(first["latitude"]), float(first["longitude"])
    name = first.get("name") or venue
    cache[key] = {"lat": lat, "lon": lon, "name": name}
    save_geocode_cache(cache)
    time.sleep(0.3)  # be polite
    return lat, lon, name

# ---------- datetime parsing ----------
def parse_kickoff_local(dt_str: str) -> Optional[pd.Timestamp]:
    """
    Parse a string like 'Fri 20th September 2024, 08:35pm SAST' into tz-aware Timestamp.
    If TZ abbr is mapped, localize to that tz; else assume UTC when naive.
    """
    parts = dt_str.strip().split()
    tz_abbr = parts[-1] if parts else ""
    tz_name = TZ_ABBR.get(tz_abbr.upper(), None)

    if tz_name:
        dt_wo_tz = " ".join(parts[:-1])
        dt_naive = dtparser.parse(dt_wo_tz, dayfirst=True)
        return pd.Timestamp(dt_naive).tz_localize(tz.gettz(tz_name))
    else:
        dt_parsed = dtparser.parse(dt_str, dayfirst=True)
        ts = pd.Timestamp(dt_parsed)
        if ts.tzinfo is None or ts.tz is None:
            return ts.tz_localize("UTC")
        return ts

# ---------- weather fetch ----------
def weather_cache_path(lat: float, lon: float, dt_utc: pd.Timestamp) -> Path:
    key = f"{lat:.4f}_{lon:.4f}_{dt_utc.strftime('%Y%m%d%H')}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return WEATHER_CACHE_DIR / f"{h}.json"

def safe_get_hour(data: dict, field: str, idx: int):
    """Safely index an hourly field (list) at idx."""
    hourly = (data or {}).get("hourly", {})
    arr = hourly.get(field)
    if not isinstance(arr, list):
        return None
    if 0 <= idx < len(arr):
        return arr[idx]
    return None

def fetch_hour_weather(lat: float, lon: float, dt_local_sast: pd.Timestamp) -> Optional[dict]:
    """
    RugbyPass times in CSV are SAST (or other abbr). Convert kickoff to the venue's local timezone
    (from Open-Meteo response) and pick the matching hour.
    """
    # Guess day from provided local time (e.g., SAST)
    date_guess = dt_local_sast.strftime("%Y-%m-%d")

    # Cache key: UTC hour of kickoff to avoid duplicate calls
    dt_utc = dt_local_sast.tz_convert("UTC")
    cpath = weather_cache_path(lat, lon, dt_utc)
    cached = _read_json(cpath)
    if cached:
        return cached

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date_guess, "end_date": date_guess,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,weathercode",
        "timezone": "auto",  # return times in venue local tz
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {}) or {}
    times = hourly.get("time") or []
    if not times:
        return None

    # Venue tz name from API (e.g., "Europe/London", "Africa/Johannesburg")
    venue_tz_name = data.get("timezone") or "UTC"
    venue_tz = tz.gettz(venue_tz_name)

    # Convert kickoff from SAST (or given tz) to venue tz, then match hour
    kickoff_venue = dt_local_sast.tz_convert(venue_tz)
    target_date = kickoff_venue.date()
    target_hour = kickoff_venue.hour

    # Build localized timestamps for each hourly time string (strings are local w.r.t. 'timezone')
    ts_hours = pd.to_datetime(times).tz_localize(venue_tz)

    # Find index with same date & hour
    idx = None
    for j, ts in enumerate(ts_hours):
        if ts.date() == target_date and ts.hour == target_hour:
            idx = j
            break
    if idx is None:
        # fallback: closest hour on same day, otherwise closest overall hour
        same_day = [(j, abs(ts.hour - target_hour)) for j, ts in enumerate(ts_hours) if ts.date() == target_date]
        idx = (min(same_day, key=lambda x: x[1])[0] if same_day
               else int((abs(ts_hours.hour - target_hour)).astype(int).idxmin()))

    out = {
        "temp_c": safe_get_hour(data, "temperature_2m", idx),
        "precip_mm": safe_get_hour(data, "precipitation", idx),
        "wind_kph": _ms_to_kph(safe_get_hour(data, "wind_speed_10m", idx)),
        "weathercode": int(safe_get_hour(data, "weathercode", idx) or 0),
    }
    _write_json(cpath, out)
    time.sleep(0.5)  # be polite
    return out

def _ms_to_kph(x):
    try:
        return round(float(x) * 3.6, 1)
    except Exception:
        return None

def wx_summary(code: Optional[int]) -> Optional[str]:
    if code is None:
        return None
    return WX_MAP.get(int(code), None)

# ---------- main ----------
def main():
    df = pd.read_csv(IN_CSV)
    if not {"Date_time","Venue"}.issubset(df.columns):
        raise SystemExit("CSV must contain Date_time and Venue columns")

    geocache = load_geocode_cache()

    # Prepare new columns
    df["wx_temp_c"] = pd.Series(dtype="float")
    df["wx_precip_mm"] = pd.Series(dtype="float")
    df["wx_wind_kph"] = pd.Series(dtype="float")
    df["wx_weathercode"] = pd.Series(dtype="Int64")
    df["wx_summary"] = pd.Series(dtype="string")

    # Iterate rows
    for i, row in df.iterrows():
        dt_str = str(row["Date_time"])
        venue  = str(row["Venue"]) if pd.notna(row["Venue"]) else ""

        # 1) Parse kickoff (tz-aware)
        try:
            kickoff_local = parse_kickoff_local(dt_str)
        except Exception:
            kickoff_local = None

        if not venue or not kickoff_local:
            continue

        # 2) Geocode venue (with overrides & cache)
        g = geocode_venue(venue, geocache)
        if not g:
            continue
        lat, lon, _pretty = g

        # 3) Fetch weather for the corresponding local hour at venue
        w = fetch_hour_weather(lat, lon, kickoff_local)
        if not w:
            continue

        # 4) Populate columns
        df.at[i, "wx_temp_c"] = w.get("temp_c")
        df.at[i, "wx_precip_mm"] = w.get("precip_mm")
        df.at[i, "wx_wind_kph"] = w.get("wind_kph")
        code = w.get("weathercode")
        df.at[i, "wx_weathercode"] = pd.NA if code is None else int(code)
        df.at[i, "wx_summary"] = wx_summary(code)

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved weather-enriched CSV -> {OUT_CSV}")

if __name__ == "__main__":
    main()


