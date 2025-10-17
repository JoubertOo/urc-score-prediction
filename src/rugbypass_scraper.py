"""
RugbyPass URC scraper (scrape-only; precise FT/HT + venue from details block)

Outputs:
  Home_team, Away_team, Date_time, Venue,
  Fulltime_score_home, Fulltime_score_away,
  Halftime_score_home, Halftime_score_away

Run (defaults):
  python src/rugbypass_scraper.py

Debug pages with missing FT:
  python src/rugbypass_scraper.py --debug-missing
"""

from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import requests
import requests_cache
from bs4 import BeautifulSoup, Tag, NavigableString
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------- config ----------------------

CACHE_NAME = "cache/http"
CACHE_TTL_SECONDS = 24 * 3600
DEFAULT_URLS = "data/urls_matches.txt"
DEFAULT_OUT = "data/processed/matches.csv"
DEFAULT_DELAY = (5.0, 9.0)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Connection": "close",
}

TEAM_NORMALIZE = {
    "cardiff blues": "Cardiff",
    "cardiff rugby": "Cardiff",
    "cardiff": "Cardiff",
    "zebre parma": "Zebre",
    "zebre": "Zebre",
    "vodacom bulls": "Bulls",
    "bulls": "Bulls",
    "dhl stormers": "Stormers",
    "stormers": "Stormers",
    "hollywoodbets sharks": "Sharks",
    "cell c sharks": "Sharks",
    "sharks": "Sharks",
    "emirates lions": "Lions",
    "lions": "Lions",
    "leinster": "Leinster",
    "munster": "Munster",
    "ulster": "Ulster",
    "connacht": "Connacht",
    "glasgow warriors": "Glasgow Warriors",
    "glasgow": "Glasgow Warriors",
    "edinburgh": "Edinburgh",
    "dragons": "Dragons",
    "ospreys": "Ospreys",
    "scarlets": "Scarlets",
}

# Venue hint words (used only for weaker fallbacks)
VENUE_HINTS = (
    " park", " stadium", " arena", " ground", " field",
    " parc", " stadio", " stade"
)
VENUE_STOPWORDS = {
    "login","tickets","ticket","sign in","watch","highlights",
    "news","preview","report","results","fixtures","standings","table",
    "video","videos","shop","store","subscribe","menu"
}

# Score / interval detection
SCORE_RE = re.compile(r"^\s*(\d{1,3})\s*[-–]\s*(\d{1,3})\s*$")
FT_MARKERS = re.compile(r"\b(Full\s*-?\s*Time|FT|Final\s*Score)\b", re.I)
HT_MARKERS = re.compile(r"\b(Half\s*-?\s*Time|HT)\b", re.I)

# ---------------------- helpers ----------------------

def build_session() -> requests.Session:
    s = requests_cache.CachedSession(CACHE_NAME, backend="sqlite", expire_after=CACHE_TTL_SECONDS)
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def sleep_politely(lo: float, hi: float, enabled: bool = True):
    if enabled:
        time.sleep(random.uniform(lo, hi))

def soft_strip(x) -> Optional[str]:
    return str(x).strip() if x is not None and pd.notna(x) else None

def normalize_team(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    return TEAM_NORMALIZE.get(name.lower().strip(), name.strip())

def split_home_away(label: str) -> tuple[Optional[str], Optional[str]]:
    m = re.search(r"(.+?)\s+v(?:s\.?)?\s+(.+)", label or "", flags=re.I)
    return (soft_strip(m.group(1)), soft_strip(m.group(2))) if m else (None, None)

def first_span_matching(soup: BeautifulSoup, pattern: re.Pattern) -> Optional[str]:
    for sp in soup.find_all("span"):
        t = sp.get_text(" ", strip=True)
        if pattern.search(t):
            return t
    return None

# ---- venue helpers (prefer details block, then minimal fallbacks) ----

def extract_venue_from_details(soup: BeautifulSoup) -> Optional[str]:
    """
    Only return the line in the 'details' block that contains the goal-posts icon.
    Prefer the visible span; fallback to the img's alt.
    """
    details = soup.find("div", class_="details")
    if not details:
        return None

    for line in details.find_all("div", class_="line"):
        icon_img = line.find("img")
        if not icon_img:
            continue

        # collect possible src attributes to check for goal-posts icon
        src_candidates = [
            icon_img.get("src") or "",
            icon_img.get("data-src") or "",
            icon_img.get("srcset") or "",
            icon_img.get("data-srcset") or "",
        ]
        src_lower = " ".join(src_candidates).lower()

        looks_like_goalposts = (
            "goal-posts-icon" in src_lower or
            "goalposts" in src_lower or
            "goal-post" in src_lower
        )
        if not looks_like_goalposts:
            continue

        title_span = line.select_one(".title span")
        venue_from_span = title_span.get_text(" ", strip=True) if title_span else None
        alt_txt = (icon_img.get("alt") or "").strip()

        return venue_from_span or alt_txt

    return None

def extract_venue_from_jsonld(soup: BeautifulSoup) -> Optional[str]:
    """Structured data as a last resort (may differ from display names)."""
    for node in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(node.string or "")
        except Exception:
            continue
        objs = data if isinstance(data, list) else [data]
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            loc = obj.get("location") or obj.get("eventVenue")
            if isinstance(loc, dict):
                name = loc.get("name")
                if isinstance(name, str):
                    txt = name.strip()
                    if txt:
                        return txt
    return None

def _is_nav_like(tag: Tag) -> bool:
    cur: Optional[Tag] = tag
    while isinstance(cur, Tag) and cur is not None:
        cls = " ".join(cur.get("class") or []).lower()
        name = (cur.name or "").lower()
        if name in {"nav","header","footer"} or any(w in cls for w in ["nav","menu","breadcrumbs","breadcrumb","header","footer"]):
            return True
        cur = cur.parent
    return False

def extract_venue_fallbacks(soup: BeautifulSoup) -> Optional[str]:
    # obvious venue/stadium classes
    node = soup.find(attrs={"class": re.compile(r"venue|stadium", re.I)})
    if node:
        txt = node.get_text(" ", strip=True)
        if txt:
            return txt
    # generic hint-based search (avoid nav)
    for el in soup.find_all(True):
        if _is_nav_like(el):
            continue
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        tl = txt.lower()
        if any(sw in tl for sw in VENUE_STOPWORDS):
            continue
        if any(h in tl for h in VENUE_HINTS) and 3 <= len(txt) <= 80:
            if re.search(r"\bv(?:s\.?)?\b", txt, flags=re.I):
                continue
            return txt
    return None

def find_venue(soup: BeautifulSoup) -> Optional[str]:
    v = extract_venue_from_details(soup)
    if v:
        return v
    v = extract_venue_fallbacks(soup)
    if v:
        return v
    return extract_venue_from_jsonld(soup)

# ---- precise Key Events traversal (robust FT/HT) ----

def find_interval_tag(soup: BeautifulSoup, marker_regex: re.Pattern) -> Optional[Tag]:
    # Try explicit class first
    for div in soup.find_all("div", class_="interval"):
        if marker_regex.search(div.get_text(" ", strip=True) or ""):
            return div
    # Fallback: any tag whose visible text matches
    for tag in soup.find_all(True):
        if marker_regex.search(tag.get_text(" ", strip=True) or ""):
            return tag
    return None

def score_from_node_text(tag: Tag) -> Optional[tuple[int, int]]:
    if not tag or not hasattr(tag, "get_text"):
        return None
    label = tag.select_one(".score .label")
    if label:
        txt = label.get_text(" ", strip=True)
        m = SCORE_RE.match(txt)
        if m:
            return int(m.group(1)), int(m.group(2))
    txt = tag.get_text(" ", strip=True)
    m = SCORE_RE.match(txt)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def find_first_score_between(start: Tag, stop_regex: re.Pattern) -> tuple[Optional[int], Optional[int]]:
    for nxt in start.find_all_next(True):
        if stop_regex.search(nxt.get_text(" ", strip=True) or ""):
            break
        got = score_from_node_text(nxt)
        if got:
            return got
    return None, None

def find_score_after_interval_variants(soup: BeautifulSoup, marker_regex: re.Pattern) -> tuple[Optional[int], Optional[int]]:
    start = find_interval_tag(soup, marker_regex)
    if not start:
        return None, None
    NEXT_INTERVAL = re.compile(r"\b(Full\s*-?\s*Time|FT|Final\s*Score|Half\s*-?\s*Time|HT|Kick[-\s]*Off|Start)\b", re.I)
    return find_first_score_between(start, NEXT_INTERVAL)

# ---------------------- parser ----------------------

def parse_match_page(html: str, source_url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # Teams
    label = first_span_matching(soup, re.compile(r"\bv(?:s\.?)?\b"))
    if not label:
        h1 = soup.find("h1")
        label = h1.get_text(" ", strip=True) if h1 else None
    if not label and soup.title:
        label = soup.title.get_text(" ", strip=True)
    home_team, away_team = split_home_away(label or "")
    home_team = normalize_team(home_team)
    away_team = normalize_team(away_team)

    # Date/time
    date_time = first_span_matching(
        soup, re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b.*\b(am|pm)\b", re.I)
    ) or first_span_matching(
        soup, re.compile(r"\b(SAST|BST|GMT|UTC)\b", re.I)
    )
    if not date_time:
        tt = soup.find("time")
        if tt:
            date_time = tt.get("datetime") or tt.get_text(" ", strip=True)

    # Venue (prefer details block text exactly as shown)
    venue = find_venue(soup)

    # Scores
    full_home, full_away = find_score_after_interval_variants(soup, FT_MARKERS)
    half_home, half_away = find_score_after_interval_variants(soup, HT_MARKERS)

    return {
        "Home_team": home_team,
        "Away_team": away_team,
        "Date_time": soft_strip(date_time),
        "Venue": soft_strip(venue),
        "Fulltime_score_home": full_home,
        "Fulltime_score_away": full_away,
        "Halftime_score_home": half_home,
        "Halftime_score_away": half_away,
    }

# ---------------------- CLI ----------------------

def run_scrape(urls_file: str, out_csv: str, delay_lo: float, delay_hi: float, debug_missing: bool):
    session = build_session()

    path = Path(urls_file)
    if not path.exists():
        print(f"!! URLs file not found: {urls_file}")
        sys.exit(1)

    raw = [u for u in path.read_text(encoding="utf-8").splitlines() if u.strip() and not u.strip().startswith("#")]
    urls = [u.strip().replace(" ", "") for u in raw]

    if not urls:
        print(f"!! No URLs found in {urls_file}")
        sys.exit(1)

    one_url_only = (len(urls) == 1)
    print(f"Scraping {len(urls)} match page(s) from {urls_file} -> {out_csv}")

    rows: List[dict] = []
    for i, url in enumerate(urls, 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=(15, 45))
            r.raise_for_status()
            from_cache = getattr(r, "from_cache", False)
            print(f"[{i}/{len(urls)}] {url} (from_cache={from_cache})")
            row = parse_match_page(r.text, url)

            if debug_missing and (row["Fulltime_score_home"] is None or row["Fulltime_score_away"] is None):
                dbg_dir = Path("data/debug"); dbg_dir.mkdir(parents=True, exist_ok=True)
                name = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
                (dbg_dir / f"{i:03d}_{name}.html").write_text(r.text, encoding="utf-8")
                print(f"  .. debug dump -> data/debug/{i:03d}_{name}.html")

            rows.append(row)
        except Exception as e:
            print(f"  !! Failed: {e}")
        sleep_politely(delay_lo, delay_hi, enabled=not one_url_only)

    if not rows:
        print("No rows scraped.")
        return

    df = pd.DataFrame(rows)
    cols = [
        "Home_team","Away_team","Date_time","Venue",
        "Fulltime_score_home","Fulltime_score_away",
        "Halftime_score_home","Halftime_score_away",
    ]
    df = df[cols]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} match(es) -> {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default=DEFAULT_URLS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--delay-lo", type=float, default=DEFAULT_DELAY[0])
    ap.add_argument("--delay-hi", type=float, default=DEFAULT_DELAY[1])
    ap.add_argument("--debug-missing", action="store_true", help="dump HTML for pages missing FT scores")
    args = ap.parse_args()
    run_scrape(args.urls, args.out, args.delay_lo, args.delay_hi, args.debug_missing)

if __name__ == "__main__":
    main()
