# /// script
# dependencies = [ "requests", "beautifulsoup4", "openai", "tqdm", "typer" ]
# ///
# top_velocity_openalex.py
# Requires: Python 3.9+, requests, pandas, beautifulsoup4, openai
# Usage: python top_velocity_openalex.py
# Notes: OpenAlex API rate-limits. Add a mailto argument to be a good citizen:
#   base = "https://api.openalex.org/works?mailto=your_email@example.com&..."
import requests
import math
import time
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from openai import OpenAI
import typer

app = typer.Typer()

BASE = "https://api.openalex.org/works"
# adjust years to last 10 years; current year 2025 per our conversation
START_YEAR = 2015
END_YEAR = 2025
MAX_PAGES = 50            # OpenAlex basic paging limit: 10,000 results (50 pages × 200)
PER_PAGE = 200            # OpenAlex supports up to 200 per page
TOP_N = 200               # how many top results to output
SLEEP_BETWEEN = 0.5       # be polite; increase if you hit rate limits
MAILTO = "your_email@example.com"  # replace with your email for OpenAlex polite usage

# Global config (will be updated by CLI)
CACHE_DIR = Path("cache")
VERBOSE = False
OFFLINE = False
SKIP_DOWNLOAD = False
FORCE_REFRESH = False
SKIP_SUMMARY = True
REFRESH_YEARS = []

def get_cache_path(key):
    CACHE_DIR.mkdir(exist_ok=True)
    hash_key = hashlib.md5(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{hash_key}.json"

def fetch_page(year, page):
    """Fetch a page from OpenAlex for a specific year. Returns (data, from_cache) tuple."""
    # filter: works published in specific year, exclude reviews
    filter_q = f"from_publication_date:{year}-01-01,to_publication_date:{year}-12-31,type:!review"
    params = {
        "filter": filter_q,
        "sort": "cited_by_count:desc",
        "page": page,
        "per_page": PER_PAGE,
        "mailto": MAILTO
    }

    # Cache key based on params
    cache_key = json.dumps(params, sort_keys=True)
    cache_path = get_cache_path(cache_key)

    # Check if we should force refresh this specific year
    force_this_year = FORCE_REFRESH or (year in REFRESH_YEARS)

    if not force_this_year and cache_path.exists():
        # Simple cache validity check (e.g. 24 hours) could be added here
        # For now, we assume if it's cached, it's good (smart caching)
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                if VERBOSE:
                    print(f"  [Cache] Hit for year {year} page {page}")
                return json.load(f), True  # from_cache=True
        except Exception:
            pass # invalid cache, fetch again

    if OFFLINE or SKIP_DOWNLOAD:
        if VERBOSE:
            print(f"  [Offline] Skipping download for year {year} page {page}")
        return {"results": []}, False

    if VERBOSE:
        print(f"  [API] Fetching year {year} page {page}...")
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Save to cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data, False  # from_cache=False

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    # Inverted index is { "word": [pos1, pos2], ... }
    # We need to reconstruct the list of words in order
    word_map = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            word_map[pos] = word

    sorted_positions = sorted(word_map.keys())
    return " ".join(word_map[pos] for pos in sorted_positions)

def get_full_text_content(url):
    if not url:
        return None
    try:
        # Simple timeout and user-agent
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            soup = BeautifulSoup(r.text, "html.parser")
            # Very naive text extraction: paragraphs
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            text = "\n".join(paragraphs)
            return text[:10000] # Limit to first 10k chars to avoid huge payloads
    except Exception:
        pass
    return None

def generate_summary(title, abstract, full_text_url):
    if SKIP_SUMMARY:
        return None

    # Check cache for summary - use work title + abstract prefix for cache key
    safe_abstract = abstract or ""
    summary_key = f"summary_{title}_{safe_abstract[:20]}"
    cache_path = get_cache_path(summary_key)

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                cached_summary = cached.get("summary")
                # Return if we have a non-empty summary
                if cached_summary:
                    print("  Using cached summary.")
                    return cached_summary
                # Empty summary = retry generation
        except Exception:
            pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  OPENAI_API_KEY not found, skipping summary.")
        return None

    content_source = None

    # Try to get full text if available
    if full_text_url:
        print(f"  Fetching full text from {full_text_url}...")
        full_text = get_full_text_content(full_text_url)
        if full_text and len(full_text) > 500:
            content_source = "Full Text Start: " + full_text
            print("  Got full text.")
        else:
            print("  Could not get full text, using abstract.")

    # Fall back to abstract if no full text
    if not content_source and safe_abstract.strip():
        content_source = "Abstract: " + safe_abstract

    # If we have no content at all, skip (don't cache - maybe next time we'll have content)
    if not content_source:
        print(f"  No abstract or full text available for: {title[:50]}...")
        return None

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant. Summarize the following academic paper in exactly 3 sentences. Focus on the main contribution and impact."},
                {"role": "user", "content": f"Title: {title}\n\n{content_source}"}
            ],
            max_completion_tokens=500
        )
        summary = response.choices[0].message.content.strip()

        # Cache it (update existing cache if present)
        cache_data = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict):
                        cache_data = existing
            except Exception:
                pass

        cache_data["summary"] = summary

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        return summary
    except Exception as e:
        print(f"  Error generating summary: {e}")
        return None

def compute_metrics(work):
    """
    Returns dict with:
      - id, title, doi, pub_year, cited_by_count
      - avg_velocity = total_citations / years_since_pub
      - recent_2yr_avg = avg citations per year over last 2 available years (if counts_by_year available)
      - recent_slope = simple slope (most_recent - prev)/1
    """
    now_year = END_YEAR
    pub_year = work.get("publication_year") or now_year
    total_cites = work.get("cited_by_count", 0)

    years = max(1, now_year - pub_year + 1)
    avg_velocity = total_cites / years

    # prepare recent metrics
    counts_by_year = {}
    for e in work.get("counts_by_year", []):
        # some OpenAlex endpoints provide entries like {'year':2022,'cited_by_count':...}
        y = e.get("year")
        c = e.get("cited_by_count") or e.get("cited_by_count", 0)
        if y:
            counts_by_year[int(y)] = int(c)

    # compute recent 2-year avg using the two most recent years present (if available)
    recent_years = sorted([y for y in counts_by_year.keys() if y <= now_year], reverse=True)
    recent_2yr_avg = None
    recent_slope = None
    if len(recent_years) >= 2:
        y1, y0 = recent_years[0], recent_years[1]  # y1 = most recent, y0 = one before
        c1, c0 = counts_by_year[y1], counts_by_year[y0]
        recent_2yr_avg = (c1 + c0) / 2.0
        recent_slope = (c1 - c0)  # per-year change
    elif len(recent_years) == 1:
        y1 = recent_years[0]
        recent_2yr_avg = counts_by_year[y1]
        recent_slope = None

    # Reconstruct abstract
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    return {
        "id": work.get("id"),
        "title": work.get("display_name") or work.get("title"),
        "doi": (work.get("ids") or {}).get("doi"),
        "publication_year": pub_year,
        "cited_by_count": total_cites,
        "avg_velocity": avg_velocity,
        "recent_2yr_avg": recent_2yr_avg,
        "recent_slope": recent_slope,
        "openalex_url": work.get("id"),
        "counts_by_year": counts_by_year,
        "abstract": abstract,
        "best_oa_url": (work.get("best_oa_location") or {}).get("pdf_url") or (work.get("best_oa_location") or {}).get("landing_page_url")
    }

@app.command()
def cli(
    cache_location: Path = typer.Option("cache", help="Directory to store cache files"),
    skip_download: bool = typer.Option(False, help="Skip downloading new data from OpenAlex"),
    skip_summary: bool = typer.Option(True, help="Skip generating AI summaries"),
    force_refresh: bool = typer.Option(False, help="Force refresh of OpenAlex data (ignore cache)"),
    offline: bool = typer.Option(False, help="Run in offline mode (implies skip_download and skip_summary)"),
    verbose: bool = typer.Option(False, help="Enable verbose logging"),
    start_year: int = typer.Option(2015, help="Start year for analysis"),
    end_year: int = typer.Option(2025, help="End year for analysis"),
    max_pages: int = typer.Option(10, help="Maximum number of pages to fetch per year from OpenAlex"),
    top_n: int = typer.Option(200, help="Number of top papers to analyze"),
    refresh_year: list[int] = typer.Option([], help="Specific years to force refresh (can be used multiple times)"),
):
    # Update globals
    global CACHE_DIR, VERBOSE, OFFLINE, SKIP_DOWNLOAD, FORCE_REFRESH, SKIP_SUMMARY, REFRESH_YEARS
    global START_YEAR, END_YEAR, MAX_PAGES, TOP_N

    CACHE_DIR = cache_location
    VERBOSE = verbose
    OFFLINE = offline
    SKIP_DOWNLOAD = skip_download or offline
    FORCE_REFRESH = force_refresh
    SKIP_SUMMARY = skip_summary or offline
    REFRESH_YEARS = refresh_year
    START_YEAR = start_year
    END_YEAR = end_year
    MAX_PAGES = max_pages
    TOP_N = top_n

    CACHE_DIR.mkdir(exist_ok=True)

    if VERBOSE:
        print(f"Configuration:")
        print(f"  Cache Dir: {CACHE_DIR}")
        print(f"  Offline: {OFFLINE}")
        print(f"  Skip Download: {SKIP_DOWNLOAD}")
        print(f"  Skip Summary: {SKIP_SUMMARY}")
        print(f"  Force Refresh: {FORCE_REFRESH}")
        print(f"  Refresh Years: {REFRESH_YEARS}")
        print(f"  Years: {START_YEAR}-{END_YEAR}")
        print(f"  Pages per Year: {MAX_PAGES}")

    out = []
    fetched = 0
    print(f"Fetching OpenAlex works ({START_YEAR}-{END_YEAR}, {MAX_PAGES} pages/year)...")

    try:
        for year in range(START_YEAR, END_YEAR + 1):
            if VERBOSE:
                print(f"Processing year {year}...")

            for page in range(1, MAX_PAGES + 1):
                data, from_cache = fetch_page(year, page)
                results = data.get("results", [])
                if not results:
                    break
                for w in results:
                    fetched += 1
                    m = compute_metrics(w)
                    out.append(m)

                # Only sleep if we actually hit the API
                if not from_cache:
                    time.sleep(SLEEP_BETWEEN)

                # defensive stop if API says there's no next
                if not results:
                    break
    except Exception as e:
        print("Error fetching OpenAlex:", e)

    print(f"Fetched {fetched} works; computing ranks...")

    # Rank by average velocity (descending) and also show recent_2yr_avg ranking
    out_sorted_by_avg = sorted(out, key=lambda x: (x["avg_velocity"] or 0), reverse=True)[:TOP_N]
    out_sorted_by_recent = sorted([o for o in out if o["recent_2yr_avg"] is not None],
                                 key=lambda x: x["recent_2yr_avg"], reverse=True)[:TOP_N]

    # write CSV of average velocity top list
    with open("top_velocity_by_avg.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id","title","doi","publication_year","cited_by_count",
            "avg_velocity","recent_2yr_avg","recent_slope","openalex_url"
        ], extrasaction='ignore')
        writer.writeheader()
        for r in out_sorted_by_avg:
            writer.writerow(r)

    # write CSV of recent velocity top list
    with open("top_velocity_by_recent2yr.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id","title","doi","publication_year","cited_by_count",
            "avg_velocity","recent_2yr_avg","recent_slope","openalex_url"
        ], extrasaction='ignore')
        writer.writeheader()
        for r in out_sorted_by_recent:
            writer.writerow(r)

    # Compute top 20 by citations-in-year for EACH year
    # This is the main output: for each year, which papers got the most citations that year
    import json

    # Sanity check threshold: max plausible citations in a single year
    # Even the most viral papers rarely exceed 30k citations/year
    # (there's an outlier with like 800k citations/year but it's clearly a data error)
    MAX_CITATIONS_PER_YEAR = 50000

    # First, compute citations-per-year ranking for ALL papers
    # Structure: year -> list of (work, citations_in_year) sorted desc
    year_leaderboards = {}
    for y in range(START_YEAR, END_YEAR + 1):
        leaderboard = []
        for w in out:
            pub_year = w["publication_year"]
            if y >= pub_year:  # Paper must exist
                c_by_y = w.get("counts_by_year", {})
                citations_in_y = c_by_y.get(y, 0)
                # Filter out obvious data errors
                if citations_in_y > MAX_CITATIONS_PER_YEAR:
                    print(f"  Warning: Filtering out {w['title'][:50]}... with {citations_in_y} citations in {y} (likely data error)")
                    citations_in_y = 0
                leaderboard.append((w, citations_in_y))
        # Sort by citations in that year desc
        leaderboard.sort(key=lambda x: x[1], reverse=True)
        year_leaderboards[y] = leaderboard

    # Build rank maps for each year (for rank history)
    year_rank_maps = {}
    for y, leaderboard in year_leaderboards.items():
        year_rank_maps[y] = {w["id"]: i for i, (w, _) in enumerate(leaderboard, 1)}

    # Select top 20 for each year and compute their rank history
    out_top_per_year = []
    seen_ids = set()  # Track unique papers across all years

    for y in sorted(year_leaderboards.keys(), reverse=True):
        leaderboard = year_leaderboards[y]
        top_20 = leaderboard[:20]

        for w, citations_in_y in top_20:
            # Create a copy with year-specific info
            work_entry = w.copy()
            work_entry["display_year"] = y  # The year this entry is for
            work_entry["citations_in_year"] = citations_in_y

            # Compute rank history (rank by citations-in-that-year for each year since publication)
            rank_history = {}
            pub_year = w["publication_year"]
            for hist_year in range(max(pub_year, START_YEAR), END_YEAR + 1):
                if w["id"] in year_rank_maps[hist_year]:
                    rank_history[hist_year] = year_rank_maps[hist_year][w["id"]]
            work_entry["rank_history"] = json.dumps(rank_history)

            out_top_per_year.append(work_entry)
            seen_ids.add(w["id"])

    # Generate summaries in parallel (limit workers to avoid rate limits)
    def gen_summary_for_work(w):
        summary = generate_summary(w["title"], w["abstract"], w["best_oa_url"])
        return w["id"], summary

    SUMMARY_WORKERS = 20  # Conservative to avoid OpenAI rate limits
    with ThreadPoolExecutor(max_workers=SUMMARY_WORKERS) as executor:
        futures = {executor.submit(gen_summary_for_work, w): w for w in out_top_per_year}
        for future in tqdm(as_completed(futures), total=len(out_top_per_year), desc="Generating Summaries"):
            work_id, summary = future.result()
            # Find the work and assign summary
            for w in out_top_per_year:
                if w["id"] == work_id:
                    w["summary"] = summary
                    break

    # write CSV of average velocity top list
    with open("top_velocity_by_avg.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id","title","doi","publication_year","cited_by_count",
            "avg_velocity","recent_2yr_avg","recent_slope","openalex_url"
        ], extrasaction='ignore')
        writer.writeheader()
        for r in out_sorted_by_avg:
            writer.writerow(r)

    # write CSV of recent velocity top list
    with open("top_velocity_by_recent2yr.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id","title","doi","publication_year","cited_by_count",
            "avg_velocity","recent_2yr_avg","recent_slope","openalex_url"
        ], extrasaction='ignore')
        writer.writeheader()
        for r in out_sorted_by_recent:
            writer.writerow(r)

    # write CSV of top citations by year (top 20 papers by citations received in each year)
    with open("top_velocity_by_year.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id","title","doi","publication_year","cited_by_count",
            "avg_velocity","citations_in_year","display_year","openalex_url", "rank_history", "summary"
        ], extrasaction='ignore')
        writer.writeheader()
        for r in out_top_per_year:
            writer.writerow(r)

    print("Wrote top_velocity_by_avg.csv, top_velocity_by_recent2yr.csv, and top_velocity_by_year.csv")
    print("Done.")

if __name__ == "__main__":
    app()

