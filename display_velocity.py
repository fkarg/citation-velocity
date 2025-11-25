# /// script
# dependencies = [ "typer" ]
# ///
import csv
import sys
import json
import typer
from pathlib import Path
from enum import Enum

app = typer.Typer()

class SortOption(str, Enum):
    citations = "citations"
    velocity = "velocity"
    published = "published"

@app.command()
def main(
    filename: Path = typer.Option("top_velocity_by_year.csv", help="Path to the CSV file"),
    year: int = typer.Option(None, help="Filter by specific display year"),
    limit: int = typer.Option(20, help="Number of papers to show per year"),
    show_summary: bool = typer.Option(True, help="Show AI summaries"),
    show_history: bool = typer.Option(True, help="Show rank history"),
    sort_by: SortOption = typer.Option(SortOption.citations, help="Sort criteria within the year"),
):
    if not filename.exists():
        print(f"Error: {filename} not found. Please run citation_velocity.py first.")
        raise typer.Exit(code=1)

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by display_year (the year for which we're showing top citations)
    by_year = {}
    for r in rows:
        y = int(r["display_year"])
        if year and y != year:
            continue
        if y not in by_year:
            by_year[y] = []
        by_year[y].append(r)

    # Display
    print(f"\n{'='*100}")
    print("TOP PAPERS BY CITATIONS RECEIVED IN EACH YEAR")
    print(f"{'='*100}\n")

    for display_year in sorted(by_year.keys(), reverse=True):
        print(f"YEAR: {display_year}")
        print(f"{'-'*100}")

        # Get works for this year
        works_list = by_year[display_year]

        # Find max citations in this year for bar scaling (always based on citations)
        max_cites = max([float(w["citations_in_year"]) for w in works_list]) if works_list else 1.0

        # Sort based on criteria
        if sort_by == SortOption.citations:
            works = sorted(works_list, key=lambda x: float(x["citations_in_year"]), reverse=True)
        elif sort_by == SortOption.velocity:
            works = sorted(works_list, key=lambda x: float(x["avg_velocity"]), reverse=True)
        elif sort_by == SortOption.published:
            works = sorted(works_list, key=lambda x: int(x["publication_year"]), reverse=True)
        else:
            works = works_list

        for i, w in enumerate(works[:limit], 1):
            cites_in_year = int(float(w["citations_in_year"]))
            avg_vel = float(w["avg_velocity"])
            pub_year = w["publication_year"]
            title = w["title"]
            doi = w.get("doi", "")
            rank_hist_str = w.get("rank_history", "{}")

            # Truncate title
            if len(title) > 55:
                title = title[:52] + "..."

            # Simple bar
            bar_len = int((cites_in_year / max_cites) * 20)
            bar = "█" * bar_len

            print(f"{i:2}. {title:<55} | Cites {display_year}: {cites_in_year:6} {bar} (pub:{pub_year}, avg_vel:{avg_vel:.0f}) {doi}")

            # Parse and display rank history if available
            if show_history:
                try:
                    rh = json.loads(rank_hist_str)
                    if rh:
                        sorted_years = sorted([int(y) for y in rh.keys()])
                        hist_parts = []
                        for y in sorted_years:
                            r = rh[str(y)]
                            hist_parts.append(f"{y}(#{r})")
                        print(f"    Rank History: {' -> '.join(hist_parts)}")
                except json.JSONDecodeError:
                    pass

            # Display summary
            if show_summary:
                summary = w.get("summary")
                if summary:
                    print(f"    Summary: {summary}")

            print("")
        print("\n")

if __name__ == "__main__":
    app()