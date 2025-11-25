# citation-velocity

Analyze citation velocity of research papers using OpenAlex data.

Run with `uv run citation_velocity.py`. Optionally provide an openai API key and `--no-skip-summary` if you want automatic summaries (doesn't always work though).

Usage:

```python
> uv run citation_velocity.py --help

 Usage: citation_velocity.py [OPTIONS]

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────╮
│ --cache-location                              PATH     Directory to store cache files          │
│                                                        [default: cache]                        │
│ --skip-download         --no-skip-download             Skip downloading new data from OpenAlex │
│                                                        [default: no-skip-download]             │
│ --skip-summary          --no-skip-summary              Skip generating AI summaries            │
│                                                        [default: skip-summary]                 │
│ --force-refresh         --no-force-refresh             Force refresh of OpenAlex data (ignore  │
│                                                        cache)                                  │
│                                                        [default: no-force-refresh]             │
│ --offline               --no-offline                   Run in offline mode (implies            │
│                                                        skip_download and skip_summary)         │
│                                                        [default: no-offline]                   │
│ --verbose               --no-verbose                   Enable verbose logging                  │
│                                                        [default: no-verbose]                   │
│ --start-year                                  INTEGER  Start year for analysis [default: 2015] │
│ --end-year                                    INTEGER  End year for analysis [default: 2025]   │
│ --max-pages                                   INTEGER  Maximum number of pages to fetch per    │
│                                                        year from OpenAlex                      │
│                                                        [default: 10]                           │
│ --top-n                                       INTEGER  Number of top papers to analyze         │
│                                                        [default: 200]                          │
│ --refresh-year                                INTEGER  Specific years to force refresh (can be │
│                                                        used multiple times)                    │
│ --install-completion                                   Install completion for the current      │
│                                                        shell.                                  │
│ --show-completion                                      Show completion for the current shell,  │
│                                                        to copy it or customize the             │
│                                                        installation.                           │
│ --help                                                 Show this message and exit.             │
╰────────────────────────────────────────────────────────────────────────────────────────────────╯
```
