#!/usr/bin/env python
"""Load a one-year baseline i4b run and print the available record columns.

Reads the flat per-step table written by ``run_baseline.py``
(``val_records_step0.parquet``) into a pandas DataFrame and prints its columns,
as a starting point for the CCTA timing analysis.

Examples:
--------
    python scripts/i4b/ccta_timings.py
    python scripts/i4b/ccta_timings.py output/year_baseline/val_records_step0.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("output/year_baseline/val_records_step0.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_PATH,
        help=f"Per-step parquet table (default: {DEFAULT_PATH}).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.path)
    print(f"Loaded {len(df)} rows from {args.path}")
    print(f"\n{len(df.columns)} columns:")
    for c in df.columns:
        print(f"  {c}")


if __name__ == "__main__":
    main()
