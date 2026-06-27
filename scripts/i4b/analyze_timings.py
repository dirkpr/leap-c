#!/usr/bin/env python
"""Aggregate raw per-step i4b run records into computation-time statistics.

Reads the flat per-step table written by the i4b run scripts
(``val_records_step*.parquet`` from ``run_baseline.py`` / ``run_sac_zop.py``;
the baseline NPZ ``val_log_step*.npz`` scalars are also supported) and reports
mean / max / percentile compute times for the overall policy and the acados
solver. Aggregation lives here, not in the trainers, so long (one-year) runs
record raw data cheaply and the statistics are computed offline.

The two headline signals:

- ``policy_time_s``      - wall-clock of the full policy evaluation per step.
- ``solver.time_tot``    - acados total solve time per step (the dominant cost).
- ``policy_overhead_s``  - their difference: the actor's MLP + parameter-setting
                            overhead (≈ 0 for the baseline MPC).

Examples:
--------
    python scripts/i4b/analyze_timings.py output/baseline_i4b_seed0
    python scripts/i4b/analyze_timings.py output/run/val_records_step0.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PERCENTILES = (50, 95, 99)


def _step_of(p: Path) -> int:
    digits = "".join(c for c in p.stem if c.isdigit())
    return int(digits) if digits else -1


def _npz_to_df(path: Path) -> pd.DataFrame:
    """Flat table from the baseline NPZ: keep only the 1-D scalar channels."""
    data = np.load(path, allow_pickle=True)
    cols = {k: data[k] for k in data.files if data[k].ndim == 1}
    n = max((len(v) for v in cols.values()), default=0)
    cols = {k: v for k, v in cols.items() if len(v) == n}
    return pd.DataFrame(cols)


def _load_table(path: Path) -> tuple[pd.DataFrame, Path]:
    """Load a per-step table from a parquet/npz file or a run directory.

    For a directory, the most recent validation (largest step in the filename)
    is used; parquet is preferred over the NPZ fallback.
    """
    if path.is_dir():
        parquets = sorted(path.glob("val_records_step*.parquet"))
        if parquets:
            src = max(parquets, key=_step_of)
            return pd.read_parquet(src), src
        npzs = sorted(path.glob("val_log_step*.npz"))
        if npzs:
            src = max(npzs, key=_step_of)
            return _npz_to_df(src), src
        raise FileNotFoundError(
            f"No val_records_step*.parquet or val_log_step*.npz found in {path}"
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path), path
    if path.suffix == ".npz":
        return _npz_to_df(path), path
    raise ValueError(f"Unsupported file type: {path}")


def _summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {}
    out = {
        "count": int(x.size),
        "mean_ms": float(x.mean() * 1e3),
        "std_ms": float(x.std() * 1e3),
        "min_ms": float(x.min() * 1e3),
        "max_ms": float(x.max() * 1e3),
        "total_s": float(x.sum()),
    }
    for p in PERCENTILES:
        out[f"p{p}_ms"] = float(np.percentile(x, p) * 1e3)
    return out


def summarize(df: pd.DataFrame) -> dict:
    """Compute per-metric statistics for the compute-time columns."""
    metrics: dict[str, dict] = {}

    time_cols = [c for c in df.columns if c == "policy_time_s" or c.startswith("solver.time")]
    for c in time_cols:
        s = _summary(df[c].to_numpy())
        if s:
            metrics[c] = s

    if "policy_time_s" in df.columns and "solver.time_tot" in df.columns:
        overhead = df["policy_time_s"].to_numpy() - df["solver.time_tot"].to_numpy()
        s = _summary(overhead)
        if s:
            metrics["policy_overhead_s"] = s

    status_col = next((c for c in ("solver.status", "solver_status") if c in df.columns), None)
    n_fail = int((df[status_col].to_numpy() != 0).sum()) if status_col is not None else None

    return {"n_steps": int(len(df)), "solver_failures": n_fail, "metrics": metrics}


def print_table(summary: dict, src: Path) -> None:
    print(f"\nComputation-time statistics  ({summary['n_steps']} steps)  [{src}]")
    if summary["solver_failures"] is not None:
        print(f"solver failures (status != 0): {summary['solver_failures']}")
    cols = ["count", "mean_ms", "max_ms", "p95_ms", "p99_ms", "std_ms", "total_s"]
    header = f"{'metric':<22}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, stats in summary["metrics"].items():
        row = f"{name:<22}"
        for c in cols:
            v = stats.get(c)
            row += f"{v:>12.3f}" if isinstance(v, float) else f"{v:>12}"
        print(row)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path, help="Run directory or a parquet/npz per-step table.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write timing_summary.json (default: alongside the source).",
    )
    args = parser.parse_args()

    df, src = _load_table(args.path)
    summary = summarize(df)
    summary["source"] = str(src)
    print_table(summary, src)

    out = args.out if args.out is not None else (src.parent / "timing_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to: {out}")


if __name__ == "__main__":
    main()
