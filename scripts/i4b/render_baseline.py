"""Plotly HTML dashboard for i4b baseline channel logs.

Loads ``val_log_step*.npz`` + ``val_log_step*.json`` produced by
``run_baseline.py`` and emits a self-contained HTML dashboard — one
``go.Figure`` per panel group.

Panel rules:
- ``line`` panel: scalar time series + 1-step-ahead MPC prediction overlay.
  Setpoint pair ``T_set_lower`` / ``T_set_upper`` becomes a shaded comfort band.
- ``sequence`` panels also get a secondary heatmap section showing the full
  (T, K) prediction horizon grid.
- ``matrix`` panel: mean-over-time heatmap with symmetric RdBu colorscale.

The ``figures`` dict in ``main()`` is the extension point — add a
``name -> go.Figure`` entry to grow the dashboard.

Usage
-----
    python scripts/i4b/render_baseline.py --run-dir output/i4b_baseline
    python scripts/i4b/render_baseline.py  # finds latest run automatically
    python scripts/i4b/render_baseline.py --run-dir <dir> --out-html <file.html>
"""

from __future__ import annotations

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


@dataclass
class Panel:
    name: str
    kind: str  # "line" | "matrix"
    channels: list[dict]  # channel metadata dicts


def _find_latest_run(output_root: Path = Path("output")) -> Path:
    candidates = sorted(
        (p for p in output_root.rglob("val_log_step*.npz")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No val_log_step*.npz under {output_root}")
    return candidates[-1].parent


def load(run_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    npz_candidates = sorted(run_dir.glob("val_log_step*.npz"))
    json_candidates = sorted(run_dir.glob("val_log_step*.json"))
    if not npz_candidates or not json_candidates:
        raise FileNotFoundError(f"Missing val_log files in {run_dir}")
    arrays = dict(np.load(npz_candidates[-1], allow_pickle=False))
    metadata = json.loads(json_candidates[-1].read_text())
    return arrays, metadata


def _group_panels(channel_meta: list[dict]) -> list[Panel]:
    order: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for ch in channel_meta:
        if not ch["kinds"]:
            continue
        p = ch["panel"]
        if p not in grouped:
            grouped[p] = []
            order.append(p)
        grouped[p].append(ch)

    panels: list[Panel] = []
    for name in order:
        chans = grouped[name]
        kind = "matrix" if any("matrix" in c["kinds"] for c in chans) else "line"
        panels.append(Panel(name=name, kind=kind, channels=chans))
    return panels


def _infer_T(arrays: dict[str, np.ndarray], metadata: dict) -> int:
    for ch in metadata["channels"]:
        for kind in ("scalar", "sequence", "matrix"):
            if kind in ch["kinds"]:
                return int(arrays[ch["keys"][kind]].shape[0])
    raise RuntimeError("Could not infer T from any channel.")


def _line_panel_figure(
    panel: Panel,
    arrays: dict[str, np.ndarray],
    t_cl: np.ndarray,
    dt_h: float,
) -> go.Figure:
    """One go.Figure for a line panel: scalars + 1-step-ahead predictions."""
    fig = go.Figure()
    by_name = {c["name"]: c for c in panel.channels}
    has_band = {"T_set_lower", "T_set_upper"} <= by_name.keys()
    ylabels = [c["ylabel"] for c in panel.channels if c["ylabel"]]

    # Comfort band: lower trace first, then upper with fill='tonexty'.
    if has_band:
        lo_key = by_name["T_set_lower"]["keys"].get("scalar")
        hi_key = by_name["T_set_upper"]["keys"].get("scalar")
        if lo_key and hi_key:
            fig.add_trace(
                go.Scatter(
                    x=t_cl,
                    y=arrays[lo_key],
                    name="T_set_lower",
                    mode="lines",
                    line=dict(color="green", width=0.8, dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=t_cl,
                    y=arrays[hi_key],
                    name="comfort band",
                    mode="lines",
                    line=dict(color="green", width=0.8, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(0,128,0,0.08)",
                )
            )

    for ch in panel.channels:
        name = ch["name"]
        if has_band and name in {"T_set_lower", "T_set_upper"}:
            continue  # already drawn as band above
        if "scalar" in ch["kinds"]:
            fig.add_trace(
                go.Scatter(x=t_cl, y=arrays[ch["keys"]["scalar"]], name=name, mode="lines")
            )
        if "scalars_dict" in ch["kinds"]:
            prefix = f"{name}."
            for k in sorted(key for key in arrays if key.startswith(prefix)):
                fig.add_trace(go.Scatter(x=t_cl, y=arrays[k], name=k[len(prefix) :], mode="lines"))
        if "sequence" in ch["kinds"]:
            seq = arrays[ch["keys"]["sequence"]]  # (T, K)
            fig.add_trace(
                go.Scatter(
                    x=t_cl,
                    y=seq[:, 0],
                    name=f"{name} (1-step pred)",
                    mode="lines",
                    line=dict(dash="dash", width=1.2),
                )
            )

    fig.update_layout(
        height=340,
        title=panel.name,
        xaxis_title="time [h]",
        yaxis_title=ylabels[0] if ylabels else "",
        legend=dict(orientation="h"),
    )
    return fig


def _sequence_heatmap_figure(
    panel: Panel,
    arrays: dict[str, np.ndarray],
    t_cl: np.ndarray,
    dt_h: float,
) -> go.Figure | None:
    """Heatmap of the full (T, K) prediction grid for sequence channels.

    Rows = time step (realised), columns = horizon offset. Shows the full
    planning history in one view — more informative than the old slider.
    """
    seq_channels = [c for c in panel.channels if "sequence" in c["kinds"]]
    if not seq_channels:
        return None

    ch = seq_channels[0]
    seq = arrays[ch["keys"]["sequence"]]  # (T, K)
    K = seq.shape[1]
    horizon_offsets = np.arange(K) * dt_h

    zmid = float(np.nanmedian(seq))
    fig = go.Figure(
        go.Heatmap(
            z=seq,
            x=horizon_offsets,
            y=t_cl,
            colorscale="RdBu",
            zmid=zmid,
            colorbar=dict(title=ch["ylabel"] or ch["name"]),
        )
    )
    fig.update_layout(
        height=320,
        title=f"{panel.name} — prediction horizon (T × K)",
        xaxis_title="horizon offset [h]",
        yaxis_title="time [h]",
    )
    return fig


def _matrix_panel_figure(panel: Panel, arrays: dict[str, np.ndarray]) -> go.Figure:
    """Mean-over-time heatmap for a matrix channel (e.g. sensitivity du/dp)."""
    ch = next(c for c in panel.channels if "matrix" in c["kinds"])
    stacked = arrays[ch["keys"]["matrix"]]  # (T, H, W)
    mean_mat = stacked.mean(axis=0)
    vmax = float(np.abs(mean_mat).max()) or 1.0

    fig = go.Figure(
        go.Heatmap(
            z=mean_mat,
            colorscale=ch.get("cmap", "RdBu"),
            zmin=-vmax,
            zmax=vmax,
        )
    )
    fig.update_layout(
        height=420,
        title=ch.get("ylabel") or ch["name"],
        xaxis_title="parameter index j",
        yaxis_title="horizon step k",
    )
    return fig


def build_dashboard(figures: dict[str, go.Figure], out_path: Path, title: str) -> None:
    """Concatenate plotly figures into one self-contained HTML file."""
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title></head><body>",
        f"<h1>{title}</h1>",
    ]
    for i, (name, fig) in enumerate(figures.items()):
        parts.append(f"<h2>{name}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))


def main(run_dir: Path, out_html: Path) -> None:
    arrays, metadata = load(run_dir)
    header = metadata["header"]
    dt_h = float(header["delta_t_s"]) / 3600.0

    panels = _group_panels(metadata["channels"])
    if not panels:
        raise RuntimeError("No channels with data to render.")

    T = _infer_T(arrays, metadata)
    t_cl = np.arange(T) * dt_h

    figures: dict[str, go.Figure] = {}
    for panel in panels:
        if panel.kind == "matrix":
            figures[panel.name] = _matrix_panel_figure(panel, arrays)
        else:
            figures[panel.name] = _line_panel_figure(panel, arrays, t_cl, dt_h)
            hmap = _sequence_heatmap_figure(panel, arrays, t_cl, dt_h)
            if hmap is not None:
                figures[f"{panel.name} — horizon"] = hmap

    build_dashboard(figures, out_html, title=f"i4b baseline — {run_dir.name}")
    print(f"Dashboard saved to: {out_html.resolve()}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Render i4b baseline channel log as a plotly HTML dashboard.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Output HTML path (default: <run-dir>/dashboard.html)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _find_latest_run()
    out_html = args.out_html or run_dir / "dashboard.html"
    print(f"Rendering from: {run_dir}")
    main(run_dir, out_html)
