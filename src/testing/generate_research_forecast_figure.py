"""Generate a publication-ready probabilistic forecast figure.

The script reads the latest forecast evaluation output by default, filters the
requested date window, and saves a clean figure with labeled axes for use in a
research paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTING_ROOT = PROJECT_ROOT / "models" / "testing"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "paper"
DEFAULT_OUTPUT_BASENAME = "forecast_2026-04-05_to_2026-04-06"


def _select_predictions_csv(start_date: str, end_date: str) -> Path:
    """Return the forecast CSV that fully covers the requested date range."""

    candidates = sorted(TESTING_ROOT.glob("*/test_predictions_vs_actual.csv"))
    if not candidates:
        raise FileNotFoundError(f"No forecast evaluation CSV found under {TESTING_ROOT}")

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    coverage = []

    for csv_path in candidates:
        timestamps = pd.read_csv(csv_path, usecols=["timestamp"])
        timestamps["timestamp"] = pd.to_datetime(timestamps["timestamp"])
        coverage.append(
            {
                "path": csv_path,
                "start": timestamps["timestamp"].min(),
                "end": timestamps["timestamp"].max(),
            }
        )

    matching = [item for item in coverage if item["start"] <= start_timestamp and item["end"] >= end_timestamp]
    if matching:
        return max(matching, key=lambda item: item["path"].parent.name)["path"]

    latest_coverage = max(coverage, key=lambda item: item["end"])
    raise ValueError(
        f"No forecast CSV fully covers {start_date} to {end_date}. "
        f"Closest available run is {latest_coverage['path']} with coverage "
        f"{latest_coverage['start']} to {latest_coverage['end']}."
    )


def _load_date_slice(csv_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load the forecast CSV and keep the inclusive date range requested."""

    df = pd.read_csv(csv_path)
    required_columns = {"timestamp", "actual_load_mw", "p10", "p50", "p90"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing_columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    date_slice = df.loc[(df["timestamp"] >= start_timestamp) & (df["timestamp"] < end_timestamp)].copy()

    if date_slice.empty:
        raise ValueError(
            f"No rows found in {csv_path} for the requested range {start_date} to {end_date}."
        )

    return date_slice.sort_values("timestamp").reset_index(drop=True)


def _format_axis(ax: plt.Axes) -> None:
    """Apply paper-friendly axis formatting."""

    ax.set_xlabel("Date and time (IST)", fontsize=12)
    ax.set_ylabel("Electricity demand (MW)", fontsize=12)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, which="major", color="#d9e2ec", linewidth=0.8)
    ax.grid(True, which="minor", color="#edf2f7", linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def _build_figure(date_slice: pd.DataFrame, output_basename: str) -> tuple[plt.Figure, plt.Axes]:
    """Create the publication-ready forecast plot."""

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
        }
    )

    fig, ax = plt.subplots(figsize=(13.2, 5.6))

    band_color = "#5DADE2"
    actual_color = "#1F2937"
    p50_color = "#F59E0B"
    p10_color = "#10B981"
    p90_color = "#8B5CF6"

    ax.fill_between(
        date_slice["timestamp"],
        date_slice["p10"],
        date_slice["p90"],
        color=band_color,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        date_slice["timestamp"],
        date_slice["actual_load_mw"],
        color=actual_color,
        linewidth=1.8,
        label="Actual / observed",
        zorder=4,
    )
    ax.plot(
        date_slice["timestamp"],
        date_slice["p50"],
        color=p50_color,
        linewidth=2.2,
        label="P50 forecast",
        zorder=3,
    )
    ax.plot(
        date_slice["timestamp"],
        date_slice["p10"],
        color=p10_color,
        linewidth=1.2,
        linestyle=(0, (2, 2)),
        label="P10 forecast",
        zorder=2,
    )
    ax.plot(
        date_slice["timestamp"],
        date_slice["p90"],
        color=p90_color,
        linewidth=1.2,
        linestyle=(0, (2, 2)),
        label="P90 forecast",
        zorder=2,
    )

    peak_row = date_slice.loc[date_slice["p50"].idxmax()]
    ax.annotate(
        f"Peak P50\n{peak_row['p50']:.0f} MW",
        xy=(peak_row["timestamp"], peak_row["p50"]),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=9,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cbd5e1", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.0),
        zorder=5,
    )

    ax.set_xlim(date_slice["timestamp"].iloc[0], date_slice["timestamp"].iloc[-1])
    y_min = min(date_slice[["actual_load_mw", "p10"]].min().min(), date_slice["p50"].min())
    y_max = max(date_slice[["actual_load_mw", "p90"]].max().max(), date_slice["p50"].max())
    padding = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - padding, y_max + padding)

    day_boundary = pd.Timestamp(date_slice["timestamp"].dt.normalize().iloc[0] + pd.Timedelta(days=1))
    if date_slice["timestamp"].min() < day_boundary < date_slice["timestamp"].max():
        ax.axvline(day_boundary, color="#94a3b8", linestyle="--", linewidth=0.9, alpha=0.7, zorder=0)

    _format_axis(ax)

    title = "48-hour probabilistic electricity demand forecast"
    subtitle = "5-6 April 2026, 15-minute resolution"
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=14)
    ax.text(
        0.0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        fontsize=11,
        color="#4b5563",
        ha="left",
        va="bottom",
    )

    legend_handles = [
        Line2D([0], [0], color=actual_color, lw=1.8, label="Actual / observed"),
        Line2D([0], [0], color=p50_color, lw=2.2, label="P50 forecast"),
        Line2D([0], [0], color=p10_color, lw=1.2, linestyle=(0, (2, 2)), label="P10 forecast"),
        Line2D([0], [0], color=p90_color, lw=1.2, linestyle=(0, (2, 2)), label="P90 forecast"),
        Patch(facecolor=band_color, edgecolor=band_color, alpha=0.18, label="P10-P90 coverage band"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig, ax


def _save_outputs(fig: plt.Figure, output_dir: Path, output_basename: str) -> dict[str, Path]:
    """Save the figure in paper-friendly formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": output_dir / f"{output_basename}.pdf",
        "svg": output_dir / f"{output_basename}.svg",
        "png": output_dir / f"{output_basename}.png",
    }

    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to test_predictions_vs_actual.csv. Defaults to the newest run under models/testing.",
    )
    parser.add_argument("--start-date", default="2026-04-05", help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default="2026-04-06", help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figure files will be written.",
    )
    parser.add_argument(
        "--output-basename",
        default=DEFAULT_OUTPUT_BASENAME,
        help="Base filename used for the saved figure outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and save the requested paper figure."""

    args = parse_args()
    input_csv = args.input if args.input is not None else _select_predictions_csv(args.start_date, args.end_date)
    date_slice = _load_date_slice(input_csv, args.start_date, args.end_date)
    fig, _ = _build_figure(date_slice, args.output_basename)
    outputs = _save_outputs(fig, args.output_dir, args.output_basename)
    plt.close(fig)

    peak_row = date_slice.loc[date_slice["p50"].idxmax()]
    print(f"Input CSV: {input_csv}")
    print(f"Selected rows: {len(date_slice)}")
    print(f"Date range: {date_slice['timestamp'].iloc[0]} to {date_slice['timestamp'].iloc[-1]}")
    print(f"Peak P50: {peak_row['p50']:.2f} MW at {peak_row['timestamp']}")
    for label, path in outputs.items():
        print(f"Saved {label.upper()}: {path}")


if __name__ == "__main__":
    main()