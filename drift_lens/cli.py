"""
drift-lens CLI — compare, watch, report, dashboard.

Usage::

    drift-lens compare --baseline ./snapshots/day1.parquet --current ./snapshots/day14.parquet
    drift-lens watch --snapshots ./snapshots --method mmd --threshold 0.4
    drift-lens report --snapshots ./snapshots --output drift_report.html
    drift-lens dashboard --snapshots ./snapshots
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import click
import numpy as np

from drift_lens.alert import AlertEngine
from drift_lens.detector import DriftDetector
from drift_lens.logger import EmbeddingLogger


def _load_embeddings(path: str) -> np.ndarray:
    """
    Load embeddings from a parquet file or a directory of parquet files.

    Parameters
    ----------
    path : str
        Path to a ``.parquet`` file or a directory containing them.

    Returns
    -------
    np.ndarray
        Stacked embeddings ``(n, d)``.
    """
    p = Path(path)
    logger = EmbeddingLogger(path=str(p.parent if p.is_file() else p), window="1d")

    if p.is_file():
        return logger.load_snapshot(p)

    files = sorted(p.glob("*.parquet"))
    if not files:
        raise click.ClickException(f"No .parquet files found in {p}")
    arrays = [logger.load_snapshot(f) for f in files]
    return np.vstack(arrays)


@click.group()
@click.version_option(version="0.1.0", prog_name="drift-lens")
def cli() -> None:
    """drift-lens: Your metrics lie. Your embeddings don't."""


@cli.command()
@click.option("--baseline", required=True, help="Path to baseline snapshot (file or directory).")
@click.option("--current", required=True, help="Path to current snapshot (file or directory).")
@click.option(
    "--method",
    type=click.Choice(["frechet", "mmd", "topology"], case_sensitive=False),
    default="frechet",
    help="Detection method.",
)
@click.option("--threshold", type=float, default=0.3, help="Alert threshold (0-1).")
def compare(baseline: str, current: str, method: str, threshold: float) -> None:
    """Compare two embedding snapshots for drift."""
    click.echo(f"Loading baseline from {baseline} ...")
    base_emb = _load_embeddings(baseline)
    click.echo(f"  → {base_emb.shape[0]} embeddings, dim={base_emb.shape[1]}")

    click.echo(f"Loading current from {current} ...")
    curr_emb = _load_embeddings(current)
    click.echo(f"  → {curr_emb.shape[0]} embeddings, dim={curr_emb.shape[1]}")

    click.echo(f"\nRunning {method} detection ...")
    detector = DriftDetector(method=method, threshold=threshold)
    result = detector.compare(base_emb, curr_emb)

    severity_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "bright_red"}
    from drift_lens.alert import _severity_for_score
    severity = _severity_for_score(result.drift_score)

    click.echo("\n" + "=" * 50)
    click.secho(
        f"  DRIFT SCORE: {result.drift_score:.4f}  ({severity.upper()})",
        fg=severity_colors[severity],
        bold=True,
    )
    click.echo(f"  Is Drift:    {result.is_drift}")
    if result.p_value is not None:
        click.echo(f"  p-value:     {result.p_value:.4f}")
    click.echo("=" * 50)

    click.echo("\nDetails:")
    for k, v in result.details.items():
        click.echo(f"  {k}: {v}")


@cli.command()
@click.option("--snapshots", required=True, help="Directory containing snapshot parquet files.")
@click.option(
    "--method",
    type=click.Choice(["frechet", "mmd", "topology"], case_sensitive=False),
    default="frechet",
)
@click.option("--threshold", type=float, default=0.3)
@click.option("--interval", type=int, default=60, help="Polling interval in seconds.")
def watch(snapshots: str, method: str, threshold: float, interval: int) -> None:
    """Watch a snapshot directory and alert on drift."""
    snap_dir = Path(snapshots)
    if not snap_dir.is_dir():
        raise click.ClickException(f"Not a directory: {snap_dir}")

    logger = EmbeddingLogger(path=str(snap_dir), window="1d")
    detector = DriftDetector(method=method, threshold=threshold)
    alert_engine = AlertEngine(threshold=threshold, alert_dir=str(snap_dir))

    click.echo(f"Watching {snap_dir} for new snapshots (method={method}, interval={interval}s)")
    click.echo("Press Ctrl+C to stop.\n")

    seen_files: set[str] = set()

    try:
        while True:
            files = logger.list_snapshots()
            file_names = {str(f) for f in files}
            new_files = file_names - seen_files

            if new_files and len(files) >= 2:
                click.echo(f"[{time.strftime('%H:%M:%S')}] New snapshot detected: {len(new_files)} file(s)")

                baseline = logger.load_snapshot(files[0])
                current = logger.load_snapshot(files[-1])

                result = detector.compare(baseline, current)
                alert = alert_engine.check(result)

                if alert.fired:
                    click.secho(f"  ALERT: {alert.message}", fg="red", bold=True)
                    click.echo(f"  Action: {alert.recommended_action}")
                else:
                    click.echo(f"  Score: {result.drift_score:.4f} — {alert.message}")

            seen_files = file_names
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped watching.")


@cli.command()
@click.option("--snapshots", required=True, help="Directory containing snapshot parquet files.")
@click.option("--output", default="drift_report.html", help="Output HTML report path.")
@click.option(
    "--method",
    type=click.Choice(["frechet", "mmd", "topology"], case_sensitive=False),
    default="frechet",
)
def report(snapshots: str, output: str, method: str) -> None:
    """Generate a standalone HTML drift report."""
    snap_dir = Path(snapshots)
    logger = EmbeddingLogger(path=str(snap_dir), window="1d")
    files = logger.list_snapshots()

    if len(files) < 2:
        raise click.ClickException(
            f"Need at least 2 snapshots to generate a report, found {len(files)} in {snap_dir}."
        )

    click.echo(f"Found {len(files)} snapshots in {snap_dir}")

    baseline = logger.load_snapshot(files[0])
    detector = DriftDetector(method=method)

    scores = []
    for f in files:
        emb = logger.load_snapshot(f)
        result = detector.compare(baseline, emb)
        scores.append({"file": f.name, "drift_score": result.drift_score})

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[s["file"] for s in scores],
            y=[s["drift_score"] for s in scores],
            mode="lines+markers",
            name="Drift Score",
            line=dict(color="#FF9800", width=3),
        )
    )
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Alert threshold")
    fig.update_layout(
        title="drift-lens Report",
        xaxis_title="Snapshot",
        yaxis_title="Drift Score",
        yaxis=dict(range=[0, 1.05]),
        height=500,
    )

    from drift_lens.alert import _severity_for_score

    html_rows = ""
    for s in scores:
        sev = _severity_for_score(s["drift_score"])
        html_rows += (
            f"<tr><td>{s['file']}</td>"
            f"<td>{s['drift_score']:.4f}</td>"
            f"<td>{sev.upper()}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>drift-lens Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; }}
        h1 {{ color: #333; }}
        .tagline {{ color: #888; font-style: italic; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>drift-lens Report</h1>
    <p class="tagline">Your metrics lie. Your embeddings don't.</p>
    <p>Method: <strong>{method}</strong> | Snapshots: <strong>{len(files)}</strong></p>

    {fig.to_html(full_html=False, include_plotlyjs="cdn")}

    <h2>Snapshot Details</h2>
    <table>
        <tr><th>Snapshot</th><th>Drift Score</th><th>Severity</th></tr>
        {html_rows}
    </table>

    <p style="color:#888; margin-top:2rem;">Generated by drift-lens v0.1.0</p>
</body>
</html>"""

    Path(output).write_text(html)
    click.echo(f"Report saved to {output}")


@cli.command()
@click.option("--snapshots", default=None, help="Path to snapshots (uses demo data if omitted).")
def dashboard(snapshots: str | None) -> None:
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    if snapshots:
        cmd.extend(["--", "--snapshots", snapshots])
    click.echo("Launching drift-lens dashboard ...")
    subprocess.run(cmd)


if __name__ == "__main__":
    cli()
