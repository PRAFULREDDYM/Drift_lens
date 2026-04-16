"""
drift-lens dashboard — the wow-demo.

Launch::

    streamlit run drift_lens/dashboard.py

Or with custom data::

    streamlit run drift_lens/dashboard.py -- --snapshots ./my_data/snapshots
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Ensure drift_lens is importable when running from repo root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from drift_lens.constants import KMEANS_K_MAX, KMEANS_K_MIN, SEVERITY_THRESHOLDS
from drift_lens.detector import DriftDetector
from drift_lens.logger import EmbeddingLogger
from drift_lens.projector import DriftProjector

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="drift-lens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished, dark-feeling look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .big-score {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .severity-low    { color: #4CAF50; }
    .severity-medium { color: #FF9800; }
    .severity-high   { color: #F44336; }
    .severity-critical { color: #B71C1C; }
    .tagline {
        font-size: 0.95rem;
        color: #888;
        margin-top: -0.3rem;
    }
    .early-warning {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FF9800;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading demo data …")
def load_demo_data() -> tuple[list[np.ndarray], dict]:
    """Load pre-generated demo data from examples/demo_data."""
    demo_dir = _PROJECT_ROOT / "examples" / "demo_data"
    if not demo_dir.exists():
        st.error(
            "Demo data not found. Run `python examples/quickstart.py` first "
            "to generate synthetic embeddings."
        )
        st.stop()

    snapshot_dir = demo_dir / "snapshots"
    logger = EmbeddingLogger(path=str(snapshot_dir), window="1d")
    files = logger.list_snapshots()

    snapshots = [logger.load_snapshot(f) for f in files]

    timeline_path = demo_dir / "timeline.json"
    with open(timeline_path) as f:
        timeline = json.load(f)

    return snapshots, timeline


def _auto_k(X: np.ndarray) -> int:
    """Select optimal K for KMeans via silhouette score."""
    best_k, best_score = KMEANS_K_MIN, -1.0
    n = X.shape[0]
    k_max = min(KMEANS_K_MAX, n - 1)
    for k in range(KMEANS_K_MIN, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        if s > best_score:
            best_score = s
            best_k = k
    return best_k


@st.cache_data(show_spinner="Computing UMAP projections …")
def project_pair(
    baseline: np.ndarray, current: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit UMAP on baseline, transform both sets."""
    projector = DriftProjector(method="umap", n_components=2)
    proj_base = projector.fit_transform(baseline)
    proj_curr = projector.transform(current)
    return proj_base, proj_curr


@st.cache_data(show_spinner="Computing drift …")
def compute_drift(
    baseline: np.ndarray, current: np.ndarray, method: str
) -> dict:
    """Run drift detection and return result as a dict (for caching)."""
    detector = DriftDetector(method=method)
    result = detector.compare(baseline, current)
    return {
        "drift_score": result.drift_score,
        "is_drift": result.is_drift,
        "p_value": result.p_value,
        "method": result.method,
        "details": result.details,
    }


def _severity_label(score: float) -> str:
    if score >= SEVERITY_THRESHOLDS["critical"]:
        return "critical"
    if score >= SEVERITY_THRESHOLDS["high"]:
        return "high"
    if score >= SEVERITY_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _severity_emoji(severity: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}[severity]


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
def main() -> None:
    snapshots, timeline = load_demo_data()
    days = timeline["days"]
    drift_scores = timeline["drift_scores"]
    accuracy_vals = timeline["accuracy"]

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("## drift-lens")
        st.markdown('<p class="tagline">Your metrics lie. Your embeddings don\'t.</p>', unsafe_allow_html=True)
        st.divider()
        baseline_day = st.selectbox("Baseline day", days, index=0)
        current_day = st.selectbox("Current day", days, index=len(days) - 1)
        method = st.selectbox("Detection method", ["frechet", "mmd", "topology"])
        st.divider()
        st.caption("v0.1.0 · open source · no API keys")

    baseline_idx = baseline_day - 1
    current_idx = current_day - 1
    baseline_emb = snapshots[baseline_idx]
    current_emb = snapshots[current_idx]

    # Compute drift for selected pair
    result = compute_drift(baseline_emb, current_emb, method)
    score = result["drift_score"]
    severity = _severity_label(score)
    emoji = _severity_emoji(severity)

    # --- Header row ---
    hcol1, hcol2 = st.columns([2, 3])
    with hcol1:
        st.markdown(
            f'<div class="big-score severity-{severity}">'
            f"DRIFT SCORE: {score:.2f} {emoji} {severity.upper()}</div>",
            unsafe_allow_html=True,
        )
    with hcol2:
        # Find when drift first exceeded 0.3
        first_alert_day = next(
            (d for d, s in zip(days, drift_scores) if s >= 0.3 and d > 1), None
        )
        first_drop_day = next(
            (d for d, a in zip(days, accuracy_vals) if a < 0.90), None
        )
        if first_alert_day and first_drop_day and first_drop_day > first_alert_day:
            lead = first_drop_day - first_alert_day
            st.markdown(
                f'<div class="early-warning">'
                f"DETECTED {lead} DAYS BEFORE ACCURACY DROP</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Alert fired: Day {first_alert_day} · Accuracy dropped: Day {first_drop_day}"
            )
        elif first_alert_day:
            st.markdown(
                f'<div class="early-warning">ALERT FIRST FIRED ON DAY {first_alert_day}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # --- UMAP scatter plots ---
    st.subheader("Embedding Space Comparison")
    ucol1, ucol2 = st.columns(2)

    proj_base, proj_curr = project_pair(baseline_emb, current_emb)

    # Cluster with shared K so colours are comparable
    k_base = _auto_k(proj_base)
    k_curr = _auto_k(proj_curr)
    k_shared = max(k_base, k_curr)

    labels_base = KMeans(n_clusters=k_shared, n_init=5, random_state=42).fit_predict(proj_base)
    labels_curr = KMeans(n_clusters=k_shared, n_init=5, random_state=42).fit_predict(proj_curr)

    color_seq = px.colors.qualitative.Set2

    with ucol1:
        df_base = pd.DataFrame({"x": proj_base[:, 0], "y": proj_base[:, 1], "cluster": labels_base.astype(str)})
        fig1 = px.scatter(
            df_base, x="x", y="y", color="cluster",
            title=f"Baseline — Day {baseline_day}",
            color_discrete_sequence=color_seq,
        )
        fig1.update_traces(marker=dict(size=4, opacity=0.7))
        fig1.update_layout(
            height=420,
            showlegend=False,
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with ucol2:
        df_curr = pd.DataFrame({"x": proj_curr[:, 0], "y": proj_curr[:, 1], "cluster": labels_curr.astype(str)})
        fig2 = px.scatter(
            df_curr, x="x", y="y", color="cluster",
            title=f"Current — Day {current_day}",
            color_discrete_sequence=color_seq,
        )
        fig2.update_traces(marker=dict(size=4, opacity=0.7))
        fig2.update_layout(
            height=420,
            showlegend=False,
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Drift timeline + method details ---
    tcol1, tcol2 = st.columns([3, 2])

    with tcol1:
        st.subheader("Drift Timeline")

        fig_timeline = go.Figure()

        fig_timeline.add_trace(
            go.Scatter(
                x=days,
                y=drift_scores,
                mode="lines+markers",
                name="Drift Score",
                line=dict(color="#FF9800", width=3),
                marker=dict(size=8),
            )
        )

        fig_timeline.add_trace(
            go.Scatter(
                x=days,
                y=accuracy_vals,
                mode="lines+markers",
                name="Accuracy",
                line=dict(color="#2196F3", width=2, dash="dot"),
                marker=dict(size=6),
                yaxis="y",
            )
        )

        # Threshold line
        fig_timeline.add_hline(
            y=0.3,
            line_dash="dash",
            line_color="rgba(244,67,54,0.5)",
            annotation_text="Alert threshold",
            annotation_position="top left",
        )

        # Annotation for first alert
        if first_alert_day:
            fig_timeline.add_vline(
                x=first_alert_day,
                line_dash="dash",
                line_color="rgba(255,152,0,0.6)",
                annotation_text=f"Alert fired (Day {first_alert_day})",
                annotation_position="top",
            )
        if first_drop_day:
            fig_timeline.add_vline(
                x=first_drop_day,
                line_dash="dash",
                line_color="rgba(33,150,243,0.6)",
                annotation_text=f"Accuracy drops (Day {first_drop_day})",
                annotation_position="top right",
            )

        fig_timeline.update_layout(
            height=380,
            xaxis_title="Day",
            yaxis_title="Score / Accuracy",
            yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with tcol2:
        st.subheader("Detection Details")

        tab_fed, tab_mmd, tab_topo = st.tabs(["Fréchet (FED)", "MMD", "Topology"])

        with tab_fed:
            r_fed = compute_drift(baseline_emb, current_emb, "frechet")
            st.metric("Drift Score", f"{r_fed['drift_score']:.4f}")
            st.metric("Raw FED", f"{r_fed['details'].get('raw_distance', 0):.2f}")
            st.metric("Mean Shift (L2)", f"{r_fed['details'].get('mean_shift', 0):.4f}")
            st.caption("Assumes Gaussian embedding distributions. Fast, interpretable.")

        with tab_mmd:
            r_mmd = compute_drift(baseline_emb, current_emb, "mmd")
            st.metric("Drift Score", f"{r_mmd['drift_score']:.4f}")
            if r_mmd["p_value"] is not None:
                st.metric("p-value", f"{r_mmd['p_value']:.4f}")
            st.metric("Raw MMD²", f"{r_mmd['details'].get('raw_mmd_squared', 0):.6f}")
            st.metric("Bandwidth σ", f"{r_mmd['details'].get('bandwidth_sigma', 0):.4f}")
            st.caption("Non-parametric, kernel-based. No Gaussian assumption.")

        with tab_topo:
            r_topo = compute_drift(baseline_emb, current_emb, "topology")
            st.metric("Drift Score", f"{r_topo['drift_score']:.4f}")
            st.metric(
                "Wasserstein Distance",
                f"{r_topo['details'].get('total_wasserstein', 0):.4f}",
            )
            per_dim = r_topo["details"].get("per_dimension", {})
            for dim_label, dist in per_dim.items():
                st.metric(f"{dim_label} distance", f"{dist:.4f}")
            st.caption("Persistent homology — detects cluster merges, splits, holes.")

    st.divider()

    # --- Alert history + recommended actions ---
    acol1, acol2 = st.columns([3, 2])

    with acol1:
        st.subheader("Alert History")
        alert_data = []
        for day_idx, (d, s) in enumerate(zip(days, drift_scores)):
            sev = _severity_label(s)
            alert_data.append(
                {
                    "Day": d,
                    "Drift Score": round(s, 4),
                    "Severity": f"{_severity_emoji(sev)} {sev.upper()}",
                    "Would Alert": "YES" if s >= 0.3 and d > 1 else "—",
                }
            )
        st.dataframe(
            pd.DataFrame(alert_data),
            use_container_width=True,
            hide_index=True,
            height=350,
        )

    with acol2:
        st.subheader("Recommended Actions")
        current_severity = _severity_label(score)
        if current_severity == "critical":
            st.error("**CRITICAL** — Immediate action required")
            st.markdown(
                "1. **Halt** auto-decisions if safety-critical\n"
                "2. **Re-evaluate** model on current production data\n"
                "3. **Consider** fine-tuning or retraining on recent data"
            )
        elif current_severity == "high":
            st.warning("**HIGH** — Investigate promptly")
            st.markdown(
                "1. **Run evaluation** on current production data\n"
                "2. **Compare** baseline vs current metrics\n"
                "3. **Check** data pipeline for upstream shifts"
            )
        elif current_severity == "medium":
            st.info("**MEDIUM** — Monitor closely")
            st.markdown(
                "1. **Review** recent data pipeline changes\n"
                "2. **Re-evaluate** on a fresh sample\n"
                "3. **Set up** more frequent monitoring"
            )
        else:
            st.success("**LOW** — No action required")
            st.markdown("Embeddings are stable. Continue routine monitoring.")

    # --- Footer ---
    st.divider()
    st.caption(
        "drift-lens v0.1.0 · "
        "Your metrics lie. Your embeddings don't. · "
        "[GitHub](https://github.com/PRAFULREDDYM/drift-lens)"
    )


if __name__ == "__main__":
    main()
