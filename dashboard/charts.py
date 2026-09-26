"""Plotly Visualization Utilities for Streamlit SOC Dashboard."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_risk_distribution_chart(incidents: List[Dict[str, Any]]) -> go.Figure:
    """Create a pie chart summarizing threat risk levels."""
    if not incidents:
        fig = go.Figure()
        fig.update_layout(title="No Incident Data Available")
        return fig

    df = pd.DataFrame(incidents)
    
    def get_risk_tier(score: float) -> str:
        if score >= 0.8:
            return "Critical Risk"
        elif score >= 0.5:
            return "High Risk"
        elif score >= 0.3:
            return "Medium Risk"
        return "Low Risk"

    df["risk_tier"] = df["risk_score"].apply(get_risk_tier)
    counts = df["risk_tier"].value_counts().reset_index()
    counts.columns = ["Risk Tier", "Count"]

    fig = px.pie(
        counts,
        values="Count",
        names="Risk Tier",
        color="Risk Tier",
        color_discrete_map={
            "Critical Risk": "#ef4444",
            "High Risk": "#f97316",
            "Medium Risk": "#eab308",
            "Low Risk": "#22c55e",
        },
        hole=0.4,
        title="Threat Incident Risk Level Breakdown",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), template="plotly_white")
    return fig


def create_attack_category_chart(incidents: List[Dict[str, Any]]) -> go.Figure:
    """Create a bar chart showing incidents by predicted attack category."""
    if not incidents:
        fig = go.Figure()
        fig.update_layout(title="No Data")
        return fig

    df = pd.DataFrame(incidents)
    counts = df["attack_type"].value_counts().reset_index()
    counts.columns = ["Attack Type", "Incidents"]

    fig = px.bar(
        counts,
        x="Attack Type",
        y="Incidents",
        color="Attack Type",
        title="Flagged Threats by Classification Category",
        text="Incidents",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), template="plotly_white", showlegend=False)
    return fig


def create_network_latency_chart(latency_df: pd.DataFrame) -> go.Figure:
    """Create a latency benchmark breakdown chart."""
    fig = px.bar(
        latency_df,
        x="Stage",
        y="Latency_us",
        color="Stage",
        title="NIDS Pipeline Subsystem Latency Benchmark (Microseconds)",
        labels={"Latency_us": "Latency (μs)"},
        text="Latency_us",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), template="plotly_white", showlegend=False)
    return fig
