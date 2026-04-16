from pathlib import Path
import sys
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow `python src/app3.py` to resolve sibling imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import (
    SKILL_DISTRIBUTIONS_PATH,
    render_ai_ml_cooccurrence_view,
    render_occupation_similarity_view,
)


VIEW_LABELS = {
    "occupation_similarity": "Similarity vs AI Exposure",
    "ai_ml_cooccurrence": "AI/ML Co-occurrence",
    "skill_distributions": "Skill Distributions",
}

AI_EVENTS = [
    {"label": "GH Copilot preview", "period": "2021-Q2"},
    {"label": "GH Copilot GA", "period": "2022-Q2"},
    {"label": "ChatGPT", "period": "2022-Q4"},
    {"label": "GPT-4", "period": "2023-Q1"},
    {"label": "Claude 1", "period": "2023-Q1"},
    {"label": "Claude 2", "period": "2023-Q3"},
    {"label": "Claude 3", "period": "2024-Q1"},
    {"label": "GPT-4o", "period": "2024-Q2"},
    {"label": "Claude 3.5 Sonnet", "period": "2024-Q2"},
    {"label": "Claude Code", "period": "2025-Q1"},
    {"label": "GPT-4.5 / o3", "period": "2025-Q1"},
]


@st.cache_data(show_spinner="Loading data...")
def load_agg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "QUARTER",
            "YEAR",
            "SKILL_NAME",
            "SKILL_SUBCATEGORY_NAME",
            "SKILL_CATEGORY_NAME",
            "CNT",
        ],
    )
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["QUARTER"] = pd.to_numeric(df["QUARTER"], errors="coerce")
    df["CNT"] = pd.to_numeric(df["CNT"], errors="coerce")
    df = df.dropna(
        subset=[
            "YEAR",
            "QUARTER",
            "SKILL_NAME",
            "SKILL_SUBCATEGORY_NAME",
            "SKILL_CATEGORY_NAME",
            "CNT",
        ]
    ).copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df["QUARTER"] = df["QUARTER"].astype(int)
    df["CNT"] = df["CNT"].astype(int)
    df["time_ord"] = df["YEAR"] * 10 + df["QUARTER"]
    df["period"] = df["YEAR"].astype(str) + "-Q" + df["QUARTER"].astype(str)
    quarter_start_month = (df["QUARTER"] - 1) * 3 + 1
    df["QUARTER_START"] = pd.to_datetime(
        {
            "year": df["YEAR"],
            "month": quarter_start_month,
            "day": 1,
        }
    )
    df["QUARTER_LABEL"] = "Q" + df["QUARTER"].astype(str) + " " + df["YEAR"].astype(str)

    agg = (
        df.groupby(
            [
                "time_ord",
                "period",
                "QUARTER_START",
                "QUARTER_LABEL",
                "SKILL_NAME",
                "SKILL_SUBCATEGORY_NAME",
                "SKILL_CATEGORY_NAME",
            ],
            observed=True,
        )["CNT"]
        .sum()
        .reset_index()
    )
    total = (
        df.groupby(["time_ord"], observed=True)["CNT"]
        .sum()
        .rename("total_CNT")
        .reset_index()
    )
    agg = agg.merge(total, on="time_ord")
    agg["share"] = agg["CNT"] / agg["total_CNT"]
    return agg.sort_values("time_ord").reset_index(drop=True)


def add_event_markers(fig: go.Figure) -> None:
    by_period: dict[str, list[str]] = defaultdict(list)
    for event in AI_EVENTS:
        by_period[event["period"]].append(event["label"])

    for period in sorted(by_period):
        text = "<br>".join(by_period[period])
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=period,
            x1=period,
            y0=0,
            y1=1,
            line=dict(color="rgba(120,120,120,0.4)", width=1, dash="dot"),
        )
        fig.add_annotation(
            xref="x",
            yref="paper",
            x=period,
            y=0.80,
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=12, color="rgba(255,255,255,0.5)"),
        )


def line_chart(agg: pd.DataFrame, dim_col: str, selected: list[str], metric: str) -> go.Figure:
    fig = go.Figure()
    for name in selected:
        sub = (
            agg[agg[dim_col] == name]
            .groupby(["time_ord", "period"], observed=True)["CNT"]
            .sum()
            .reset_index()
            .sort_values("time_ord")
        )
        if metric == "Relative growth (%)":
            baseline = sub["CNT"].iloc[0] if not sub.empty else None
            y_vals = (sub["CNT"] / baseline - 1) * 100 if baseline else sub["CNT"] * 0
            hover = "<b>%{fullData.name}</b><br>%{x}<br>Growth: %{y:+.1f}%<extra></extra>"
        else:
            baseline = sub["CNT"].iloc[0] if not sub.empty else 0
            y_vals = sub["CNT"] - baseline
            hover = "<b>%{fullData.name}</b><br>%{x}<br>Δ CNT: %{y:,}<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=sub["period"],
                y=y_vals,
                mode="lines+markers",
                name=name,
                hovertemplate=hover,
            )
        )

    y_label = {
        "Relative growth (%)": "% change from earliest period",
        "Absolute Δ CNT": "Δ CNT from earliest period",
    }[metric]
    fig.update_layout(
        xaxis_title="Quarter",
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=500,
        margin=dict(t=60),
    )
    add_event_markers(fig)
    return fig


@st.cache_data(show_spinner="Computing slopes...")
def compute_slopes(agg: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = agg.groupby([group_col, "time_ord"], observed=True)["CNT"].sum().reset_index()
    grouped["xy"] = grouped["time_ord"] * grouped["CNT"]
    grouped["xx"] = grouped["time_ord"] ** 2
    slopes = grouped.groupby(group_col, observed=True).agg(
        n=("time_ord", "count"),
        sum_x=("time_ord", "sum"),
        sum_y=("CNT", "sum"),
        sum_xx=("xx", "sum"),
        sum_xy=("xy", "sum"),
    )
    denom = slopes["n"] * slopes["sum_xx"] - slopes["sum_x"] ** 2
    slopes["slope"] = (slopes["n"] * slopes["sum_xy"] - slopes["sum_x"] * slopes["sum_y"]) / denom
    return slopes["slope"].reset_index().rename(columns={group_col: "name"})


def slope_bar_chart(slopes: pd.DataFrame, title: str, top_n: int = 50, flop: bool = False) -> go.Figure:
    top = (
        slopes.nsmallest(top_n, "slope").sort_values("slope", ascending=True)
        if flop
        else slopes.nlargest(top_n, "slope").sort_values("slope", ascending=False)
    )
    color = "tomato" if flop else "steelblue"
    fig = go.Figure(
        go.Bar(
            x=top["slope"],
            y=top["name"],
            orientation="h",
            marker_color=color,
            hovertemplate="%{y}<br>Slope: %{x:,.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="OLS slope (CNT / quarter)",
        yaxis_title=None,
        height=max(420, top_n * 22),
        margin=dict(l=10, r=20, t=50, b=40),
    )
    return fig


def render_skill_distributions_view() -> None:
    st.subheader("Skill Distributions Over Time")
    st.caption(
        "Tracks skill-category, subcategory, and skill growth through time using "
        f"`{SKILL_DISTRIBUTIONS_PATH.name}`."
    )

    if not SKILL_DISTRIBUTIONS_PATH.exists():
        st.error(
            f"Missing `{SKILL_DISTRIBUTIONS_PATH.name}`. Generate the file before launching this dashboard."
        )
        return

    agg = load_agg(SKILL_DISTRIBUTIONS_PATH)
    all_skills = sorted(agg["SKILL_NAME"].unique())
    all_categories = sorted(agg["SKILL_CATEGORY_NAME"].unique())
    all_subcategories = sorted(agg["SKILL_SUBCATEGORY_NAME"].unique())

    dimensions = {
        "Skill Category": ("SKILL_CATEGORY_NAME", all_categories),
        "Skill Subcategory": ("SKILL_SUBCATEGORY_NAME", all_subcategories),
        "Skill": ("SKILL_NAME", all_skills),
    }
    default_dimension = "Skill Subcategory"
    default_selection = [
        "Artificial Intelligence and Machine Learning (AI/ML)",
        "Nursing and Patient Care",
        "Communication",
        "Web Design and Development",
        "Software Development",
    ]

    with st.sidebar:
        st.header("Controls")
        dim_label = st.selectbox(
            "Dimension",
            options=list(dimensions.keys()),
            index=list(dimensions.keys()).index(default_dimension),
            key="app3_skill_dist_dimension",
        )
        dim_col, dim_options = dimensions[dim_label]
        preselect = [value for value in default_selection if value in dim_options] if dim_label == default_dimension else []
        selected = st.multiselect(
            f"Select {dim_label.lower()}(s)",
            options=dim_options,
            default=preselect,
            placeholder="Choose one or more...",
            key="app3_skill_dist_selected",
        )
        metric = st.radio(
            "Metric",
            ["Relative growth (%)", "Absolute Δ CNT"],
            horizontal=True,
            key="app3_skill_dist_metric",
        )

    st.subheader("CNT Change Over Time")
    if selected:
        st.plotly_chart(
            line_chart(agg, dim_col, selected, metric),
            use_container_width=True,
        )
    else:
        st.info("Select at least one item above.")

    st.divider()
    st.subheader("Top N Fastest-Growing")

    with st.sidebar:
        slope_dim_label = st.selectbox(
            "Slope dimension",
            options=["Skill Subcategory", "Skill"],
            key="app3_slope_dim",
        )
        top_n = st.slider(
            "Top N",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            key="app3_top_n",
        )

    slope_dim_col = dimensions[slope_dim_label][0]
    slopes = compute_slopes(agg, slope_dim_col)
    col_top, col_flop = st.columns(2)
    with col_top:
        st.plotly_chart(
            slope_bar_chart(slopes, f"Top {top_n} {slope_dim_label.lower()}s", top_n),
            use_container_width=True,
        )
    with col_flop:
        st.plotly_chart(
            slope_bar_chart(slopes, f"Flop {top_n} {slope_dim_label.lower()}s", top_n, flop=True),
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Skill Analytics",
        page_icon=":material/show_chart:",
        layout="wide",
    )
    st.title("Skill Analytics")

    with st.sidebar:
        view = st.radio(
            "View",
            options=list(VIEW_LABELS),
            format_func=lambda key: VIEW_LABELS[key],
        )

    if view == "occupation_similarity":
        render_occupation_similarity_view()
    elif view == "ai_ml_cooccurrence":
        render_ai_ml_cooccurrence_view()
    else:
        render_skill_distributions_view()


if __name__ == "__main__":
    main()
