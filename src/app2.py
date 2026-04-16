from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Skill Distributions", layout="wide")
st.title("Skill Distributions Over Time")

BASE_DIR = Path(__file__).resolve().parents[1]
SKILL_DISTRIBUTIONS_PATH = BASE_DIR / "outputs" / "skill_distributions2.csv"


@st.cache_data(show_spinner="Loading data…")
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
    agg.sort_values("time_ord", inplace=True)
    return agg


# ── Configurable AI event markers ────────────────────────────────────────────
# Add, remove, or edit entries here. "period" must match the YYYY-QN format.
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


def add_event_markers(fig: go.Figure) -> None:
    # Group same-quarter events so labels stack instead of overlap
    from collections import defaultdict

    by_period: dict = defaultdict(list)
    for e in AI_EVENTS:
        by_period[e["period"]].append(e["label"])

    periods_sorted = sorted(by_period)

    for period in periods_sorted:
        labels = by_period[period]
        text = "<br>".join(labels)
        y = 0.80
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
            y=y,
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=12, color="rgba(255,255,255,0.5)"),
        )


def delta_from_root(series_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with CNT replaced by change vs. the earliest available period."""
    baseline = pd.to_numeric(
        series_df.loc[series_df["time_ord"].idxmin(), "CNT"], errors="coerce"
    )
    out = series_df.copy()
    out["delta"] = pd.to_numeric(out["CNT"], errors="coerce") - baseline
    return out


def line_chart(
    agg: pd.DataFrame, dim_col: str, selected: list[str], metric: str
) -> go.Figure:
    fig = go.Figure()
    for name in selected:
        sub = agg[agg[dim_col] == name]
        sub = (
            sub.groupby(["time_ord", "period"], observed=True)["CNT"]
            .sum()
            .reset_index()
            .sort_values("time_ord")
        )
        if metric == "Relative growth (%)":
            baseline = sub["CNT"].iloc[0] if not sub.empty else None
            y_vals = (sub["CNT"] / baseline - 1) * 100 if baseline else sub["CNT"] * 0
            hover = (
                "<b>%{fullData.name}</b><br>%{x}<br>Growth: %{y:+.1f}%<extra></extra>"
            )
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


@st.cache_data(show_spinner="Computing slopes…")
def compute_slopes(_agg: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = _agg.groupby([group_col, "time_ord"], observed=True)["CNT"].sum().reset_index()
    g["xy"] = g["time_ord"] * g["CNT"]
    g["xx"] = g["time_ord"] ** 2
    s = g.groupby(group_col, observed=True).agg(
        n=("time_ord", "count"),
        sum_x=("time_ord", "sum"),
        sum_y=("CNT", "sum"),
        sum_xx=("xx", "sum"),
        sum_xy=("xy", "sum"),
    )
    denom = s["n"] * s["sum_xx"] - s["sum_x"] ** 2
    s["slope"] = (s["n"] * s["sum_xy"] - s["sum_x"] * s["sum_y"]) / denom
    return s["slope"].reset_index().rename(columns={group_col: "name"})


def slope_bar_chart(
    slopes: pd.DataFrame, title: str, top_n: int = 50, flop: bool = False
) -> go.Figure:
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


if not SKILL_DISTRIBUTIONS_PATH.exists():
    st.error(
        f"Missing `{SKILL_DISTRIBUTIONS_PATH}`. Generate the file before launching this dashboard."
    )
    st.stop()

agg = load_agg(SKILL_DISTRIBUTIONS_PATH)
all_skills = sorted(agg["SKILL_NAME"].unique())
all_categories = sorted(agg["SKILL_CATEGORY_NAME"].unique())
all_subcategories = sorted(agg["SKILL_SUBCATEGORY_NAME"].unique())

# ── CNT change over time ──────────────────────────────────────────
st.header("CNT change over time")

DIMENSIONS = {
    "Skill Category": ("SKILL_CATEGORY_NAME", all_categories),
    "Skill Subcategory": ("SKILL_SUBCATEGORY_NAME", all_subcategories),
    "Skill": ("SKILL_NAME", all_skills),
}

DEFAULT_DIMENSION = "Skill Subcategory"
DEFAULT_SELECTION = [
    "Artificial Intelligence and Machine Learning (AI/ML)",
    "Nursing and Patient Care",
    "Communication",
    "Web Design and Development",
    "Software Development",
]

dim_label = st.selectbox(
    "Dimension",
    options=list(DIMENSIONS.keys()),
    index=list(DIMENSIONS.keys()).index(DEFAULT_DIMENSION),
)
dim_col, dim_options = DIMENSIONS[dim_label]

preselect = (
    [v for v in DEFAULT_SELECTION if v in dim_options]
    if dim_label == DEFAULT_DIMENSION
    else []
)
selected = st.multiselect(
    f"Select {dim_label.lower()}(s)",
    options=dim_options,
    default=preselect,
    placeholder="Choose one or more…",
)

metric = st.radio(
    "Metric",
    ["Relative growth (%)", "Absolute Δ CNT"],
    horizontal=True,
)

if selected:
    st.plotly_chart(
        line_chart(agg, dim_col, selected, metric), use_container_width=True
    )
else:
    st.info("Select at least one item above.")

st.divider()

# ── Top N by slope ───────────────────────────────────────────────
st.header("Top N fastest-growing (by OLS slope)")
slope_dim_label = st.selectbox(
    "Dimension",
    options=["Skill Subcategory", "Skill"],
    key="slope_dim",
)
slope_dim_col = DIMENSIONS[slope_dim_label][0]
top_n = st.slider("Top N", min_value=5, max_value=100, value=50, step=5)

slopes = compute_slopes(agg, slope_dim_col)
col_top, col_flop = st.columns(2)
with col_top:
    st.plotly_chart(
        slope_bar_chart(slopes, f"Top {top_n} {slope_dim_label.lower()}s", top_n),
        use_container_width=True,
    )
with col_flop:
    st.plotly_chart(
        slope_bar_chart(
            slopes, f"Flop {top_n} {slope_dim_label.lower()}s", top_n, flop=True
        ),
        use_container_width=True,
    )
