from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DIVERGENCE_PATH = BASE_DIR / "outputs" / "distribution_divergence.csv"
AI_SKILL_SOURCE_PATH = BASE_DIR / "outputs" / "skill_distributions2.csv"
DATASCIENTIST_COOCCUR_PATH = BASE_DIR / "outputs" / "datascientist_cooccur.csv"
AI_ML_SUBCATEGORY_NAME = "Artificial Intelligence and Machine Learning (AI/ML)"

VIEW_LABELS = {
    "divergence": "Occupation Divergence",
    "ai_ml_cooccurrence": "AI/ML Co-occurrence",
}
METRIC_LABELS = {
    "KL_FROM_PREVIOUS": "Quarter-on-quarter KL divergence",
    "KL_FROM_BASELINE": "KL divergence from baseline quarter",
}


@st.cache_data(show_spinner=False)
def load_divergence_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["YEAR"] = df["YEAR"].astype(int)
    df["QUARTER"] = df["QUARTER"].astype(int)
    df["TOTAL_SKILL_COUNT"] = df["TOTAL_SKILL_COUNT"].astype(int)
    df["ACTIVE_SKILL_COUNT"] = df["ACTIVE_SKILL_COUNT"].astype(int)
    df["PREVIOUS_GAP_QUARTERS"] = pd.to_numeric(df["PREVIOUS_GAP_QUARTERS"], errors="coerce")
    df["KL_FROM_PREVIOUS"] = pd.to_numeric(df["KL_FROM_PREVIOUS"], errors="coerce")
    df["KL_FROM_BASELINE"] = pd.to_numeric(df["KL_FROM_BASELINE"], errors="coerce")
    df["QUARTER_START"] = pd.PeriodIndex.from_fields(year=df["YEAR"], quarter=df["QUARTER"], freq="Q").start_time
    df["QUARTER_LABEL"] = "Q" + df["QUARTER"].astype(str) + " " + df["YEAR"].astype(str)
    df["OCCUPATION_LABEL"] = df["ONET_2019_NAME"] + " (" + df["ONET_2019"] + ")"
    return df.sort_values(["OCCUPATION_LABEL", "YEAR", "QUARTER"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ai_ml_skill_names(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=["SKILL_NAME", "SKILL_SUBCATEGORY_NAME"])
    return (
        df.loc[df["SKILL_SUBCATEGORY_NAME"].eq(AI_ML_SUBCATEGORY_NAME), "SKILL_NAME"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner="Loading AI/ML co-occurrence data...")
def load_ai_ml_cooccurrence_data(skill_source_path: Path, cooccur_path: Path) -> pd.DataFrame:
    ai_skills = set(load_ai_ml_skill_names(skill_source_path).tolist())
    if not ai_skills:
        return pd.DataFrame(
            columns=[
                "YEAR",
                "QUARTER",
                "QUARTER_LABEL",
                "QUARTER_START",
                "AI_SKILL",
                "OTHER_SKILL",
                "CNT",
            ]
        )

    frames: list[pd.DataFrame] = []
    chunk_reader = pd.read_csv(
        cooccur_path,
        usecols=["YEAR", "QUARTER", "SKILL_1", "SKILL_2", "CNT"],
        chunksize=250_000,
    )
    for chunk in chunk_reader:
        chunk = chunk.dropna(subset=["YEAR", "QUARTER", "SKILL_1", "SKILL_2", "CNT"]).copy()
        mask = chunk["SKILL_1"].isin(ai_skills) | chunk["SKILL_2"].isin(ai_skills)
        if not mask.any():
            continue

        filtered = chunk.loc[mask, ["YEAR", "QUARTER", "SKILL_1", "SKILL_2", "CNT"]].copy()
        filtered["YEAR"] = filtered["YEAR"].astype(int)
        filtered["QUARTER"] = filtered["QUARTER"].astype(int)
        filtered["CNT"] = pd.to_numeric(filtered["CNT"], errors="coerce")
        filtered = filtered.dropna(subset=["CNT"])
        filtered["CNT"] = filtered["CNT"].astype(int)

        left = filtered["SKILL_1"]
        right = filtered["SKILL_2"]
        filtered["PAIR_LEFT"] = left.where(left <= right, right)
        filtered["PAIR_RIGHT"] = right.where(left <= right, left)
        frames.append(filtered[["YEAR", "QUARTER", "PAIR_LEFT", "PAIR_RIGHT", "CNT"]])

    if not frames:
        return pd.DataFrame(
            columns=[
                "YEAR",
                "QUARTER",
                "QUARTER_LABEL",
                "QUARTER_START",
                "AI_SKILL",
                "OTHER_SKILL",
                "CNT",
            ]
        )

    deduped = (
        pd.concat(frames, ignore_index=True)
        .groupby(["YEAR", "QUARTER", "PAIR_LEFT", "PAIR_RIGHT"], as_index=False)["CNT"]
        .max()
    )

    pair_left_is_ai = deduped["PAIR_LEFT"].isin(ai_skills)
    pair_right_is_ai = deduped["PAIR_RIGHT"].isin(ai_skills)
    relevant = deduped.loc[pair_left_is_ai ^ pair_right_is_ai].copy()
    relevant["AI_SKILL"] = relevant["PAIR_LEFT"].where(relevant["PAIR_LEFT"].isin(ai_skills), relevant["PAIR_RIGHT"])
    relevant["OTHER_SKILL"] = relevant["PAIR_RIGHT"].where(relevant["PAIR_LEFT"].isin(ai_skills), relevant["PAIR_LEFT"])
    relevant["QUARTER_LABEL"] = "Q" + relevant["QUARTER"].astype(str) + " " + relevant["YEAR"].astype(str)
    relevant["QUARTER_START"] = pd.PeriodIndex.from_fields(
        year=relevant["YEAR"],
        quarter=relevant["QUARTER"],
        freq="Q",
    ).start_time

    return relevant.sort_values(["YEAR", "QUARTER", "OTHER_SKILL", "AI_SKILL"]).reset_index(drop=True)


def build_time_series_chart(data: pd.DataFrame, metric: str) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("QUARTER_START:T", title="Quarter"),
            y=alt.Y(f"{metric}:Q", title=METRIC_LABELS[metric]),
            color=alt.Color("OCCUPATION_LABEL:N", title="Occupation"),
            tooltip=[
                alt.Tooltip("OCCUPATION_LABEL:N", title="Occupation"),
                alt.Tooltip("QUARTER_LABEL:N", title="Quarter"),
                alt.Tooltip(f"{metric}:Q", title=METRIC_LABELS[metric], format=".4f"),
                alt.Tooltip("TOTAL_SKILL_COUNT:Q", title="Skill count", format=","),
                alt.Tooltip("ACTIVE_SKILL_COUNT:Q", title="Active skills", format=","),
            ],
        )
        .properties(height=420)
    )


def build_quarter_bar_chart(data: pd.DataFrame, metric: str) -> alt.Chart:
    ranked = data.sort_values(metric, ascending=False).head(20).sort_values(metric, ascending=True)
    return (
        alt.Chart(ranked)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric}:Q", title=METRIC_LABELS[metric]),
            y=alt.Y("OCCUPATION_LABEL:N", sort=None, title="Occupation"),
            color=alt.Color(f"{metric}:Q", title=METRIC_LABELS[metric]),
            tooltip=[
                alt.Tooltip("OCCUPATION_LABEL:N", title="Occupation"),
                alt.Tooltip("QUARTER_LABEL:N", title="Quarter"),
                alt.Tooltip(f"{metric}:Q", title=METRIC_LABELS[metric], format=".4f"),
                alt.Tooltip("TOTAL_SKILL_COUNT:Q", title="Skill count", format=","),
                alt.Tooltip("ACTIVE_SKILL_COUNT:Q", title="Active skills", format=","),
            ],
        )
        .properties(height=520)
    )


def build_top_occupations_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    summary = (
        df.dropna(subset=[metric])
        .groupby(["OCCUPATION_LABEL", "ONET_2019", "ONET_2019_NAME"], as_index=False)
        .agg(
            latest_quarter=("QUARTER_LABEL", "last"),
            latest_value=(metric, "last"),
            peak_value=(metric, "max"),
            mean_value=(metric, "mean"),
        )
        .sort_values("peak_value", ascending=False)
    )
    return summary


def build_ai_ml_skill_summary(df: pd.DataFrame) -> pd.DataFrame:
    quarterly_ranks = build_ai_ml_quarterly_rank_data(df)
    summary = (
        df.groupby("OTHER_SKILL", as_index=False)
        .agg(
            total_cnt=("CNT", "sum"),
            active_quarters=("QUARTER_LABEL", "nunique"),
            ai_skill_count=("AI_SKILL", "nunique"),
        )
        .sort_values(["total_cnt", "active_quarters", "OTHER_SKILL"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    summary["overall_rank"] = summary["total_cnt"].rank(method="dense", ascending=False).astype(int)
    best_quarter_rank = (
        quarterly_ranks.groupby("OTHER_SKILL", as_index=False)["quarter_rank"]
        .min()
        .rename(columns={"quarter_rank": "best_quarter_rank"})
    )
    return summary.merge(best_quarter_rank, on="OTHER_SKILL", how="left").sort_values(
        ["overall_rank", "OTHER_SKILL"]
    )


def build_ai_ml_quarterly_rank_data(df: pd.DataFrame) -> pd.DataFrame:
    quarterly = (
        df.groupby(["YEAR", "QUARTER", "QUARTER_LABEL", "QUARTER_START", "OTHER_SKILL"], as_index=False)
        .agg(
            quarterly_cnt=("CNT", "sum"),
            ai_skill_count=("AI_SKILL", "nunique"),
        )
        .sort_values(["YEAR", "QUARTER", "quarterly_cnt", "OTHER_SKILL"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )
    quarterly["quarter_rank"] = (
        quarterly.groupby(["YEAR", "QUARTER"])["quarterly_cnt"].rank(method="dense", ascending=False).astype(int)
    )
    return quarterly.sort_values(["YEAR", "QUARTER", "quarter_rank", "OTHER_SKILL"]).reset_index(drop=True)


def build_ai_ml_top_skills_chart(data: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_circle(size=140)
        .encode(
            x=alt.X(
                "overall_rank:Q",
                title="Overall co-occurrence rank (1 = most frequent)",
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y("OTHER_SKILL:N", sort=alt.SortField("overall_rank", order="ascending"), title="Skill"),
            color=alt.Color("overall_rank:Q", title="Overall rank"),
            tooltip=[
                alt.Tooltip("OTHER_SKILL:N", title="Skill"),
                alt.Tooltip("overall_rank:Q", title="Overall rank", format=".0f"),
                alt.Tooltip("active_quarters:Q", title="Active quarters", format=","),
                alt.Tooltip("ai_skill_count:Q", title="AI/ML skills paired", format=","),
                alt.Tooltip("best_quarter_rank:Q", title="Best quarter rank", format=".0f"),
            ],
        )
        .properties(height=520)
    )


def build_ai_ml_trend_chart(data: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("QUARTER_START:T", title="Quarter"),
            y=alt.Y(
                "quarter_rank:Q",
                title="Quarterly co-occurrence rank (1 = most frequent)",
                axis=alt.Axis(tickMinStep=1),
                scale=alt.Scale(reverse=True),
            ),
            color=alt.Color("OTHER_SKILL:N", title="Skill"),
            tooltip=[
                alt.Tooltip("OTHER_SKILL:N", title="Skill"),
                alt.Tooltip("QUARTER_LABEL:N", title="Quarter"),
                alt.Tooltip("quarter_rank:Q", title="Quarter rank", format=".0f"),
            ],
        )
        .properties(height=420)
    )


def render_divergence_view() -> None:
    st.subheader("Occupation Skill Divergence")
    st.caption(
        "Quarterly KL divergence for occupation-specific skill distributions derived from "
        f"`{DIVERGENCE_PATH.name}`."
    )

    if not DIVERGENCE_PATH.exists():
        st.error(
            f"Missing `{DIVERGENCE_PATH}`. Run `python -m src.distribution_divergence` first to generate the input data."
        )
        return

    df = load_divergence_data(DIVERGENCE_PATH)
    available_metric_rows = {metric: df[metric].notna().sum() for metric in METRIC_LABELS}

    with st.sidebar:
        st.header("Controls")
        metric = st.selectbox(
            "Metric",
            options=list(METRIC_LABELS),
            format_func=lambda key: METRIC_LABELS[key],
        )

        quarter_options = df["QUARTER_LABEL"].drop_duplicates().tolist()
        selected_range = st.select_slider(
            "Quarter range",
            options=list(range(len(quarter_options))),
            value=(0, len(quarter_options) - 1),
            format_func=lambda idx: quarter_options[idx],
        )
        start_label = quarter_options[selected_range[0]]
        end_label = quarter_options[selected_range[1]]

        ranked_occupations = build_top_occupations_summary(df, metric)
        default_occupations = ranked_occupations["OCCUPATION_LABEL"].head(5).tolist()
        selected_occupations = st.multiselect(
            "Occupations",
            options=ranked_occupations["OCCUPATION_LABEL"].tolist(),
            default=default_occupations,
            help="Pick one or more occupations to plot over time.",
        )

        spotlight_quarter = st.selectbox(
            "Quarter spotlight",
            options=quarter_options,
            index=len(quarter_options) - 1,
            help="Shows the largest KL divergence values observed in a single quarter.",
        )

    filtered = df[df["QUARTER_LABEL"].isin(quarter_options[selected_range[0] : selected_range[1] + 1])]
    plotted = filtered[filtered["OCCUPATION_LABEL"].isin(selected_occupations)] if selected_occupations else filtered.iloc[0:0]
    spotlight = filtered[filtered["QUARTER_LABEL"] == spotlight_quarter].dropna(subset=[metric])

    total_occupations = df["OCCUPATION_LABEL"].nunique()
    total_rows = int(df[metric].notna().sum())
    peak_row = df.dropna(subset=[metric]).sort_values(metric, ascending=False).iloc[0]

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Occupations", f"{total_occupations:,}")
    metric_2.metric("Metric rows", f"{total_rows:,}")
    metric_3.metric("Quarter range", f"{start_label} to {end_label}")
    metric_4.metric(
        "Peak divergence",
        f"{peak_row[metric]:.3f}",
        delta=f"{peak_row['ONET_2019_NAME']} in {peak_row['QUARTER_LABEL']}",
    )

    st.subheader("Time Series")
    if plotted.empty:
        st.info("Select at least one occupation to draw the time series.")
    else:
        st.altair_chart(build_time_series_chart(plotted.dropna(subset=[metric]), metric), use_container_width=True)

    left_col, right_col = st.columns([1.2, 1.0])

    with left_col:
        st.subheader("Quarter Spotlight")
        st.altair_chart(build_quarter_bar_chart(spotlight, metric), use_container_width=True)

    with right_col:
        st.subheader("Top Occupations")
        summary = build_top_occupations_summary(filtered, metric).rename(
            columns={
                "latest_quarter": "Latest quarter",
                "latest_value": "Latest value",
                "peak_value": "Peak value",
                "mean_value": "Mean value",
            }
        )
        st.dataframe(
            summary[["OCCUPATION_LABEL", "Latest quarter", "Latest value", "Peak value", "Mean value"]].head(25),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Filtered Data")
    st.dataframe(
        filtered[
            [
                "OCCUPATION_LABEL",
                "QUARTER_LABEL",
                "KL_FROM_PREVIOUS",
                "KL_FROM_BASELINE",
                "TOTAL_SKILL_COUNT",
                "ACTIVE_SKILL_COUNT",
                "BASELINE_QUARTER_LABEL",
            ]
        ].sort_values(["OCCUPATION_LABEL", "YEAR", "QUARTER"]),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        f"Loaded {len(df):,} rows from `{DIVERGENCE_PATH.name}`. "
        f"{METRIC_LABELS['KL_FROM_PREVIOUS']} rows: {available_metric_rows['KL_FROM_PREVIOUS']:,}. "
        f"{METRIC_LABELS['KL_FROM_BASELINE']} rows: {available_metric_rows['KL_FROM_BASELINE']:,}."
    )


def render_ai_ml_cooccurrence_view() -> None:
    st.subheader("Data Scientist AI/ML Co-occurrence")
    st.caption(
        "Uses the unique `SKILL_NAME` values from `skill_distributions2.csv` where "
        f"`SKILL_SUBCATEGORY_NAME` is `{AI_ML_SUBCATEGORY_NAME}` to identify AI/ML skills in "
        "`datascientist_cooccur.csv`, then ranks the non-AI skills that pair with them most often "
        "across the selected period."
    )

    missing_paths = [
        path.name
        for path in (AI_SKILL_SOURCE_PATH, DATASCIENTIST_COOCCUR_PATH)
        if not path.exists()
    ]
    if missing_paths:
        st.error(f"Missing required input file(s): {', '.join(missing_paths)}.")
        return

    ai_skill_names = load_ai_ml_skill_names(AI_SKILL_SOURCE_PATH)
    ai_ml_pairs = load_ai_ml_cooccurrence_data(AI_SKILL_SOURCE_PATH, DATASCIENTIST_COOCCUR_PATH)

    if ai_skill_names.empty:
        st.warning(f"No AI/ML skills were found in `{AI_SKILL_SOURCE_PATH.name}`.")
        return
    if ai_ml_pairs.empty:
        st.warning(f"No AI/ML skill matches were found in `{DATASCIENTIST_COOCCUR_PATH.name}`.")
        return

    with st.sidebar:
        st.header("Controls")
        quarter_options = ai_ml_pairs["QUARTER_LABEL"].drop_duplicates().tolist()
        selected_range = st.select_slider(
            "Quarter range",
            options=list(range(len(quarter_options))),
            value=(0, len(quarter_options) - 1),
            format_func=lambda idx: quarter_options[idx],
        )
        selected_ai_skills = st.multiselect(
            "AI/ML skills",
            options=sorted(ai_ml_pairs["AI_SKILL"].unique().tolist()),
            default=[],
            help="Leave empty to include all AI/ML skills found in the co-occurrence file.",
        )
        top_n = st.slider("Top skills", min_value=10, max_value=50, value=20, step=5)

    filtered = ai_ml_pairs[
        ai_ml_pairs["QUARTER_LABEL"].isin(quarter_options[selected_range[0] : selected_range[1] + 1])
    ]
    if selected_ai_skills:
        filtered = filtered[filtered["AI_SKILL"].isin(selected_ai_skills)]

    if filtered.empty:
        st.info("The current filters do not leave any AI/ML co-occurrence rows.")
        return

    summary = build_ai_ml_skill_summary(filtered)
    quarterly_ranks = build_ai_ml_quarterly_rank_data(filtered)
    trend_skills = summary["OTHER_SKILL"].head(min(top_n, len(summary))).tolist()
    trend_data = quarterly_ranks[quarterly_ranks["OTHER_SKILL"].isin(trend_skills)]

    st.subheader("Quarterly Rank Trend")
    st.altair_chart(build_ai_ml_trend_chart(trend_data), use_container_width=True)

    st.caption(
        f"Plotting the top {len(trend_skills):,} ranked skills from `{DATASCIENTIST_COOCCUR_PATH.name}`. "
        "Mirrored pairs such as `A,B` and `B,A` are collapsed before ranking, and ranks are dense ranks computed from the deduped co-occurrence counts."
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

    if view == "divergence":
        render_divergence_view()
    else:
        render_ai_ml_cooccurrence_view()


if __name__ == "__main__":
    main()
