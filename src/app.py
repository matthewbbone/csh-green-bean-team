from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DIVERGENCE_PATH = BASE_DIR / "outputs" / "distribution_divergence.csv"
SKILL_DISTRIBUTIONS_PATH = BASE_DIR / "outputs" / "skill_distributions2.csv"
FELTEN_EXPOSURE_PATH = BASE_DIR / "outputs" / "felten_language_modeling_aioe.csv"
AI_SKILL_SOURCE_PATH = SKILL_DISTRIBUTIONS_PATH
DATASCIENTIST_COOCCUR_PATH = BASE_DIR / "outputs" / "datascientist_cooccur.csv"
AI_ML_SUBCATEGORY_NAME = "Artificial Intelligence and Machine Learning (AI/ML)"

VIEW_LABELS = {
    "divergence": "Occupation Divergence",
    "ai_ml_cooccurrence": "AI/ML Co-occurrence",
    "occupation_similarity": "Occupation Similarity",
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


@st.cache_data(show_spinner="Loading occupation skill distributions...")
def load_skill_distribution_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "YEAR",
            "QUARTER",
            "ONET_2019",
            "ONET_2019_NAME",
            "SKILL_ID",
            "SKILL_NAME",
            "CNT",
        ],
    )
    df["YEAR"] = df["YEAR"].astype(int)
    df["QUARTER"] = df["QUARTER"].astype(int)
    df["CNT"] = pd.to_numeric(df["CNT"], errors="coerce").fillna(0.0)
    df["SKILL_KEY"] = df["SKILL_ID"].fillna(df["SKILL_NAME"])
    df["SOC_CODE"] = df["ONET_2019"].str.replace(r"\.\d+$", "", regex=True)
    soc_labels = (
        df[["SOC_CODE", "ONET_2019_NAME"]]
        .dropna()
        .sort_values(["SOC_CODE", "ONET_2019_NAME"])
        .drop_duplicates(subset=["SOC_CODE"])
        .rename(columns={"ONET_2019_NAME": "SOC_OCCUPATION_NAME"})
    )
    df = df.merge(soc_labels, on="SOC_CODE", how="left")
    df["SOC_LABEL"] = df["SOC_OCCUPATION_NAME"] + " (" + df["SOC_CODE"] + ")"
    df["QUARTER_LABEL"] = "Q" + df["QUARTER"].astype(str) + " " + df["YEAR"].astype(str)
    df["OCCUPATION_LABEL"] = df["ONET_2019_NAME"] + " (" + df["ONET_2019"] + ")"
    return df.sort_values(["YEAR", "QUARTER", "OCCUPATION_LABEL", "SKILL_KEY"]).reset_index(drop=True)


@st.cache_data(show_spinner="Loading Felten exposure data...")
def load_felten_exposure_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["SOC Code", "Occupation Title", "Language Modeling AIOE"],
    ).rename(
        columns={
            "SOC Code": "SOC_CODE",
            "Occupation Title": "FELTEN_OCCUPATION_TITLE",
            "Language Modeling AIOE": "LM_AIOE",
        }
    )
    df["SOC_CODE"] = df["SOC_CODE"].astype(str)
    df["SOC_LABEL"] = df["FELTEN_OCCUPATION_TITLE"] + " (" + df["SOC_CODE"] + ")"
    df["LM_AIOE"] = pd.to_numeric(df["LM_AIOE"], errors="coerce")
    return df.dropna(subset=["SOC_CODE", "LM_AIOE"]).sort_values("SOC_CODE").reset_index(drop=True)


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


def build_quarter_skill_data(df: pd.DataFrame, year: int, quarter: int) -> pd.DataFrame:
    quarter_df = df[(df["YEAR"] == year) & (df["QUARTER"] == quarter)].copy()
    if quarter_df.empty:
        return quarter_df

    return (
        quarter_df.groupby(
            [
                "YEAR",
                "QUARTER",
                "QUARTER_LABEL",
                "SOC_CODE",
                "SOC_OCCUPATION_NAME",
                "SOC_LABEL",
                "SKILL_KEY",
                "SKILL_NAME",
            ],
            as_index=False,
        )["CNT"]
        .sum()
        .sort_values(["SOC_LABEL", "SKILL_KEY"])
        .reset_index(drop=True)
    )


def build_similar_occupation_table(
    quarter_skill_data: pd.DataFrame,
    selected_occupation_label: str,
    top_n: int | None = 3,
) -> pd.DataFrame:
    occupation_cols = ["SOC_CODE", "SOC_OCCUPATION_NAME", "SOC_LABEL"]
    selected = quarter_skill_data[quarter_skill_data["SOC_LABEL"] == selected_occupation_label][
        occupation_cols + ["SKILL_KEY", "SKILL_NAME", "CNT"]
    ].copy()
    if selected.empty:
        return pd.DataFrame()

    selected_norm = float((selected["CNT"] ** 2).sum() ** 0.5)
    if selected_norm == 0.0:
        return pd.DataFrame()

    others = quarter_skill_data[quarter_skill_data["SOC_LABEL"] != selected_occupation_label][
        occupation_cols + ["SKILL_KEY", "CNT"]
    ].copy()
    if others.empty:
        return pd.DataFrame()

    selected_counts = selected[["SKILL_KEY", "CNT"]].rename(columns={"CNT": "SELECTED_CNT"})
    shared = others.merge(selected_counts, on="SKILL_KEY", how="inner")
    if shared.empty:
        return pd.DataFrame()

    similarity = (
        shared.assign(DOT_COMPONENT=shared["CNT"] * shared["SELECTED_CNT"])
        .groupby(occupation_cols, as_index=False)
        .agg(
            DOT_PRODUCT=("DOT_COMPONENT", "sum"),
            SHARED_SKILL_COUNT=("SKILL_KEY", "nunique"),
        )
    )
    norms = (
        others.groupby(occupation_cols, as_index=False)
        .agg(
            CANDIDATE_NORM=("CNT", lambda series: float((series**2).sum() ** 0.5)),
            CANDIDATE_TOTAL_COUNT=("CNT", "sum"),
            CANDIDATE_ACTIVE_SKILLS=("SKILL_KEY", "nunique"),
        )
    )
    similarity = similarity.merge(norms, on=occupation_cols, how="inner")
    similarity["COSINE_SIMILARITY"] = similarity["DOT_PRODUCT"] / (
        similarity["CANDIDATE_NORM"] * selected_norm
    )
    ranked = (
        similarity[similarity["COSINE_SIMILARITY"] > 0]
        .sort_values(["COSINE_SIMILARITY", "SHARED_SKILL_COUNT", "SOC_LABEL"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return ranked.head(top_n).reset_index(drop=True) if top_n is not None else ranked


def build_skill_gap_table(
    quarter_skill_data: pd.DataFrame,
    selected_occupation_label: str,
    comparison_occupation_label: str,
    top_n: int = 10,
) -> pd.DataFrame:
    selected = quarter_skill_data[quarter_skill_data["SOC_LABEL"] == selected_occupation_label][
        ["SKILL_KEY", "SKILL_NAME", "CNT"]
    ].copy()
    comparison = quarter_skill_data[quarter_skill_data["SOC_LABEL"] == comparison_occupation_label][
        ["SKILL_KEY", "SKILL_NAME", "CNT"]
    ].copy()
    if selected.empty or comparison.empty:
        return pd.DataFrame()

    selected_total = float(selected["CNT"].sum())
    comparison_total = float(comparison["CNT"].sum())

    merged = selected.merge(
        comparison,
        on="SKILL_KEY",
        how="outer",
        suffixes=("_SELECTED", "_COMPARISON"),
    )
    merged["SKILL_NAME"] = merged["SKILL_NAME_SELECTED"].fillna(merged["SKILL_NAME_COMPARISON"])
    merged["CNT_SELECTED"] = merged["CNT_SELECTED"].fillna(0.0)
    merged["CNT_COMPARISON"] = merged["CNT_COMPARISON"].fillna(0.0)
    merged["SELECTED_SHARE"] = merged["CNT_SELECTED"] / selected_total if selected_total else 0.0
    merged["COMPARISON_SHARE"] = merged["CNT_COMPARISON"] / comparison_total if comparison_total else 0.0
    merged["SHARE_GAP"] = merged["COMPARISON_SHARE"] - merged["SELECTED_SHARE"]
    merged["ABS_SHARE_GAP"] = merged["SHARE_GAP"].abs()
    merged["GAP_DIRECTION"] = merged["SHARE_GAP"].map(
        lambda value: comparison_occupation_label if value > 0 else selected_occupation_label
    )
    return (
        merged[
            [
                "SKILL_NAME",
                "CNT_SELECTED",
                "CNT_COMPARISON",
                "SELECTED_SHARE",
                "COMPARISON_SHARE",
                "SHARE_GAP",
                "ABS_SHARE_GAP",
                "GAP_DIRECTION",
            ]
        ]
        .sort_values(["ABS_SHARE_GAP", "SKILL_NAME"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )


def build_similarity_exposure_scatter_data(
    quarter_skill_data: pd.DataFrame,
    selected_occupation_label: str,
    exposure_df: pd.DataFrame,
) -> pd.DataFrame:
    similarity = build_similar_occupation_table(
        quarter_skill_data=quarter_skill_data,
        selected_occupation_label=selected_occupation_label,
        top_n=None,
    )
    if similarity.empty:
        return similarity

    return (
        similarity.merge(exposure_df[["SOC_CODE", "FELTEN_OCCUPATION_TITLE", "LM_AIOE"]], on="SOC_CODE", how="inner")
        .sort_values(["COSINE_SIMILARITY", "LM_AIOE"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_similarity_exposure_chart(
    data: pd.DataFrame,
    selected_occupation_exposure: float | None = None,
) -> alt.Chart:
    point_selection = alt.selection_point(name="transition_select", fields=["SOC_LABEL"])
    points = (
        alt.Chart(data)
        .mark_circle(size=90)
        .encode(
            x=alt.X("COSINE_SIMILARITY:Q", title="Cosine similarity to selected occupation"),
            y=alt.Y("LM_AIOE:Q", title="Felten language-modeling AI exposure"),
            color=alt.condition(point_selection, alt.value("#d04a02"), alt.value("#1f77b4")),
            opacity=alt.condition(point_selection, alt.value(1.0), alt.value(0.55)),
            tooltip=[
                alt.Tooltip("SOC_LABEL:N", title="Transition occupation"),
                alt.Tooltip("COSINE_SIMILARITY:Q", title="Cosine similarity", format=".3f"),
                alt.Tooltip("LM_AIOE:Q", title="LM AIOE", format=".3f"),
                alt.Tooltip("SHARED_SKILL_COUNT:Q", title="Shared skills", format=","),
                alt.Tooltip("FELTEN_OCCUPATION_TITLE:N", title="Felten occupation"),
            ],
        )
        .add_params(point_selection)
    )
    if selected_occupation_exposure is None or pd.isna(selected_occupation_exposure):
        return points.properties(height=460)

    rule = alt.Chart(pd.DataFrame({"LM_AIOE": [selected_occupation_exposure]})).mark_rule(
        color="#888888",
        strokeDash=[6, 4],
    ).encode(y="LM_AIOE:Q")
    return (rule + points).properties(height=460)


def extract_selected_occupation_from_event(event: object, valid_labels: list[str]) -> str | None:
    if not event:
        return None

    valid = set(valid_labels)
    stack = [event]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            if current in valid:
                return current
            continue
        if isinstance(current, dict):
            stack.extend(current.values())
            continue
        if isinstance(current, list):
            stack.extend(current)
    return None


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
        st.altair_chart(build_time_series_chart(plotted.dropna(subset=[metric]), metric), width="stretch")

    left_col, right_col = st.columns([1.2, 1.0])

    with left_col:
        st.subheader("Quarter Spotlight")
        st.altair_chart(build_quarter_bar_chart(spotlight, metric), width="stretch")

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
            width="stretch",
        )

    st.subheader("Filtered Data")
    st.dataframe(
        filtered.sort_values(["OCCUPATION_LABEL", "YEAR", "QUARTER"])[
            [
                "OCCUPATION_LABEL",
                "QUARTER_LABEL",
                "KL_FROM_PREVIOUS",
                "KL_FROM_BASELINE",
                "TOTAL_SKILL_COUNT",
                "ACTIVE_SKILL_COUNT",
                "BASELINE_QUARTER_LABEL",
            ]
        ],
        hide_index=True,
        width="stretch",
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

    filtered = ai_ml_pairs[
        ai_ml_pairs["QUARTER_LABEL"].isin(quarter_options[selected_range[0] : selected_range[1] + 1])
    ]

    if filtered.empty:
        st.info("The current filters do not leave any AI/ML co-occurrence rows.")
        return

    summary = build_ai_ml_skill_summary(filtered)
    quarterly_ranks = build_ai_ml_quarterly_rank_data(filtered)
    max_top_n = int(len(summary))

    with st.sidebar:
        top_n = int(
            st.number_input(
                "Top N skills",
                min_value=1,
                max_value=max_top_n,
                value=min(20, max_top_n),
                step=1,
                help="Defines the ranked complementary-skill pool available for the line chart.",
            )
        )
        top_ranked_skills = summary["OTHER_SKILL"].head(top_n).tolist()
        selected_complementary_skills = st.multiselect(
            "Complementary skills",
            options=top_ranked_skills,
            default=top_ranked_skills,
            help="Choose which complementary skills from the current top-N ranking to plot.",
        )

    trend_data = quarterly_ranks[quarterly_ranks["OTHER_SKILL"].isin(selected_complementary_skills)]

    if trend_data.empty:
        st.info("Select at least one complementary skill to draw the rank trend.")
        return

    st.subheader("Quarterly Rank Trend")
    st.altair_chart(build_ai_ml_trend_chart(trend_data), width="stretch")

    st.caption(
        f"Plotting {len(selected_complementary_skills):,} selected complementary skills from the top {top_n:,} ranks in `{DATASCIENTIST_COOCCUR_PATH.name}`. "
        "Mirrored pairs such as `A,B` and `B,A` are collapsed before ranking, and ranks are dense ranks computed from the deduped co-occurrence counts."
    )


def render_occupation_similarity_view() -> None:
    st.subheader("Occupation Similarity")
    st.caption(
        "Builds quarter-specific SOC skill vectors from `skill_distributions2.csv`, places "
        "transition occupations on a similarity-versus-Felten-exposure scatter plot, and uses the "
        "selected point to show the biggest skill-share gaps."
    )

    missing_paths = [
        path.name
        for path in (SKILL_DISTRIBUTIONS_PATH, FELTEN_EXPOSURE_PATH)
        if not path.exists()
    ]
    if missing_paths:
        st.error(f"Missing required input file(s): {', '.join(missing_paths)}.")
        return

    skill_df = load_skill_distribution_data(SKILL_DISTRIBUTIONS_PATH)
    exposure_df = load_felten_exposure_data(FELTEN_EXPOSURE_PATH)
    if skill_df.empty:
        st.warning(f"No rows were found in `{SKILL_DISTRIBUTIONS_PATH.name}`.")
        return
    if exposure_df.empty:
        st.warning(f"No exposure rows were found in `{FELTEN_EXPOSURE_PATH.name}`.")
        return

    quarter_lookup = (
        skill_df[["YEAR", "QUARTER", "QUARTER_LABEL"]]
        .drop_duplicates()
        .sort_values(["YEAR", "QUARTER"])
        .reset_index(drop=True)
    )
    quarter_options = quarter_lookup["QUARTER_LABEL"].tolist()

    with st.sidebar:
        st.header("Controls")
        quarter_label = st.selectbox(
            "Quarter",
            options=quarter_options,
            index=len(quarter_options) - 1,
        )

    selected_quarter = quarter_lookup.loc[quarter_lookup["QUARTER_LABEL"] == quarter_label].iloc[0]
    quarter_skill_data = build_quarter_skill_data(
        skill_df,
        year=int(selected_quarter["YEAR"]),
        quarter=int(selected_quarter["QUARTER"]),
    )
    occupation_options = quarter_skill_data["SOC_LABEL"].drop_duplicates().sort_values().tolist()

    with st.sidebar:
        selected_occupation = st.selectbox(
            "Occupation",
            options=occupation_options,
            index=0,
        )

    scatter_data = build_similarity_exposure_scatter_data(
        quarter_skill_data=quarter_skill_data,
        selected_occupation_label=selected_occupation,
        exposure_df=exposure_df,
    )
    if scatter_data.empty:
        st.info("No occupations with both shared skills and Felten exposure scores were found for this selection.")
        return

    selected_soc_code = quarter_skill_data.loc[
        quarter_skill_data["SOC_LABEL"] == selected_occupation,
        "SOC_CODE",
    ].iloc[0]
    selected_exposure_rows = exposure_df[exposure_df["SOC_CODE"] == selected_soc_code]
    selected_exposure = (
        float(selected_exposure_rows["LM_AIOE"].iloc[0]) if not selected_exposure_rows.empty else None
    )

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Transition occupations in plot", f"{len(scatter_data):,}")
    metric_2.metric("Selected occupation LM AIOE", f"{selected_exposure:.3f}" if selected_exposure is not None else "Unavailable")
    metric_3.metric("Best similarity", f"{scatter_data['COSINE_SIMILARITY'].max():.3f}")

    st.subheader("Similarity vs AI Exposure")
    event = st.altair_chart(
        build_similarity_exposure_chart(scatter_data, selected_occupation_exposure=selected_exposure),
        width="stretch",
        on_select="rerun",
        selection_mode="transition_select",
        key="occupation_similarity_scatter",
    )

    selected_transition = extract_selected_occupation_from_event(
        event,
        valid_labels=scatter_data["SOC_LABEL"].tolist(),
    )
    if selected_transition is None:
        st.info("Click a point in the scatter plot to inspect that transition occupation's skill gaps.")
        return

    selected_transition_row = scatter_data.loc[
        scatter_data["SOC_LABEL"] == selected_transition
    ].iloc[0]
    gap_table = build_skill_gap_table(
        quarter_skill_data,
        selected_occupation_label=selected_occupation,
        comparison_occupation_label=selected_transition,
        top_n=10,
    )
    if gap_table.empty:
        st.info("No skill-gap rows were available for the selected transition occupation.")
        return

    st.subheader("Selected Transition Occupation")
    st.markdown(
        f"**{selected_transition}**  \n"
        f"Cosine similarity: {selected_transition_row['COSINE_SIMILARITY']:.3f} | "
        f"LM AIOE: {selected_transition_row['LM_AIOE']:.3f} | "
        f"Shared skills: {int(selected_transition_row['SHARED_SKILL_COUNT']):,}"
    )
    st.dataframe(
        gap_table.rename(
            columns={
                "SKILL_NAME": "Skill",
                "CNT_SELECTED": "Selected count",
                "CNT_COMPARISON": "Transition count",
                "SELECTED_SHARE": "Selected share",
                "COMPARISON_SHARE": "Transition share",
                "SHARE_GAP": "Share gap",
                "ABS_SHARE_GAP": "Absolute share gap",
                "GAP_DIRECTION": "More concentrated in",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.caption(
        f"Quarter: {quarter_label}. Selected occupation: {selected_occupation}. "
        f"Felten exposure matches are available for {scatter_data['SOC_CODE'].nunique():,} transition occupations in this quarter. "
        "Skill gaps are ranked by absolute difference in within-occupation skill share, not raw posting volume."
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
    elif view == "ai_ml_cooccurrence":
        render_ai_ml_cooccurrence_view()
    else:
        render_occupation_similarity_view()


if __name__ == "__main__":
    main()
