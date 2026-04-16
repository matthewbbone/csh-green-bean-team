from pathlib import Path

from snowflake.snowpark.functions import col, count, lit, quarter, year

from src.utils.snowpark_connection import SnowPark


DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2026-01-01"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "skill_distributions.csv"


def build_skill_distributions():
    session = SnowPark.get_shared_session()
    postings = session.table("BGI_POSTINGS_BACKUPS.APR_26.US_POSTINGS").alias("p")
    skills = session.table("BGI_POSTINGS_BACKUPS.APR_26.US_POSTINGS_SKILLS").alias("s")

    joined = (
        postings.join(skills, postings["ID"] == skills["ID"], how="left")
        .filter(col("POSTED") >= DEFAULT_START_DATE)
        .filter(col("POSTED") < DEFAULT_END_DATE)
        .select(
            quarter(postings["POSTED"]).alias("QUARTER"),
            year(postings["POSTED"]).alias("YEAR"),
            postings["ONET_2019"].alias("ONET_2019"),
            postings["ONET_2019_NAME"].alias("ONET_2019_NAME"),
            skills["SKILL_ID"].alias("SKILL_ID"),
            skills["SKILL_NAME"].alias("SKILL_NAME"),
            skills["SKILL_SUBCATEGORY"].alias("SKILL_SUBCATEGORY"),
            skills["SKILL_SUBCATEGORY_NAME"].alias("SKILL_SUBCATEGORY_NAME"),
            skills["SKILL_CATEGORY_NAME"].alias("SKILL_CATEGORY_NAME"),
            skills["SKILL_CATEGORY"].alias("SKILL_CATEGORY"),
        )
    )

    return (
        joined.group_by(
            "QUARTER",
            "YEAR",
            "ONET_2019",
            "ONET_2019_NAME",
            "SKILL_ID",
            "SKILL_NAME",
            "SKILL_SUBCATEGORY",
            "SKILL_SUBCATEGORY_NAME",
            "SKILL_CATEGORY",
        )
        .agg(count(lit(1)).alias("CNT"))
        .filter(col("CNT") > 100)
        .select(
            col("QUARTER"),
            col("YEAR"),
            col("ONET_2019"),
            col("ONET_2019_NAME"),
            col("SKILL_ID"),
            col("SKILL_NAME"),
            col("SKILL_SUBCATEGORY"),
            col("SKILL_CATEGORY_NAME"),
            col("SKILL_CATEGORY"),
            col("SKILL_CATEGORY_NAME").alias("SKILL_CATEGORY_NAME_2"),
            col("CNT"),
        )
    )


def save_skill_distributions(output_path: Path = OUTPUT_PATH) -> Path:
    skill_distributions = build_skill_distributions()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skill_distributions.to_pandas().to_csv(output_path, index=False)
    return output_path


def main() -> None:
    output_path = save_skill_distributions()
    print(f"Saved skill distributions to {output_path}")


if __name__ == "__main__":
    main()
