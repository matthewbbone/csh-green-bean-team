import argparse
import re
from pathlib import Path

from snowflake.snowpark.functions import col, count, quarter, year

from src.utils.snowpark_connection import SnowPark


DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2026-01-01"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def _slugify_onet_code(onet_code: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", onet_code).strip("_")


def build_skill_cooccurrence(onet_code: str):
    session = SnowPark.get_shared_session()
    postings = session.table("BGI_POSTINGS_BACKUPS.APR_26.US_POSTINGS").alias("p")
    skills = session.table("BGI_POSTINGS_BACKUPS.APR_26.US_POSTINGS_SKILLS").alias("s")

    filtered_skills = (
        skills.join(postings, skills["ID"] == postings["ID"], how="inner")
        .filter(col("POSTED") >= DEFAULT_START_DATE)
        .filter(col("POSTED") < DEFAULT_END_DATE)
        .filter(postings["ONET_2019"] == onet_code)
        .select(
            skills["SKILL_NAME"].alias("SKILL_NAME"),
            skills["ID"].alias("ID"),
            postings["POSTED"].alias("POSTED"),
        )
    )

    skill_1 = filtered_skills.alias("s1")
    skill_2 = filtered_skills.alias("s2")

    return (
        skill_1.join(skill_2, skill_1["ID"] == skill_2["ID"], how="inner")
        .group_by(
            quarter(skill_1["POSTED"]),
            year(skill_1["POSTED"]),
            skill_1["SKILL_NAME"],
            skill_2["SKILL_NAME"],
        )
        .agg(count(skill_1["ID"]).alias("CNT"))
        .filter(col("CNT") > 10)
        .select(
            quarter(skill_1["POSTED"]).alias("QUARTER"),
            year(skill_1["POSTED"]).alias("YEAR"),
            skill_1["SKILL_NAME"].alias("SKILL_1"),
            skill_2["SKILL_NAME"].alias("SKILL_2"),
            col("CNT"),
        )
    )


def save_skill_cooccurrence(onet_code: str, output_path: Path | None = None) -> Path:
    output_path = output_path or OUTPUT_DIR / f"skill_cooccur_{_slugify_onet_code(onet_code)}.csv"
    skill_cooccurrence = build_skill_cooccurrence(onet_code)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skill_cooccurrence.to_pandas().to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export skill co-occurrence counts for an O*NET code.")
    parser.add_argument("onet_code", help="O*NET 2019 code, for example 15-2051.00", default="15-2051.00")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV path. Defaults to skill_cooccur_<onet_code>.csv at the repo root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = save_skill_cooccurrence(args.onet_code, args.output)
    print(f"Saved skill co-occurrence to {output_path}")


if __name__ == "__main__":
    main()
