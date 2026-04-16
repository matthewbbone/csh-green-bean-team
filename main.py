import argparse
from pathlib import Path

from src.skill_cooccur import _slugify_onet_code, save_skill_cooccurrence
from src.skill_distributions import save_skill_distributions
from src.utils.snowpark_connection import SnowPark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export skill distributions and co-occurrence CSVs.")
    parser.add_argument("onet_code", help="O*NET 2019 code for the co-occurrence export, for example 15-2051.00")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where the CSV files will be written. Defaults to the repo root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    distributions_path = output_dir / "skill_distributions.csv"
    cooccur_path = output_dir / f"skill_cooccur_{_slugify_onet_code(args.onet_code)}.csv"

    try:
        saved_distributions_path = save_skill_distributions(distributions_path)
        saved_cooccur_path = save_skill_cooccurrence(args.onet_code, cooccur_path)
    finally:
        SnowPark.close_shared_session()

    print(f"Saved skill distributions to {saved_distributions_path}")
    print(f"Saved skill co-occurrence to {saved_cooccur_path}")


if __name__ == "__main__":
    main()
