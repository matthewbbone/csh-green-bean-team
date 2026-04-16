import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
DEFAULT_INPUT_PATH = OUTPUT_DIR / "skill_distributions.csv"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "distribution_divergence.csv"
DEFAULT_EPSILON = 1e-9


def quarter_label(year: int, quarter: int) -> str:
    return f"{year}Q{quarter}"


def quarter_index(year: int, quarter: int) -> int:
    return year * 4 + quarter - 1


def quarter_gap(previous_quarter: tuple[int, int], current_quarter: tuple[int, int]) -> int:
    previous_year, previous_q = previous_quarter
    current_year, current_q = current_quarter
    return quarter_index(current_year, current_q) - quarter_index(previous_year, previous_q)


def kl_divergence(
    current_counts: dict[str, float],
    reference_counts: dict[str, float],
    epsilon: float,
) -> float:
    skill_ids = set(current_counts) | set(reference_counts)
    if not skill_ids:
        return 0.0

    current_total = sum(current_counts.values()) + epsilon * len(skill_ids)
    reference_total = sum(reference_counts.values()) + epsilon * len(skill_ids)

    divergence = 0.0
    for skill_id in skill_ids:
        current_probability = (current_counts.get(skill_id, 0.0) + epsilon) / current_total
        reference_probability = (reference_counts.get(skill_id, 0.0) + epsilon) / reference_total
        divergence += current_probability * math.log(current_probability / reference_probability)
    return divergence


def load_skill_distributions(
    input_path: Path,
) -> tuple[
    dict[tuple[str, str], dict[tuple[int, int], dict[str, float]]],
    set[tuple[int, int]],
]:
    occupation_quarter_skill_counts: dict[tuple[str, str], dict[tuple[int, int], dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    all_quarters: set[tuple[int, int]] = set()

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            occupation_key = (row["ONET_2019"], row["ONET_2019_NAME"])
            quarter_key = (int(row["YEAR"]), int(row["QUARTER"]))
            skill_key = row["SKILL_ID"] or row["SKILL_NAME"]
            count = float(row["CNT"])

            occupation_quarter_skill_counts[occupation_key][quarter_key][skill_key] += count
            all_quarters.add(quarter_key)

    return occupation_quarter_skill_counts, all_quarters


def compute_distribution_divergence(
    input_path: Path = DEFAULT_INPUT_PATH,
    baseline_quarter: tuple[int, int] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> list[dict[str, object]]:
    occupation_quarter_skill_counts, all_quarters = load_skill_distributions(input_path)
    if baseline_quarter is not None and baseline_quarter not in all_quarters:
        available_quarters = sorted(all_quarters)
        first_available = available_quarters[0]
        last_available = available_quarters[-1]
        requested_label = quarter_label(*baseline_quarter)
        raise ValueError(
            f"Requested baseline {requested_label} is not present in {input_path}. "
            f"Available quarters run from {quarter_label(*first_available)} to {quarter_label(*last_available)}."
        )

    rows: list[dict[str, object]] = []
    for (onet_code, onet_name), quarter_skill_counts in sorted(occupation_quarter_skill_counts.items()):
        ordered_quarters = sorted(quarter_skill_counts)
        baseline_for_occupation = baseline_quarter or ordered_quarters[0]
        baseline_counts = quarter_skill_counts.get(baseline_for_occupation)

        previous_quarter: tuple[int, int] | None = None
        previous_counts: dict[str, float] | None = None

        for current_quarter in ordered_quarters:
            current_counts = quarter_skill_counts[current_quarter]
            current_year, current_q = current_quarter
            total_skill_count = sum(current_counts.values())
            active_skill_count = len(current_counts)

            kl_from_previous = None
            previous_gap_quarters = None
            previous_label_value = None
            previous_year_value = None
            previous_quarter_value = None
            if previous_counts is not None and previous_quarter is not None:
                kl_from_previous = kl_divergence(current_counts, previous_counts, epsilon)
                previous_gap_quarters = quarter_gap(previous_quarter, current_quarter)
                previous_year_value, previous_quarter_value = previous_quarter
                previous_label_value = quarter_label(previous_year_value, previous_quarter_value)

            baseline_year, baseline_q = baseline_for_occupation
            baseline_label_value = quarter_label(baseline_year, baseline_q)
            kl_from_baseline = None
            if baseline_counts is not None:
                kl_from_baseline = kl_divergence(current_counts, baseline_counts, epsilon)

            rows.append(
                {
                    "ONET_2019": onet_code,
                    "ONET_2019_NAME": onet_name,
                    "YEAR": current_year,
                    "QUARTER": current_q,
                    "QUARTER_LABEL": quarter_label(current_year, current_q),
                    "TOTAL_SKILL_COUNT": int(total_skill_count),
                    "ACTIVE_SKILL_COUNT": active_skill_count,
                    "PREVIOUS_YEAR": previous_year_value,
                    "PREVIOUS_QUARTER": previous_quarter_value,
                    "PREVIOUS_QUARTER_LABEL": previous_label_value,
                    "PREVIOUS_GAP_QUARTERS": previous_gap_quarters,
                    "KL_FROM_PREVIOUS": kl_from_previous,
                    "BASELINE_YEAR": baseline_year,
                    "BASELINE_QUARTER": baseline_q,
                    "BASELINE_QUARTER_LABEL": baseline_label_value,
                    "BASELINE_SOURCE": "requested" if baseline_quarter is not None else "first_available_for_occupation",
                    "KL_FROM_BASELINE": kl_from_baseline,
                }
            )

            previous_quarter = current_quarter
            previous_counts = current_counts

    return rows


def save_distribution_divergence(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    baseline_quarter: tuple[int, int] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> Path:
    rows = compute_distribution_divergence(
        input_path=input_path,
        baseline_quarter=baseline_quarter,
        epsilon=epsilon,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ONET_2019",
        "ONET_2019_NAME",
        "YEAR",
        "QUARTER",
        "QUARTER_LABEL",
        "TOTAL_SKILL_COUNT",
        "ACTIVE_SKILL_COUNT",
        "PREVIOUS_YEAR",
        "PREVIOUS_QUARTER",
        "PREVIOUS_QUARTER_LABEL",
        "PREVIOUS_GAP_QUARTERS",
        "KL_FROM_PREVIOUS",
        "BASELINE_YEAR",
        "BASELINE_QUARTER",
        "BASELINE_QUARTER_LABEL",
        "BASELINE_SOURCE",
        "KL_FROM_BASELINE",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure occupation-quarter skill distribution change using KL divergence."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--baseline-year",
        type=int,
        help="Optional baseline year used for KL(current || baseline).",
    )
    parser.add_argument(
        "--baseline-quarter",
        type=int,
        choices=[1, 2, 3, 4],
        help="Optional baseline quarter used for KL(current || baseline).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Additive smoothing constant used in the KL calculation. Defaults to {DEFAULT_EPSILON}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.baseline_year is None) != (args.baseline_quarter is None):
        raise ValueError("Provide both --baseline-year and --baseline-quarter, or neither.")

    baseline_quarter = None
    if args.baseline_year is not None and args.baseline_quarter is not None:
        baseline_quarter = (args.baseline_year, args.baseline_quarter)

    output_path = save_distribution_divergence(
        input_path=args.input,
        output_path=args.output,
        baseline_quarter=baseline_quarter,
        epsilon=args.epsilon,
    )
    if baseline_quarter is None:
        print(
            "Saved distribution divergence to "
            f"{output_path} using each occupation's first available quarter as the baseline."
        )
    else:
        print(
            "Saved distribution divergence to "
            f"{output_path} using {quarter_label(*baseline_quarter)} as the baseline quarter."
        )


if __name__ == "__main__":
    main()
