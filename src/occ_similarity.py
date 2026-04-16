import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
DEFAULT_INPUT_PATH = OUTPUT_DIR / "skill_distributions.csv"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "occupation_similarity_network.csv"


OccupationKey = tuple[str, str]
QuarterKey = tuple[int, int]
SkillVector = dict[str, float]


def quarter_label(year: int, quarter: int) -> str:
    return f"{year}Q{quarter}"


def soc_code_from_onet(onet_code: str) -> str:
    return onet_code.split(".", 1)[0]


def _pair_key(left: OccupationKey, right: OccupationKey) -> tuple[OccupationKey, OccupationKey]:
    return (left, right) if left <= right else (right, left)


def load_occupation_skill_vectors(
    input_path: Path,
) -> dict[QuarterKey, dict[OccupationKey, SkillVector]]:
    quarter_vectors: dict[QuarterKey, dict[OccupationKey, SkillVector]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    occupation_name_by_code: dict[str, str] = {}

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            year = int(row["YEAR"])
            quarter = int(row["QUARTER"])
            soc_code = soc_code_from_onet(row["ONET_2019"])
            skill_key = row.get("SKILL_ID") or row.get("SKILL_NAME")
            if not skill_key:
                continue

            occupation_name_by_code.setdefault(soc_code, row["ONET_2019_NAME"])
            count = float(row["CNT"])
            quarter_vectors[(year, quarter)][
                (soc_code, occupation_name_by_code[soc_code])
            ][skill_key] += count

    return quarter_vectors


def compute_quarter_similarity_rows(
    quarter_key: QuarterKey,
    occupation_vectors: dict[OccupationKey, SkillVector],
    min_similarity: float = 0.0,
) -> list[dict[str, object]]:
    year, quarter = quarter_key
    quarter_name = quarter_label(year, quarter)

    occupation_norms: dict[OccupationKey, float] = {}
    inverted_index: dict[str, list[tuple[OccupationKey, float]]] = defaultdict(list)

    for occupation_key, skill_counts in occupation_vectors.items():
        norm = math.sqrt(sum(value * value for value in skill_counts.values()))
        if norm == 0.0:
            continue

        occupation_norms[occupation_key] = norm
        for skill_key, value in skill_counts.items():
            if value != 0.0:
                inverted_index[skill_key].append((occupation_key, value))

    dot_products: dict[tuple[OccupationKey, OccupationKey], float] = defaultdict(float)
    shared_skill_counts: dict[tuple[OccupationKey, OccupationKey], int] = defaultdict(int)

    for occupation_values in inverted_index.values():
        ordered_values = sorted(occupation_values, key=lambda item: item[0])
        for left_index in range(len(ordered_values)):
            left_occupation, left_value = ordered_values[left_index]
            for right_index in range(left_index + 1, len(ordered_values)):
                right_occupation, right_value = ordered_values[right_index]
                pair = _pair_key(left_occupation, right_occupation)
                dot_products[pair] += left_value * right_value
                shared_skill_counts[pair] += 1

    rows: list[dict[str, object]] = []
    for (left_occupation, right_occupation), dot_product in sorted(dot_products.items()):
        left_norm = occupation_norms[left_occupation]
        right_norm = occupation_norms[right_occupation]
        similarity = dot_product / (left_norm * right_norm)
        if similarity < min_similarity:
            continue

        left_code, left_name = left_occupation
        right_code, right_name = right_occupation
        rows.append(
            {
                "YEAR": year,
                "QUARTER": quarter,
                "QUARTER_LABEL": quarter_name,
                "SOC_CODE_1": left_code,
                "SOC_NAME_1": left_name,
                "SOC_CODE_2": right_code,
                "SOC_NAME_2": right_name,
                "COSINE_SIMILARITY": similarity,
                "DOT_PRODUCT": dot_product,
                "NORM_1": left_norm,
                "NORM_2": right_norm,
                "SHARED_SKILL_COUNT": shared_skill_counts[(left_occupation, right_occupation)],
            }
        )

    return rows


def apply_top_k_per_occupation(
    rows: list[dict[str, object]],
    top_k_per_occupation: int | None,
) -> list[dict[str, object]]:
    if top_k_per_occupation is None:
        return rows

    rows_by_quarter: dict[QuarterKey, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_quarter[(int(row["YEAR"]), int(row["QUARTER"]))].append(row)

    filtered_rows: list[dict[str, object]] = []
    for quarter_key in sorted(rows_by_quarter):
        quarter_rows = rows_by_quarter[quarter_key]
        rows_for_occupation: dict[str, list[dict[str, object]]] = defaultdict(list)

        for row in quarter_rows:
            rows_for_occupation[str(row["SOC_CODE_1"])].append(row)
            rows_for_occupation[str(row["SOC_CODE_2"])].append(row)

        kept_ids: set[int] = set()
        for occupation_rows in rows_for_occupation.values():
            ranked_rows = sorted(
                occupation_rows,
                key=lambda row: (
                    -float(row["COSINE_SIMILARITY"]),
                    str(row["SOC_CODE_1"]),
                    str(row["SOC_CODE_2"]),
                ),
            )
            for row in ranked_rows[:top_k_per_occupation]:
                kept_ids.add(id(row))

        filtered_rows.extend(row for row in quarter_rows if id(row) in kept_ids)

    return filtered_rows


def compute_occupation_similarity_network(
    input_path: Path = DEFAULT_INPUT_PATH,
    min_similarity: float = 0.0,
    top_k_per_occupation: int | None = None,
) -> list[dict[str, object]]:
    quarter_vectors = load_occupation_skill_vectors(input_path)
    rows: list[dict[str, object]] = []

    for quarter_key in sorted(quarter_vectors):
        quarter_rows = compute_quarter_similarity_rows(
            quarter_key=quarter_key,
            occupation_vectors=quarter_vectors[quarter_key],
            min_similarity=min_similarity,
        )
        rows.extend(quarter_rows)

    return apply_top_k_per_occupation(rows, top_k_per_occupation)


def save_occupation_similarity_network(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    min_similarity: float = 0.0,
    top_k_per_occupation: int | None = None,
) -> Path:
    rows = compute_occupation_similarity_network(
        input_path=input_path,
        min_similarity=min_similarity,
        top_k_per_occupation=top_k_per_occupation,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "YEAR",
        "QUARTER",
        "QUARTER_LABEL",
        "SOC_CODE_1",
        "SOC_NAME_1",
        "SOC_CODE_2",
        "SOC_NAME_2",
        "COSINE_SIMILARITY",
        "DOT_PRODUCT",
        "NORM_1",
        "NORM_2",
        "SHARED_SKILL_COUNT",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a quarter-by-quarter SOC similarity network from skill-count vectors."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input skill-distribution CSV path. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Optional cosine-similarity floor. Defaults to 0.0.",
    )
    parser.add_argument(
        "--top-k-per-occupation",
        type=int,
        help="Optional per-quarter top-k filter that keeps an edge if it is in either SOC occupation's top k neighbors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_similarity < 0.0:
        raise ValueError("--min-similarity must be non-negative.")
    if args.top_k_per_occupation is not None and args.top_k_per_occupation < 1:
        raise ValueError("--top-k-per-occupation must be at least 1.")

    output_path = save_occupation_similarity_network(
        input_path=args.input,
        output_path=args.output,
        min_similarity=args.min_similarity,
        top_k_per_occupation=args.top_k_per_occupation,
    )
    filter_label = (
        f" and top-k-per-occupation={args.top_k_per_occupation}"
        if args.top_k_per_occupation is not None
        else ""
    )
    print(
        f"Saved SOC occupation similarity network to {output_path} "
        f"with min_similarity={args.min_similarity}{filter_label}."
    )


if __name__ == "__main__":
    main()
