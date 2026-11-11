"""Load and validate polynomial coefficient datasets.

The loader runs in O(r) time for ``r`` CSV data rows, excluding the sorting
of each polynomial's coefficient indices.  It returns coefficients in
ascending order.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeAlias


PolynomialPair: TypeAlias = tuple[list[int], list[int]]


def _parse_integer(value: str, field: str, row_number: int) -> int:
    """Parse an integer field and raise a descriptive validation error."""
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"Row {row_number}: {field} must be an integer, got {value!r}."
        ) from error


def _is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and value & (value - 1) == 0


def load_dataset(path: str | Path) -> dict[int, PolynomialPair]:
    """Load and validate polynomial pairs from a CSV file.

    The CSV must contain ``n``, ``coefficient_index``, ``A_coefficient``, and
    ``B_coefficient`` columns.
    """
    dataset: dict[int, list[tuple[int, int, int]]] = {}
    csv_path = Path(path)

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            expected_fields = {
                "n",
                "coefficient_index",
                "A_coefficient",
                "B_coefficient",
            }
            if reader.fieldnames is None:
                raise ValueError("Dataset is missing a CSV header row.")
            if set(reader.fieldnames) != expected_fields:
                raise ValueError(
                    "Dataset must have exactly these columns: "
                    "n, coefficient_index, A_coefficient, B_coefficient."
                )

            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Row {row_number}: missing a column value.")
                if all(value == "" for value in row.values()):
                    continue

                n = _parse_integer(row["n"], "n", row_number)
                coefficient_index = _parse_integer(
                    row["coefficient_index"], "coefficient_index", row_number
                )
                a_coefficient = _parse_integer(
                    row["A_coefficient"], "A_coefficient", row_number
                )
                b_coefficient = _parse_integer(
                    row["B_coefficient"], "B_coefficient", row_number
                )

                if not _is_power_of_two(n):
                    raise ValueError(
                        f"Row {row_number}: n must be a positive power of two, got {n}."
                    )
                if not 0 <= a_coefficient <= 9:
                    raise ValueError(
                        f"Row {row_number}: A_coefficient must be in 0..9, "
                        f"got {a_coefficient}."
                    )
                if not 0 <= b_coefficient <= 9:
                    raise ValueError(
                        f"Row {row_number}: B_coefficient must be in 0..9, "
                        f"got {b_coefficient}."
                    )

                dataset.setdefault(n, []).append(
                    (coefficient_index, a_coefficient, b_coefficient)
                )
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Dataset file not found: {csv_path}") from error

    result: dict[int, PolynomialPair] = {}
    for n, rows in dataset.items():
        if len(rows) != n:
            raise ValueError(
                f"n={n} must have exactly {n} rows, but found {len(rows)}."
            )

        rows.sort(key=lambda row: row[0])
        indices = [row[0] for row in rows]
        expected_indices = list(range(n))
        if indices != expected_indices:
            raise ValueError(
                f"n={n} must have coefficient indices 0..{n - 1} exactly once; "
                f"got {indices}."
            )

        result[n] = (
            [row[1] for row in rows],
            [row[2] for row in rows],
        )

    return result


if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "Dataset.csv"
    loaded_dataset = load_dataset(dataset_path)
    for n in sorted(loaded_dataset):
        a_coefficients, b_coefficients = loaded_dataset[n]
        displayed_a = a_coefficients[:8]
        displayed_b = b_coefficients[:8]
        suffix = " ..." if n > 8 else ""
        print(f"n={n}: A={displayed_a}{suffix}, B={displayed_b}{suffix}")