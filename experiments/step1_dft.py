"""Benchmark the Vandermonde-matrix DFT for every dataset size."""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.dft import (
    build_dft_matrix,
    complex_multiplication_count,
    dft_horner_at_roots,
    dft_vandermonde,
)


def _repeat_count(n: int) -> int:
    """Choose enough repetitions for small inputs without over-running large ones."""
    if n <= 16:
        return 20
    if n <= 64:
        return 10
    return 5


def _measure(
    A: list[int],
    B: list[int],
    repeats: int,
) -> tuple[float, float, float]:
    """Return median matrix-build, multiply, and combined times in seconds."""
    build_times: list[float] = []
    multiply_times: list[float] = []
    total_times: list[float] = []

    for _ in range(repeats):
        total_start = time.perf_counter()
        build_start = time.perf_counter()
        matrix = build_dft_matrix(len(A))
        build_times.append(time.perf_counter() - build_start)

        multiply_start = time.perf_counter()
        dft_vandermonde(A, matrix)
        dft_vandermonde(B, matrix)
        multiply_times.append(time.perf_counter() - multiply_start)
        total_times.append(time.perf_counter() - total_start)

    return (
        statistics.median(build_times),
        statistics.median(multiply_times),
        statistics.median(total_times),
    )


def _max_error(
    A: list[int],
    B: list[int],
    matrix: list[list[complex]],
) -> float:
    """Compare both custom transforms with NumPy and direct Horner evaluation."""
    errors: list[float] = []
    for coefficients in (A, B):
        actual = dft_vandermonde(coefficients, matrix)
        direct = dft_horner_at_roots(coefficients)
        expected = np.fft.fft(coefficients)
        errors.extend(
            abs(actual_value - expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
        errors.extend(
            abs(actual_value - direct_value)
            for actual_value, direct_value in zip(actual, direct)
        )
    return max(errors, default=0.0)


def _print_n4_table(A: list[int], B: list[int]) -> None:
    """Print the complete n=4 DFT of A and B."""
    matrix = build_dft_matrix(4)
    transformed_A = dft_vandermonde(A, matrix)
    transformed_B = dft_vandermonde(B, matrix)
    print("\nFull DFT for n=4")
    print("j | DFT(A)                 | DFT(B)")
    print("--+------------------------+------------------------")
    for j, (value_A, value_B) in enumerate(zip(transformed_A, transformed_B)):
        print(f"{j} | {value_A.real:9.5f} {value_A.imag:+9.5f}i | "
              f"{value_B.real:9.5f} {value_B.imag:+9.5f}i")


def _save_plot(rows: list[dict[str, float | int]], output_path: Path) -> None:
    """Save measured total time and a scaled n² reference line."""
    sizes = [int(row["n"]) for row in rows]
    measured = [float(row["total_time"]) for row in rows]
    reference_base = measured[0] / (sizes[0] ** 2)
    reference = [reference_base * n**2 for n in sizes]

    plt.figure()
    plt.loglog(sizes, measured, "o-", label="Vandermonde DFT total")
    plt.loglog(sizes, reference, "--", label="scaled n² reference")
    plt.xlabel("n")
    plt.ylabel("median time (seconds)")
    plt.title("1-D DFT via Vandermonde matrix")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Run all step 1 experiments and save CSV and plot results."""
    data_path = PROJECT_ROOT / "data" / "Dataset.csv"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    dataset = load_dataset(data_path)
    rows: list[dict[str, float | int]] = []

    for n in sorted(dataset):
        A, B = dataset[n]
        matrix = build_dft_matrix(n)
        repeats = _repeat_count(n)
        build_time, multiply_time, total_time = _measure(A, B, repeats)
        rows.append(
            {
                "n": n,
                "build_time": build_time,
                "multiply_time": multiply_time,
                "total_time": total_time,
                "complex_mults": 2 * complex_multiplication_count(n),
                "max_error_vs_numpy": _max_error(A, B, matrix),
            }
        )
        print(
            f"n={n}: build={build_time:.6e}s, multiply={multiply_time:.6e}s, "
            f"total={total_time:.6e}s, repeats={repeats}"
        )

        if n == 4:
            _print_n4_table(A, B)

    csv_path = results_dir / "step1_dft_timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "n",
                "build_time",
                "multiply_time",
                "total_time",
                "complex_mults",
                "max_error_vs_numpy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    _save_plot(rows, results_dir / "step1_dft.png")
    print(f"\nSaved timing data to {csv_path}")
    print(f"Saved plot to {results_dir / 'step1_dft.png'}")


if __name__ == "__main__":
    main()
