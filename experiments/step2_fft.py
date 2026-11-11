"""Compare recursive radix-2 FFT timing and accuracy with the Step 1 DFT."""

from __future__ import annotations

import csv
import math
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
    dft_vandermonde,
    complex_multiplication_count as dft_multiplication_count,
)
from src.fft import (
    complex_multiplication_count as fft_multiplication_count,
    fft,
)


def _repeat_count(n: int) -> int:
    """Match the Step 1 repetition schedule."""
    if n <= 16:
        return 20
    if n <= 64:
        return 10
    return 5


def _measure(
    A: list[int],
    B: list[int],
    repeats: int,
) -> tuple[float, float]:
    """Return median DFT and FFT times for transforming A and B."""
    dft_times: list[float] = []
    fft_times: list[float] = []
    for _ in range(repeats):
        dft_start = time.perf_counter()
        matrix = build_dft_matrix(len(A))
        dft_vandermonde(A, matrix)
        dft_vandermonde(B, matrix)
        dft_times.append(time.perf_counter() - dft_start)

        fft_start = time.perf_counter()
        fft(A)
        fft(B)
        fft_times.append(time.perf_counter() - fft_start)
    return statistics.median(dft_times), statistics.median(fft_times)


def _max_diff(A: list[int], B: list[int]) -> float:
    """Return the largest absolute FFT-versus-DFT difference for A and B."""
    matrix = build_dft_matrix(len(A))
    differences = []
    for coefficients in (A, B):
        dft_result = dft_vandermonde(coefficients, matrix)
        fft_result = fft(coefficients)
        differences.extend(
            abs(fft_value - dft_value)
            for fft_value, dft_value in zip(fft_result, dft_result)
        )
    return max(differences, default=0.0)


def _print_n8_table(A: list[int], B: list[int]) -> None:
    """Print the first eight FFT and DFT values for the n=8 sample."""
    matrix = build_dft_matrix(8)
    dft_A = dft_vandermonde(A, matrix)
    dft_B = dft_vandermonde(B, matrix)
    fft_A = fft(A)
    fft_B = fft(B)
    print("\nFirst 8 outputs for n=8")
    print("j | FFT(A)                | DFT(A)                | FFT(B)                | DFT(B)")
    print("--+-----------------------+-----------------------+-----------------------+-----------------------")
    for j in range(8):
        print(
            f"{j} | {fft_A[j].real:8.4f}{fft_A[j].imag:+8.4f}i | "
            f"{dft_A[j].real:8.4f}{dft_A[j].imag:+8.4f}i | "
            f"{fft_B[j].real:8.4f}{fft_B[j].imag:+8.4f}i | "
            f"{dft_B[j].real:8.4f}{dft_B[j].imag:+8.4f}i"
        )


def _save_plot(rows: list[dict[str, float | int]], output_path: Path) -> None:
    """Save timing curves and scaled n² and n log2(n) reference curves."""
    sizes = [int(row["n"]) for row in rows]
    dft_times = [float(row["dft_time"]) for row in rows]
    fft_times = [float(row["fft_time"]) for row in rows]
    dft_reference = [dft_times[0] * (n / sizes[0]) ** 2 for n in sizes]
    fft_reference = [
        fft_times[0] * (n * math.log2(n)) / (sizes[0] * math.log2(sizes[0]))
        for n in sizes
    ]

    plt.figure()
    plt.loglog(sizes, dft_times, "o-", label="Vandermonde DFT")
    plt.loglog(sizes, fft_times, "s-", label="recursive FFT")
    plt.loglog(sizes, dft_reference, "--", label="scaled n² reference")
    plt.loglog(sizes, fft_reference, ":", label="scaled n log2(n) reference")
    plt.xlabel("n")
    plt.ylabel("median time for A and B (seconds)")
    plt.title("DFT versus recursive radix-2 FFT")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Run Step 2 comparisons and save the CSV and plot."""
    dataset = load_dataset(PROJECT_ROOT / "data" / "Dataset.csv")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    rows: list[dict[str, float | int]] = []

    for n in sorted(dataset):
        A, B = dataset[n]
        dft_time, fft_time = _measure(A, B, _repeat_count(n))
        max_diff = _max_diff(A, B)
        row: dict[str, float | int] = {
            "n": n,
            "dft_time": dft_time,
            "fft_time": fft_time,
            "speedup": dft_time / fft_time,
            "dft_mults": 2 * dft_multiplication_count(n),
            "fft_mults": 2 * fft_multiplication_count(n),
            "max_diff": max_diff,
        }
        rows.append(row)
        print(
            f"n={n}: DFT={dft_time:.6e}s, FFT={fft_time:.6e}s, "
            f"speedup={row['speedup']:.2f}x, max |FFT-DFT|={max_diff:.3e}, "
            f"within 1e-9={max_diff <= 1e-9}"
        )
        if n == 8:
            _print_n8_table(A, B)

    csv_path = results_dir / "step2_fft_vs_dft.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "n",
                "dft_time",
                "fft_time",
                "speedup",
                "dft_mults",
                "fft_mults",
                "max_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    _save_plot(rows, results_dir / "step2_fft_vs_dft.png")
    print(f"\nSaved timing data to {csv_path}")
    print(f"Saved plot to {results_dir / 'step2_fft_vs_dft.png'}")


if __name__ == "__main__":
    main()
