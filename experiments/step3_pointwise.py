"""Verify pointwise frequency-domain multiplication for every dataset input."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.convolution import naive_convolution
from src.fft import fft
from src.data_loader import load_dataset
from src.pointwise import pipeline_step3, pad_to_length


def _inverse_fft(values: list[complex]) -> list[complex]:
    """Compute an inverse FFT using the forward FFT and conjugation identity."""
    N = len(values)
    return [value.conjugate() / N for value in fft([value.conjugate() for value in values])]


def _max_error(actual: list[complex], expected: list[complex]) -> float:
    """Return the largest elementwise absolute difference."""
    return max(
        (abs(actual_value - expected_value)
         for actual_value, expected_value in zip(actual, expected)),
        default=0.0,
    )


def _print_n4_c_hat(A: list[int], B: list[int]) -> None:
    """Print the complete n=4 pointwise product spectrum."""
    _, _, _, C_hat = pipeline_step3(A, B)
    print("\nC_hat for n=4 (padded to N=8)")
    print("k | real                 | imaginary")
    print("--+----------------------+----------------------")
    for k, value in enumerate(C_hat):
        print(f"{k} | {value.real:20.10f} | {value.imag:20.10f}")


def _print_circular_convolution_demo(A: list[int], B: list[int]) -> None:
    """Show why length-n FFT multiplication wraps the product circularly."""
    A_hat = fft(A)
    B_hat = fft(B)
    circular_product = _inverse_fft(
        [a_value * b_value for a_value, b_value in zip(A_hat, B_hat)]
    )
    true_product = naive_convolution(A, B)
    print("\nCircular-convolution demonstration for n=4 (no padding)")
    print(f"True linear product:     {true_product}")
    print(
        "Inverse length-n result: "
        f"{[round(value.real, 10) for value in circular_product]}"
    )
    print("Equal to true product:   ", len(circular_product) == len(true_product)
          and all(abs(value.real - expected) <= 1e-9
                  for value, expected in zip(circular_product, true_product)))


def main() -> None:
    """Run Step 3 verification and save the error CSV."""
    dataset = load_dataset(PROJECT_ROOT / "data" / "Dataset.csv")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    rows: list[dict[str, float | int]] = []

    for n in sorted(dataset):
        A, B = dataset[n]
        N, _, _, C_hat = pipeline_step3(A, B)
        reference = fft(pad_to_length(naive_convolution(A, B), N))
        max_error = _max_error(C_hat, reference)
        within_tolerance = max_error <= 1e-9
        rows.append(
            {"n": n, "N": N, "max_error_vs_reference": max_error}
        )
        print(
            f"n={n}, N={N}: max error={max_error:.3e}, "
            f"within 1e-9={within_tolerance}"
        )

        if n == 4:
            _print_n4_c_hat(A, B)
            _print_circular_convolution_demo(A, B)

    csv_path = results_dir / "step3_pointwise.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["n", "N", "max_error_vs_reference"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved verification data to {csv_path}")


if __name__ == "__main__":
    main()
