"""Run the complete Advanced Algorithms assignment pipeline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.convolution import naive_convolution
from src.data_loader import load_dataset
from src.dft import build_dft_matrix, dft_vandermonde
from src.encoding import decode_complex_vector, encode_complex_vector
from src.fft import fft
from src.ifft import ifft
from src.pointwise import pipeline_step3
from src.rsa_utils import decrypt_int, encrypt_int, generate_keypair


def _max_difference(left: list[complex], right: list[complex]) -> float:
    """Return the largest elementwise absolute difference."""
    return max(
        (abs(a - b) for a, b in zip(left, right)),
        default=0.0,
    )


def main() -> None:
    """Run all algorithm stages and write a compact summary CSV."""
    dataset = load_dataset(PROJECT_ROOT / "data" / "Dataset.csv")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    summary: list[dict[str, int | float | str | bool]] = []

    print("Step | n | key bits | metric | value")
    print("-----+---+----------+--------+------")
    for n in sorted(dataset):
        A, B = dataset[n]
        matrix = build_dft_matrix(n)
        dft_A = dft_vandermonde(A, matrix)
        fft_A = fft(A)
        fft_error = _max_difference(dft_A, fft_A)
        N, _, _, C_hat = pipeline_step3(A, B)
        product_reference = fft([*naive_convolution(A, B), *([0] * (N - (2 * n - 1)))])
        pointwise_error = _max_difference(C_hat, product_reference)

        for step, metric, value in (
            ("DFT", "max_error_vs_fft", fft_error),
            ("pointwise", "max_error_vs_reference", pointwise_error),
        ):
            print(f"{step:5} | {n:3} | {'-':8} | {metric:20} | {value:.3e}")
            summary.append(
                {
                    "step": step,
                    "n": n,
                    "key_bits": "",
                    "metric": metric,
                    "value": value,
                    "passed": value <= 1e-9,
                }
            )

        reference = naive_convolution(A, B)
        for key_bits in (128, 256, 512):
            public_key, private_key, _, _ = generate_keypair(key_bits)
            encoded, parameters = encode_complex_vector(
                C_hat, scale_bits=32, modulus=public_key[0]
            )
            encrypted = [encrypt_int(value, public_key) for value in encoded]
            decrypted = [decrypt_int(value, private_key) for value in encrypted]
            decoded = decode_complex_vector(decrypted, parameters)
            recovered = ifft(decoded)
            recovered_coefficients = [
                round(value.real) for value in recovered[: 2 * n - 1]
            ]
            passed = recovered_coefficients == reference
            max_distance = max(
                (abs(value.real - round(value.real)) for value in recovered),
                default=0.0,
            )
            print(
                f"{'RSA+IFFT':8} | {n:3} | {key_bits:8} | "
                f"max_dist_to_int      | {max_distance:.3e} "
                f"({'PASS' if passed else 'FAIL'})"
            )
            summary.append(
                {
                    "step": "RSA+IFFT",
                    "n": n,
                    "key_bits": key_bits,
                    "metric": "max_dist_to_int",
                    "value": max_distance,
                    "passed": passed,
                }
            )
            if not passed:
                raise AssertionError(f"Full pipeline failed for n={n}, key_bits={key_bits}.")

    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", "n", "key_bits", "metric", "value", "passed"],
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
