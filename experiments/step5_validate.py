"""Validate the complete FFT, RSA, and inverse-FFT polynomial pipeline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.convolution import naive_convolution
from src.data_loader import load_dataset
from src.encoding import decode_complex_vector, encode_complex_vector
from src.ifft import ifft
from src.pointwise import pipeline_step3
from src.rsa_utils import decrypt_int, encrypt_int, generate_keypair


DEFAULT_TOLERANCE = 1e-6


def _max_real_distance_to_integer(values: list[complex]) -> float:
    """Return the largest distance from a real part to its nearest integer."""
    return max(
        (abs(value.real - round(value.real)) for value in values),
        default=0.0,
    )


def _recover_product(
    A: list[int],
    B: list[int],
    key_bits: int,
    tolerance: float,
) -> tuple[list[complex], list[int], float, float]:
    """Run the full encoded RSA product pipeline and return validation values."""
    N, _, _, C_hat = pipeline_step3(A, B)
    public_key, private_key, _, _ = generate_keypair(key_bits)
    encoded, encoding_parameters = encode_complex_vector(
        C_hat,
        scale_bits=32,
        modulus=public_key[0],
    )
    encrypted = [encrypt_int(value, public_key) for value in encoded]
    recovered_integers = [decrypt_int(value, private_key) for value in encrypted]
    assert recovered_integers == encoded

    decoded_C_hat = decode_complex_vector(recovered_integers, encoding_parameters)
    recovered = ifft(decoded_C_hat)
    max_imag = max((abs(value.imag) for value in recovered), default=0.0)
    max_dist_to_int = _max_real_distance_to_integer(recovered)
    assert max_imag < tolerance, (
        f"max imaginary part {max_imag} is not below tolerance {tolerance}"
    )
    assert max_dist_to_int < tolerance, (
        f"max distance to integer {max_dist_to_int} is not below tolerance {tolerance}"
    )
    return recovered, naive_convolution(A, B), max_imag, max_dist_to_int


def main() -> None:
    """Run all dataset/key-size validations and save the required CSV."""
    dataset = load_dataset(PROJECT_ROOT / "data" / "Dataset.csv")
    rows: list[dict[str, int | float | bool]] = []

    # Each encoded component has at most 0.5 / 2**32 quantization error.
    # The inverse FFT averages rotated component errors, so 1e-6 is a
    # conservative tolerance with a large margin over that bound.
    tolerance = DEFAULT_TOLERANCE
    for n in sorted(dataset):
        A, B = dataset[n]
        for key_bits in (128, 256, 512):
            recovered, reference, max_imag, max_dist_to_int = _recover_product(
                A, B, key_bits, tolerance
            )
            rounded = [round(value.real) for value in recovered[: 2 * n - 1]]
            match = rounded == reference
            max_before_rounding = max(
                (
                    abs(value - expected)
                    for value, expected in zip(recovered[: 2 * n - 1], reference)
                ),
                default=0.0,
            )
            print(
                f"n={n}, key_bits={key_bits}: max_imag={max_imag:.3e}, "
                f"max_dist_to_int={max_dist_to_int:.3e}, "
                f"max difference before rounding={max_before_rounding:.3e}, "
                f"match={match}"
            )
            if n in (4, 8):
                print(f"  recovered: {rounded}")
                print(f"  reference: {reference}")
            rows.append(
                {
                    "n": n,
                    "key_bits": key_bits,
                    "max_imag": max_imag,
                    "max_dist_to_int": max_dist_to_int,
                    "tolerance": tolerance,
                    "match_with_convolution": match,
                }
            )
            assert match, f"Recovered product differs from convolution for n={n}"

    csv_path = PROJECT_ROOT / "results" / "step5_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "n",
                "key_bits",
                "max_imag",
                "max_dist_to_int",
                "tolerance",
                "match_with_convolution",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved validation results to {csv_path}")


if __name__ == "__main__":
    main()
