"""Optional scale-bit experiment for the largest assignment dataset."""

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


def main() -> None:
    """Sweep scale bits for n=128 with a 512-bit RSA modulus."""
    A, B = load_dataset(PROJECT_ROOT / "data" / "Dataset.csv")[128]
    N, _, _, C_hat = pipeline_step3(A, B)
    reference = naive_convolution(A, B)
    public_key, private_key, _, _ = generate_keypair(512)
    rows: list[dict[str, int | float | bool]] = []

    print("scale_bits | max_dist_to_int | match_with_convolution")
    print("------------+------------------+-----------------------")
    for scale_bits in (8, 16, 24, 32):
        encoded, parameters = encode_complex_vector(
            C_hat, scale_bits=scale_bits, modulus=public_key[0]
        )
        recovered_integers = [
            decrypt_int(encrypt_int(value, public_key), private_key)
            for value in encoded
        ]
        recovered = ifft(decode_complex_vector(recovered_integers, parameters))
        max_distance = max(
            (abs(value.real - round(value.real)) for value in recovered),
            default=0.0,
        )
        rounded = [round(value.real) for value in recovered[: 2 * len(A) - 1]]
        match = rounded == reference
        print(f"{scale_bits:10} | {max_distance:16.6e} | {match}")
        rows.append(
            {
                "scale_bits": scale_bits,
                "max_dist_to_int": max_distance,
                "match_with_convolution": match,
            }
        )

    output_path = PROJECT_ROOT / "results" / "scale_sweep.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["scale_bits", "max_dist_to_int", "match_with_convolution"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved scale sweep to {output_path}")


if __name__ == "__main__":
    main()
