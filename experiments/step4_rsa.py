"""Encrypt Step 3 frequency-domain values with textbook RSA."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.encoding import decode_complex_vector, encode_complex_vector
from src.pointwise import pipeline_step3
from src.rsa_utils import decrypt_int, encrypt_int, generate_keypair


def _truncate_hex(value: int, digits: int = 20) -> str:
    """Return a readable prefix of an integer's hexadecimal representation."""
    text = f"{value:x}"
    return text if len(text) <= digits else f"{text[:digits]}..."


def main() -> None:
    """Run RSA encoding, recovery, and timing experiments."""
    dataset = load_dataset(PROJECT_ROOT / "data" / "Dataset.csv")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    rows: list[dict[str, int | float | bool]] = []

    for n in sorted(dataset):
        A, B = dataset[n]
        N, _, _, C_hat = pipeline_step3(A, B)
        for key_bits in (128, 256, 512):
            keygen_start = time.perf_counter()
            public_key, private_key, _, _ = generate_keypair(key_bits)
            keygen_s = time.perf_counter() - keygen_start
            modulus, _ = public_key
            encoded, parameters = encode_complex_vector(
                C_hat, scale_bits=32, modulus=modulus
            )
            print(
                f"n={n}, key_bits={key_bits}: max encoded integer="
                f"{parameters.max_encoded_value}, encoded bits="
                f"{parameters.encoded_bit_width}, N bits={modulus.bit_length()}"
            )

            encrypt_start = time.perf_counter()
            ciphertexts = [encrypt_int(value, public_key) for value in encoded]
            encrypt_s = time.perf_counter() - encrypt_start

            decrypt_start = time.perf_counter()
            recovered = [decrypt_int(value, private_key) for value in ciphertexts]
            decrypt_s = time.perf_counter() - decrypt_start
            exact = recovered == encoded
            assert exact, f"RSA integer recovery failed for n={n}, key_bits={key_bits}"

            decoded = decode_complex_vector(recovered, parameters)
            max_decode_error = max(
                (abs(actual - expected) for actual, expected in zip(decoded, C_hat)),
                default=0.0,
            )
            rows.append(
                {
                    "n": n,
                    "key_bits": key_bits,
                    "num_integers": len(encoded),
                    "max_encoded_bits": parameters.encoded_bit_width,
                    "keygen_s": keygen_s,
                    "encrypt_s": encrypt_s,
                    "decrypt_s": decrypt_s,
                    "ints_recovered_exactly": exact,
                    "max_decode_error": max_decode_error,
                }
            )

            if n == 4 and key_bits == 128:
                print("\nWorked RSA example (n=4, 128-bit key)")
                print(f"C_hat[0] = {C_hat[0]}")
                print(f"encoded real,imag = {encoded[:2]}")
                print(
                    "ciphertexts (hex, truncated) = "
                    f"{_truncate_hex(ciphertexts[0])}, {_truncate_hex(ciphertexts[1])}"
                )
                print(f"decrypted integers = {recovered[:2]}")

    csv_path = results_dir / "step4_rsa.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "n",
                "key_bits",
                "num_integers",
                "max_encoded_bits",
                "keygen_s",
                "encrypt_s",
                "decrypt_s",
                "ints_recovered_exactly",
                "max_decode_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved RSA experiment results to {csv_path}")


if __name__ == "__main__":
    main()
