"""Readable n=4 walkthrough of the polynomial FFT/RSA workflow."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dft import dft_vandermonde
from fft import fft, ifft
from rsa_utils import (
    decode_component,
    encode_component,
    generate_keypair,
    rsa_decrypt,
    rsa_encrypt,
)


ROOT = Path(__file__).resolve().parent


def load_n4_polynomials() -> tuple[np.ndarray, np.ndarray]:
    """Load the n=4 coefficient vectors from the assignment dataset."""
    data_path = ROOT / "data" / "Dataset.csv"
    rows: list[tuple[int, int, int]] = []
    with data_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if int(row["n"]) == 4:
                rows.append(
                    (
                        int(row["coefficient_index"]),
                        int(row["A_coefficient"]),
                        int(row["B_coefficient"]),
                    )
                )

    if len(rows) != 4:
        raise ValueError(f"Expected four n=4 rows in {data_path}, found {len(rows)}.")
    rows.sort()
    return (
        np.array([row[1] for row in rows], dtype=float),
        np.array([row[2] for row in rows], dtype=float),
    )


def format_complex(value: complex) -> str:
    """Format a complex value compactly for tables."""
    return f"{value.real:10.4f} {value.imag:+10.4f}i"


def print_frequency_table(
    roots: np.ndarray,
    a_values: np.ndarray,
    b_values: np.ndarray,
    product_values: np.ndarray,
) -> None:
    """Print root locations and evaluated frequency-domain values."""
    print("\nEvaluation at the N=8 roots of unity")
    print(" k | root                 | A(root)              | B(root)              | A(root)*B(root)")
    print("---+----------------------+----------------------+----------------------+----------------------")
    for index, (root, a_value, b_value, product) in enumerate(
        zip(roots, a_values, b_values, product_values)
    ):
        print(
            f"{index:2d} | {format_complex(root)} | {format_complex(a_value)} | "
            f"{format_complex(b_value)} | {format_complex(product)}"
        )


def main() -> None:
    """Run and display the complete workflow for n=4 only."""
    A, B = load_n4_polynomials()
    n = len(A)
    N = 2 * n
    A_padded = np.pad(A, (0, N - n))
    B_padded = np.pad(B, (0, N - n))

    print("=== Polynomial multiplication demo: n=4 ===")
    print(f"A(x) coefficients: {A.astype(int).tolist()}")
    print(f"B(x) coefficients: {B.astype(int).tolist()}")
    print(f"\nZero-padding to N=2n={N}:")
    print(f"A padded: {A_padded.astype(int).tolist()}")
    print(f"B padded: {B_padded.astype(int).tolist()}")

    roots = np.exp(-2j * np.pi * np.arange(N) / N)
    A_hat = fft(A_padded)
    B_hat = fft(B_padded)
    C_hat = A_hat * B_hat
    direct_A_hat = dft_vandermonde(A_padded)
    direct_B_hat = dft_vandermonde(B_padded)

    print_frequency_table(roots, A_hat, B_hat, C_hat)
    print(
        "\nFFT/DFT cross-check: "
        f"max A difference={np.max(np.abs(A_hat - direct_A_hat)):.3e}, "
        f"max B difference={np.max(np.abs(B_hat - direct_B_hat)):.3e}"
    )

    public_key, private_key = generate_keypair(128)
    modulus = public_key[0]
    scale = 1000
    offset = int(np.ceil(np.max(np.abs(C_hat)) * scale)) + 1
    encoded: list[tuple[int, int]] = []
    encrypted: list[tuple[int, int]] = []
    decrypted: list[tuple[int, int]] = []
    decoded_values: list[complex] = []

    for value in C_hat:
        encoded_pair = (
            encode_component(float(value.real), scale, offset),
            encode_component(float(value.imag), scale, offset),
        )
        if any(not 0 <= component < modulus for component in encoded_pair):
            raise ValueError("An encoded component does not fit in the RSA modulus.")
        encrypted_pair = tuple(
            rsa_encrypt(component, public_key) for component in encoded_pair
        )
        decrypted_pair = tuple(
            rsa_decrypt(component, private_key) for component in encrypted_pair
        )
        encoded.append(encoded_pair)
        encrypted.append(encrypted_pair)
        decrypted.append(decrypted_pair)
        decoded_values.append(
            decode_component(decrypted_pair[0], scale, offset)
            + 1j * decode_component(decrypted_pair[1], scale, offset)
        )

    print("\nRSA encoding details")
    print(f"Key size: {modulus.bit_length()} bits")
    print(f"Scale: {scale}; offset: {offset}; modulus: {modulus}")
    print(" k | encoded (real, imag)        | ciphertexts (hex, truncated)")
    print("---+-----------------------------+--------------------------------")
    for index, (encoded_pair, encrypted_pair) in enumerate(zip(encoded, encrypted)):
        ciphertext_text = ", ".join(f"{value:x}"[:18] + "..." for value in encrypted_pair)
        print(f"{index:2d} | {str(encoded_pair):27s} | {ciphertext_text}")

    recovered = ifft(np.array(decoded_values))
    reference = np.convolve(A.astype(int), B.astype(int))
    recovered_coefficients = np.rint(recovered.real[: 2 * n - 1]).astype(int)

    print("\nAfter RSA decryption and inverse FFT")
    print(f"Decoded C_hat[0]: {format_complex(decoded_values[0])}")
    print(f"Maximum IFFT imaginary magnitude: {np.max(np.abs(recovered.imag)):.3e}")
    print(
        "Maximum distance to nearest integer: "
        f"{np.max(np.abs(recovered.real - np.rint(recovered.real))):.3e}"
    )
    print(f"Recovered coefficients: {recovered_coefficients.tolist()}")
    print(f"Direct convolution:     {reference.tolist()}")
    print(f"Exact match: {np.array_equal(recovered_coefficients, reference)}")


if __name__ == "__main__":
    main()
