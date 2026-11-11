"""Pytest coverage for the assignment's algorithm stages."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.convolution import naive_convolution
from src.data_loader import load_dataset
from src.dft import build_dft_matrix, dft_horner_at_roots, dft_vandermonde
from src.encoding import decode_complex_vector, encode_complex_vector
from src.fft import fft
from src.ifft import ifft
from src.pointwise import pad_to_length, pipeline_step3
from src.rsa_utils import decrypt_int, encrypt_int, generate_keypair


ROOT = Path(__file__).resolve().parents[1]


def test_loader_validation_and_dataset() -> None:
    dataset = load_dataset(ROOT / "data" / "Dataset.csv")
    assert sorted(dataset) == [4, 8, 16, 32, 64, 128]
    assert all(len(pair[0]) == n and len(pair[1]) == n for n, pair in dataset.items())

    invalid = ROOT / "tests" / "invalid_dataset.csv"
    invalid.write_text(
        "n,coefficient_index,A_coefficient,B_coefficient\n"
        "3,0,1,1\n3,1,1,1\n3,2,1,1\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(ValueError, match="power of two"):
            load_dataset(invalid)
    finally:
        invalid.unlink()


@pytest.mark.parametrize("coefficients", [[2, 2, 9, 1], [1, 0, 3, 2, 4, 1, 0, 2]])
def test_dft_matches_horner_and_numpy(coefficients: list[int]) -> None:
    actual = dft_vandermonde(coefficients, build_dft_matrix(len(coefficients)))
    assert np.allclose(actual, dft_horner_at_roots(coefficients), atol=1e-9)
    assert np.allclose(actual, np.fft.fft(coefficients), atol=1e-9)


def test_fft_matches_dft() -> None:
    coefficients = [index % 10 for index in range(16)]
    dft = dft_vandermonde(coefficients, build_dft_matrix(16))
    assert np.allclose(fft(coefficients), dft, atol=1e-9)


def test_convolution_theorem_with_padding() -> None:
    A, B = [2, 2, 9, 1], [6, 9, 2, 6]
    N, _, _, product_spectrum = pipeline_step3(A, B)
    expected = fft(pad_to_length(naive_convolution(A, B), N))
    assert np.allclose(product_spectrum, expected, atol=1e-9)


@pytest.mark.parametrize("bits", [128, 256, 512])
def test_rsa_round_trip(bits: int) -> None:
    public, private, p, q = generate_keypair(bits)
    assert public[0].bit_length() == bits
    for message in (0, 1, 2, public[0] - 1, 123456789):
        assert decrypt_int(encrypt_int(message, public), private) == message
    assert p != q


def test_encoding_round_trip() -> None:
    values = [1.25 - 2.5j, -3.0 + 0.125j]
    encoded, parameters = encode_complex_vector(values, scale_bits=24)
    decoded = decode_complex_vector(encoded, parameters)
    assert np.allclose(decoded, values, atol=parameters.theoretical_max_error * 2)


@pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 128])
def test_full_pipeline_equals_naive_convolution(n: int) -> None:
    A, B = load_dataset(ROOT / "data" / "Dataset.csv")[n]
    N, _, _, spectrum = pipeline_step3(A, B)
    public, private, _, _ = generate_keypair(128)
    encoded, parameters = encode_complex_vector(
        spectrum, scale_bits=32, modulus=public[0]
    )
    encrypted = [encrypt_int(value, public) for value in encoded]
    decrypted = [decrypt_int(value, private) for value in encrypted]
    assert decrypted == encoded
    decoded_spectrum = decode_complex_vector(decrypted, parameters)
    recovered = ifft(decoded_spectrum)
    actual = [round(value.real) for value in recovered[: 2 * n - 1]]
    assert actual == naive_convolution(A, B)
