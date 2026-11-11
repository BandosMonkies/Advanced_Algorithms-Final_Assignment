"""One-dimensional DFT using an explicit Vandermonde matrix.

The forward transform uses ``w = exp(-2*pi*i/n)`` and computes
``X[j] = sum(a[k] * w**(j*k) for k in range(n))``.  This implementation uses
only the Python standard library; NumPy is used only by the self-test as an
independent cross-check.
"""

from __future__ import annotations

import cmath
import math
from typing import TypeAlias


ComplexMatrix: TypeAlias = list[list[complex]]


def build_dft_matrix(n: int) -> ComplexMatrix:
    """Build the n-by-n forward DFT matrix in O(n²) time and space."""
    if n <= 0:
        raise ValueError(f"DFT size must be positive, got {n}.")

    root = cmath.exp(-2j * math.pi / n)
    return [[root ** (j * k) for k in range(n)] for j in range(n)]


def dft_vandermonde(
    a: list[int] | list[complex],
    matrix: ComplexMatrix | None = None,
) -> list[complex]:
    """Compute a DFT by explicit matrix-vector multiplication in O(n²) time.

    Each matrix entry and coefficient pair contributes one counted complex
    multiplication.  The count for the most recent call is available through
    :func:`get_last_complex_multiplications`.
    """
    global _last_complex_multiplications
    n = len(a)
    if n == 0:
        _last_complex_multiplications = 0
        return []

    dft_matrix = build_dft_matrix(n) if matrix is None else matrix
    if len(dft_matrix) != n or any(len(row) != n for row in dft_matrix):
        raise ValueError("The DFT matrix must be n-by-n for the input length.")

    _last_complex_multiplications = 0
    result: list[complex] = []
    for row in dft_matrix:
        value = 0j
        for coefficient, matrix_value in zip(a, row):
            value += complex(coefficient) * matrix_value
            _last_complex_multiplications += 1
        result.append(value)
    return result


_last_complex_multiplications = 0


def get_last_complex_multiplications() -> int:
    """Return the number of complex multiplications in the latest DFT call."""
    return _last_complex_multiplications


def complex_multiplication_count(n: int) -> int:
    """Return the O(n²) multiplication count for one n-point DFT."""
    if n < 0:
        raise ValueError(f"DFT size cannot be negative, got {n}.")
    return n * n


def dft_horner_at_roots(a: list[int] | list[complex]) -> list[complex]:
    """Evaluate the polynomial at all forward roots using Horner's rule.

    Time complexity is O(n²), with O(n) output space.
    """
    n = len(a)
    if n == 0:
        return []

    root = cmath.exp(-2j * math.pi / n)
    values: list[complex] = []
    for j in range(n):
        point = root**j
        value = 0j
        for coefficient in reversed(a):
            value = value * point + coefficient
        values.append(value)
    return values


if __name__ == "__main__":
    import numpy as np

    sample = [2, 2, 9, 1]
    matrix = build_dft_matrix(len(sample))
    transformed = dft_vandermonde(sample, matrix)
    direct = dft_horner_at_roots(sample)
    numpy_result = np.fft.fft(sample)
    max_error = max(
        max(abs(left - right) for left, right in zip(transformed, direct)),
        max(abs(left - right) for left, right in zip(transformed, numpy_result)),
    )
    assert max_error < 1e-10
    print(f"DFT self-test passed; max absolute difference: {max_error:.3e}")
