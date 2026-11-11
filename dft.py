"""Direct Vandermonde-matrix implementations of the DFT and inverse DFT."""

import numpy as np


def dft_vandermonde(a: np.ndarray) -> np.ndarray:
    """Return the DFT of ``a`` in O(n^2) time using an explicit matrix."""
    values = np.asarray(a, dtype=complex)
    n = len(values)
    omega = np.exp(-2j * np.pi / n)
    matrix = np.empty((n, n), dtype=complex)
    for j in range(n):
        for k in range(n):
            matrix[j, k] = omega ** (j * k)
    return matrix @ values


def idft_vandermonde(y: np.ndarray) -> np.ndarray:
    """Return the inverse DFT of ``y`` in O(n^2) time using an explicit matrix."""
    values = np.asarray(y, dtype=complex)
    n = len(values)
    omega = np.exp(2j * np.pi / n)
    matrix = np.empty((n, n), dtype=complex)
    for j in range(n):
        for k in range(n):
            matrix[j, k] = omega ** (j * k)
    return (matrix @ values) / n
