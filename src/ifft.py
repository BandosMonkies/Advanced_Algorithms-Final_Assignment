"""Inverse FFT using the recursive FFT implementation.

The inverse transform uses the identity
``IFFT(x) = conjugate(FFT(conjugate(x))) / N``.  It uses the same
``w = exp(-2*pi*i/N)`` forward convention as :mod:`src.fft`.
"""

from __future__ import annotations

try:
    from src.fft import fft
except ModuleNotFoundError as error:
    if error.name != "src":
        raise
    from fft import fft


def ifft(x: list[complex]) -> list[complex]:
    """Return the inverse FFT in O(N log N) time and O(N) space."""
    if not x:
        raise ValueError("IFFT input must be non-empty.")
    N = len(x)
    transformed = fft([value.conjugate() for value in x])
    return [value.conjugate() / N for value in transformed]


if __name__ == "__main__":
    sample = [1 + 2j, -3 + 0.5j, 2 - 1j, 0j]
    recovered = ifft(fft(sample))
    assert max(abs(actual - expected) for actual, expected in zip(recovered, sample)) < 1e-9
    print("IFFT self-test passed.")
