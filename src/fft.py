"""Recursive radix-2 Cooley-Tukey FFT.

The forward transform uses ``w = exp(-2*pi*i/n)``.  The implementation
recursively splits the input into even and odd indices and combines the
sub-transforms with butterflies.
"""

from __future__ import annotations

import cmath
import math


_last_complex_multiplications = 0


def _is_power_of_two(n: int) -> bool:
    """Return whether ``n`` is a positive power of two."""
    return n > 0 and n & (n - 1) == 0


def _fft_recursive(a: list[complex]) -> list[complex]:
    """Compute one recursive FFT after input-size validation."""
    n = len(a)
    if n == 1:
        return a.copy()

    even = _fft_recursive(a[0::2])
    odd = _fft_recursive(a[1::2])
    root = cmath.exp(-2j * math.pi / n)
    result = [0j] * n

    global _last_complex_multiplications
    for k in range(n // 2):
        twiddle = root**k
        product = twiddle * odd[k]
        _last_complex_multiplications += 1
        result[k] = even[k] + product
        result[k + n // 2] = even[k] - product
    return result


def fft(a: list[int] | list[complex]) -> list[complex]:
    """Return the radix-2 FFT of ``a`` in O(n log n) time.

    The input length must be a positive power of two.  The latest call's
    complex multiplication count is available through
    :func:`get_last_complex_multiplications`.
    """
    n = len(a)
    if not _is_power_of_two(n):
        raise ValueError(f"FFT input length must be a positive power of two, got {n}.")

    global _last_complex_multiplications
    _last_complex_multiplications = 0
    return _fft_recursive([complex(value) for value in a])


def get_last_complex_multiplications() -> int:
    """Return the number of complex multiplications in the latest FFT call."""
    return _last_complex_multiplications


def complex_multiplication_count(n: int) -> int:
    """Return the radix-2 FFT's n/2 * log2(n) multiplication count."""
    if not _is_power_of_two(n):
        raise ValueError(f"FFT size must be a positive power of two, got {n}.")
    return (n // 2) * int(math.log2(n))


if __name__ == "__main__":
    import numpy as np

    sample = [2, 2, 9, 1]
    actual = fft(sample)
    expected = np.fft.fft(sample)
    max_error = max(abs(left - right) for left, right in zip(actual, expected))
    assert max_error < 1e-10
    assert get_last_complex_multiplications() == complex_multiplication_count(4)
    print(f"FFT self-test passed; max absolute difference: {max_error:.3e}")
