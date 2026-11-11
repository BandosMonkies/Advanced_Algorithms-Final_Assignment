"""Recursive Cooley-Tukey FFT and inverse FFT implementations."""

import numpy as np


def _next_power_of_two(n: int) -> int:
    """Return the next power of two at least as large as n in O(log n) time."""
    if n < 1:
        raise ValueError("FFT input must contain at least one value.")
    size = 1
    while size < n:
        size *= 2
    return size


def _recursive_fft(values: np.ndarray, inverse: bool) -> np.ndarray:
    """Perform one recursive FFT pass in O(n log n) time."""
    n = len(values)
    if n == 1:
        return values.copy()

    angle = 2j * np.pi / n
    wn = np.exp(angle if inverse else -angle)
    w = 1.0 + 0j
    y_even = _recursive_fft(values[0::2], inverse)
    y_odd = _recursive_fft(values[1::2], inverse)
    result = np.zeros(n, dtype=complex)
    for k in range(n // 2):
        result[k] = y_even[k] + w * y_odd[k]
        result[k + n // 2] = y_even[k] - w * y_odd[k]
        w *= wn
    return result


def fft(a: np.ndarray) -> np.ndarray:
    """Return the recursive FFT in O(n log n) time, padding to a power of two."""
    values = np.asarray(a, dtype=complex)
    padded_size = _next_power_of_two(len(values))
    padded = np.pad(values, (0, padded_size - len(values)))
    return _recursive_fft(padded, inverse=False)


def ifft(y: np.ndarray) -> np.ndarray:
    """Return the inverse recursive FFT in O(n log n) time."""
    values = np.asarray(y, dtype=complex)
    padded_size = _next_power_of_two(len(values))
    padded = np.pad(values, (0, padded_size - len(values)))
    return _recursive_fft(padded, inverse=True) / padded_size
