"""Frequency-domain pointwise multiplication for polynomial products.

The polynomial coefficient convention is ascending order.  Inputs are
zero-padded to ``N = 2n`` before applying the forward FFT, so the resulting
pointwise product represents the ordinary (linear) product rather than a
wrapped circular convolution.
"""

from __future__ import annotations

try:
    from src.fft import fft
except ModuleNotFoundError as error:
    if error.name != "src":
        raise
    from fft import fft


def pad_to_length(v: list[int] | list[complex], N: int) -> list[int | complex]:
    """Return ``v`` zero-padded to length ``N`` in O(N) time and space."""
    if N < len(v):
        raise ValueError(
            f"Target length N={N} cannot be smaller than vector length {len(v)}."
        )
    return list(v) + [0] * (N - len(v))


def pointwise_multiply(
    A_hat: list[complex], B_hat: list[complex]
) -> list[complex]:
    """Multiply two frequency-domain vectors element by element in O(N) time."""
    if len(A_hat) != len(B_hat):
        raise ValueError(
            "Frequency-domain vectors must have equal lengths, "
            f"got {len(A_hat)} and {len(B_hat)}."
        )
    return [a_value * b_value for a_value, b_value in zip(A_hat, B_hat)]


def pipeline_step3(
    A: list[int], B: list[int]
) -> tuple[int, list[complex], list[complex], list[complex]]:
    """Pad A and B to 2n, FFT them, and multiply their spectra.

    Time complexity is O(n log n), and the returned tuple is
    ``(N, A_hat, B_hat, C_hat)``.
    """
    if not A or not B:
        raise ValueError("Both input polynomials must be non-empty.")
    if len(A) != len(B):
        raise ValueError(
            f"Input polynomials must have equal lengths, got {len(A)} and {len(B)}."
        )

    N = 2 * len(A)
    A_hat = fft(pad_to_length(A, N))
    B_hat = fft(pad_to_length(B, N))
    C_hat = pointwise_multiply(A_hat, B_hat)
    return N, A_hat, B_hat, C_hat


if __name__ == "__main__":
    try:
        from src.convolution import naive_convolution
    except ModuleNotFoundError as error:
        if error.name != "src":
            raise
        from convolution import naive_convolution

    A = [1, 2]
    B = [3, 4]
    N, _, _, C_hat = pipeline_step3(A, B)
    assert N == 4
    assert len(C_hat) == N
    assert pad_to_length(A, 4) == [1, 2, 0, 0]
    assert pointwise_multiply([1 + 1j], [2 - 1j]) == [3 + 1j]
    assert len(naive_convolution(A, B)) == 3
    print("pointwise self-test passed.")
