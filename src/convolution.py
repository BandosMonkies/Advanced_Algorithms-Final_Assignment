"""Naive integer polynomial convolution.

Polynomials are represented by coefficient lists in ascending order.
"""

from __future__ import annotations


def naive_convolution(A: list[int], B: list[int]) -> list[int]:
    """Return the coefficient list for ``A(x) * B(x)``.

    Time complexity is O(len(A) * len(B)); extra space is O(len(A) + len(B)).
    """
    if not A or not B:
        return []

    result = [0] * (len(A) + len(B) - 1)
    for i, a_coefficient in enumerate(A):
        for j, b_coefficient in enumerate(B):
            result[i + j] += a_coefficient * b_coefficient
    return result


if __name__ == "__main__":
    assert naive_convolution([1, 2], [3, 4]) == [3, 10, 8]
    print("naive_convolution self-test passed.")
