"""Reversible fixed-point encoding for complex frequency-domain values.

For each complex value, its real and imaginary parts are encoded as separate
non-negative integers.  With ``S = 2**scale_bits``, a component ``x`` becomes
``q = round(x * S) + offset``.  The offset is computed from the data as
``ceil(max(abs(component)) * S) + 1``, which guarantees non-negative encoded
values.  The encoded vector is stored as
``[real_0, imag_0, real_1, imag_1, ...]``.

The encoding metadata contains the offset and scale.  Decoding reverses the
mapping, and the theoretical maximum absolute round-trip error per component
is ``0.5 / 2**scale_bits``.  The maximum encoded integer and its bit-width
are computed from the data; when a target RSA modulus is supplied, every
encoded integer must satisfy ``0 <= q < N``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EncodingParameters:
    """Metadata required to decode a fixed-point complex vector."""

    scale_bits: int
    offset: int
    max_encoded_value: int
    encoded_bit_width: int

    @property
    def theoretical_max_error(self) -> float:
        """Return the maximum theoretical absolute component error."""
        return 0.5 / (2**self.scale_bits)


def encode_complex_vector(
    values: list[complex],
    scale_bits: int = 32,
    modulus: int | None = None,
) -> tuple[list[int], EncodingParameters]:
    """Encode complex values in O(len(values)) time and space."""
    if scale_bits < 0:
        raise ValueError(f"scale_bits must be non-negative, got {scale_bits}.")
    if not values:
        parameters = EncodingParameters(scale_bits, 0, 0, 0)
        return [], parameters

    scale = 2**scale_bits
    max_component = max(
        (max(abs(value.real), abs(value.imag)) for value in values),
        default=0.0,
    )
    offset = math.ceil(max_component * scale) + 1
    encoded: list[int] = []
    for value in values:
        encoded.extend(
            [round(value.real * scale) + offset, round(value.imag * scale) + offset]
        )

    max_encoded_value = max(encoded, default=0)
    if modulus is not None and any(not 0 <= q < modulus for q in encoded):
        raise ValueError(
            f"Encoded value {max_encoded_value} does not fit in the target "
            f"modulus (N has {modulus.bit_length()} bits)."
        )
    parameters = EncodingParameters(
        scale_bits=scale_bits,
        offset=offset,
        max_encoded_value=max_encoded_value,
        encoded_bit_width=max_encoded_value.bit_length(),
    )
    return encoded, parameters


def decode_complex_vector(
    encoded: list[int], parameters: EncodingParameters
) -> list[complex]:
    """Decode interleaved real/imaginary integers in O(len(encoded)) time."""
    if len(encoded) % 2 != 0:
        raise ValueError("Encoded complex vectors must contain an even number of integers.")
    if any(value < 0 for value in encoded):
        raise ValueError("Encoded values must be non-negative.")
    scale = 2**parameters.scale_bits
    return [
        complex(
            (encoded[index] - parameters.offset) / scale,
            (encoded[index + 1] - parameters.offset) / scale,
        )
        for index in range(0, len(encoded), 2)
    ]


if __name__ == "__main__":
    sample = [1.25 - 2.5j, -3.0 + 0.125j]
    encoded, parameters = encode_complex_vector(sample)
    decoded = decode_complex_vector(encoded, parameters)
    assert max(abs(left - right) for left, right in zip(sample, decoded)) <= (
        parameters.theoretical_max_error * 2
    )
    print(
        "encoding self-test passed; theoretical max component error: "
        f"{parameters.theoretical_max_error:.3e}"
    )
