"""Small RSA utilities implemented for this assignment.

This is textbook RSA without padding, so encryption is deterministic and
128-bit moduli are insecure.  It is provided for the assignment only, not for
real-world cryptographic use.
"""

from __future__ import annotations

import math
import secrets
from typing import TypeAlias


PublicKey: TypeAlias = tuple[int, int]
PrivateKey: TypeAlias = tuple[int, int]
Keypair: TypeAlias = tuple[PublicKey, PrivateKey, int, int]


def is_probable_prime(n: int, rounds: int = 40) -> bool:
    """Return whether ``n`` passes Miller-Rabin in O(rounds * log³ n) time."""
    if n in (2, 3):
        return True
    if n < 2 or n % 2 == 0:
        return False

    odd_part = n - 1
    powers_of_two = 0
    while odd_part % 2 == 0:
        powers_of_two += 1
        odd_part //= 2

    small_primes = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small_primes:
        return True
    if any(n % prime == 0 for prime in small_primes):
        return False

    for _ in range(rounds):
        base = secrets.randbelow(n - 3) + 2
        value = pow(base, odd_part, n)
        if value in (1, n - 1):
            continue
        for _ in range(powers_of_two - 1):
            value = pow(value, 2, n)
            if value == n - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int) -> int:
    """Generate a probable prime with exactly ``bits`` bits."""
    if bits < 4:
        raise ValueError(f"Prime size must be at least 4 bits, got {bits}.")
    while True:
        candidate = secrets.randbits(bits)
        candidate |= (1 << (bits - 1)) | (1 << (bits - 2)) | 1
        if is_probable_prime(candidate):
            return candidate


def generate_keypair(modulus_bits: int) -> Keypair:
    """Generate ``(public, private, p, q)`` with an exact-size modulus."""
    if modulus_bits < 8 or modulus_bits % 2 != 0:
        raise ValueError("modulus_bits must be an even integer of at least 8.")

    prime_bits = modulus_bits // 2
    while True:
        p = generate_prime(prime_bits)
        q = generate_prime(prime_bits)
        if p == q:
            continue
        modulus = p * q
        if modulus.bit_length() != modulus_bits:
            continue
        phi = (p - 1) * (q - 1)
        exponent = 65537
        if math.gcd(exponent, phi) != 1:
            continue
        private_exponent = pow(exponent, -1, phi)
        return (modulus, exponent), (modulus, private_exponent), p, q


def encrypt_int(m: int, public_key: PublicKey) -> int:
    """Encrypt a non-negative integer smaller than N using textbook RSA."""
    modulus, exponent = public_key
    if not 0 <= m < modulus:
        raise ValueError(f"Message must satisfy 0 <= m < N; got m={m}.")
    return pow(m, exponent, modulus)


def decrypt_int(c: int, private_key: PrivateKey) -> int:
    """Decrypt an RSA ciphertext using textbook RSA."""
    modulus, exponent = private_key
    if not 0 <= c < modulus:
        raise ValueError(f"Ciphertext must satisfy 0 <= c < N; got c={c}.")
    return pow(c, exponent, modulus)


if __name__ == "__main__":
    for modulus_bits in (128, 256, 512):
        public_key, private_key, p, q = generate_keypair(modulus_bits)
        modulus, _ = public_key
        assert modulus.bit_length() == modulus_bits
        assert p != q and p.bit_length() == q.bit_length() == modulus_bits // 2
        edge_values = [0, 1, 2, modulus - 1]
        edge_values.extend(secrets.randbelow(modulus) for _ in range(3))
        for message in edge_values:
            assert decrypt_int(encrypt_int(message, public_key), private_key) == message
        print(f"RSA {modulus_bits}-bit self-test passed.")
