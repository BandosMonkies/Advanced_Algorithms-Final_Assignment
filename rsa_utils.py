"""Small, self-contained RSA helpers for the polynomial demonstration."""

import math
import secrets
from functools import lru_cache


def is_probable_prime(n: int, rounds: int = 20) -> bool:
    """Test probable primality with Miller-Rabin in O(rounds * log(n)^3) time."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small_primes:
        return True
    if n % 2 == 0:
        return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(rounds):
        base = secrets.randbelow(n - 3) + 2
        value = pow(base, d, n)
        if value in (1, n - 1):
            continue
        for _ in range(s - 1):
            value = pow(value, 2, n)
            if value == n - 1:
                break
        else:
            return False
    return True


def _random_prime(bit_length: int) -> int:
    """Generate a probable prime with the requested bit length."""
    while True:
        candidate = secrets.randbits(bit_length)
        candidate |= (1 << (bit_length - 1)) | 1
        if is_probable_prime(candidate):
            return candidate


@lru_cache(maxsize=None)
def generate_keypair(bits: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generate and cache an RSA keypair in expected O(log(n)^3) trial time."""
    if bits not in (128, 256, 512):
        raise ValueError("bits must be one of 128, 256, or 512.")
    half_bits = bits // 2
    while True:
        p = _random_prime(half_bits)
        q = _random_prime(half_bits)
        if p == q:
            continue
        modulus = p * q
        phi = (p - 1) * (q - 1)
        for exponent in (65537, 3, 5, 17, 257):
            if math.gcd(exponent, phi) == 1:
                private_exponent = pow(exponent, -1, phi)
                return ((modulus, exponent), (modulus, private_exponent))


def rsa_encrypt(m: int, public_key: tuple[int, int]) -> int:
    """Encrypt one integer in O(log(e) * log(n)^2) modular-arithmetic time."""
    modulus, exponent = public_key
    if not isinstance(m, int) or not 0 <= m < modulus:
        raise ValueError("RSA plaintext must be an integer in the range [0, modulus).")
    return pow(m, exponent, modulus)


def rsa_decrypt(c: int, private_key: tuple[int, int]) -> int:
    """Decrypt one integer in O(log(d) * log(n)^2) modular-arithmetic time."""
    modulus, exponent = private_key
    if not isinstance(c, int) or not 0 <= c < modulus:
        raise ValueError("RSA ciphertext must be an integer in the range [0, modulus).")
    return pow(c, exponent, modulus)


def encode_component(x: float, scale: int = 1000, offset: int = 10**6) -> int:
    """Map a float to a non-negative integer in O(1) time for RSA encoding."""
    # This maps a float to a non-negative int smaller than the modulus so RSA
    # (which only works on integers mod n) can encrypt it; offset must exceed
    # any possible |x*scale| in this dataset.
    return round(x * scale) + offset


def decode_component(y: int, scale: int = 1000, offset: int = 10**6) -> float:
    """Map an encoded integer back to a float in O(1) time."""
    return (y - offset) / scale
