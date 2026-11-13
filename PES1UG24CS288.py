"""Run the polynomial DFT/FFT, RSA round-trip, and timing experiment."""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

from dft import dft_vandermonde
from fft import fft, ifft
from rsa_utils import (
    decode_component,
    encode_component,
    generate_keypair,
    rsa_decrypt,
    rsa_encrypt,
)


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sample.csv"
REPORT_DIR = ROOT / "report_data"
SIZES = (4, 8, 16, 32, 64, 128)


def load_polynomials(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load coefficient rows grouped by n in O(number of CSV rows) time."""
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            n = int(row["n"])
            grouped.setdefault(n, []).append(
                (
                    int(row["coefficient_index"]),
                    int(row["A_coefficient"]),
                    int(row["B_coefficient"]),
                )
            )
    result = {}
    for n, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        result[n] = (
            np.array([row[1] for row in rows], dtype=float),
            np.array([row[2] for row in rows], dtype=float),
        )
    return result


def encrypt_frequency_product(
    values: np.ndarray,
    public_key: tuple[int, int],
    private_key: tuple[int, int],
    scale: int = 1000,
    offset: int = 10**6,
) -> np.ndarray:
    """RSA round-trip real and imaginary parts in O(n * log(modulus)^3) time."""
    recovered = np.empty(len(values), dtype=complex)
    modulus = public_key[0]
    for index, value in enumerate(values):
        real_encoded = int(encode_component(float(value.real), scale, offset))
        imaginary_encoded = int(encode_component(float(value.imag), scale, offset))
        for encoded in (real_encoded, imaginary_encoded):
            if not 0 <= encoded < modulus:
                raise ValueError(
                    f"Encoded frequency component {encoded} must be in [0, {modulus})."
                )
        real = decode_component(
            rsa_decrypt(rsa_encrypt(real_encoded, public_key), private_key),
            scale,
            offset,
        )
        imaginary = decode_component(
            rsa_decrypt(rsa_encrypt(imaginary_encoded, public_key), private_key),
            scale,
            offset,
        )
        recovered[index] = real + 1j * imaginary
    return recovered


def save_and_plot(results: list[dict[str, float]]) -> None:
    """Save timing data and three plots in O(number of tested sizes) time."""
    REPORT_DIR.mkdir(exist_ok=True)
    with (REPORT_DIR / "timings.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("n", "dft_time", "fft_time", "speedup"))
        writer.writeheader()
        writer.writerows(results)

    n_values = np.array([row["n"] for row in results])
    dft_times = np.array([row["dft_time"] for row in results])
    fft_times = np.array([row["fft_time"] for row in results])
    speedups = dft_times / fft_times

    plt.figure()
    plt.plot(n_values, dft_times, "o-", label="DFT O(n²)")
    plt.plot(n_values, fft_times, "o-", label="FFT O(n log n)")
    plt.xlabel("n (number of coefficients)")
    plt.ylabel("time (seconds)")
    plt.title("DFT and FFT timing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "timing_linear.png")
    plt.close()

    plt.figure()
    plt.loglog(n_values, dft_times, "o-", label="DFT O(n²)")
    plt.loglog(n_values, fft_times, "o-", label="FFT O(n log n)")
    plt.xlabel("n (log scale)")
    plt.ylabel("time (seconds, log scale)")
    plt.title("DFT and FFT log-log timing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "timing_loglog.png")
    plt.close()

    plt.figure()
    plt.plot(n_values, speedups, "o-", label="DFT time / FFT time")
    plt.xlabel("n (number of coefficients)")
    plt.ylabel("speedup")
    plt.title("FFT speedup over DFT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "speedup.png")
    plt.close()


def main() -> None:
    """Run all polynomial sizes and report correctness and empirical growth."""
    data_path = DATA_PATH if DATA_PATH.exists() else ROOT / "data" / "Dataset.csv"
    polynomials = load_polynomials(data_path)
    # One cached 512-bit keypair is reused for every n.  128/256-bit keys use
    # the identical process, generated once each too, for the RSA report demo.
    public_key, private_key = generate_keypair(512)
    results = []

    for n in SIZES:
        print(f"\n\n======================= n={n} ===========================")
        a, b = polynomials[n]
        a_padded = np.pad(a, (0, n))
        b_padded = np.pad(b, (0, n))

        start = time.perf_counter()
        a_dft = dft_vandermonde(a_padded)
        b_dft = dft_vandermonde(b_padded)
        dft_time = time.perf_counter() - start

        start = time.perf_counter()
        a_fft = fft(a_padded)
        b_fft = fft(b_padded)
        fft_time = time.perf_counter() - start

        if n == 16:
            print(a_dft)
            print(b_dft)
            
        print(f"DFT time: {dft_time:.8f} seconds")
        print(f"FFT time: {fft_time:.8f} seconds")

        max_difference = max(
            np.max(np.abs(a_dft - a_fft)), np.max(np.abs(b_dft - b_fft))
        )
        print(f"Max diff DFT vs FFT: {max_difference:.3e}")

        product = a_fft * b_fft


        if n==16:
            print("Product = ", product)


        # The assignment's default offset is 10**6.  This dataset's largest
        # frequency product needs a larger offset after scale=1000.
        safe_offset = int(np.ceil(np.max(np.abs(product)) * 1000)) + 1
        recovered_product = encrypt_frequency_product(
            product, public_key, private_key, scale=1000, offset=safe_offset
        )
        print(
            "Max RSA quantization error: "
            f"{np.max(np.abs(product - recovered_product)):.3e}"
        )

        recovered = ifft(recovered_product)
        imaginary_error = np.max(np.abs(recovered.imag))
        rounded = np.rint(recovered.real[: 2 * n - 1]).astype(int)
        direct = np.convolve(a.astype(int), b.astype(int))
        print(f"Max inverse FFT imaginary error: {imaginary_error:.3e}")
        print(f"Convolution match: {np.array_equal(rounded, direct)}")
        if imaginary_error >= 1e-6:
            raise AssertionError("Inverse FFT imaginary error exceeded 1e-6.")
        results.append(
            {"n": n, "dft_time": dft_time, "fft_time": fft_time, "speedup": dft_time / fft_time}
        )

    save_and_plot(results)
    print("\n=== Timing summary ===")
    print("Saved: report_data/timings.csv, timing_linear.png, timing_loglog.png, speedup.png")
    dft_normalized = [row["dft_time"] / row["n"] ** 2 for row in results]
    fft_normalized = [
        row["fft_time"] / (row["n"] * np.log2(row["n"])) for row in results
    ]
    print(
        "Empirical growth summary: DFT time/n² ranges "
        f"{min(dft_normalized):.3e}-{max(dft_normalized):.3e}; "
        "FFT time/(n log2 n) ranges "
        f"{min(fft_normalized):.3e}-{max(fft_normalized):.3e}; "
        "time(n)/n is not constant."
    )


if __name__ == "__main__":
    main()
