# UE24CS342AA1 Advanced Algorithms

This project implements fast polynomial multiplication using a Vandermonde
DFT, a recursive radix-2 FFT, pointwise multiplication, textbook RSA
encryption of frequency-domain values, and inverse-FFT validation.

## Requirements

Python 3.10+ with:

- `numpy`
- `matplotlib`
- `pytest`

The provided project environment can be used with `.venv/bin/python`.

## Running

Run the complete integration pipeline:

```bash
.venv/bin/python main.py
```

Run individual stages:

```bash
.venv/bin/python src/data_loader.py
.venv/bin/python src/dft.py
.venv/bin/python src/fft.py
.venv/bin/python src/pointwise.py
.venv/bin/python src/rsa_utils.py
.venv/bin/python src/encoding.py
.venv/bin/python src/ifft.py
```

Run the experiment scripts:

```bash
.venv/bin/python -m experiments.step1_dft
.venv/bin/python -m experiments.step2_fft
.venv/bin/python -m experiments.step3_pointwise
.venv/bin/python -m experiments.step4_rsa
.venv/bin/python -m experiments.step5_validate
.venv/bin/python -m experiments.scale_sweep
```

Run the test suite:

```bash
.venv/bin/python -m pytest -q
```

## Project structure

```text
data/Dataset.csv                 Input polynomial coefficients
src/data_loader.py               CSV loading and validation
src/convolution.py               O(n^2) reference convolution
src/dft.py                       Vandermonde DFT
src/fft.py                       Recursive radix-2 FFT
src/pointwise.py                 Padding and frequency multiplication
src/rsa_utils.py                 Assignment RSA implementation
src/encoding.py                  Complex fixed-point encoding
src/ifft.py                      Inverse FFT
experiments/                     Per-step benchmarks and validation
tests/                           Pytest tests
results/                         CSV and plot outputs
main.py                          End-to-end integration runner
```

## Transform convention and padding

Polynomials use ascending coefficient order:

```text
A(x) = A[0] + A[1]x + ...
```

The forward transform uses:

```text
w = exp(-2*pi*i/N)
```

The inverse uses the conjugate-root identity:

```text
IFFT(x) = conjugate(FFT(conjugate(x))) / N
```

For two degree-`n-1` inputs, the product has `2n-1` coefficients.  Each input
is therefore zero-padded to `N = 2n` before the FFT.  Since the dataset sizes
are powers of two, `2n` is also a valid radix-2 FFT length.  Without padding,
frequency-domain multiplication produces circular convolution and wraps the
high-degree coefficients.

## Complex-value encoding

Each frequency-domain value is encoded as two separate integers.  For a
component `x`, with `S = 2**scale_bits`:

```text
q = round(x * S) + offset
```

The offset is computed from the maximum absolute component so all encoded
values are non-negative.  Values are stored as:

```text
[real_0, imag_0, real_1, imag_1, ...]
```

Decoding subtracts the offset and divides by `S`.  The theoretical maximum
component quantization error is `0.5 / 2**scale_bits`.

