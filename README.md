# PASTA
 Pipeline for Automated Spectral Analysis.
 
 Example usage in `example_fit_hermes.py` and `check_fit_hermes.py`.

## Installation
The code is written in Rust, with an interface to Python. It can be installed as a Python package, either from the pre-built wheel or by compiling the source code.


### Pre-built wheel
A pre-built wheel for x86_64 Linux is provided under the Github [releases](https://github.com/SimonVanSchuylenbergh/PASTA/releases). It can be installed in the active Python environment with `pip`:
```sh
pip install pasta-0.1.0-cp37-abi3-manylinux_2_35_x86_64.whl
```

### Compiling from source
The Python package is built from source using [maturin](https://github.com/PyO3/maturin). To install in the active Python environment use
```sh
maturin develop --release
```
