# PASTA
 Pipeline for Automated Spectral Analysis

## Installation
### Python
This folder functions as a Python package that can be installed in an environement using pip:
    pip install -e .
The `-e` is optional and links the package to this folder such that modifications to the files are applied immediately without needing to reinstall the package.
The necessary dependencies are specified in the `environment.yaml` file

### Rust
Part of the code relies on functions written in Rust that provide bindings to Python.
A pre-compiled binary wheel is provided and can be installed with pip:
`pip intall `
