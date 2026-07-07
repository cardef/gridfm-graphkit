# Installation

The steps below mirror the [README](https://github.com/gridfm/gridfm-graphkit/blob/main/README.md#installation).

Create and activate a virtual environment (make sure you use the right python version = 3.10, 3.11 or 3.12. I highly recommend 3.12)

```bash
python -m venv venv
source venv/bin/activate
```

Install gridfm-graphkit from PyPI

```bash
pip install gridfm-graphkit
```

`torch-scatter` / `torch-sparse` are **no longer required**: all scatter and
sparse operations use native PyTorch (`gridfm_graphkit/utils/scatter.py`,
parity-tested in `tests/test_native_scatter.py`).


For documentation generation and unit testing, install with the optional `dev` and `test` extras:

```bash
pip install "gridfm-graphkit[dev,test]"
```
