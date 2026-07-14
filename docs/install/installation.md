# Installation

The research fork must be installed from its checkout; the package published
under `gridfm-graphkit` on PyPI may represent the upstream repository instead.

```bash
git clone https://github.com/cardef/gridfm-graphkit.git
cd gridfm-graphkit
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

`torch-scatter` / `torch-sparse` are **no longer required**: all scatter and
sparse operations use native PyTorch (`gridfm_graphkit/utils/scatter.py`,
parity-tested in `tests/test_native_scatter.py`).


For editable development, documentation, and unit testing:

```bash
python -m pip install -e ".[dev,test]"
```
