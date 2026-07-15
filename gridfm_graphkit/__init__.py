"""GridFM GraphKit.

Built-in components are loaded lazily by :mod:`gridfm_graphkit.io.param_handler`.
Keeping package import side-effect free is required by the confirmatory
FM-scaling entry point: importing the package must not silently import the
legacy hierarchy or fitted per-grid normalizers.
"""

__all__: list[str] = []
