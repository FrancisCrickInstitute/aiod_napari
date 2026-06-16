from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aiod-napari")
except PackageNotFoundError:
    __version__ = "unknown"
