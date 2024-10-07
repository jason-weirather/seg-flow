try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version('seg-flow')
except PackageNotFoundError:
    # If the package is not installed, fall back to a hardcoded version
    __version__ = "0.0.99"
