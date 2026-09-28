"""Package-wide exceptions."""


class PyTimeSeriesError(Exception):
    """Base class for all pytimeseries errors."""


class NotFittedError(PyTimeSeriesError):
    """Raised when predict/update is called before fit."""
