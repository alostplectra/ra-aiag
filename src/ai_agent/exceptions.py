class DataSourceConnectionError(RuntimeError):
    """Raised when a data source cannot be reached."""


class DataSourceQueryError(RuntimeError):
    """Raised when a data source query cannot be executed safely."""

