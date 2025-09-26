from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class DataSource(ABC):
    """Base class for all data sources."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """Attempt to establish a connection to the underlying source."""

    @abstractmethod
    def fetch_preview(self, limit: int = 5) -> Any:
        """Pull a small sample of data for verification or prompting."""

    @property
    def is_connected(self) -> bool:
        return self._connected

    def mark_connected(self) -> None:
        self._connected = True

    def mark_disconnected(self) -> None:
        self._connected = False

    def get_metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "connected": self._connected}
