from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..exceptions import DataSourceConnectionError
from .base import DataSource

try:
    import oracledb  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    oracledb = None  # type: ignore


@dataclass(slots=True)
class OracleConfig:
    dsn: str
    user: str
    password: str
    preview_query: Optional[str] = None


class OracleDataSource(DataSource):
    def __init__(self, config: OracleConfig) -> None:
        super().__init__(name="oracle")
        self._config = config
        self._connection: Optional["oracledb.Connection"] = None

    def connect(self) -> None:
        if oracledb is None:
            raise DataSourceConnectionError("El paquete 'oracledb' no esta instalado en el entorno actual.")

        try:
            self._connection = oracledb.connect(
                dsn=self._config.dsn,
                user=self._config.user,
                password=self._config.password,
            )
            self._test_connection()
            self.mark_connected()
        except Exception as exc:  # pragma: no cover - safety for runtime failures
            self.mark_disconnected()
            raise DataSourceConnectionError(f"No se pudo conectar a Oracle: {exc}") from exc

    def _test_connection(self) -> None:
        if self._connection is None:
            raise DataSourceConnectionError("No hay conexion activa para validar.")
        test_query = "SELECT 1 FROM dual"
        with self._connection.cursor() as cursor:
            cursor.execute(test_query)
            cursor.fetchone()

    def fetch_preview(self, limit: int = 5) -> Sequence[tuple]:
        if not self.is_connected or self._connection is None:
            raise DataSourceConnectionError("La conexion a Oracle no esta activa.")

        preview_query = self._config.preview_query or "SELECT 1 FROM dual"
        limited_query = f"SELECT * FROM ({preview_query}) WHERE ROWNUM <= :limit"
        with self._connection.cursor() as cursor:
            cursor.execute(limited_query, limit=limit)
            return cursor.fetchall()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self.mark_disconnected()

