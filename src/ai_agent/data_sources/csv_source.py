from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from ..exceptions import DataSourceConnectionError, DataSourceQueryError
from .base import DataSource

SUPPORTED_CSV_SUFFIXES = {".csv", ".txt"}
SUPPORTED_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}
ALLOWED_AGGREGATIONS = {"sum": "sum", "avg": "mean", "mean": "mean", "max": "max", "min": "min", "count": "count"}
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SAFE_EXPRESSION = re.compile(r"^[A-Za-z0-9_\s\+\-\*/\(\)\.]+$")


def _normalize_aggregation(name: str) -> str:
    key = name.lower()
    if key not in ALLOWED_AGGREGATIONS:
        raise DataSourceQueryError(f"Funcion de agregacion no soportada: {name}")
    return ALLOWED_AGGREGATIONS[key]


def _ensure_identifier(identifier: str, field_name: str) -> str:
    if not SAFE_IDENTIFIER.match(identifier):
        raise DataSourceQueryError(f"{field_name} invalido: {identifier}")
    return identifier


def _ensure_expression(expression: str) -> str:
    if not SAFE_EXPRESSION.match(expression.replace(" ** ", "")):
        raise DataSourceQueryError("Expresion contiene caracteres no permitidos.")
    return expression


def _to_datetime(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce")
    if converted.isna().all():
        raise DataSourceQueryError("No se pudo convertir la columna a fecha/hora para usar granularidades.")
    return converted


def _build_alias(path: Path) -> str:
    stem = path.stem.lower()
    alias = re.sub(r"\W+", "_", stem)
    if not alias:
        alias = "dataset"
    if alias[0].isdigit():
        alias = f"t_{alias}"
    return alias


@dataclass(slots=True)
class QueryResult:
    description: str
    rows: pd.DataFrame


class TabularFileDataSource(DataSource):
    def __init__(
        self,
        path: str | Path,
        delimiter: str = ",",
        sheet_name: Optional[str] = None,
        name: str = "archivo",
        preview_rows: int = 5,
        max_query_rows: Optional[int] = 200,
    ) -> None:
        super().__init__(name=name)
        self._path = Path(path)
        self._delimiter = delimiter
        self._sheet_name = sheet_name
        self._preview_rows = preview_rows
        self._max_query_rows = max_query_rows
        self._dataframe: Optional[pd.DataFrame] = None
        self._schema_signature: list[tuple[str, str]] = []
        self._cached_preview: Optional[pd.DataFrame] = None
        self._table_alias = _build_alias(self._path)

    @property
    def table_alias(self) -> str:
        return self._table_alias

    def connect(self) -> None:
        if not self._path.exists():
            raise DataSourceConnectionError(f"El archivo {self._path} no existe.")

        try:
            df = self._load_full_dataframe()
            self._dataframe = df
            self._schema_signature = [(column, str(dtype)) for column, dtype in df.dtypes.items()]
            self._cached_preview = df.head(self._preview_rows)
            self.mark_connected()
        except Exception as exc:  # pragma: no cover - seguridad en runtime
            self._dataframe = None
            self._schema_signature = []
            self._cached_preview = None
            self.mark_disconnected()
            raise DataSourceConnectionError(f"No se pudo leer el archivo {self._path}: {exc}") from exc

    def _load_full_dataframe(self) -> pd.DataFrame:
        suffix = self._path.suffix.lower()
        if suffix in SUPPORTED_CSV_SUFFIXES:
            return pd.read_csv(self._path, delimiter=self._delimiter)
        if suffix in SUPPORTED_EXCEL_SUFFIXES:
            return pd.read_excel(self._path, sheet_name=self._sheet_name)
        raise DataSourceConnectionError(f"El formato {suffix} no esta soportado.")

    def _get_dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            self._dataframe = self._load_full_dataframe()
        return self._dataframe.copy()

    def fetch_preview(self, limit: int = 5) -> pd.DataFrame:
        if not self.is_connected:
            raise DataSourceConnectionError("La conexion al archivo no esta activa.")

        if self._cached_preview is None or len(self._cached_preview) < limit:
            df = self._get_dataframe()
            return df.head(limit)
        return self._cached_preview.head(limit)

    @property
    def schema_signature(self) -> list[tuple[str, str]]:
        return list(self._schema_signature)

    @property
    def row_count(self) -> Optional[int]:
        if self._dataframe is not None:
            return len(self._dataframe)
        try:
            df = self._load_full_dataframe()
        except Exception:
            return None
        self._dataframe = df
        return len(df)

    def execute_plan(self, plan: Mapping[str, Any], limit: Optional[int] = None) -> QueryResult:
        if not self.is_connected:
            raise DataSourceConnectionError("La conexion al archivo no esta activa.")

        try:
            df = self._apply_plan(self._get_dataframe(), plan)
        except DataSourceQueryError:
            raise
        except Exception as exc:  # pragma: no cover
            raise DataSourceQueryError(f"Fallo al ejecutar el plan: {exc}") from exc

        effective_limit = self._resolve_limit(limit)
        if effective_limit is not None:
            df = df.head(effective_limit)

        description = json.dumps(plan, ensure_ascii=False)
        return QueryResult(description=description, rows=df)

    def _resolve_limit(self, limit: Optional[int]) -> Optional[int]:
        if limit is not None and limit > 0:
            if self._max_query_rows is not None:
                return min(limit, self._max_query_rows)
            return limit
        if limit is not None and limit <= 0:
            return None
        return self._max_query_rows

    def _apply_plan(self, df: pd.DataFrame, plan: Mapping[str, Any]) -> pd.DataFrame:
        working = df.copy()

        for derived in plan.get("derived_columns", []):
            name = _ensure_identifier(str(derived.get("name")), "Nombre de columna derivada")
            expression = _ensure_expression(str(derived.get("expression")))
            working[name] = working.eval(expression, engine="python")

        for filt in plan.get("filters", []):
            column = _ensure_identifier(str(filt.get("column")), "Columna de filtro")
            if column not in working.columns:
                raise DataSourceQueryError(f"La columna {column} no existe para aplicar filtro.")
            operator = str(filt.get("operator"))
            value = filt.get("value")
            working = self._apply_filter(working, column, operator, value)

        group_instructions = plan.get("group_by") or []
        if group_instructions:
            group_columns: List[str] = []
            grouping_df = working
            for instruction in group_instructions:
                column = _ensure_identifier(str(instruction.get("column")), "Columna de agrupacion")
                if column not in working.columns:
                    raise DataSourceQueryError(f"La columna {column} no existe para agrupar.")
                granularity = instruction.get("granularity")
                alias = instruction.get("alias")
                if granularity:
                    alias_name = _ensure_identifier(alias, "Alias de agrupacion") if alias else f"{column}_{granularity}"
                    grouping_df = grouping_df.assign(
                        **{alias_name: self._apply_granularity(working[column], granularity)}
                    )
                    group_columns.append(alias_name)
                else:
                    group_columns.append(column)
            agg_defs = plan.get("aggregations") or []
            if not agg_defs:
                raise DataSourceQueryError("Se requieren agregaciones cuando se agrupa.")
            named_aggs: Dict[str, tuple[str, str]] = {}
            for agg in agg_defs:
                target = _ensure_identifier(str(agg.get("target")), "Columna de agregacion")
                if target not in grouping_df.columns:
                    raise DataSourceQueryError(f"La columna {target} no existe para agregar.")
                function = _normalize_aggregation(str(agg.get("function")))
                alias = agg.get("alias") or f"{target}_{function}"
                alias_name = _ensure_identifier(str(alias), "Alias de agregacion")
                named_aggs[alias_name] = (target, function)
            aggregated = grouping_df.groupby(group_columns, dropna=False).agg(**named_aggs).reset_index()
            working = aggregated
        else:
            agg_defs = plan.get("aggregations") or []
            if agg_defs:
                named_aggs = {}
                for agg in agg_defs:
                    target = _ensure_identifier(str(agg.get("target")), "Columna de agregacion")
                    if target not in working.columns:
                        raise DataSourceQueryError(f"La columna {target} no existe para agregar.")
                    function = _normalize_aggregation(str(agg.get("function")))
                    alias = agg.get("alias") or f"{target}_{function}"
                    alias_name = _ensure_identifier(str(alias), "Alias de agregacion")
                    named_aggs[alias_name] = (target, function)
                working = working.agg(**named_aggs)
                if isinstance(working, pd.Series):
                    working = working.to_frame().T

        select_columns = plan.get("select")
        if select_columns:
            columns = [_ensure_identifier(str(col), "Columna a seleccionar") for col in select_columns]
            missing = [col for col in columns if col not in working.columns]
            if missing:
                raise DataSourceQueryError(f"Las columnas {missing} no existen en el resultado de la operacion.")
            working = working[columns]

        for sort_instruction in plan.get("sort", []):
            column = _ensure_identifier(str(sort_instruction.get("by")), "Columna de ordenamiento")
            if column not in working.columns:
                raise DataSourceQueryError(f"La columna {column} no existe para ordenar.")
            ascending = str(sort_instruction.get("direction", "asc")).lower() != "desc"
            working = working.sort_values(by=column, ascending=ascending)

        limit = plan.get("limit")
        if isinstance(limit, int) and limit > 0:
            working = working.head(limit)

        return working.reset_index(drop=True)

    def _apply_filter(self, df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.DataFrame:
        series = df[column]
        op_map = {
            "==": "eq",
            "!=": "ne",
            ">": "gt",
            "<": "lt",
            ">=": "ge",
            "<=": "le",
        }
        if operator in op_map:
            method = getattr(series, op_map[operator])
            try:
                condition = method(value)
            except Exception as exc:
                raise DataSourceQueryError(f"No se pudo aplicar el filtro {column} {operator} {value}: {exc}") from exc
            return df[condition]
        if operator == "in":
            if not isinstance(value, list):
                raise DataSourceQueryError("El valor para el operador 'in' debe ser una lista.")
            return df[series.isin(value)]
        if operator == "not_in":
            if not isinstance(value, list):
                raise DataSourceQueryError("El valor para el operador 'not_in' debe ser una lista.")
            return df[~series.isin(value)]
        raise DataSourceQueryError(f"Operador de filtro no soportado: {operator}")

    def _apply_granularity(self, series: pd.Series, granularity: str) -> pd.Series:
        granularity_key = str(granularity).lower()
        datetime_series = _to_datetime(series)
        if granularity_key == "year":
            return datetime_series.dt.year
        if granularity_key == "quarter":
            return datetime_series.dt.to_period("Q").astype(str)
        if granularity_key == "month":
            return datetime_series.dt.to_period("M").astype(str)
        if granularity_key == "week":
            return datetime_series.dt.isocalendar().week
        if granularity_key in {"day", "date"}:
            return datetime_series.dt.date
        raise DataSourceQueryError(f"Granularidad de fecha no soportada: {granularity}")

    def schema_overview(self) -> str:
        total = f"Filas aproximadas: {self.row_count}" if self.row_count is not None else "Filas: desconocidas"
        parts = [f"- {name}: {dtype}" for name, dtype in self._schema_signature]
        return f"{total}\n" + "\n".join(parts)

