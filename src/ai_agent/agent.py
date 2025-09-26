from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Mapping, Optional

from .config import AgentConfig, FileSettings, OracleSettings, load_config
from .data_sources import (
    DataSource,
    OracleConfig,
    OracleDataSource,
    QueryResult,
    TabularFileDataSource,
)
from .exceptions import DataSourceConnectionError, DataSourceQueryError
from .ollama_client import OllamaClient


class DataAgent:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or load_config()
        self._ollama = OllamaClient(self._config.ollama_host, self._config.model)
        self._sources: List[DataSource] = []
        self._configured_errors: List[str] = []
        self._bootstrap_sources()

    def _bootstrap_sources(self) -> None:
        oracle_cfg = self._config.oracle
        file_cfg = self._config.file

        oracle_source = self._maybe_create_oracle(oracle_cfg)
        if oracle_source:
            self._sources.append(oracle_source)

        file_source = self._maybe_create_file(file_cfg)
        if file_source:
            self._sources.append(file_source)

    def _maybe_create_oracle(self, cfg: OracleSettings) -> OracleDataSource | None:
        if not cfg.enabled:
            return None
        if not all([cfg.dsn, cfg.user, cfg.password]):
            self._configured_errors.append(
                "Oracle esta habilitado pero faltan credenciales (ORACLE_DSN, ORACLE_USER, ORACLE_PASSWORD)."
            )
            return None
        oracle_config = OracleConfig(
            dsn=cfg.dsn,
            user=cfg.user,
            password=cfg.password,
            preview_query=cfg.preview_query,
        )
        return OracleDataSource(config=oracle_config)

    def _maybe_create_file(self, cfg: FileSettings) -> TabularFileDataSource | None:
        if not cfg.enabled:
            return None
        if cfg.path is None:
            self._configured_errors.append("FILE_PATH no esta definido.")
            return None
        return TabularFileDataSource(
            path=cfg.path,
            delimiter=cfg.delimiter,
            sheet_name=cfg.sheet_name,
            name=f"archivo:{cfg.path.name}",
            max_query_rows=cfg.max_query_rows,
        )

    def connect_sources(self) -> tuple[List[DataSource], List[str]]:
        connected: List[DataSource] = []
        errors: List[str] = list(self._configured_errors)

        for source in self._sources:
            try:
                source.connect()
                connected.append(source)
            except DataSourceConnectionError as exc:
                errors.append(str(exc))

        return connected, errors

    def run(self, user_query: str, temperature: float = 0.0) -> str:
        connected, errors = self.connect_sources()

        if not connected:
            details = "; ".join(errors) if errors else "Sin detalles adicionales"
            return f"Error de conexion a la fuente: {details}"

        dynamic_context = self._collect_dynamic_context(connected, user_query)
        context = self._build_context_snippets(connected, dynamic_context)
        system_prompt = (
            "Eres un asistente de analitica de datos. Usa unicamente los datos que se reciben en el contexto. "
            "Si la informacion no es suficiente, explica la limitacion. Responde siempre en espanol claro."
        )
        user_prompt = (
            "Contexto de datos:\n"
            f"{context}\n\n"
            "Instrucciones: analiza el contexto previo y responde a la pregunta del usuario."
        )
        full_prompt = f"{user_prompt}\n\nPregunta del usuario: {user_query}"

        return self._ollama.simple_chat(system_prompt=system_prompt, user_prompt=full_prompt, temperature=temperature)

    def _collect_dynamic_context(
        self, sources: Iterable[DataSource], user_query: str
    ) -> Dict[str, List[str]]:
        context_parts: Dict[str, List[str]] = {}

        for source in sources:
            if isinstance(source, TabularFileDataSource):
                plan = self._generate_plan_for_file(source, user_query)
                if plan is None:
                    continue
                try:
                    result = source.execute_plan(plan)
                except (DataSourceQueryError, DataSourceConnectionError) as exc:
                    context_parts.setdefault(source.name, []).append(
                        f"Calculo dinamico fallido: {exc}"
                    )
                    continue
                context_parts.setdefault(source.name, []).extend(
                    self._format_plan_result(result)
                )
        return context_parts

    def _generate_plan_for_file(self, source: TabularFileDataSource, user_query: str) -> Optional[Mapping[str, object]]:
        schema_lines = [f"{name} ({dtype})" for name, dtype in source.schema_signature]
        schema_block = "\n".join(schema_lines) if schema_lines else "No disponible"
        row_info = source.row_count if source.row_count is not None else "desconocido"
        table_hint = (
            f"Puedes usar el nombre de tabla dataset o el alias {source.table_alias}."
            if hasattr(source, "table_alias")
            else "Usa el conjunto de datos tal como se describe."
        )

        preview_df = source.fetch_preview(limit=5)
        try:
            preview_block = preview_df.to_markdown()
        except Exception:
            preview_block = preview_df.to_string()

        system_prompt = (
            "Eres un planificador de analisis de datos. Dispones de TODAS las filas del dataset. "
            "Debes convertir la instruccion en un unico PLAN JSON que describe filtros, columnas derivadas, "
            "agrupaciones y agregaciones usando el esquema proporcionado. NUNCA respondas con texto libero ni digas "
            "que faltan datos.\n\n"
            "Tu respuesta DEBE ser un unico objeto JSON con las claves opcionales:\n"
            "{\n"
            "  \"filters\": [ { \"column\": str, \"operator\": str, \"value\": any } ],\n"
            "  \"derived_columns\": [ { \"name\": str, \"expression\": str } ],\n"
            "  \"group_by\": [ { \"column\": str, \"granularity\": str | null, \"alias\": str | null } ],\n"
            "  \"aggregations\": [ { \"target\": str, \"function\": str, \"alias\": str | null } ],\n"
            "  \"select\": [str],\n"
            "  \"sort\": [ { \"by\": str, \"direction\": \"asc\"|\"desc\" } ],\n"
            "  \"limit\": int\n"
            "}\n"
            "Reglas adicionales:\n"
            "- Usa solo funciones sum, avg, mean, max, min, count.\n"
            "- Usa operadores ==, !=, >, <, >=, <=, in, not_in.\n"
            "- Las expresiones derivadas solo pueden usar columnas y aritmetica (+, -, *, /, parentesis).\n"
            "- Cuando la pregunta hable de importe, puedes crear una columna derivada como Quantity * UnitPrice.\n"
            "- Para agrupar por periodos, define una granularidad (year, quarter, month, week, day).\n"
            "- Incluye siempre las columnas relevantes en \"select\" y ordena si se solicita comparar tendencias.\n"
            "- Si realmente la pregunta no se puede responder con los datos, responde exactamente NO_PLAN.\n"
            "- De lo contrario, produce el JSON sin comentarios ni texto adicional."
        )
        user_prompt = (
            "Esquema aproximado (columna y tipo):\n"
            f"{schema_block}\n"
            f"Filas aproximadas: {row_info}\n"
            f"{table_hint}\n"
            "Muestra de datos:\n"
            f"{preview_block}\n\n"
            "Instruccion del usuario:\n"
            f"{user_query}\n\n"
            "Responde solo con el objeto JSON descrito o NO_PLAN."
        )

        response = self._ollama.simple_chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)
        plan = self._extract_plan(response)
        return plan

    def _extract_plan(self, response: str) -> Optional[Mapping[str, object]]:
        cleaned = response.strip()
        if cleaned.upper() == "NO_PLAN":
            return None
        fence_match = re.search(r"```json\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        json_blob = fence_match.group(1) if fence_match else cleaned
        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _format_plan_result(self, result: QueryResult) -> List[str]:
        rows = result.rows
        total_rows = len(rows)
        max_rows = 200
        display_rows = rows.head(max_rows) if total_rows > max_rows else rows
        try:
            table_markdown = display_rows.to_markdown()
        except Exception:
            table_markdown = display_rows.to_string()
        summary_line = f"Filas del resultado: {total_rows}"
        if total_rows > max_rows:
            summary_line += f" (se muestran las primeras {max_rows})"
        return [
            "Plan ejecutado:",
            result.description,
            summary_line,
            "Resultado del plan:",
            table_markdown,
        ]

    def _build_context_snippets(
        self, sources: Iterable[DataSource], dynamic_context: Dict[str, List[str]]
    ) -> str:
        snippets: List[str] = []
        for source in sources:
            parts: List[str] = [f"Fuente {source.name}:"]
            if isinstance(source, TabularFileDataSource):
                parts.append(source.schema_overview())
                summary_snippets = source.summary_snippets()
                if summary_snippets:
                    parts.append("Resumen estadistico:")
                    parts.extend(summary_snippets)
            try:
                preview = source.fetch_preview(limit=5)
            except Exception as exc:
                parts.append(f"Error al obtener vista previa ({exc}).")
            else:
                try:
                    formatted_preview = preview.to_markdown()
                except Exception:
                    formatted_preview = str(preview)
                parts.append("Vista previa:")
                parts.append(formatted_preview)

            if source.name in dynamic_context:
                parts.extend(dynamic_context[source.name])

            snippets.append("\n".join(parts))
        return "\n\n".join(snippets)

