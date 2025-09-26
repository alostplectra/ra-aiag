from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class OracleSettings:
    enabled: bool
    dsn: Optional[str]
    user: Optional[str]
    password: Optional[str]
    preview_query: Optional[str]


@dataclass(slots=True)
class FileSettings:
    enabled: bool
    path: Optional[Path]
    delimiter: str
    sheet_name: Optional[str]
    max_query_rows: Optional[int]


@dataclass(slots=True)
class AgentConfig:
    ollama_host: str
    model: str
    oracle: OracleSettings
    file: FileSettings


def _parse_max_query_rows(default: Optional[int] = 200) -> Optional[int]:
    raw = os.getenv("FILE_MAX_QUERY_ROWS")
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed <= 0:
        return None
    return parsed


def load_config() -> AgentConfig:
    oracle_enabled = os.getenv("ORACLE_ENABLED", "false").lower() == "true"
    file_enabled = os.getenv("FILE_ENABLED", "true").lower() == "true"

    oracle_settings = OracleSettings(
        enabled=oracle_enabled,
        dsn=os.getenv("ORACLE_DSN"),
        user=os.getenv("ORACLE_USER"),
        password=os.getenv("ORACLE_PASSWORD"),
        preview_query=os.getenv("ORACLE_PREVIEW_QUERY"),
    )

    file_path = os.getenv("FILE_PATH")
    file_settings = FileSettings(
        enabled=file_enabled,
        path=Path(file_path) if file_path else None,
        delimiter=os.getenv("FILE_DELIMITER", ","),
        sheet_name=os.getenv("FILE_SHEET_NAME"),
        max_query_rows=_parse_max_query_rows(),
    )

    return AgentConfig(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
        oracle=oracle_settings,
        file=file_settings,
    )

