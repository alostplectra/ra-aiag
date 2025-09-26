from __future__ import annotations

import argparse
import sys

from .agent import DataAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el agente conectado a Oracle o archivos locales utilizando Ollama."
    )
    parser.add_argument("consulta", help="Pregunta o instruccion para el agente")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura para el modelo de Ollama (0.0 a 1.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agent = DataAgent()

    try:
        respuesta = agent.run(args.consulta, temperature=args.temperature)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error ejecutando el agente: {exc}", file=sys.stderr)
        return 1

    print(respuesta)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

