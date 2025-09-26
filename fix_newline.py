from pathlib import Path

path = Path("src/ai_agent/data_sources/csv_source.py")
text = path.read_text()
old = "        insights.append(\r\n            \"Top periodos por ingresos:\r\n\" + self._format_table(aggregated)\r\n        )\r\n"
new = "        insights.append(\n            \"Top periodos por ingresos:\\n\" + self._format_table(aggregated)\n        )\n"
if old not in text:
    raise SystemExit("Patron no encontrado para la cadena de ingresos")
text = text.replace(old, new, 1)
path.write_text(text)
