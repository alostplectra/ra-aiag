<<<<<<< HEAD
# ra-aiag
=======
# Agente de IA con Ollama y fuentes Oracle/CSV

Este proyecto crea un agente que utiliza el modelo `gpt-oss-20b` desde una instancia local de Ollama. El agente intenta conectarse tanto a una base de datos Oracle como a un archivo CSV o Excel, usa las conexiones que se establezcan correctamente y devuelve un error cuando ninguna fuente esta disponible.

## Requisitos

- Python 3.10 o superior
- [Ollama](https://ollama.com/) ejecutandose localmente con el modelo `gpt-oss-20b`
- Dependencias del proyecto (`pip install -e .[excel]` si se quieren leer archivos de Excel)
- Cliente Oracle autonomo si se utiliza la biblioteca `oracledb` en modo cliente grueso (consultar la documentacion oficial)

## Instalacion

En sistemas tipo Unix (bash):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Para soporte Excel
pip install -e .[excel]
```

En Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
# Para soporte Excel
pip install -e .[excel]
```

## Configuracion

Las variables se pueden definir en un archivo `.env` en la raiz o como variables de entorno.

```env
# Conexion a Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss-20b

# Oracle
ORACLE_ENABLED=true
ORACLE_DSN=localhost/orclpdb1
ORACLE_USER=usuario
ORACLE_PASSWORD=clave
# Opcional: consulta usada para vista previa
ORACLE_PREVIEW_QUERY=SELECT * FROM tabla_ejemplo

# Archivo CSV o Excel
FILE_ENABLED=true
FILE_PATH=data/ejemplo.csv
FILE_DELIMITER=;
# Para Excel
FILE_SHEET_NAME=Hoja1
# Limite de filas en los resultados dinamicos (0 para sin limite)
FILE_MAX_QUERY_ROWS=200
```

> Si `ORACLE_ENABLED` o `FILE_ENABLED` estan en `false`, el agente omitira esa fuente.

## Uso

Una vez configurado, ejecutar:

```bash
python -m ai_agent.cli "Cual es el resumen de ventas del archivo?"
```

En Windows (PowerShell) la forma modular equivalente es:

```powershell
python -m ai_agent.cli "Cual es el resumen de ventas del archivo?"
```

O instalar el script y usar:

```bash
aio-agent "Resume la informacion del dataset"
```

En Windows el script instalado se invoca igual desde PowerShell:

```powershell
aio-agent "Resume la informacion del dataset"
```

El agente intentara conectarse primero a las fuentes configuradas y generara una respuesta en espanol usando el modelo de Ollama. Si ninguna fuente esta disponible, devolvera un mensaje de error detallando los problemas detectados.

### Calculos dinamicos sobre archivos

- Para archivos CSV y Excel, el agente genera un plan JSON (filtros, columnas derivadas, agrupaciones y agregaciones) y lo ejecuta directamente con pandas, sin motores SQL externos.
- El plan soporta agregaciones `sum`, `avg/mean`, `max`, `min` y `count`, filtros con operadores `==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `not_in`, y granularidades de fecha (a?o, trimestre, mes, semana, dia).
- El resultado del plan se incorpora al contexto de Ollama y se limita a `FILE_MAX_QUERY_ROWS` filas (ajustar a 0 para ver todas).
- Solo se permiten operaciones de lectura; si no se puede generar un plan valido, el agente recurrira a la vista previa limitada.

## Desarrollo

- El modulo principal esta en `src/ai_agent/agent.py`.
- Los conectores de datos estan en `src/ai_agent/data_sources/`.
- El cliente de Ollama se encuentra en `src/ai_agent/ollama_client.py`.

Contribuciones y mejoras son bienvenidas.

>>>>>>> 86bc446 (cargar proyecto)
