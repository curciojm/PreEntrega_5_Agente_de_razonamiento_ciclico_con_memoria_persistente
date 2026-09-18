# Agente de razonamiento cíclico con memoria persistente (Pre-Entrega 5)

Proyecto correspondiente a la Pre-Entrega 5 del curso de AI Engineering.

## Descripción

Agente de razonamiento cíclico implementado con LangGraph, LangChain y Google Gemini, con capacidad para utilizar herramientas de recuperación de información y mantener memoria persistente entre interacciones mediante un identificador de conversación (`thread_id`).

El proyecto implementa un agente basado en un grafo de estados que permite al modelo decidir de manera autónoma cuándo utilizar una herramienta, procesar el resultado obtenido y continuar el ciclo hasta generar una respuesta final.

El sistema utiliza herramientas de recuperación híbrida sobre documentos académicos relacionados con metodología de la investigación y estadística.

El agente permite:

* Recibir consultas en lenguaje natural.
* Decidir de manera autónoma cuándo utilizar una herramienta.
* Consultar información mediante herramientas especializadas.
* Buscar conceptos y explicaciones en los documentos disponibles.
* Buscar fuentes y páginas relacionadas con un tema.
* Ejecutar múltiples llamadas a herramientas dentro de una misma interacción.
* Procesar los resultados de las herramientas y continuar el ciclo.
* Mantener el historial de mensajes entre diferentes turnos mediante `thread_id`.
* Persistir el estado del agente utilizando SQLite.
* Controlar el número máximo de iteraciones mediante `recursion_limit`.
* Registrar eventos relevantes durante la ejecución.
* Generar una traza del ciclo de ejecución en formato JSON.
* Clasificar errores provenientes de los proveedores LLM.
* Ejecutar operaciones de manera asíncrona mediante `asyncio`.
* Ejecutar pruebas automatizadas mediante `pytest`.
* Utilizar herramientas simuladas (*mocking*) para probar el ciclo del agente sin realizar llamadas reales al modelo.

El agente utiliza Google Gemini como proveedor LLM y las herramientas se encuentran vinculadas al modelo mediante `bind_tools()`.

## Requisitos

### Mínimos

* Python 3.12+
* API key de Google Gemini.
* API key de Pinecone.
* Un índice de Pinecone configurado para almacenar los embeddings de los documentos.

### Extras

* API key de OpenAI.
* API key de Anthropic.

Los proveedores OpenAI y Anthropic son opcionales y se encuentran contemplados como alternativas dentro de la configuración del proyecto.

## Tecnologías utilizadas

* Python
* asyncio
* Pydantic
* LangChain
* LangChain Core
* LangGraph
* LangGraph Checkpoint SQLite
* LangChain Google GenAI
* Pinecone
* Hugging Face
* Sentence Transformers
* tiktoken
* pytest
* pytest-asyncio
* Ruff

## Variables de entorno

El proyecto utiliza las siguientes variables de entorno:

* `GOOGLE_API_KEY`
* `OPENAI_API_KEY`
* `ANTHROPIC_API_KEY`
* `PINECONE_API_KEY`
* `INDEX_NAME`

Crear un archivo `.env` a partir de `.env.example` y completar las variables correspondientes, en caso de no encontrarse configuradas como variables de entorno del sistema.

Las claves reales no se incluyen en el repositorio.

Las variables `OPENAI_API_KEY` y `ANTHROPIC_API_KEY` son opcionales para la configuración utilizada en esta entrega. Pueden permanecer configuradas con valores no reales cuando no se utilizan esos proveedores.

## Arquitectura del agente

El agente se implementa mediante `StateGraph` utilizando `MessagesState` como estado compartido.

La arquitectura principal está compuesta por:

* **Nodo LLM:** recibe el estado de la conversación y consulta al modelo.
* **Nodo de herramientas:** ejecuta las herramientas solicitadas por el modelo.
* **Condición de herramientas:** determina, a partir de la respuesta del modelo, si el grafo debe continuar hacia el nodo de herramientas o finalizar.
* **Checkpointer:** persiste el estado de la conversación entre diferentes turnos.

La estructura del grafo es:

```text
                 ┌───────────────┐
                 │      LLM      │
                 └───────┬───────┘
                         │
                   tools_condition
                    ┌────┴────┐
                    │         │
                   Tools     END
                    │
                    ▼
                 ┌───────────────┐
                 │     Tools     │
                 └───────┬───────┘
                         │
                         ▼
                        LLM
```

El ciclo entre `LLM` y `Tools` permite que el agente realice múltiples llamadas a herramientas antes de producir una respuesta final.

## Ciclo de razonamiento y uso de herramientas

El modelo dispone de dos herramientas especializadas:

* `buscar_concepto`: recupera información relacionada con la definición o explicación de un concepto.
* `buscar_fuente`: recupera información relacionada con las fuentes y páginas donde puede encontrarse información sobre un tema.

Las herramientas se registran mediante `@tool` y posteriormente se vinculan al modelo mediante:

```python
llm.bind_tools(tools)
```

La selección de herramientas no se implementa mediante condiciones manuales en el código. El modelo determina qué herramienta necesita utilizar a partir de la consulta y de las descripciones proporcionadas mediante los *docstrings* de las herramientas.

Luego, `tools_condition` permite al grafo determinar si debe continuar hacia el nodo de herramientas o finalizar la ejecución.

Por ejemplo, ante una consulta que solicite explicar un concepto y además indicar dónde encontrar información adicional, el agente puede realizar el siguiente ciclo:

```text
Pregunta del usuario
        ↓
       LLM
        ↓
buscar_concepto
        ↓
      Tools
        ↓
       LLM
        ↓
buscar_fuente
        ↓
      Tools
        ↓
       LLM
        ↓
Respuesta final
```

De esta manera, una misma interacción puede requerir múltiples llamadas a herramientas antes de alcanzar una respuesta final.

Este comportamiento fue verificado durante la ejecución del agente mediante `main.py` y mediante pruebas simuladas del grafo, sin depender de llamadas reales al modelo para las pruebas automatizadas.

## Memoria persistente

La memoria del agente se implementa mediante `AsyncSqliteSaver`.

El estado de cada conversación se identifica mediante un `thread_id`, configurado en `main.py`.

```python
CONFIG = {
    "configurable": {
        "thread_id": "conversacion-rag-1"
    },
    "recursion_limit": 10,
}
```

El mismo `thread_id` permite que diferentes llamadas al grafo recuperen el historial previo de la conversación.

La persistencia se almacena localmente en SQLite mediante un archivo de checkpoints.

El parámetro `recursion_limit` establece un límite máximo de iteraciones del grafo para evitar ciclos indefinidos.

## Ejecución

El script principal permite ejecutar una conversación con el agente.

Para ejecutar el sistema:

```bash
python main.py
```

Durante la ejecución, el agente puede:

1. Recibir la consulta del usuario.
2. Analizar si necesita utilizar una herramienta.
3. Ejecutar una o más herramientas.
4. Incorporar los resultados obtenidos al estado de la conversación.
5. Continuar el ciclo.
6. Generar la respuesta final.
7. Guardar la traza de la conversación en formato JSON.

El archivo SQLite utilizado para la persistencia se genera localmente y no se incluye en el repositorio pero se genera automaticamente durante la ejecución.

## Manejo de errores

Se implementó una clasificación de errores mediante `LLMErrorType` y la excepción personalizada `LLMError`.

Actualmente se contemplan:

* `KEY`: credenciales inválidas o no autorizadas.
* `RATE_LIMIT`: límite de solicitudes o cuota alcanzada.
* `UNKNOWN`: error no contemplado específicamente.

Los errores se clasifican antes de ser registrados para evitar exponer información sensible del proveedor en los logs.

## Logging

Se incorporó el módulo estándar `logging` de Python para registrar eventos relevantes durante la ejecución del agente.

Los mensajes se clasifican según su nivel de importancia:

* `INFO`: información sobre proveedores utilizados, inicialización del retriever y ejecución de herramientas.
* `WARNING`: errores recuperables de proveedores LLM.
* `ERROR`: errores que impiden completar una operación.

La configuración utiliza el nivel `INFO`, por lo que se muestran en consola los mensajes `INFO` y los niveles superiores.

Los logs permiten observar el ciclo de ejecución y facilitan la identificación de errores sin registrar claves de API u otra información sensible.

## Testing

Se incorporaron pruebas automatizadas utilizando pytest y pytest-asyncio.

Las pruebas permiten verificar los distintos componentes del sistema sin depender de llamadas reales al modelo cuando no es necesario.

Se incluyen pruebas para:

Ejecución del nodo LLM.
Clasificación de errores del modelo.
Ejecución de las herramientas.
Clasificación de errores de las herramientas.
Funcionamiento del grafo.
Ejecución de ciclos con múltiples llamadas a herramientas mediante mocking.
Persistencia de memoria entre diferentes turnos.
Serialización de la traza de ejecución.
Guardado de trazas independientes.
Ejecución de los tests

Los tests automatizados pueden ejecutarse mediante:

```bash
pytest -v
```

La integración real del agente también fue verificada mediante la ejecución de main.py, utilizando el modelo y las herramientas reales. Esta prueba permitió observar una interacción que requirió múltiples herramientas y una segunda interacción utilizando el mismo estado persistente de conversación.

## Traza de ejecución

El proyecto genera una traza de las interacciones realizadas por el agente en formato JSON.

La traza permite observar el ciclo de ejecución, los mensajes intercambiados y las llamadas a herramientas realizadas durante la conversación.

Un flujo de múltiples pasos puede representarse de la siguiente manera:

```text
HumanMessage
      ↓
AIMessage + tool_call
      ↓
ToolMessage
      ↓
AIMessage + tool_call
      ↓
ToolMessage
      ↓
AIMessage final
```

Se incluye una traza de ejemplo en:

```text
traces/conversacion-rag-0.json
```

La traza permite verificar que el agente puede realizar más de una llamada a herramientas antes de producir una respuesta final.

La traza representa el ciclo de ejecución observable del agente; no contiene ni pretende representar razonamientos internos no expuestos por el modelo.

## Estructura del proyecto

```text
├── tests/
│   ├── test_graph.py
│   ├── test_memory.py
│   ├── test_nodes.py
│   ├── test_tools.py
│   └── test_trace_utils.py
│
├── traces/
│   └── conversacion-rag-0.json
│
├── agent.py                  # Construcción y conexión del StateGraph
├── chunking.py               # Limpieza y división de documentos
├── db_config.py              # Configuración de embeddings y Pinecone
├── db_ingest.py              # Ingesta y configuración de la infraestructura RAG
├── errors.py                 # Clasificación y manejo de errores
├── logging_config.py         # Configuración del sistema de logs
├── main.py                   # Punto de entrada y ejecución del agente
├── model.py                  # Configuración de los proveedores LLM
├── nodes.py                  # Nodo LLM y configuración del ToolNode
├── prompt_config.py          # Configuración del prompt del agente
├── retriever.py              # Configuración del retriever híbrido
├── schemas.py                # Modelos y tipos utilizados por el sistema
├── tools.py                  # Herramientas disponibles para el agente
├── trace_utils.py            # Serialización y guardado de trazas
├── .env.example              # Ejemplo de variables de entorno
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

Los archivos SQLite utilizados para los checkpoints y otros archivos generados durante la ejecución se almacenan localmente y se encuentran excluidos del repositorio mediante `.gitignore`.

## Calidad y buenas prácticas

El proyecto utiliza Ruff como herramienta de análisis y formateo del código.

Para formatear automáticamente el proyecto:

```bash
ruff format .
```

Para analizar el código sin modificarlo:

```bash
ruff check .
```

También se utilizan:

* Type Hints.
* Funciones asíncronas mediante `async def`.
* `await` para operaciones asíncronas.
* Excepciones personalizadas.
* *Mocking* en pruebas unitarias.
* Variables de entorno para las credenciales.
* `recursion_limit` para limitar los ciclos del grafo.
* Persistencia mediante SQLite.

## Sobre el código

El proyecto fue desarrollado tomando como referencia:

* Código y ejemplos proporcionados por el profesor como guía para la Pre-Entrega 5.
* Ejemplos y contenidos incluidos en el temario de la plataforma sobre agentes, LangGraph, herramientas y memoria persistente.
* Documentación oficial y recursos disponibles en Internet.
* ChatGPT como herramienta de asistencia durante el desarrollo.