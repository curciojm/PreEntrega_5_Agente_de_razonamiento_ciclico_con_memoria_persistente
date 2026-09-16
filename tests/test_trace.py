import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from trace_utils import extraer_texto, guardar_traza, serializar_traza


def test_extraer_texto():

    mensaje = AIMessage(
        content="Esta es una respuesta de prueba."
    )

    resultado = extraer_texto(mensaje)

    assert resultado == "Esta es una respuesta de prueba."


def test_serializar_traza():

    mensajes = [
        HumanMessage(
            content="¿Qué es la correlación?"
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "buscar_concepto",
                    "args": {
                        "consulta": "correlación"
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Información recuperada.",
            name="buscar_concepto",
            tool_call_id="call-1",
        ),
        AIMessage(
            content="La correlación estudia la relación entre dos variables."
        ),
    ]

    resultado = serializar_traza(mensajes)

    assert isinstance(resultado, list)
    assert len(resultado) == 4

    assert resultado[0]["tipo"] == "HumanMessage"

    assert resultado[1]["tipo"] == "AIMessage"
    assert resultado[1]["tool_calls"][0]["nombre"] == (
        "buscar_concepto"
    )
    assert resultado[1]["tool_calls"][0]["argumentos"] == {
        "consulta": "correlación"
    }

    assert resultado[2]["tipo"] == "ToolMessage"
    assert resultado[2]["herramienta"] == "buscar_concepto"

    assert resultado[3]["tipo"] == "AIMessage"
    assert "correlación" in resultado[3]["contenido"]


def test_guardar_dos_trazas_independientes(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    traza_1 = [
        {
            "tipo": "HumanMessage",
            "contenido": "¿Qué es la correlación?",
        },
        {
            "tipo": "AIMessage",
            "contenido": "La correlación estudia la relación entre variables.",
        },
    ]

    traza_2 = [
        {
            "tipo": "HumanMessage",
            "contenido": "¿Qué es el diseño experimental?",
        },
        {
            "tipo": "AIMessage",
            "contenido": "El diseño experimental organiza las condiciones de un estudio.",
        },
    ]

    thread_id_1 = "conversacion-rag-1"
    thread_id_2 = "conversacion-rag-2"

    guardar_traza(
        traza_1,
        thread_id_1,
    )

    guardar_traza(
        traza_2,
        thread_id_2,
    )

    ruta_1 = tmp_path / "traces" / "conversacion-rag-1.json"
    ruta_2 = tmp_path / "traces" / "conversacion-rag-2.json"

    assert ruta_1.exists()
    assert ruta_2.exists()

    with open(ruta_1, "r", encoding="utf-8") as archivo:
        resultado_1 = json.load(archivo)

    with open(ruta_2, "r", encoding="utf-8") as archivo:
        resultado_2 = json.load(archivo)

    assert resultado_1 == traza_1
    assert resultado_2 == traza_2

    assert resultado_1 != resultado_2

    assert resultado_1[0]["contenido"] == (
        "¿Qué es la correlación?"
    )

    assert resultado_2[0]["contenido"] == (
        "¿Qué es el diseño experimental?"
    )
