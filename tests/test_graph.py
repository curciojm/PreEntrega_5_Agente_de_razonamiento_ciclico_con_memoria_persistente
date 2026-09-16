import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@pytest.mark.asyncio
async def test_graph_realiza_ciclo_multistep(monkeypatch):

    respuestas_llm = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "buscar_concepto_test",
                    "args": {
                        "consulta": "diseño experimental"
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "buscar_fuente_test",
                    "args": {
                        "tema": "diseño experimental"
                    },
                    "id": "call-2",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content=(
                "El diseño experimental es un método de investigación "
                "que permite estudiar relaciones entre variables."
            )
        ),
    ]

    llamadas_llm = []

    class FakeLLM:

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            llamadas_llm.append(messages)
            return respuestas_llm[len(llamadas_llm) - 1]

    fake_llm = FakeLLM()

    import nodes

    monkeypatch.setattr(
        nodes,
        "get_model",
        lambda _: fake_llm,
    )

    @tool
    async def buscar_concepto_test(consulta: str) -> list[dict]:
        """Busca información teórica sobre un concepto para el test."""
        return [
            {
                "fuente": "Pagano 2006",
                "contenido": "Definición del diseño experimental.",
            }
        ]

    @tool
    async def buscar_fuente_test(tema: str) -> list[dict]:
        """Busca la fuente y página de un tema para el test."""
        return [
            {
                "fuente": "Pagano 2006",
                "pagina": 42,
            }
        ]

    tools_test = [
        buscar_concepto_test,
        buscar_fuente_test,
    ]

    graph_test = StateGraph(MessagesState)

    graph_test.add_node("llm", nodes.call_model)
    graph_test.add_node("tools", ToolNode(tools_test))

    graph_test.set_entry_point("llm")

    graph_test.add_conditional_edges(
        "llm",
        tools_condition,
    )

    graph_test.add_edge("tools", "llm")

    app = graph_test.compile()

    config = {
        "configurable": {
            "thread_id": "test-graph-multistep"
        },
        "recursion_limit": 10,
    }

    resultado = await app.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Explicame qué es el diseño experimental "
                        "y además decime en qué fuente y página "
                        "puedo leer más sobre este tema."
                    )
                )
            ]
        },
        config=config,
    )

    mensajes = resultado["messages"]

    tool_calls = []

    for mensaje in mensajes:
        if hasattr(mensaje, "tool_calls") and mensaje.tool_calls:
            tool_calls.extend(mensaje.tool_calls)

    assert len(llamadas_llm) == 3
    assert len(tool_calls) == 2

    assert tool_calls[0]["name"] == "buscar_concepto_test"
    assert tool_calls[1]["name"] == "buscar_fuente_test"

    assert mensajes[-1].content == (
        "El diseño experimental es un método de investigación "
        "que permite estudiar relaciones entre variables."
    )