import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import graph


@pytest.mark.asyncio
async def test_memory_same_thread(monkeypatch, tmp_path):

    respuestas = [
        AIMessage(
            content="La correlación mide la relación entre dos variables."
        ),
        AIMessage(
            content="Sí, la correlación puede ser positiva o negativa."
        ),
    ]

    llamadas = []

    class FakeLLM:

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):

            llamadas.append(messages)

            return respuestas[len(llamadas) - 1]

    import nodes

    fake_llm = FakeLLM()

    monkeypatch.setattr(
        nodes,
        "get_model",
        lambda _: fake_llm,
    )

    database = tmp_path / "memory.sqlite"

    config = {
        "configurable": {
            "thread_id": "test-memory-1"
        },
        "recursion_limit": 10,
    }

    async with AsyncSqliteSaver.from_conn_string(
        str(database)
    ) as checkpointer:

        app = graph.compile(
            checkpointer=checkpointer
        )

        await app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="¿Qué es la correlación?"
                    )
                ]
            },
            config=config,
        )

        await app.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="¿Y puede ser positiva?"
                    )
                ]
            },
            config=config,
        )

    assert len(llamadas) == 2

    mensajes_segunda_llamada = llamadas[1]

    assert any(
        mensaje.content == "¿Qué es la correlación?"
        for mensaje in mensajes_segunda_llamada
    )

    assert any(
        mensaje.content == (
            "La correlación mide la relación entre dos variables."
        )
        for mensaje in mensajes_segunda_llamada
    )

    assert any(
        mensaje.content == "¿Y puede ser positiva?"
        for mensaje in mensajes_segunda_llamada
    )
