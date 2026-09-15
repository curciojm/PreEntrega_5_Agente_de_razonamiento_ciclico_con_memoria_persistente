import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import graph

CONFIG = {
    "configurable": {
        "thread_id": "test-multistep-1"
    },
    "recursion_limit": 10,
}


@pytest.mark.asyncio
async def test_agent_multistep():
    pregunta = (
        "Explicame qué es el diseño experimental "
        "y además decime en qué fuente y página puedo leer "
        "más sobre este tema."
    )

    async with AsyncSqliteSaver.from_conn_string(
        "test_checkpoints.sqlite"
    ) as checkpointer:

        app = graph.compile(checkpointer=checkpointer)

        resultado = await app.ainvoke(
            {
                "messages": [
                    HumanMessage(content=pregunta)
                ]
            },
            config=CONFIG,
        )

    mensajes = resultado["messages"]

    tool_calls = []

    for mensaje in mensajes:
        if hasattr(mensaje, "tool_calls") and mensaje.tool_calls:
            tool_calls.extend(mensaje.tool_calls)

    print("\n" + "=" * 80)
    print("TEST MULTI-STEP")
    print("=" * 80)

    print(f"\nCantidad de llamadas a herramientas: {len(tool_calls)}")

    for i, tool_call in enumerate(tool_calls, start=1):
        print(
            f"{i}. {tool_call['name']} "
            f"-> {tool_call['args']}"
        )

    print("\nRespuesta final:")
    print(mensajes[-1].content)

    assert len(tool_calls) >= 2, (
        "El agente debería realizar al menos "
        "dos llamadas a herramientas."
    )