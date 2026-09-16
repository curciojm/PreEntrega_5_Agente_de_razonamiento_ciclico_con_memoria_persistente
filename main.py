import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import graph
from schemas import LLMError
from trace_utils import extraer_texto, guardar_traza, serializar_traza

CONFIG = {
    "configurable": {
         # Identificador persistente de la conversación.
        "thread_id": "conversacion-rag-1"
    },
    "recursion_limit": 10,
}

async def main():

    async with AsyncSqliteSaver.from_conn_string(
        "checkpoints.sqlite"
    ) as checkpointer:

        app = graph.compile(checkpointer=checkpointer)

        print("=" * 80)
        print("🧑 Turno 1: '¿Qué es la regresión?'")
        print("=" * 80)

        try:
            resultado1 = await app.ainvoke(
                {"messages": [HumanMessage(content="¿Qué es la regresión?")]},
                config=CONFIG,
            )
        except LLMError as e:
                    print(f"\n❌ Error: {e}")
                    return

        print("\n🤖 RESPUESTA:")
        print(extraer_texto(resultado1["messages"][-1]))

        print("\n" + "=" * 80)
        print("🧑 Turno 2: '¿Y dónde puedo leer más sobre eso?'")
        print("=" * 80)

        try:
            resultado2 = await app.ainvoke(
                {"messages": [HumanMessage(content="¿Y dónde puedo leer más sobre eso?")]},
                config=CONFIG,
            )
        except LLMError as e:
            print(f"\n❌ Error: {e}")
            return

        print("\n🤖 RESPUESTA:")
        print(extraer_texto(resultado2["messages"][-1]))

        traza = serializar_traza(resultado2["messages"])

        guardar_traza(
            traza,
            CONFIG["configurable"]["thread_id"],
        )

        print("\n" + "=" * 80)
        print("🔎 TRAZA REACT")
        print("=" * 80)

        for paso in traza:
            print(paso)

        return resultado1, resultado2

if __name__ == "__main__":
    asyncio.run(main())