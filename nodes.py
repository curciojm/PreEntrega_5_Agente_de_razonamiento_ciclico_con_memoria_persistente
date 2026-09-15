import asyncio

from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from errors import classify_error
from logging_config import logger
from model import get_model
from prompt_config import SYSTEM_PROMPT
from tools import tools


async def call_model(state: MessagesState) -> dict:

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    last_error = None

    for provider in ["openai", "anthropic","gemini"]:

        try:
            logger.info("Intentando utilizar proveedor LLM: %s", provider)

            # BIND DE TOOL AL NODO
            llm = get_model(provider)
            llm_tools = llm.bind_tools(tools)

            # Latencia para evitar saturación del proveedor
            await asyncio.sleep(5)

            response = await llm_tools.ainvoke(messages)

            logger.info(
                "Proveedor LLM utilizado correctamente: %s",
                provider,
            )

            return {"messages": [response]}

        
        except Exception as e:
            last_error = e

            error = classify_error(e)

            # Se captura el error para evitar exponer partes de las claves
            logger.warning(
                "El proveedor %s falló: %s",
                provider,
                error.message,
            )

            continue

    logger.error("Todos los proveedores LLM fallaron")

    raise classify_error(last_error)


tool_node = ToolNode(tools)