from langchain_core.tools import tool

from errors import classify_error
from logging_config import logger
from retriever import get_retriever_hibrido
from schemas import ResultadoConcepto, ResultadoFuente


@tool
async def buscar_concepto(consulta: str) -> list[ResultadoConcepto]:
    """
    Busca información teórica sobre un concepto de estadística
    y metodología.

    Usar esta herramienta cuando el usuario solicite una definición
    o explicación de un concepto.

    Devuelve el contenido relevante junto con la fuente.
    """
    try:
        logger.info("Ejecutando buscar_concepto: %s", consulta)

        retriever_hibrido = await get_retriever_hibrido()
        docs = await retriever_hibrido.ainvoke(consulta)

        return [
            ResultadoConcepto(
                fuente=doc.metadata.get("fuente", "desconocida"),
                contenido=doc.page_content,
            )
            for doc in docs[:5]
        ]

    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        raise classify_error(e)


@tool
async def buscar_fuente(tema: str) -> list[ResultadoFuente]:
    """
    Busca en la base de conocimientos dónde se encuentra información
    relevante sobre un tema de estadística y metodología.

    Usar esta herramienta cuando el usuario pregunte dónde encontrar
    información, qué fuente consultar o en qué página aparece un tema.

    Devuelve la fuente y el número entero de página correspondiente.
    """
    try:
        logger.info("Ejecutando buscar_fuente: %s", tema)

        retriever_hibrido = await get_retriever_hibrido()
        docs = await retriever_hibrido.ainvoke(tema)

        return [
            ResultadoFuente(
                fuente=doc.metadata.get("fuente", "desconocida"),
                pagina=int(doc.metadata.get("pagina", 0)),
            )
            for doc in docs[:5]
        ]

    except Exception as e:
        error = classify_error(e)
        logger.error(
            "Error durante la ejecución de buscar_concepto: %s",
            error.message,
        )
    raise error


tools = [buscar_concepto, buscar_fuente]