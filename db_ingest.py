import asyncio

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from db_config import DIMENSIONS, EMBEDDINGS, INDEX_NAME, NAMESPACE, PINECONE_API_KEY
from logging_config import logger
from setup import procesamiento_desde_pdfs, recuperar_documentos_de_pinecone


async def setup_vector_infrastructure(
    index_name: str = INDEX_NAME,
    dimensions: int = DIMENSIONS,
):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if index_name not in pc.list_indexes().names():
        try:
            logger.info("Índice inexistente — creando índice")

            pc.create_index(
                name=index_name,
                dimension=dimensions,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

            while not pc.describe_index(index_name).status["ready"]:
                await asyncio.sleep(5)

        except Exception as e:
            logger.exception(
                "Error durante la creación del índice: %s",
                e,
            )
            raise

    index = pc.Index(index_name)

    try:
        stats = index.describe_index_stats()

        vector_count = (
            stats["namespaces"]
            .get(NAMESPACE, {})
            .get("vector_count", 0)
        )

        if vector_count == 0:
            logger.info(
                "Índice disponible pero sin vectores — "
                "procesando e indexando documentos por primera vez"
            )

            documentos_procesados = procesamiento_desde_pdfs()

            vectorstore = PineconeVectorStore.from_documents(
                documents=documentos_procesados,
                embedding=EMBEDDINGS,
                index_name=index_name,
                namespace=NAMESPACE,
            )

        else:
            logger.info(
                "Índice existente con %s vectores — "
                "reutilizando infraestructura",
                vector_count,
            )

            documentos_procesados = recuperar_documentos_de_pinecone(index)

            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=EMBEDDINGS,
                namespace=NAMESPACE,
            )

    except Exception as e:
        logger.exception(
            "Error durante el procesamiento e indexación de los documentos: %s",
            e,
        )
        raise

    print("📦 Vectores en el namespace:",
          stats["namespaces"].get(NAMESPACE, {}),
          )

    logger.info(f"Estado del índice: {stats}")

    return index, vectorstore, documentos_procesados