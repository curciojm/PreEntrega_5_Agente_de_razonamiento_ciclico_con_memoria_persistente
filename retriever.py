from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from db_config import DIMENSIONS, INDEX_NAME
from db_ingest import setup_vector_infrastructure

# En esta PreEntrega se crea el retriever como variable temporal para evitar la creacion del index por cada turno donde
# se ejecuta una tool
_retriever_hibrido = None


async def get_retriever_hibrido() -> EnsembleRetriever:
    global _retriever_hibrido

    if _retriever_hibrido is None:
        _, vectorstore, documentos_procesados = (
            # primera creacion de lindex y dimensiones
            await setup_vector_infrastructure(
                INDEX_NAME,
                DIMENSIONS,
            )
        )

        retriever_bm25 = BM25Retriever.from_documents(
            documentos_procesados
        )

        retriever_bm25.k = 5

        retriever_vectorial = vectorstore.as_retriever(
            search_kwargs={"k": 5},
        )

        # Los mejores resultados de Recall@5 y Precision@5
        # se obtuvieron con estos pesos (ver reporte PreEntrega 4).
        _retriever_hibrido = EnsembleRetriever(
            retrievers=[retriever_bm25, retriever_vectorial],
            weights=[0.25, 0.75],
        )

    return _retriever_hibrido