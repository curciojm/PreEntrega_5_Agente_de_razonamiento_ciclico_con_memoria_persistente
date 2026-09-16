from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from db_config import DIMENSIONS, INDEX_NAME
from db_ingest import setup_vector_infrastructure

# Cache del retriever para reutilizarlo durante la ejecución del proceso.
_retriever_hibrido = None


async def get_retriever_hibrido() -> EnsembleRetriever:
    global _retriever_hibrido

    if _retriever_hibrido is None:
        _, vectorstore, documentos_procesados = (
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

        # Pesos seleccionados a partir de la evaluación de Precision@5 y Recall@5 de la PreEntrega 4.
        _retriever_hibrido = EnsembleRetriever(
            retrievers=[retriever_bm25, retriever_vectorial],
            weights=[0.25, 0.75],
        )

    return _retriever_hibrido