import pytest
from langchain_core.documents import Document

import tools
from schemas import LLMError, ResultadoConcepto, ResultadoFuente
from tools import buscar_concepto, buscar_fuente


@pytest.mark.asyncio
async def test_buscar_concepto(monkeypatch):

    documentos = [
        Document(
            page_content="La correlación estudia la relación entre dos variables.",
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 10,
            },
        ),
        Document(
            page_content="La correlación puede ser positiva o negativa.",
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 11,
            },
        ),
    ]

    class FakeRetriever:

        async def ainvoke(self, query):
            assert query == "¿Qué es la correlación?"
            return documentos

    async def fake_get_retriever():
        return FakeRetriever()

    monkeypatch.setattr(
        tools,
        "get_retriever_hibrido",
        fake_get_retriever
    )

    resultado = await buscar_concepto.ainvoke(
        {"consulta": "¿Qué es la correlación?"}
    )

    assert isinstance(resultado, list)
    assert len(resultado) == 2

    assert all(
        isinstance(item, ResultadoConcepto)
        for item in resultado
    )

    assert resultado[0].fuente == "Pagano 2006"
    assert resultado[0].contenido == (
        "La correlación estudia la relación entre dos variables."
    )


@pytest.mark.asyncio
# mark.asyncio: Esta función de test es asíncrona (async def) y necesita ejecutarse dentro de un event loop de asyncio
async def test_buscar_fuente(monkeypatch):

    documentos = [
        Document(
            page_content="Contenido sobre correlación.",
            metadata={
                "fuente": "Pagano 2006",
                "pagina": 10.0,
            },
        ),
        Document(
            page_content="Más contenido sobre correlación.",
            metadata={
                "fuente": "Sampieri",
                "pagina": 25,
            },
        ),
    ]

    class FakeRetriever:

        async def ainvoke(self, query):
            assert query == "correlación"
            return documentos

    import tools

    async def fake_get_retriever():
        return FakeRetriever()

    monkeypatch.setattr(
        tools,
        "get_retriever_hibrido",
        fake_get_retriever
    )

    resultado = await buscar_fuente.ainvoke(
        {"tema": "correlación"}
    )

    assert isinstance(resultado, list)
    assert len(resultado) == 2

    assert all(
        isinstance(item, ResultadoFuente)
        for item in resultado
    )

    assert resultado[0].fuente == "Pagano 2006"
    assert resultado[0].pagina == 10
    assert isinstance(resultado[0].pagina, int)

    assert resultado[1].fuente == "Sampieri"
    assert resultado[1].pagina == 25


@pytest.mark.asyncio
async def test_buscar_concepto_clasifica_error(monkeypatch):

    class FakeRetriever:

        async def ainvoke(self, query):
            raise ValueError("Error de prueba")

    import tools

    async def fake_get_retriever():
        return FakeRetriever()

    monkeypatch.setattr(
        tools,
        "get_retriever_hibrido",
        fake_get_retriever
    )

    with pytest.raises(LLMError) as exc_info:
        await buscar_concepto.ainvoke(
            {"consulta": "¿Qué es la correlación?"}
        )

    assert exc_info.value.error_type.name == "UNKNOWN"