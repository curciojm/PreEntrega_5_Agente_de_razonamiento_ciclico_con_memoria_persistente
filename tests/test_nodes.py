import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from errors import LLMError, LLMErrorType
from nodes import call_model


@pytest.mark.asyncio
async def test_call_model(monkeypatch):
    respuesta = AIMessage(content="Respuesta de prueba")

    class FakeLLM:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            assert isinstance(messages[0], SystemMessage)
            assert isinstance(messages[-1], HumanMessage)
            return respuesta

    fake_llm = FakeLLM()

    import nodes

    monkeypatch.setattr(
        nodes,
        "get_model",
        lambda provider: fake_llm,
    )

    state = {
        "messages": [
            HumanMessage(content="¿Qué es la regresión?")
        ]
    }

    resultado = await call_model(state)

    assert resultado["messages"] == [respuesta]


@pytest.mark.asyncio
async def test_call_model_clasifica_error(monkeypatch):
    class FakeLLM:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            raise ValueError("Error simulado")

    fake_llm = FakeLLM()

    import nodes

    monkeypatch.setattr(
        nodes,
        "get_model",
        lambda provider: fake_llm,
    )

    state = {
        "messages": [
            HumanMessage(content="¿Qué es la regresión?")
        ]
    }

    with pytest.raises(LLMError) as exc_info:
        await call_model(state)

    assert exc_info.value.error_type == LLMErrorType.UNKNOWN