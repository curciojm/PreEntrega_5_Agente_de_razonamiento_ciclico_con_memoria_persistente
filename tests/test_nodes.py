import pytest
from langchain_core.messages import AIMessage, HumanMessage

from nodes import call_model
from schemas import LLMError, LLMErrorType


@pytest.mark.asyncio
async def test_call_model(monkeypatch):

    respuesta = AIMessage(
        content="La correlación estudia la relación entre dos variables."
    )

    class FakeLLM:

        async def ainvoke(self, messages):
            assert isinstance(messages[0], type(messages[0]))
            assert isinstance(messages[-1], HumanMessage)
            assert messages[-1].content == "¿Qué es la correlación?"

            return respuesta

    import nodes

    monkeypatch.setattr(
        nodes,
        "llm_tools",
        FakeLLM()
    )

    state = {
        "messages": [
            HumanMessage(content="¿Qué es la correlación?")
        ]
    }

    resultado = await call_model(state)

    assert "messages" in resultado
    assert len(resultado["messages"]) == 1
    assert resultado["messages"][0] == respuesta


@pytest.mark.asyncio
async def test_call_model_clasifica_error(monkeypatch):

    class FakeLLM:

        async def ainvoke(self, messages):
            raise ValueError("Error de prueba")

    import nodes

    monkeypatch.setattr(
        nodes,
        "llm_tools",
        FakeLLM()
    )

    state = {
        "messages": [
            HumanMessage(content="¿Qué es la correlación?")
        ]
    }

    with pytest.raises(LLMError) as exc_info:
        await call_model(state)

    assert exc_info.value.error_type == LLMErrorType.UNKNOWN