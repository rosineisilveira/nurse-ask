import pytest
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from tests.ollama_judge import OllamaJudge
from app.application.rag_workflow.graph import nurseask_engine


@pytest.mark.asyncio
async def test_faithfulness():
    pergunta = " tubos"

    config = {"configurable": {"thread_id": "test_faithfulness_session"}}
    result = await nurseask_engine.ainvoke(
        {
            "question": pergunta,
            "documents": [],
            "context_valid": False,
            "retry_count": 0,
        },
        config=config,
    )

    contexto = []
    for doc in result.get("documents", []):
        if isinstance(doc, dict):
            texto = doc.get("page_content") or doc.get("text") or doc.get("content") or str(doc)
        else:
            texto = getattr(doc, "page_content", str(doc))
        if texto:
            contexto.append(texto)

    resposta = result.get("generation", "")

    assert resposta, "A resposta gerada veio vazia."
    assert contexto, f"O contexto resgatado veio vazio! Estrutura dos docs: {result.get('documents')}"

    local_judge = OllamaJudge(model_name="llama3.1")

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.8,
        model=local_judge,
        include_reason=True,
        verbose_mode=False,
    )

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta,
        retrieval_context=contexto,
    )

    faithfulness_metric.measure(test_case)

    print(f"\n📊 Nota: {faithfulness_metric.score:.2f}")
    print(f"🧠 Razão: {faithfulness_metric.reason}\n")

    assert faithfulness_metric.is_successful(), (
        f"Faithfulness falhou ({faithfulness_metric.score:.2f}): {faithfulness_metric.reason}"
    )
    
    # pytest tests/test_rag_faithfulness.py