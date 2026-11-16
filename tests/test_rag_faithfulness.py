import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from tests.ollama_judge import OllamaJudge
from ollama_judge import OllamaJudge
from app.application.rag_workflow.graph import nurseask_engine


@pytest.mark.asyncio
async def test_faithfulness():
    pergunta = "posiçoes terapeuticas ?"

    # 1. Executa o grafo
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

    # 🟢 2. Extração flexível dos textos resgatados pelo Retriever
    contexto = []
    for doc in result.get("documents", []):
        if isinstance(doc, dict):
            texto = doc.get("page_content") or doc.get("text") or doc.get("content") or str(doc)
        else:
            texto = getattr(doc, "page_content", str(doc))
        if texto:
            contexto.append(texto)

    resposta = result.get("generation", "")

    # Validações preventivas antes de chamar a LLM Juíza
    assert resposta, "A resposta gerada veio vazia."
    assert contexto, f"O contexto resgatado veio vazio! Estrutura dos docs: {result.get('documents')}"

    # 3. LLM Juíza local (Ollama com JSON mode)
    local_judge = OllamaJudge(model_name="llama3.1")

    # 4. Avaliação de Faithfulness
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.8,
        model=local_judge
    )

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta,
        retrieval_context=contexto
    )

    assert_test(test_case, [faithfulness_metric])

    #pytest tests/test_rag_faithfulness.py