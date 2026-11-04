import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.application.rag_workflow.graph import nurseask_engine

# 1. Definição de métricas rígidas para área clínica
faithfulness_metric = FaithfulnessMetric(
    threshold=0.9, # Exige 90%+ de fidelidade ao texto base (zero alucinação)
    model="gpt-4o-mini" # LLM Juiz rápido e barato
)

# Métrica customizada para conformidade de protocolo de enfermagem
protocol_compliance_metric = GEval(
    name="Conformidade Clínico-Assistencial",
    criteria="Verifique se a resposta contém apenas procedimentos seguros de enfermagem e se cita a dosagem/unidade correta.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    threshold=0.8,
    model="gpt-4o-mini"
)

@pytest.mark.asyncio
async def test_consulta_vancomicina():
    pergunta = "Qual o tempo mínimo de infusão de vancomicina 1g em adultos?"
    
    # Executa o seu LangGraph
    config = {"configurable": {"thread_id": "test_session_vanco"}}
    result = await nurseask_engine.ainvoke(
        {"question": pergunta, "documents": [], "context_valid": False, "retry_count": 0},
        config=config
    )
    
    # Extrai o contexto resgatado e a resposta
    contexto = [doc.get("page_content", "") for doc in result.get("documents", [])]
    resposta = result.get("answer", "")
    
    # Monta o caso de teste para o DeepEval
    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta,
        retrieval_context=contexto
    )
    
    # Executa a asserção (se falhar, o pytest aponta exatamente o erro)
    assert_test(test_case, [faithfulness_metric, protocol_compliance_metric])