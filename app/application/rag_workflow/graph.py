from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated, TypedDict, List, Any
from app.application.rag_workflow.nodes import retriever_node, generator_node
from app.application.rag_workflow.nodes import retriever_node, evaluator_node, generator_node, rewrite_node
from langgraph.graph import StateGraph, END


# Estado simples sem reducers ou acoplamento
class GraphState(TypedDict):
    question: str           # Pergunta atual (que muda na reescrita)
    original_question: str  # Pergunta original do usuário (nunca muda)
    documents: List[Any]
    context_valid: bool
    retry_count: int        # Contador de reescritas (0, 1, 2, 3)

def decide_to_generate(state: GraphState) -> str:
    valid = state.get("context_valid", False)
    count = state.get("retry_count", 0)
    
    print(f"\n🚦 [ROTEADOR] Válido: {valid} | Tentativas de reescrita realizadas: {count}/3")
    
    # Se o contexto é válido OU se já reescrevemos 3 vezes, manda gerar
    if valid or count >= 3:
        if not valid:
            print("🛑 [ROTEADOR] Limite de 3 tentativas atingido! Forçando ida para o Gerador.")
        else:
            print("✅ [ROTEADOR] Contexto Aprovado! Indo para o Gerador.")
        return "generate"
    
    print("🔄 [ROTEADOR] Contexto Reprovado -> Enviando para o Rewriter.")
    return "rewrite"

builder = StateGraph(GraphState)

builder.add_node("retriever", retriever_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generator_node)

builder.set_entry_point("retriever")
builder.add_edge("retriever", "evaluator")

builder.add_conditional_edges(
    "evaluator",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

builder.add_edge("rewrite", "retriever")
builder.add_edge("generate", END)

nurseask_engine = builder.compile()