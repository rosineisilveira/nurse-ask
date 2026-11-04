# from langgraph.graph import StateGraph, START, END
# from typing_extensions import TypedDict
# from typing import Annotated, TypedDict, List, Any
# from app.application.rag_workflow.nodes import retriever_node, generator_node
# from app.application.rag_workflow.nodes import retriever_node, evaluator_node, generator_node, rewrite_node
# from langgraph.graph import StateGraph, END


# # Estado simples sem reducers ou acoplamento
# class GraphState(TypedDict):
#     question: str           # Pergunta atual (que muda na reescrita)
#     original_question: str  # Pergunta original do usuário (nunca muda)
#     documents: List[Any]
#     context_valid: bool
#     retry_count: int        # Contador de reescritas (0, 1, 2, 3)

# def decide_to_generate(state: GraphState) -> str:
#     valid = state.get("context_valid", False)
#     count = state.get("retry_count", 0)
    
#     print(f"\n🚦 [ROTEADOR] Válido: {valid} | Tentativas de reescrita realizadas: {count}/3")
    
#     # Se o contexto é válido OU se já reescrevemos 3 vezes, manda gerar
#     if valid or count >= 3:
#         if not valid:
#             print("🛑 [ROTEADOR] Limite de 3 tentativas atingido! Forçando ida para o Gerador.")
#         else:
#             print("✅ [ROTEADOR] Contexto Aprovado! Indo para o Gerador.")
#         return "generate"
    
#     print("🔄 [ROTEADOR] Contexto Reprovado -> Enviando para o Rewriter.")
#     return "rewrite"

# builder = StateGraph(GraphState)

# builder.add_node("retriever", retriever_node)
# builder.add_node("evaluator", evaluator_node)
# builder.add_node("rewrite", rewrite_node)
# builder.add_node("generate", generator_node)

# builder.set_entry_point("retriever")
# builder.add_edge("retriever", "evaluator")

# builder.add_conditional_edges(
#     "evaluator",
#     decide_to_generate,
#     {
#         "generate": "generate",
#         "rewrite": "rewrite"
#     }
# )

# builder.add_edge("rewrite", "retriever")
# builder.add_edge("generate", END)

# nurseask_engine = builder.compile()

import os
from redis import Redis
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.application.rag_workflow.state import RAGState
from app.application.rag_workflow.nodes import (
    condense_question_node,
    retriever_node,
    evaluator_node,
    rewrite_node,
    generator_node,
    summarize_node,
)

# 1. Conexão com o Redis Checkpointer
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
checkpointer = AsyncRedisSaver(redis_url=REDIS_URL)


# 2. Roteador Condicional (Utilizando RAGState e limite de 3 tentativas)
def decide_to_generate(state: RAGState) -> str:
    valid = state.get("context_valid", False)
    count = state.get("retry_count", 0)

    print(f"\n🚦 [ROTEADOR] Contexto Válido: {valid} | Tentativas de reescrita realizadas: {count}/3")

    # Se o contexto é válido OU se já reescrevemos 3 vezes, avança para o gerador
    if valid or count >= 3:
        if not valid:
            print("🛑 [ROTEADOR] Limite de 3 tentativas atingido! Forçando ida para o Gerador.")
        else:
            print("✅ [ROTEADOR] Contexto Aprovado! Indo para o Gerador.")
        return "generate"

    print("🔄 [ROTEADOR] Contexto Reprovado -> Enviando para o Rewriter.")
    return "rewrite"


# 3. Construção do Grafo com RAGState
builder = StateGraph(RAGState)

# Registro de todos os nós do pipeline
builder.add_node("condense_question", condense_question_node)
builder.add_node("retriever", retriever_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generator_node)
builder.add_node("summarize", summarize_node)

# 4. Definição do Fluxo e Arestas

# Entrada: Resolve anáforas do chat usando histórico do Redis antes da busca
builder.add_edge(START, "condense_question")
builder.add_edge("condense_question", "retriever")
builder.add_edge("retriever", "evaluator")

# Aresta condicional após avaliação do contexto
builder.add_conditional_edges(
    "evaluator",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)

# Loop de reescrita: retorna diretamente para o retriever (preserva 'question' intacta)
builder.add_edge("rewrite", "retriever")

# Saída com controle de memória: Gerador -> Sumarização/Limpeza -> END
builder.add_edge("generate", "summarize")
builder.add_edge("summarize", END)

# 5. Compilação com Checkpointer Nativo no Redis
nurseask_engine = builder.compile(checkpointer=checkpointer)