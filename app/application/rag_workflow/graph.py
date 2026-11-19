
import os
from redis import Redis
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from app.application.rag_workflow.state import RAGState
from app.application.rag_workflow.schemas import RouteTarget

from app.application.rag_workflow.state import RAGState
from app.application.rag_workflow.nodes import (
    rewrite_node,
    fallback_node,
    retriever_node,
    guardrail_router_node,
    generator_node,
    route_after_guardrail,
    condense_question_node,
    summarize_node,
    evaluator_node
    
)

## 1. Conexão com o Redis Checkpointer
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
checkpointer = AsyncRedisSaver(redis_url=REDIS_URL)

   
def decide_to_generate(state: RAGState) -> str:
    valid = state.get("context_valid", False)
    count = state.get("retry_count", 0)

    print(f"\n🚦 [ROTEADOR] Contexto Válido: {valid} | Tentativas de reescrita realizadas: {count}/3")

    if valid:
        print("✅ [ROTEADOR] Contexto Aprovado! Indo para o Gerador.")
        return "generate"

    if count >= 3:
        print("🛑 [ROTEADOR] Limite de 3 tentativas atingido! Abstendo via Fallback.")
        return "fallback"

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
builder.add_node("guardrail_router", guardrail_router_node)
builder.add_node("fallback", fallback_node)

# 4. Definição do Fluxo e Arestas

# Entrada: Resolve anáforas do chat usando histórico do Redis antes da busca
builder.add_edge(START, "guardrail_router")

builder.add_conditional_edges(
    "guardrail_router",
    route_after_guardrail,
    {
        # Se for seguro (RouteTarget.RETRIEVER), mapeamos para o condense_question 
        # para preservar seu fluxo original de tratamento de histórico
        RouteTarget.RETRIEVER.value: "condense_question",
        
        # Se for bloqueado, vai para o fallback
        RouteTarget.FALLBACK.value: "fallback"
    }
)

# [NOVO] Encerra o fluxo se cair no fallback
builder.add_edge("fallback", END)

builder.add_edge("condense_question", "retriever")
builder.add_edge("retriever", "evaluator")

# Aresta condicional após avaliação do contexto
builder.add_conditional_edges(
    "evaluator",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "fallback": "fallback",
    },
)

# Loop de reescrita: retorna diretamente para o retriever (preserva 'question' intacta)
builder.add_edge("rewrite", "retriever")

# Saída com controle de memória: Gerador -> Sumarização/Limpeza -> END
builder.add_edge("generate", "summarize")
builder.add_edge("summarize", END)

# 5. Compilação com Checkpointer Nativo no Redis
nurseask_engine = builder.compile(checkpointer=checkpointer)



