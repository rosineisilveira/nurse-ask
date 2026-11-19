
from __future__ import annotations

from typing import Annotated
from typing_extensions import NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# DOCUMENT SCHEMA (Contrato do ElasticMotor)
# ============================================================

class RetrievedDocument(TypedDict):
    """Documento pai normalizado retornado pelo ElasticMotor."""

    parent_id: str
    chunk_ids: list[str]
    content: str

    source: str
    title: str
    page: str | int
    pages: list[str | int]
    pages_spanned: str
    
    score: float  # Nota canônica final definida pelo ElasticMotor

    rrf_score: float
    retrieval_score: float
    reranker_score: NotRequired[float]  # Presente apenas quando o reranker está ativo


# ============================================================
# GRAPH STATE (Contrato Compartilhado do LangGraph)
# ============================================================

class RAGState(TypedDict, total=False):
    """Representa o estado compartilhado entre todos os nós do workflow."""

    # Entrada e Intenção
    question: str
    standalone_question: str
    intent: str

    # Recuperação e Telemetria
    documents: list[RetrievedDocument]
    top_score: float

    # Avaliação e Controle de Loop
    context_valid: bool
    retry_count: int
    is_hallucinating: bool

    # Guardrail e Fallback
    is_out_of_scope: bool
    fallback_reason: str

    # Saída Final (Mantida em str livre para não bloquear o streaming SSE)
    answer: str
    generation: str

    # Memória e Histórico
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str

    # Rastreamento de Falhas
    error: str | None


# ============================================================
# LLM OUTPUT SCHEMAS (Apenas Nós Intermediários)
# ============================================================

