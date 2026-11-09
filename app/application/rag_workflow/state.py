from typing import TypedDict, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """
    Represents the state of the LangGraph workflow at any given node.
    """
    question: str
    intent: str
    documents: List[Dict[str, Any]]
    answer: str
    is_hallucinating: bool
    context_valid: bool
    retry_count: int
    is_out_of_scope: bool
    fallback_reason: str
                        #  campos para suporte à Memória e Reescrita (RRF/RAG):
    messages: Annotated[List[AnyMessage], add_messages]
    summary: str
    standalone_question: str
                        # para o LangGraph registrar o retornov
    generation: str  