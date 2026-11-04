from typing import TypedDict, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """
    Represents the state of the LangGraph workflow at any given node.
    """
    # Seus campos originais mantidos intactos:
    question: str
    intent: str
    documents: List[Dict[str, Any]]
    answer: str
    is_hallucinating: bool
    context_valid: bool
    retry_count: int

    # Novos campos para suporte à Memória e Reescrita (RRF/RAG):
    messages: Annotated[List[AnyMessage], add_messages]
    summary: str
    standalone_question: str