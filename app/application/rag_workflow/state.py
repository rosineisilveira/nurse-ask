from typing import TypedDict, List, Dict, Any

class RAGState(TypedDict):
    """
    Represents the state of the LangGraph workflow at any given node.
    """
    question: str
    intent: str
    documents: List[Dict[str, Any]]
    answer: str
    is_hallucinating: bool