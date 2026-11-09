# app/application/rag_workflow/schemas.py
from enum import Enum

class RouteTarget(str, Enum):
    RETRIEVER = "retriever"
    FALLBACK = "fallback"
    GENERATOR = "generator"
    REWRITER = "rewriter"