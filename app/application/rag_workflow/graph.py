from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from app.application.rag_workflow.nodes import retriever_node, generator_node

# Definição simples do estado
class GraphState(TypedDict):
    question: str
    documents: list
    generation: str

# Montagem do Grafo Enxuto
builder = StateGraph(GraphState)

# Adiciona apenas os dois nós essenciais
builder.add_node("retriever", retriever_node)
builder.add_node("generate", generator_node)

# Conecta o fluxo em linha reta
builder.add_edge(START, "retriever")
builder.add_edge("retriever", "generate")
builder.add_edge("generate", END)

# Compila o motor
nurseask_engine = builder.compile()