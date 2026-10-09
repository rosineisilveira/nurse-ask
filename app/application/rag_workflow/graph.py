from langgraph.graph import StateGraph, END
from app.application.rag_workflow.state import RAGState
from app.application.rag_workflow.nodes import (
    router_node, 
    hybrid_retriever_node, 
    generator_node,
    critic_node,
    guardrail_node
)

def route_after_intent(state: RAGState) -> str:
    """
    Conditional edge: Evaluates the intent returned by Jev AI
    to decide the next node in the graph.
    """
    intent = state.get("intent")
    
    if intent == "medical_rag":
        return "retriever"
    else:
        return "guardrail"

def build_nurseask_graph():
    """
    Builds, configures and compiles the LangGraph state machine.
    """
    # 1. Initialize Graph with typed state
    workflow = StateGraph(RAGState)
    
    # 2. Add all Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", hybrid_retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("guardrail", guardrail_node)
    
    # 3. Set the starting point
    workflow.set_entry_point("router")
    
    # 4. Conditional routing based on user intent
    workflow.add_conditional_edges(
        "router",
        route_after_intent,
        {
            "retriever": "retriever",
            "guardrail": "guardrail"
        }
    )
    
    # 5. Linear flow for standard RAG
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "critic")
    
    # 6. Exit points
    workflow.add_edge("critic", END)
    workflow.add_edge("guardrail", END)
    
    return workflow.compile()

# Export a compiled instance for the FastAPI app to consume
nurseask_engine = build_nurseask_graph()