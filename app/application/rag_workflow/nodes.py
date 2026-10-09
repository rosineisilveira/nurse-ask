from app.application.rag_workflow.state import RAGState
from app.infrastructure.ai_clients.jev_client import JevClient

jev_client = JevClient()

async def router_node(state: RAGState) -> dict:
    """
    Analyzes the user's question using Jev AI and determines the workflow path.
    """
    question = state.get("question", "")
    
    prompt = f"Classify the intent of the following healthcare user query: {question}"
    
    # Execute the deterministic choice primitive
    response = await jev_client.choice(
        input_text=prompt,
        options=["medical_rag", "math_calculator", "chitchat"]
    )
    
    return {"intent": response.selected_option}


async def hybrid_retriever_node(state: RAGState) -> dict:
    """
    Retrieves medical guidelines using Vector (Qdrant) and Lexical (Elasticsearch) search.
    Applies Cohere Rerank. (Currently mocked)
    """
    # MOCK (Dados simulados para testar o fluxo antes de ligar o Qdrant real)
    mock_docs = [
        {
            "source": "protocolo_sepse_2024.pdf", 
            "content": "Administrar antibiótico de amplo espectro na primeira hora."
        }
    ]
    return {"documents": mock_docs}


async def generator_node(state: RAGState) -> dict:
    """
    Generates the final response anchored strictly to the retrieved context.
    """
    # MOCK (Simulação do LLM escrevendo a resposta)
    mock_answer = "Com base no protocolo de sepse, deve-se administrar antibiótico na primeira hora [1]."
    return {"answer": mock_answer}


async def critic_node(state: RAGState) -> dict:
    """
    Uses Jev AI (Noul primitive) to evaluate if the generated response hallucinates.
    """
    answer = state.get("answer", "")
    
    # Avalia alucinação. Retorna True (Sim, alucinou) ou False (Não, está seguro)
    response = await jev_client.noul(input_text=f"Evaluating response for hallucinations: {answer}")
    
    return {"is_hallucinating": response.boolean_result}


async def guardrail_node(state: RAGState) -> dict:
    """
    Fallback node to block non-medical conversations.
    """
    return {"answer": "Sou um assistente clínico. Por favor, faça perguntas focadas em protocolos ou diretrizes médicas."}