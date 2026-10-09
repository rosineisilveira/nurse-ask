import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator

# Importa o motor real que criamos com o LangGraph
from app.application.rag_workflow.graph import nurseask_engine

app = FastAPI(
    title="NURSEASK API",
    description="API do assistente clínico usando Advanced RAG, LangGraph e Jev AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    question: str

async def stream_langgraph_response(question: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Executa o grafo do LangGraph e faz o stream dos eventos e tokens.
    """
    initial_state = {
        "question": question,
        "intent": "",
        "documents": [],
        "answer": "",
        "is_hallucinating": False
    }
    
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # astream_events captura cada passo dos nós em tempo real
        async for event in nurseask_engine.astream_events(
            initial_state, 
            config=config, 
            version="v2"
        ):
            kind = event["event"]
            node_name = event.get("name", "")

            # 1. Eventos de Status (Para a UI mostrar o que está acontecendo)
            if kind == "on_chain_start":
                if node_name == "router":
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Analisando a intenção da pergunta (Jev AI)...'})}\n\n"
                elif node_name == "retriever":
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Buscando diretrizes clínicas...'})}\n\n"
                elif node_name == "critic":
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Validando segurança e alucinações (Jev AI)...'})}\n\n"

            # 2. Streaming Token a Token (Quando conectarmos o LLM real no generator)
            elif kind == "on_chat_model_stream" and node_name == "generator":
                content_chunk = event["data"]["chunk"].content
                if content_chunk:
                    yield f"data: {json.dumps({'type': 'token', 'content': content_chunk})}\n\n"
            
            # Tratamento temporário para o nosso Mock de geração atual
            elif kind == "on_chain_end" and node_name == "generator":
                answer = event["data"].get("output", {}).get("answer", "")
                if answer: # Simula envio rápido caso não haja stream real ainda
                    yield f"data: {json.dumps({'type': 'token', 'content': answer})}\n\n"

            # 3. Finalização do Grafo
            elif kind == "on_chain_end" and node_name == "LangGraph": # O nome padrão do workflow base
                final_state = event["data"].get("output", {})
                
                if final_state.get("is_hallucinating"):
                    yield f"data: {json.dumps({'type': 'error', 'content': '[ALERTA] A resposta foi bloqueada pelo Jev AI (Falha de Fidelidade).'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Erro interno: {str(e)}'})}\n\n"


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Endpoint principal que recebe a requisição e devolve o fluxo (stream).
    """
    return StreamingResponse(
        stream_langgraph_response(question=request.question, session_id=request.session_id),
        media_type="text/event-stream"
    )

#uvicorn main:app --reload