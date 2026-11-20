import json
import sys
import asyncio
from fastapi import FastAPI
from fastapi import Request
import uuid
from app.application.rag_workflow.graph import nurseask_engine 
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage
from contextlib import asynccontextmanager
from app.application.rag_workflow.graph import checkpointer

from app.application.rag_workflow.graph import nurseask_engine

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Executado na INICIALIZAÇÃO do servidor
#     print("🔄 [REDIS] Criando índices do RediSearch...")
#     await checkpointer.setup()
#     print("✅ [REDIS] Índices prontos!")
    
#     yield  # A aplicação roda enquanto estiver neste ponto
    
#     # Executado no DESLIGAMENTO do servidor (se precisar fechar conexões)
#     print("🛑 [SERVER] Encerrando aplicação...")

# app = FastAPI(lifespan=lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Mantém a configuração original do Redis
    print("🔄 [REDIS] Criando índices do RediSearch...")
    await checkpointer.setup()
    print("✅ [REDIS] Índices prontos!")

    # 2. Executa o seu script de ingestão automaticamente na subida
    print("📥 [INGESTÃO] Verificando e indexando protocolos da pasta data/...")
    processo = await asyncio.create_subprocess_exec(
        sys.executable,
        "app/scripts/ingest_data", 
    )
    codigo_retorno = await processo.wait()

    if codigo_retorno == 0:
        print("✅ [INGESTÃO] Protocolos prontos no Elasticsearch!")
    else:
        print(f"⚠️ [INGESTÃO] Falha ao executar ingestão (código {codigo_retorno}).")

    yield

    print("🛑 [SERVER] Encerrando aplicação...")
    


app = FastAPI(
    title="NURSEASK API",
    description="API do assistente clínico usando Advanced RAG, LangGraph e Jev AI",
    version="1.0.0",
    lifespan=lifespan
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
        
@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")


@app.post("/api/v1/chat/stream")
async def chat_endpoint(request: Request):
    data = await request.json()
    question = data.get("question", "")
    session_id = data.get("session_id") or data.get("thread_id") or str(uuid.uuid4())

    async def sse_generator():
        # Estado inicial completo para satisfazer todas as chaves do RAGState
        initial_state = {
            "question": question,
            "standalone_question": question,
            "messages": [HumanMessage(content=question)],
            "summary": "",
            "intent": "",
            "documents": [],
            "answer": "",
            "is_hallucinating": False,
            "context_valid": False,
            "retry_count": 0,
        }

        config = {"configurable": {"thread_id": session_id}}

        try:
            async for event in nurseask_engine.astream_events(initial_state, config=config, version="v2"):
                kind = event["event"]
                node_name = event.get("metadata", {}).get("langgraph_node", "")

                # 1. NOTIFICAÇÕES DE STATUS
                if kind == "on_chain_start" and not event.get("name", "").startswith("Runnable"):
                    chain_name = event.get("name", "")
                    if chain_name in ["condense_question", "retriever", "evaluator", "rewrite", "generate", "summarize", "fallback_node"]:
                        yield f"data: {json.dumps({'type': 'status', 'content': f'Iniciando etapa: {chain_name}'})}\n\n"

                # 2. FILTRO DE TOKENS
                elif kind == "on_chat_model_stream" and node_name == "generate":
                    chunk = event["data"]["chunk"].content

                    if isinstance(chunk, str) and chunk != "":
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                        
                # 3.CAPTURAR A RESPOSTA FINAL DO FALLBACK
                # O fallback retorna o payload todo de uma vez no on_chain_end
                elif kind == "on_chain_end" and node_name in ["fallback", "fallback_node"]:
                    output = event.get("data", {}).get("output", {})
                    
                    if isinstance(output, dict) and "generation" in output:
                        texto_seguranca = output["generation"]
                        
                        # SIMULADOR DE STREAMING: Quebra a frase em palavras e envia uma a uma
                        for word in texto_seguranca.split():
                            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                            await asyncio.sleep(0.03) #tempo para o frontend processar

            yield "data: [DONE]\n\n"

        except Exception as e:
            # Imprime o traceback exato no terminal para identificar se é KeyError ou erro de serialização no Redis
            print(f"[ERRO SSE COMPLETO]: {repr(e)}")
            yield f"data: {json.dumps({'type': 'token', 'content': 'Erro interno.'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")