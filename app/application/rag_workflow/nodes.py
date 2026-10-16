import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig

load_dotenv()

# Configuração Qdrant (HuggingFace)
qdrant_client = QdrantClient(url="http://localhost:6333")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

qdrant_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="clinical_guidelines_hf",
    embedding=embeddings,
)

# Configuração OpenAI (Gerador)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

llm = ChatOllama(model="llama3.1", temperature=0, streaming=True)

async def retriever_node(state: dict) -> dict:
    """Busca o protocolo com diagnósticos detalhados."""
    question = state.get("question", "")
    
    print("\n" + "="*50)
    print("[DEBUG] Iniciando diagnóstico do Retriever...")
    print("="*50)
    
    # TESTE 1: A conexão com o Docker do Qdrant está de fato ativa?
    print("[DEBUG] 1. Testando conexão com o Qdrant (Docker na porta 6333)...")
    try:
        # Força um timeout rápido de 3 segundos para não travar para sempre
        from qdrant_client.http import exceptions
        info = await asyncio.to_thread(qdrant_client.get_collections)
        print(f"[✅ OK] Qdrant conectado! Coleções encontradas: {[c.name for c in info.collections]}")
    except Exception as e:
        print(f"[❌ ERRO] Não foi possível conectar ao Qdrant: {e}")
        print("[DICA] O seu Docker do Qdrant está rodando? Use 'docker ps' no terminal.")
        return {"documents": []}

    # TESTE 2: O modelo da Huggingface está travando para gerar o vetor?
    print("[DEBUG] 2. Testando modelo de Embeddings (HuggingFace)...")
    try:
        # Testa gerar o vetor de uma palavra simples
        print("[DEBUG] Tentando vetorizar a palavra 'teste'...")
        vetor = await asyncio.to_thread(embeddings.embed_query, "teste")
        print(f"[✅ OK] Embeddings funcionando! Tamanho do vetor: {len(vetor)}")
    except Exception as e:
        print(f"[❌ ERRO] O modelo HuggingFace travou ou falhou: {e}")
        print("[DICA] Isso geralmente é problema de SSL/rede bloqueada ou falta de memória.")
        return {"documents": []}

    # TESTE 3: A busca real
    print("[DEBUG] 3. Tentando a busca real no Qdrant...")
    try:
        # Executa a busca com tratamento
        results = await asyncio.to_thread(
            qdrant_store.similarity_search, question, k=1
        )
        print(f"[✅ OK] Busca finalizada com sucesso! Achou {len(results)} documentos.")
        
        documents = [
            {"source": res.metadata.get("source", "Desconhecida"), "content": res.page_content}
            for res in results
        ]
        return {"documents": documents}
        
    except Exception as e:
        print(f"[❌ ERRO] A busca final falhou: {e}")
        return {"documents": []}

async def generator_node(state: dict, config: RunnableConfig) -> dict:
    question = state.get("question", "")
    documents = state.get("documents", [])

    print("\n" + "="*50)
    print(f"[DEBUG GERADOR] Quantos documentos chegaram aqui? {len(documents)}")
    if not documents:
        print("[DEBUG GERADOR] ⚠️ ALERTA: A IA está cega! A lista 'documents' está vazia.")
    
    context_text = "\n".join([doc["content"] for doc in documents]) if documents else "Nenhuma diretriz encontrada."
    
    prompt = f"""Você é o NurseAsk, um assistente especializado para estudantes de enfermagem.
    Você deve responder à pergunta do usuário baseando-se EXCLUSIVAMENTE nas DIRETRIZES CLÍNICAS fornecidas abaixo.
    Se a resposta não estiver nas diretrizes, diga 'Não encontrei essa informação no protocolo atual'.
    NÃO invente informações. NÃO traga conhecimento externo.

    Contexto clínico: {context_text}
    
    Pergunta do usuário: {question}
    
    Responda de forma clara e baseada estritamente no contexto."""

    print(f"[DEBUG GERADOR] Prompt final enviado para IA (primeiros 200 chars):\n{prompt[:200]}...")
    print("="*50 + "\n")

    try:
        # O SEGREDO ESTÁ AQUI: Passar o 'config' avisa o LangGraph para transmitir as palavras!
        response = await llm.ainvoke(prompt, config)
        return {"generation": response.content}
    except Exception as e:
        print(f"[ERRO NO GERADOR] Falha de comunicação: {e}")
        return {"generation": "Erro ao comunicar com o Ollama local."}
    
# --- Funções de Roteamento (Esqueleto para o Graph funcionar) ---

# async def router_node(state: dict) -> dict:
#     """Determina o tipo de pergunta do usuário."""
#     # Por enquanto, assumimos que tudo é pergunta médica (triage)
#     return {"intent": "medical_rag"}


# async def supervisor_node(state: dict) -> dict:
#     """Supervisiona se a resposta tem qualidade."""
#     # Por enquanto, aprova tudo e manda encerrar o fluxo
#     return {"next_step": "end"}


# async def fallback_node(state: dict) -> dict:
#     """Responde quando não entende a pergunta."""
#     return {"answer": "Desculpe, só posso responder sobre protocolos clínicos baseados em diretrizes médicas."}


# async def critic_node(state: dict) -> dict:
#     """Avalia a qualidade da resposta gerada antes de enviar."""
#     # Por enquanto, aprova todas as respostas automaticamente
#     return {"is_valid": True}

# async def guardrail_node(state: dict) -> dict:
#     """Verifica se a pergunta ou resposta viola regras de segurança médica."""
#     # Por enquanto, deixa tudo passar e não altera o estado
#     return {}

