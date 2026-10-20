import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from database.elastic_motor import ElasticMotor
from langchain_core.messages import SystemMessage



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("🔌 [NÓ] Conectando motor do banco ao LangGraph...")
db = ElasticMotor()

load_dotenv()

# Configuração Qdrant (HuggingFace)
qdrant_client = QdrantClient(url="http://localhost:6333")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

qdrant_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="clinical_guidelines_hf",
    embedding=embeddings,
)



llm = ChatOllama(model="llama3.1", temperature=0, streaming=True)


# (Certifique-se de que o db = ElasticMotor() já está inicializado no topo do arquivo)
# db = ElasticMotor()

async def retriever_node(state: dict) -> dict:
    """Busca o protocolo com diagnósticos detalhados adaptado para Elasticsearch."""
    # Captura a pergunta de forma resiliente
    question = state.get("question", "")
    if not question and "messages" in state and len(state["messages"]) > 0:
        question = state["messages"][-1].content
        
    print("\n" + "="*50)
    print("[DEBUG] Iniciando diagnóstico do Retriever...")
    print("="*50)
    
    # TESTE 1: A conexão com o Docker do Elasticsearch está de fato ativa?
    print("[DEBUG] 1. Testando conexão com o Elasticsearch (Docker na porta 9200)...")
    try:
        # Acessa o cliente interno do Elastic via nossa classe e faz um 'ping' rápido
        info = await asyncio.to_thread(db.store.client.info)
        cluster_name = info.get("cluster_name", "desconhecido")
        print(f"[✅ OK] Elasticsearch conectado! Cluster: {cluster_name}")
    except Exception as e:
        print(f"[❌ ERRO] Não foi possível conectar ao Elasticsearch: {e}")
        print("[DICA] O seu Docker do Elasticsearch está rodando? Use 'docker ps' no terminal.")
        return {"documents": []}

    # TESTE 2: O modelo da Huggingface está travando para gerar o vetor?
    print("[DEBUG] 2. Testando modelo de Embeddings (HuggingFace)...")
    try:
        # Testa gerar o vetor de uma palavra simples usando os embeddings do motor
        print("[DEBUG] Tentando vetorizar a palavra 'teste'...")
        vetor = await asyncio.to_thread(db.embeddings.embed_query, "teste")
        print(f"[✅ OK] Embeddings funcionando! Tamanho do vetor: {len(vetor)}")
    except Exception as e:
        print(f"[❌ ERRO] O modelo HuggingFace travou ou falhou: {e}")
        print("[DICA] Isso geralmente é problema de memória RAM ou download interrompido do modelo.")
        return {"documents": []}

    # TESTE 3: A busca real
    print(f"[DEBUG] 3. Tentando a busca real para: '{question[:30]}...'")
    try:
        # Executa a busca através do nosso ElasticMotor
        results = await asyncio.to_thread(db.similarity_search, question, top_k=3)
        print(f"[✅ OK] Busca finalizada com sucesso! Achou {len(results)} documentos.")
        
        documents = []
        fontes_debug = []
        
        for res in results:
            # Extração refinada: Pegamos o nome do arquivo e a página gerados pelo PyMuPDF
            source = res.metadata.get("source", "Desconhecida")
            pagina = res.metadata.get("page", 0) + 1
            fontes_debug.append(f"Pág {pagina}")
            
            documents.append({
                "source": source,
                "page": pagina,
                "content": res.page_content
            })
            
        print(f"[DEBUG] Fontes extraídas para a IA: {fontes_debug}")
        print("="*50 + "\n")
        
        return {"documents": documents}
        
    except Exception as e:
        print(f"[❌ ERRO] A busca final falhou: {e}")
        print("="*50 + "\n")
        return {"documents": []}


async def generator_node(state: dict, config: RunnableConfig) -> dict:
    question = state.get("question", "")
    if not question and "messages" in state and len(state["messages"]) > 0:
        question = state["messages"][-1].content
        
    # Pega os documentos que o retriever_node acabou de buscar e salvar no state
    documents = state.get("documents", [])

    print("\n" + "="*50)
    print(f"[DEBUG GERADOR] Quantos documentos chegaram aqui? {len(documents)}")
    
    if not documents:
        print("[DEBUG GERADOR] ⚠️ ALERTA: A IA está cega! A lista 'documents' está vazia.")
        context_text = "Nenhuma diretriz encontrada."
        texto_fontes = "Nenhuma fonte encontrada."
    else:
        # Extrai o texto limpo
        context_text = "\n".join([doc["content"] for doc in documents])
        
        # Extrai as páginas únicas para não repetir (Ex: "Página 2, Página 5")
        paginas_unicas = list(set([str(doc["page"]) for doc in documents]))
        texto_fontes = f"Página(s): {', '.join(paginas_unicas)}"

    prompt = f"""Você é o NurseAsk, um assistente especializado para estudantes de enfermagem.
    Você deve responder à pergunta do usuário baseando-se EXCLUSIVAMENTE nas DIRETRIZES CLÍNICAS fornecidas abaixo.
    Se a resposta não estiver nas diretrizes, diga 'Não encontrei essa informação no protocolo atual'.
    NÃO invente informações. NÃO traga conhecimento externo.

    Contexto clínico: {context_text}
    
    Pergunta do usuário: {question}
    
    REGRA OBRIGATÓRIA: Ao final da sua resposta, pule uma linha e escreva EXATAMENTE:
    "**Fontes utilizadas:** {texto_fontes}"
    
    Responda de forma clara e baseada estritamente no contexto."""

    print(f"[DEBUG GERADOR] Prompt final enviado para IA (primeiros 200 chars):\n{prompt[:200]}...")
    print("="*50 + "\n")

    try:
        # O config avisa o LangGraph para transmitir as palavras (SSE)
        response = await llm.ainvoke(prompt, config)
        
        # RETORNO CORRIGIDO: Voltamos para o formato que o seu main.py espera!
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

