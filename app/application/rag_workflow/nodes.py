import asyncio
import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from database.elastic_motor import ElasticMotor
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


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

print("🧠 Carregando CrossEncoder (Re-ranker)...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

llm = ChatOllama(model="llama3.1", temperature=0.1, streaming=True, num_predict=1000)



async def retriever_node(state: dict) -> dict:
    question = state.get("question", "")
    print(f"\n🔎 [NODE] Retriever buscando e reclassificando para: '{question}'")
    
    try:
        # 1. Busca bruta no Elasticsearch (puxa 15 candidatos para dar margem)
        results = await asyncio.to_thread(db.similarity_search, question, top_k=15)
        
        if not results:
            print("❌ Nenhum resultado encontrado no Elasticsearch.")
            return {"documents": [], "question": question, "top_score": -99.0}

        # 2. Re-classificação com CrossEncoder (calcula precisão semântica)
        pares = [[question, doc.page_content] for doc in results]
        notas = reranker.predict(pares)
        
        # Agrupa documento com sua respectiva nota
        docs_com_notas = list(zip(results, notas))
        
        # Ordena do maior score para o menor
        docs_com_notas.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Consolidação e Deduplicação dos Blocos Pais / Chunks
        parent_documents = {}
        for doc, score in docs_com_notas:
            # Identificador do pai (se usar Parent-Child) ou o próprio ID do chunk
            parent_id = doc.metadata.get("parent_id") or doc.metadata.get("id") or hash(doc.page_content)
            parent_text = doc.metadata.get("parent_content", doc.page_content)
            
            # Garante que só pegamos a melhor ocorrência de uma mesma tabela/seção
            if parent_id not in parent_documents:
                parent_documents[parent_id] = {
                    "source": doc.metadata.get("source", "Desconhecido"),
                    "page": doc.metadata.get("page", "N/A"),
                    "hierarquia": doc.metadata.get("hierarquia", ""),
                    "content": parent_text,
                    "score": float(score)
                }

        # Seleciona os 3 melhores documentos/tabelas após o re-ranking
        documents = list(parent_documents.values())[:3]
        
        top_score = documents[0]["score"] if documents else -99.0
        print(f"✅ Re-ranking concluído com sucesso. Melhores trechos selecionados (Top Score: {top_score:.2f})")
        
        return {
            "documents": documents, 
            "question": question,
            "top_score": top_score
        }
        
    except Exception as e:
        print(f"❌ Erro no retriever_node: {e}")
        return {"documents": [], "question": question, "top_score": -99.0}



    

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
        context_text = ""
        fontes_unicas = []
        
        for doc in documents:
            # Extração segura: funciona para LangChain Document ou dict tradicional
            if hasattr(doc, "page_content"):
                content = doc.page_content
                meta = getattr(doc, "metadata", {})
                page = meta.get("page", 1)
                title = meta.get("title") or meta.get("source") or "Sem título"
            elif isinstance(doc, dict):
                content = doc.get("content") or doc.get("text") or ""
                page = doc.get("page", 1)
                title = doc.get("title") or doc.get("source") or "Sem título"
            else:
                content = str(doc)
                page = 1
                title = "Sem título"

            # Envia o TÍTULO e a PÁGINA para o Llama entender o contexto de onde tirou
            context_text += f"\n--- FONTE (Pág {page} | Seção: {title}) ---\n"
            context_text += content + "\n"
            
            # Guarda a referência para imprimir no frontend
            referencia = f"Pág. {page} - {title}"
            if referencia not in fontes_unicas:
                fontes_unicas.append(referencia)
                
        texto_fontes = " | ".join(fontes_unicas)

    prompt = f"""Você é o NurseAsk, um assistente clínico especializado em protocolos de enfermagem.
    Sua missão é responder à pergunta do estudante usando APENAS as informações presentes no CONTEXTO CLÍNICO EXTRAÍDO abaixo.

    REGRAS OBRIGATÓRIAS DE FORMATAÇÃO:
    1. FOCO TOTAL: Responda exatamente o que foi perguntado. Não adicione introduções ou conclusões genéricas.
    2. PROIBIDO ALUCINAR: Se a resposta não estiver no contexto, diga apenas: "Não encontrei essa informação no protocolo atual."
    3. COMO LIDAR COM TABELAS:
    - Se o usuário pedir um dado específico (ex: "dose da aspirina"), extraia apenas esse dado.
    - Se o usuário pedir uma lista (ex: "quais os tubos", "quais os sintomas"), liste TODOS os itens encontrados na tabela usando marcadores em texto (bullet points).
    - NUNCA tente recriar ou desenhar a tabela original usando barras (|) ou formatação Markdown de tabela, pois isso quebra o sistema. Converta a tabela para texto descritivo.
    4. SEM REPETIÇÃO: Jamais repita a mesma frase ou item na resposta.

    CONTEXTO CLÍNICO EXTRAÍDO:
    {context_text}

    PERGUNTA DO ESTUDANTE:
    {question}
    
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

async def evaluator_node(state: dict) -> dict:
    print(f"⚖️ [DEBUG EVALUATOR] retry_count ao entrar no Evaluator: {state.get('retry_count')}")
    print("\n⚖️ [NODE] Evaluator Iniciado...")
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        print("❌ JUIZ: Nenhum documento retornado.")
        return {"context_valid": False}
        
        
    context_text = "\n".join([doc.get("content", "") for doc in documents])
    
    prompt_juiz = f"""Responda EXATAMENTE com a palavra SIM ou NÃO, sem nenhuma pontuação ou texto adicional.
    O contexto abaixo possui a resposta para a pergunta?
    CONTEXTO: {context_text}
    PERGUNTA: {question}"""
    
    llm_juiz = ChatOllama(model="llama3.1", temperature=0.0)
    resposta = await llm_juiz.ainvoke(prompt_juiz)
    veredicto = resposta.content.strip().upper()
    
    if "SIM" in veredicto:
        print("✅ JUIZ: Aprovado.")
        return {"context_valid": True}
    else:
        print("❌ JUIZ: Reprovado.")
        return {"context_valid": False}


async def rewrite_node(state: dict) -> dict:
    orig_q = state.get("original_question") or state.get("question", "")
    curr_q = state.get("question", "")
    curr_count = state.get("retry_count", 0)
    
    proxima_tentativa = curr_count + 1
    print(f"\n🔄 [NODE] Rewriter Iniciado (Processando Tentativa {proxima_tentativa}/3)...")

    prompt = f"""Você é um especialista em busca de dados clínicos.
    A pergunta original do usuário foi: '{orig_q}'
    A busca anterior usou: '{curr_q}' e falhou.

    Esta é a TENTATIVA DE BUSCA #{proxima_tentativa}.
    Gere uma nova consulta de busca COMPLETAMENTE DIFERENTE da anterior:
    - Na tentativa 1: use sinônimos médicos ou termos formais.
    - Na tentativa 2: extraia apenas as 2 ou 3 palavras-chave fundamentais do protocolo.
    - Na tentativa 3: use o nome genérico da patologia/procedimento sem detalhes.

    Retorne APENAS a nova frase de busca, sem aspas e sem explicações."""

    # Temperature 0.4 força variações reais a cada tentativa
    llm_rewrite = ChatOllama(model="llama3.1", temperature=0.4)
    resposta = await llm_rewrite.ainvoke(prompt)
    nova_pergunta = resposta.content.strip()

    print(f"📝 [REWRITER] Nova busca (#{proxima_tentativa}): '{nova_pergunta}'")

    # Retorna a nova pergunta, limpa os documentos e INCREMENTA o contador explicitamente
    return {
        "question": nova_pergunta,
        "original_question": orig_q,
        "documents": [],
        "retry_count": proxima_tentativa
    }
    
# async def router_node(state: dict) -> dict:
#     """Determina o tipo de pergunta do usuário."""
#     # Por enquanto, assumimos que tudo é pergunta médica (triage)
#     return {"intent": "medical_rag"}




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

