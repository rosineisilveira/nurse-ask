import asyncio
import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from database.elastic_motor import ElasticMotor
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.application.rag_workflow.state import RAGState
from langchain_core.prompts import ChatPromptTemplate
from app.application.rag_workflow.schemas import RouteTarget, GuardrailDecision, EvaluationDecision
import logging
# Configuração de Logger para substituir os prints em produção
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

llm = ChatOllama(model="llama3.1", temperature=0.1, streaming=True, num_predict=1000)

motor = ElasticMotor(index_name="protocolos_enfermagem")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

async def guardrail_router_node(state: RAGState) -> dict:
    question = state.get("question", "").strip()
    
    logger.info(f"\n🛡️ [NODE] Guardrail acionado para a pergunta: '{question}'")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é o porteiro do NurseAsk. O app só aceita perguntas sobre saúde, enfermagem e medicina. Responda usando o formato JSON exigido."),
        ("human", "{question}")
    ])

    try:
        router_llm = llm.with_structured_output(GuardrailDecision)
        chain = prompt | router_llm
        
        decision = await chain.ainvoke({"question": question})

        if not decision.is_safe:
            logger.warning("🛑 [NODE] Requisição barrada pelo Guardrail. Fora de escopo clínico.")
            return {
                "is_out_of_scope": True,
                "fallback_reason": "fora_de_escopo",
                "generation": "Desculpe, sou o assistente clínico NurseAsk e sou focado exclusivamente em protocolos de enfermagem.",
                "standalone_question": question
            }

        logger.info("✅ [NODE] Guardrail aprovado. Roteando para o Elasticsearch.")
        
        # Garante a variável de busca para que o retriever_node não faça consultas vazias
        return {
            "is_out_of_scope": False,
            "standalone_question": question
        }
        
    except Exception as e:
        logger.error(f"❌ [NODE] Erro crítico no guardrail_router_node: {e}", exc_info=True)
        
        # Fail Open: Em caso de falha no modelo, permite seguir o fluxo
        # e garante a standalone_question para não quebrar o banco de dados.
        return {
            "is_out_of_scope": False,
            "standalone_question": question,
            "error": str(e)
        }

# Função roteadora condicional (a que vai na add_conditional_edges)
def route_after_guardrail(state: RAGState) -> str:
    """
    Avalia se o NeMo marcou a entrada como out-of-scope / jailbreak.
    """
    is_out_of_scope = state.get("is_out_of_scope", False)
    
    if is_out_of_scope:
        print("🛑 [ROTA] Redirecionando para o Fallback (Bloqueado).")
        return RouteTarget.FALLBACK.value
        
    print("✅ [ROTA] Entrada segura. Prosseguindo para o fluxo normal.")
    return RouteTarget.RETRIEVER.value # ou 'condense_question'

async def retriever_node(state: dict) -> dict:
    question = state.get("standalone_question") or state.get("question", "")
    
    logger.info(f"\n🔎 [NODE] Retriever acionado para a pergunta: '{question}'")

    try:
        # 1. Delegação Total: O motor faz a busca híbrida, remonta os pais, aplica o reranker e devolve o Top 3 pronto.
        # Não usamos mais asyncio.to_thread aqui, pois o método aretrieve_context já é 100% assíncrono (I/O Bound).
        documents = await motor.aretrieve_context(query=question, top_k_es=40, top_n_final=3)

        if not documents:
            logger.warning("❌ [NODE] Nenhum documento retornado pelo Elasticsearch.")
            return {"documents": [], "question": question, "top_score": -99.0}

        # 2. Extração de Telemetria: Pega a nota do documento melhor ranqueado
        top_score = documents[0].get("score", -99.0)

        logger.info(f"✅ [NODE] Recuperação concluída. {len(documents)} documentos obtidos (Top Score: {top_score:.2f})")

        # 3. Atualização do Estado (RAGState)
        return {
            "documents": documents,
            "question": question,
            "top_score": top_score
        }
    
    except Exception as e:
        # Substitui o print simples por um log de erro rastreável com stack trace
        logger.error(f"❌ [NODE] Erro crítico no retriever_node: {e}", exc_info=True)
        
        # Opcional: Adicionar a chave "error" no retorno permite que uma aresta condicional (conditional edge)
        # no LangGraph identifique a falha do banco de dados e tome uma ação amigável.
        return {
            "documents": [], 
            "question": question, 
            "top_score": -99.0,
            "error": str(e) 
        }

async def generator_node(state: dict, config: RunnableConfig) -> dict:
    question = state.get("question", "")
    if not question and "messages" in state and len(state["messages"]) > 0:
        question = state["messages"][-1].content
        
    documents = state.get("documents", [])

    print("\n" + "="*50)
    print(f"[DEBUG GERADOR] Quantos documentos chegaram aqui? {len(documents)}")
    
    if not documents:
        print("[DEBUG GERADOR] ⚠️ ALERTA: A IA está cega! A lista 'documents' está vazia.")
        context_text = "Nenhuma diretriz encontrada."
        text_sources = "Nenhuma fonte encontrada."
    else:
        context_text = ""
        unique_sources = []
        
        for doc in documents:
            # Extração segura e padronizada
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

            context_text += f"\n--- FONTE (Pág {page} | Seção: {title}) ---\n"
            context_text += content + "\n"
            
            #  Uso correto da string de referência para unicidade 
            reference = f"Pág. {page} - {title}"
            if reference not in unique_sources:
                unique_sources.append(reference)
                
        text_sources = " | ".join(unique_sources)

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
    "**Fontes utilizadas:** {text_sources}"
    
    Responda de forma clara e baseada estritamente no contexto."""

    print(f"[DEBUG GERADOR] Prompt final enviado para IA (primeiros 200 chars):\n{prompt[:200]}...")
    print("="*50 + "\n")

    try:
        response = await llm.ainvoke(prompt, config)
        return {"generation": response.content}
        
    except Exception as e:
        print(f"[ERRO NO GERADOR] Falha de comunicação: {e}")
        return {"generation": "Erro ao comunicar com o Ollama local."}

async def evaluator_node(state: dict) -> dict:
    question = state.get("standalone_question") or state.get("question", "")
    docs = state.get("documents", [])
    
    logger.info(f"\n⚖️ [NODE] Evaluator acionado para a pergunta: '{question}'")

    # 1. Extração à prova de falhas: apenas converte o que veio do banco para string.
    # Sem "ifs" que barram o texto indevidamente.
    context_text = "\n\n".join([str(doc) for doc in docs])

    if not docs:
        logger.warning("⚠️ [NODE] Nenhum documento recebido do Retriever.")
        return {"is_approved": False, "context_valid": False}

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um avaliador de relevância de contexto clínico.
SEJA PERMISSIVO. Se o documento contiver QUALQUER informação útil ou palavra-chave para a pergunta, aprove.
Responda APENAS usando o formato JSON exigido."""),
        ("human", "Pergunta: {question}\n\nDocumentos:\n{context}")
    ])

    try:
        evaluator_llm = llm.with_structured_output(EvaluationDecision)
        chain = prompt | evaluator_llm
        
        decision = await chain.ainvoke({
            "question": question, 
            "context": context_text
        })

        is_valid = decision.is_approved

        if not is_valid:
            logger.warning(f"🛑 [NODE] Contexto reprovado pelo LLM. Motivo: {decision.reason}")
        else:
            logger.info("✅ [NODE] Contexto aprovado. Roteando para a geração.")

        # 2. A CORREÇÃO CRÍTICA DO ESTADO:
        # Retornamos 'context_valid' para garantir que a sua aresta condicional reconheça a aprovação
        return {
            "is_approved": is_valid,
            "context_valid": is_valid  
        }

    except Exception as e:
        logger.error(f"❌ [NODE] Erro no evaluator_node: {e}", exc_info=True)
        # Fail Open absoluto
        return {"is_approved": True, "context_valid": True}

    
async def rewrite_node(state: dict) -> dict:
    # 1. Recupera a pergunta bruta do usuário e a consulta que acabou de falhar
    pergunta_usuario = state.get("question", "")
    consulta_anterior = state.get("standalone_question") or pergunta_usuario
    curr_count = state.get("retry_count", 0)

    proxima_tentativa = curr_count + 1
    print(f"\n🔄 [NODE] Rewriter Iniciado (Processando Tentativa {proxima_tentativa}/3)...")

    prompt = f"""Você é um especialista em busca de dados clínicos de enfermagem.
    A pergunta original do usuário é: '{pergunta_usuario}'
    A consulta de busca anterior no Elasticsearch foi: '{consulta_anterior}' e falhou em trazer o contexto ideal.

    Esta é a TENTATIVA DE REESCRITA #{proxima_tentativa}.
    Gere uma nova consulta de busca COMPLETAMENTE DIFERENTE da anterior:
    - Na tentativa 1: use sinônimos médicos, termos formais ou nomenclatura oficial de enfermagem.
    - Na tentativa 2: extraia apenas as 2 ou 3 palavras-chave fundamentais do procedimento/protocolo.
    - Na tentativa 3: use o nome genérico da patologia/procedimento sem detalhes acessórios.

    Retorne APENAS a nova frase de busca, sem aspas e sem explicações."""
    
    llm_rewrite = ChatOllama(model="llama3.1", temperature=0.4)
    resposta = await llm_rewrite.ainvoke([SystemMessage(content=prompt)])
    nova_consulta = resposta.content.strip()

    print(f"📝 [REWRITER] Nova consulta de busca (#{proxima_tentativa}): '{nova_consulta}'")

    # 2. Retorna 'standalone_question', reseta os documentos e incrementa o contador.
    # NOTA: 'question' e 'messages' NÃO são tocados para não poluir o histórico no Redis!
    return {
        "standalone_question": nova_consulta,
        "documents": [],
        "retry_count": proxima_tentativa
    }

async def condense_question_node(state: dict) -> dict:
    question = state.get("question", "")
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    # Se não houver histórico de conversa ou resumo, mantém a pergunta original
    if not summary and len(messages) <= 1:
        print("ℹ️ [NODE] Sem histórico prévio. Pergunta mantida idêntica.")
        return {"standalone_question": question}

    print("🔄 [NODE] Reformulando pergunta com base no histórico...")

    # Constrói o histórico recente (últimas 4 mensagens)
    historico_recente = ""
    for msg in messages[-4:]:
        role = "Enfermeiro" if isinstance(msg, HumanMessage) else "Assistente"
        historico_recente += f"{role}: {msg.content}\n"

    prompt_reescrita = f"""Você é um assistente técnico de reescrita para RAG de enfermagem.
    Sua tarefa é reformular a última pergunta do usuário para que ela se torne totalmente AUTÔNOMA e compreensível sem contexto anterior, preservando todas as siglas médicas, termos técnicos e medicamentos.

    Resumo do Contexto Anterior:
    {summary if summary else "Nenhum resumo anterior."}

    Histórico Recente:
    {historico_recente}

    Última pergunta do Usuário: "{question}"

    Regras:
    1. Retorne APENAS a pergunta reescrita de forma direta, sem saudações ou explicações.
    2. Se a pergunta já for autônoma (ex: "Qual a dosagem de Paracetamol?"), retorne-a exatamente como está.
    3. Se a pergunta for anafórica (ex: "Quais os materiais para isso?"), substitua "isso" pelo procedimento mencionado anteriormente.

    Pergunta Autônoma:"""
    llm_condense = ChatOllama(model="llama3.1", temperature=0)
    resposta = await llm_condense.ainvoke([SystemMessage(content=prompt_reescrita)])
    standalone = resposta.content.strip()

    print(f"🎯 Pergunta reformulada para busca: '{standalone}'")
    return {"standalone_question": standalone}


# --- NÓ 2: SUMARIZAÇÃO COM PRESERVAÇÃO CLÍNICA ---
async def summarize_node(state: dict, max_messages: int = 6) -> dict:
    messages = state.get("messages", [])

    # Só dispara a sumarização se a conversa ultrapassar o limiar
    if len(messages) <= max_messages:
        return {}

    print(f"🧹 [NODE] Limiar de mensagens atingido ({len(messages)}). Gerando resumo...")

    summary_atual = state.get("summary", "")

    # Seleciona as mensagens antigas para condensar (exceto o último turno)
    mensagens_para_resumir = messages[:-2]
    mensagens_para_manter = messages[-2:]

    texto_mensagens = ""
    for msg in mensagens_para_resumir:
        role = "Enfermeiro" if isinstance(msg, HumanMessage) else "Assistente"
        texto_mensagens += f"{role}: {msg.content}\n"

    prompt_sumarizacao = f"""Você é um sistema de memória clínica para um assistente de enfermagem.
    Sua missão é atualizar o resumo da conversa mantendo OBRIGATORIAMENTE todos os detalhes técnicos relevantes.

    Resumo Anterior:
    {summary_atual if summary_atual else "Início da conversa."}

    Novas Mensagens a Condensar:
    {texto_mensagens}

    Regras Estritas de Preservação:
    1. MANTENHA TODOS os nomes de procedimentos (ex: Posição de Sims, DDH), calibres de sondas, dosagens, materiais, vias de administração e sinais vitais citados.
    2. Descarte saudações, desculpas, agradecimentos e conversas fiadas.
    3. Escreva um resumo denso, objetivo e em parágrafo único.

    Novo Resumo Clínico:"""
    llm_sumurize = ChatOllama(model="llama3.1", temperature=0)
    resposta = await llm_sumurize.ainvoke([SystemMessage(content=prompt_sumarizacao)])
    novo_resumo = resposta.content.strip()

    # Cria instruções de remoção para apagar mensagens antigas do estado no Redis
    mensagens_removidas = [RemoveMessage(id=m.id) for m in mensagens_para_resumir if hasattr(m, "id") and m.id]

    print(f"✅ Resumo atualizado ({len(novo_resumo)} chars). Mensagens antigas removidas do Redis.")
    return {
        "summary": novo_resumo,
        "messages": mensagens_removidas
    }


async def fallback_node(state: RAGState) -> dict:
    print("🛟 [FALLBACK NODE] Encerrando com mensagem de segurança.")
    
    safe_response = state.get(
        "generation", 
        "Desculpe, sou o assistente clínico NurseAsk e não posso processar essa solicitação."
    )
    #  apenas a frase limpa vinda do neMo
    match = re.search(r'"([^"]+)"', safe_response)
    if match:
        safe_response = match.group(1) 
     
    return {"generation": safe_response}
    
    


# async def critic_node(state: dict) -> dict:
#     """Avalia a qualidade da resposta gerada antes de enviar."""
#     return {"is_valid": True}

