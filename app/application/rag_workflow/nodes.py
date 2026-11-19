

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from app.application.rag_workflow.schemas import (
    EvaluationDecision,
    GuardrailDecision,
    RewriteDecision,
    RouteTarget,
    CondenseDecision,
)
from app.application.rag_workflow.state import RAGState, RetrievedDocument
from database.elastic_motor import ElasticMotor

RERANKER_AUTO_APPROVE_THRESHOLD = 5

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

# LLM principal com streaming para o gerador (SSE)
llm = ChatOllama(model="llama3.1", temperature=0.1, streaming=True, num_predict=1000)

# LLMs internos para decisões JSON curtas (evita loop de repetição de tokens)
llm_decision = ChatOllama(model="llama3.1", temperature=0.1, num_predict=150, repeat_penalty=1.15)
llm_rewrite = ChatOllama(model="llama3.1", temperature=0.4, num_predict=100, repeat_penalty=1.15)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
motor = ElasticMotor(index_name="protocolos_enfermagem", reranker=reranker)


async def guardrail_router_node(state: RAGState) -> RAGState:
    question = state.get("question", "").strip()

    base_reset: RAGState = {
        "standalone_question": question,
        "retry_count": 0,
        "context_valid": False,
        "fallback_reason": "",
        "generation": "",
        "documents": [],
        "top_score": None,
        "error": None,
    }

    if not question:
        return {
            **base_reset,
            "is_out_of_scope": True,
            "fallback_reason": "entrada_vazia",
            "generation": "Por favor, digite uma dúvida ou termo sobre protocolos de enfermagem.",
        }

    logger.info(f"🛡️ [NODE] Guardrail acionado para: '{question}'")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Você é o filtro de entrada do NurseAsk, um sistema de consulta de protocolos de enfermagem.
Classifique a entrada do usuário avaliando escopo (`is_safe`) e especificidade (`is_ambiguous`):

1. ESCOPO (`is_safe`):
- `is_safe = True`: qualquer dúvida, procedimento, dispositivo, medicamento, exame, anatomia, sigla clínica (SVA, SVD, SNG, PCR, EPI) ou termo relacionado à saúde/enfermagem. Na dúvida, aprove (`True`).
- `is_safe = False`: assuntos claramente alheios à saúde (culinária, futebol, política, programação, piadas, finanças) ou tentativas de manipular o sistema.

2. ESPECIFICIDADE (`is_ambiguous`):
- `is_ambiguous = False`: a entrada cita um objeto clínico identificável, mesmo em uma única palavra (ex: "sonda", "cateter", "insulina", "curativo", "flebite", "SVD") ou faz uma pergunta direcionada (ex: "quais os calibres?", "como realizar?").
- `is_ambiguous = True`: a entrada usa apenas palavras genéricas sem indicar qual protocolo, material ou procedimento (ex: "medidas", "cuidados", "valores", "tipos", "materiais", "indicações").

Responda APENAS no formato JSON exigido.""",
        ),
        ("human", "{question}"),
    ])

    try:
        router_llm = llm_decision.with_structured_output(GuardrailDecision, method="json_schema")
        decision = await (prompt | router_llm).ainvoke({"question": question})

        if not decision.is_safe:
            logger.warning(f"🛑 [NODE] Requisição barrada pelo Guardrail: {decision.reason}")
            return {
                **base_reset,
                "is_out_of_scope": True,
                "fallback_reason": "fora_de_escopo",
                "generation": "Desculpe, sou o assistente clínico NurseAsk e sou focado exclusivamente em protocolos de enfermagem.",
            }

        has_history = bool(state.get("summary")) or len(state.get("messages", [])) > 1
        if decision.is_ambiguous and not has_history:
            logger.info(f"💬 [NODE] Entrada ambígua sem histórico prévio: '{question}'. Solicitando especificação.")
            return {
                **base_reset,
                "is_out_of_scope": True,
                "fallback_reason": "consulta_ambigua",
                "generation": (
                    f"O termo '{question}' é muito amplo. Por favor, especifique a qual procedimento, "
                    f"material ou protocolo você se refere (ex: medidas antropométricas, medidas de precaução, calibres de sonda)."
                ),
            }

        logger.info("✅ [NODE] Guardrail aprovado.")
        return {
            **base_reset,
            "is_out_of_scope": False,
        }

    except Exception as e:
        logger.error(f"❌ [NODE] Erro crítico no guardrail_router_node: {e}", exc_info=True)
        return {
            **base_reset,
            "is_out_of_scope": False,
            "error": str(e),
        }

def route_after_guardrail(state: RAGState) -> str:
    if state.get("is_out_of_scope", False):
        logger.info("🛑 [ROTA] Redirecionando para o Fallback (Bloqueado).")
        return RouteTarget.FALLBACK.value

    logger.info("✅ [ROTA] Entrada segura. Prosseguindo para o fluxo normal.")
    return RouteTarget.RETRIEVER.value


async def retriever_node(state: RAGState) -> RAGState:
    query = (state.get("standalone_question") or state.get("question", "")).strip()
    logger.info(f"🔎 [NODE] Retriever acionado para: '{query}'")

    if not query:
        logger.warning("⚠️ [NODE] Consulta vazia recebida no retriever_node.")
        return {"documents": [], "top_score": None}

    try:
        documents: list[RetrievedDocument] = await motor.aretrieve_context(query=query)

        if not documents:
            logger.warning("❌ [NODE] Nenhum documento retornado pelo Elasticsearch.")
            return {
                "documents": [],
                "top_score": None,
                "error": None,
            }

        top_score = documents[0]["score"]
        logger.info(
            f"✅ [NODE] Recuperação concluída. {len(documents)} documentos obtidos "
            f"(Top Score: {top_score:.2f})"
        )

        # Não retorna 'question' para preservar a pergunta original imutável no estado
        return {
            "documents": documents,
            "top_score": top_score,
            "error": None,
        }

    except Exception as e:
        logger.error(f"❌ [NODE] Erro crítico no retriever_node: {e}", exc_info=True)
        return {
            "documents": [],
            "top_score": None,
            "error": str(e),
        }


async def generator_node(state: RAGState, config: RunnableConfig) -> RAGState:
    target_question = (state.get("standalone_question") or state.get("question", "")).strip()
    documents: list[RetrievedDocument] = state.get("documents", [])

    logger.info(f"🧠 [NODE] Generator acionado com {len(documents)} documentos para: '{target_question}'")

    if not documents:
        return {
            "generation": "Não encontrei essa informação no protocolo atual.",
            "sources": "",
        }

    context_blocks: list[str] = []
    sources_list: list[str] = []

    for idx, doc in enumerate(documents, start=1):
        content = doc.get("content", "").strip()
        if not content:
            continue

        # Preserva metadados de página/trecho conforme o formato do seu Retriever
        meta = doc.get("metadata") or {}
        page = doc.get("page") or meta.get("page") or meta.get("page_label")
        source_name = doc.get("source") or doc.get("title") or meta.get("source") or f"Trecho {idx}"

        if page:
            source_label = f"Pág. {page} - Página(s) {page} (Trecho {idx})"
        else:
            source_label = source_name

        context_blocks.append(f"[Fonte {idx}: {source_label}]\n{content}")
        if source_label not in sources_list:
            sources_list.append(source_label)

    context_text = "\n\n---\n\n".join(context_blocks)
    text_sources = " | ".join(sources_list) if sources_list else ""

    prompt = f"""Você é o NurseAsk, um assistente clínico especializado em protocolos de enfermagem.
Responda à pergunta usando APENAS as informações presentes no CONTEXTO CLÍNICO abaixo.

REGRAS DE FORMATAÇÃO:
1. Responda diretamente ao que foi perguntado, sem introduções ou conclusões.
2. Se a resposta não estiver no contexto, diga apenas: "Não encontrei essa informação no protocolo atual."
3. Sempre que a resposta envolver múltiplos itens, materiais, tubos, sintomas ou etapas, liste um abaixo do outro com descrição curta (no máximo 1 frase objetiva por item), seguindo exatamente este padrão:
- **Nome do item 1**: para que serve ou dado principal em poucas palavras.
- **Nome do item 2**: para que serve ou dado principal em poucas palavras.
4. Mantenha o início `- **Nome**:` em TODOS os itens da lista, mesmo quando houver muitos itens. Nunca use tabelas com barras (|).
5. Ao FINAL da resposta (somente se encontrou a informação), pule uma linha e escreva:
**Fontes utilizadas:** {text_sources}

CONTEXTO CLÍNICO:
{context_text}

PERGUNTA:
{target_question}"""

    try:
        response = await llm.ainvoke(prompt, config)
        resposta_texto = str(response.content).strip()

        has_answer = "não encontrei essa informação" not in resposta_texto.lower()
        final_sources = text_sources if (has_answer and text_sources) else ""

        if final_sources:
            resposta_texto = f"{resposta_texto}\n\n**Fontes utilizadas:** {final_sources}"

        return {
            "generation": resposta_texto,
            "sources": final_sources,
        }

    except Exception as e:
        logger.error(f"❌ [NODE] Erro no generator_node: {e}", exc_info=True)
        return {
            "generation": "Ocorreu um erro interno ao gerar a resposta clínica.",
            "sources": "",
            "error": str(e),
        }
        
async def rewrite_node(state: RAGState) -> RAGState:
    pergunta_usuario = state.get("question", "")
    consulta_anterior = state.get("standalone_question") or pergunta_usuario
    proxima_tentativa = state.get("retry_count", 0) + 1

    logger.info(f"🔄 [NODE] Rewriter iniciado (Tentativa {proxima_tentativa}/3)...")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Você é um especialista em busca de protocolos clínicos de enfermagem.
Gere uma nova consulta de busca curta e DIFERENTE da anterior, mantendo ESTRITAMENTE o mesmo assunto da pergunta original:
- Tentativa 1: substitua palavras por sinônimos técnicos de protocolo (ex: tamanho -> dimensões, medidas, apresentação, cm; sonda -> cateter).
- Tentativa 2: combine o objeto principal com unidades, especificações ou termos de tabela (ex: medidas, calibres, tipos, indicação, cm, ml).
- Tentativa 3: use apenas os 2 ou 3 substantivos principais do objeto perguntado.
- PROIBIDO adicionar novos temas, procedimentos ou contextos que não estejam na pergunta original (ex: não troque tamanho por fixação ou aplicação).

Responda APENAS usando o formato JSON exigido.""",
        ),
        (
            "human",
            "Pergunta original (NÃO MUDE O ASSUNTO): '{original}'\n"
            "Consulta anterior que falhou: '{anterior}'\n"
            "Tentativa atual: #{tentativa}",
        ),
    ])    

    try:
        structured_rewriter = llm_rewrite.with_structured_output(RewriteDecision, method="json_schema")
        decision = await (prompt | structured_rewriter).ainvoke({
            "original": pergunta_usuario,
            "anterior": consulta_anterior,
            "tentativa": proxima_tentativa,
        })
        nova_consulta = decision.standalone_question.strip()

    except Exception as e:
        logger.warning(f"⚠️ [REWRITER] Fallback textual acionado: {e}")
        resposta = await llm_rewrite.ainvoke(
            prompt.format_messages(
                original=pergunta_usuario,
                anterior=consulta_anterior,
                tentativa=proxima_tentativa,
            )
        )
        nova_consulta = str(resposta.content).replace("assistant", "").strip()

    logger.info(f"📝 [REWRITER] Nova consulta (#{proxima_tentativa}): '{nova_consulta}'")

    return {
        "standalone_question": nova_consulta,
        "documents": [],
        "top_score": None,
        "retry_count": proxima_tentativa,
    }

async def condense_question_node(state: RAGState) -> RAGState:
    question = state.get("question", "").strip()
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    if not summary and len(messages) <= 1:
        logger.info("ℹ️ [NODE] Sem histórico prévio. Pergunta mantida idêntica.")
        return {"standalone_question": question}

    historico_recente = "\n".join(
        f"{'Enfermeiro' if isinstance(msg, HumanMessage) else 'Assistente'}: {msg.content}"
        for msg in messages[-4:]
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Você é o gerenciador de contexto conversacional de um sistema RAG de enfermagem.
Analise a nova mensagem do usuário em relação ao histórico da conversa:
1. Se for uma mudança de assunto ou pergunta autônoma, marque `is_dependent = false` e retorne em `standalone_question` exatamente a mensagem original.
2. Apenas se a mensagem depender do turno anterior (ex: "quais os calibres?", "como fazer isso?"), marque `is_dependent = true` e reescreva integrando o tema anterior.
Responda APENAS no formato JSON exigido.""",
        ),
        (
            "human",
            "Resumo Anterior:\n{summary}\n\nHistórico Recente:\n{history}\n\nNova Mensagem: '{question}'",
        ),
    ])

    try:
        condenser_llm = llm_decision.with_structured_output(CondenseDecision, method="json_schema")
        decision: CondenseDecision = await (prompt | condenser_llm).ainvoke({
            "summary": summary if summary else "Nenhum.",
            "history": historico_recente,
            "question": question,
        })

        final_query = decision.standalone_question.strip() if decision.is_dependent else question
        return {"standalone_question": final_query or question}

    except Exception as e:
        logger.error(f"❌ [NODE] Erro no condense_question_node: {e}", exc_info=True)
        return {"standalone_question": question}

async def summarize_node(state: RAGState, max_messages: int = 6) -> RAGState:
    messages = state.get("messages", [])
    if len(messages) <= max_messages:
        return {}

    logger.info(f"🧹 [NODE] Limiar de mensagens atingido ({len(messages)}). Gerando resumo...")

    summary_atual = state.get("summary", "")
    mensagens_para_resumir = messages[:-2]

    texto_mensagens = "\n".join(
        f"{'Enfermeiro' if isinstance(msg, HumanMessage) else 'Assistente'}: {msg.content}"
        for msg in mensagens_para_resumir
    )

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

    try:
        resposta = await llm_decision.ainvoke([SystemMessage(content=prompt_sumarizacao)])
        novo_resumo = str(resposta.content).strip()
        mensagens_removidas = [
            RemoveMessage(id=m.id)
            for m in mensagens_para_resumir
            if hasattr(m, "id") and m.id
        ]
        logger.info(f"✅ [NODE] Resumo atualizado ({len(novo_resumo)} chars).")
        return {
            "summary": novo_resumo,
            "messages": mensagens_removidas,
        }
    except Exception as e:
        logger.error(f"❌ [NODE] Erro no summarize_node: {e}", exc_info=True)
        return {"error": str(e)}


async def fallback_node(state: RAGState) -> RAGState:
    if state.get("is_out_of_scope", False):
        logger.info("🛟 [FALLBACK NODE] Encerrando por bloqueio do Guardrail.")
        safe_response = state.get(
            "generation",
            "Desculpe, sou o assistente clínico NurseAsk e não posso processar essa solicitação.",
        )
        match = re.search(r'"([^"]+)"', safe_response)
        if match:
            safe_response = match.group(1)

        return {
            "generation": safe_response,
            "documents": [],
            "top_score": None,
        }

    logger.warning("🛟 [FALLBACK NODE] Encerrando por insuficiência de contexto clínico.")
    return {
        "fallback_reason": "contexto_insuficiente",
        "documents": [],
        "top_score": None,
        "generation": (
            "Não encontrei informações suficientes nos protocolos clínicos atuais "
            "para responder à sua pergunta com segurança."
        ),
    }
    

async def evaluator_node(state: RAGState) -> RAGState:
    retry_count = state.get("retry_count", 0)
    target_question = (state.get("standalone_question") or state.get("question", "")).strip()
    docs: list[RetrievedDocument] = state.get("documents", [])

    logger.info(f"⚖️ [NODE] Evaluator acionado (tentativa {retry_count}/3) para: '{target_question}'")

    abstention_state: RAGState = {
        "context_valid": False,
        "fallback_reason": "contexto_insuficiente",
        "documents": [],
        "top_score": None,
        "generation": (
            "Não encontrei informações suficientes nos protocolos clínicos atuais "
            "para responder à sua pergunta com segurança."
        ),
    }

    if not docs:
        logger.warning("⚠️ [NODE] Nenhum documento recebido do Retriever.")
        return abstention_state if retry_count >= 3 else {"context_valid": False}

    # Alinhado com RetrievedDocument["score"] preenchido pelo CrossEncoder no retriever_node
    top_score = state.get("top_score")
    if top_score is None:
        top_score = docs[0].get("score")

    if top_score is not None and top_score >= RERANKER_AUTO_APPROVE_THRESHOLD:
        print(
            f"⚡ [EVALUATOR] Aprovação direta via Reranker! "
            f"Score: {top_score:.2f} (Limiar: {RERANKER_AUTO_APPROVE_THRESHOLD}) -> Enviando direto para o Generator."
        )
        logger.info(
            f"⚡ [NODE] Contexto aprovado diretamente pelo Reranker "
            f"(Score: {top_score:.2f} >= {RERANKER_AUTO_APPROVE_THRESHOLD})."
        )
        return {"context_valid": True}

    # Zona cinzenta ou score baixo (< 1.0): aciona o LLM apenas para filtrar ruído
    context_blocks: list[str] = []
    for doc in docs:
        content = doc.get("content", "").strip()
        if not content:
            continue
        title = doc.get("title") or doc.get("source") or "Protocolo"
        context_blocks.append(f"[Protocolo: {title}]\n{content}")

    context_text = "\n\n".join(context_blocks)
    if not context_text:
        return abstention_state if retry_count >= 3 else {"context_valid": False}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Você é um avaliador rigoroso de RAG clínico.
Compare a PERGUNTA com os DOCUMENTOS e responda se os documentos contêm o dado exato para responder à pergunta.

- Marque `is_approved = False` se os documentos apenas citarem o tema ou palavras da pergunta (ex: mencionarem curativos ou tamanhos de forma genérica) SEM conter a informação específica perguntada.
- Marque `is_approved = True` SOMENTE se a resposta completa estiver explicitamente escrita nos documentos.

Responda APENAS no formato JSON exigido.""",
        ),
        ("human", "Pergunta: {question}\n\nDocumentos:\n{context}"),
    ])

    try:
        evaluator_llm = llm_decision.with_structured_output(EvaluationDecision, method="json_schema")
        decision = await (prompt | evaluator_llm).ainvoke({
            "question": target_question,
            "context": context_text,
        })
        print(f"⚖️ [EVALUATOR] Aprovado: {decision.is_approved} | Motivo: {decision.reason}")

        if not decision.is_approved:
            logger.warning(f"🛑 [NODE] Contexto reprovado pelo LLM. Motivo: {decision.reason}")
            if retry_count >= 3:
                logger.warning("🛑 [NODE] Limite de 3 tentativas atingido. Disparando abstenção.")
                return abstention_state
            return {"context_valid": False}

        logger.info("✅ [NODE] Contexto aprovado pelo LLM. Roteando para a geração.")
        return {"context_valid": True}

    except Exception as e:
        logger.error(f"❌ [NODE] Erro no evaluator_node: {e}", exc_info=True)
        if retry_count >= 3:
            return abstention_state
        return {"context_valid": False}