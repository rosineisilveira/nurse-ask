
import os
import hashlib
import logging
import asyncio
from typing import List, Dict, Any
from elasticsearch import AsyncElasticsearch
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuração de Logger Enterprise
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ElasticMotor:
    def __init__(self, index_name="protocolos_enfermagem", reranker=None):
        """
        Inicializa o repositório de dados Assíncrono.
        :param reranker: O modelo de reranking (ex: BAAI/bge-reranker) injetado via injeção de dependência.
        """
        self.index_name = index_name
        self.url = os.getenv("ELASTIC_URL", "http://localhost:9200")
        self.reranker = reranker

        # Cliente assíncrono nativo para não travar o FastAPI/SSE
        self.client = AsyncElasticsearch(self.url)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"}  # Altere para "cuda" se tiver GPU
        )

    async def aretrieve_context(self, query: str, top_k_es: int = 40, top_n_final: int = 3) -> List[Dict[str, Any]]:
        """
        Executa o pipeline completo da camada de dados: 
        1. Busca Híbrida Assíncrona
        2. Deduplicação e Agrupamento Parent-Child
        3. Reranking (Reclassificação) sobre os documentos Pai
        """
        logger.info(f"🔎 [ELASTIC MOTOR] Iniciando busca e reranking para: '{query}'")
        
        try:
            # 1. Geração do vetor de forma assíncrona
            vector_query = await self.embeddings.aembed_query(query)

            # 2. Query Híbrida (Ajustada com num_candidates ideal)
            search_body = {
                "size": top_k_es,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"text": {"query": query, "boost": 0.4}}},
                            {"match": {"metadata.title": {"query": query, "boost": 1.2}}}
                        ]
                    }
                },
                "knn": {
                    "field": "vector",
                    "query_vector": vector_query,
                    "k": top_k_es,
                    "num_candidates": top_k_es * 5,  # Funil expandido para maior precisão de recall
                    "boost": 0.6
                }
            }

            # 3. Busca I/O Non-blocking no Elasticsearch
            resposta = await self.client.search(index=self.index_name, body=search_body)
            hits = resposta.get("hits", {}).get("hits", [])

            if not hits:
                logger.warning("❌ [ELASTIC MOTOR] Nenhum documento encontrado na busca híbrida.")
                return []

            # 4. Estratégia Parent-Child e Deduplicação Determinística
            parent_documents = {}
            
            for hit in hits:
                source = hit.get("_source", {})
                metadata = source.get("metadata", {})
                texto_filho = source.get("text", "")

                parent_id = metadata.get("parent_id") or metadata.get("id")
                parent_text = metadata.get("parent_content") or texto_filho
                pages_spanned = metadata.get("pages_spanned") or str(metadata.get("page", "N/A"))

                # Hash determinístico SHA-256 (Resolve o bug de concorrência em produção)
                if not parent_id:
                    parent_id = hashlib.sha256(parent_text.encode('utf-8')).hexdigest()

                if parent_id not in parent_documents:
                    parent_documents[parent_id] = {
                        "source": metadata.get("source", "Desconhecido"),
                        "page": metadata.get("page", "N/A"),
                        "pages_spanned": pages_spanned,
                        "title": metadata.get("title", ""),
                        "content": parent_text
                    }

            # Extrai os pais deduplicados
            docs_unicos = list(parent_documents.values())

            # 5. Reranking (apenas se houver o modelo injetado e documentos encontrados)
            if self.reranker and docs_unicos:
                # O Reranker avalia a pergunta contra o texto COMPLETO do Pai
                pares = [[query, doc["content"]] for doc in docs_unicos]
                
                # O modelo predict é pesado, roda em thread separada para não travar o Event Loop
                notas = await asyncio.to_thread(self.reranker.predict, pares)
                
                # Zipa os documentos com suas respectivas notas e ordena
                docs_com_notas = list(zip(docs_unicos, notas))
                docs_com_notas.sort(key=lambda x: x[1], reverse=True)
                
                # Limita ao Top N final e formata a saída
                resultados_finais = []
                for doc, score in docs_com_notas[:top_n_final]:
                    doc["score"] = float(score)
                    resultados_finais.append(doc)
                    
                top_score = resultados_finais[0]["score"] if resultados_finais else -99.0
                logger.info(f"✅ [ELASTIC MOTOR] Reranking concluído. Top Score: {top_score:.2f}")
                
                return resultados_finais

            # Retorno sem Reranker (Fallback)
            return docs_unicos[:top_n_final]

        except Exception as e:
            logger.error(f"❌ [ELASTIC MOTOR] Falha crítica na recuperação de contexto: {str(e)}", exc_info=True)
            # Ao lançar a exceção nativa, o LangGraph / LangSmith capturam o erro no dashboard
            raise e
            
    async def close(self):
        """Fecha a conexão assíncrona do Elasticsearch adequadamente (Uso em teardowns de servidor)"""
        await self.client.close()