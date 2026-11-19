
import os
import asyncio
import hashlib
import logging
from typing import List, Dict, Any
from elasticsearch import AsyncElasticsearch
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class ElasticMotor:
    def __init__(
        self, 
        index_name: str = "protocolos_enfermagem", 
        reranker: Any = None,
        rank_window_size: int = 40,
        rank_constant: int = 60,
        num_candidates_multiplier: int = 5,
        reranker_concurrency: int = 2
    ):
        """
        Inicializa o repositório de dados Assíncrono com RRF, filtro de payload e controle de concorrência.
        Mantém a assinatura original para compatibilidade total com a aplicação.
        """
        self.index_name = index_name
        self.url = os.getenv("ELASTIC_URL", "http://localhost:9200")
        self.reranker = reranker

        # Parâmetros de ajuste da busca híbrida e concorrência
        self.rank_window_size = rank_window_size
        self.rank_constant = rank_constant
        self.num_candidates_multiplier = num_candidates_multiplier
        self._reranker_semaphore = asyncio.Semaphore(reranker_concurrency)
        self._closed = False

        # Cliente assíncrono nativo
        self.client = AsyncElasticsearch(self.url)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"}  # Altere para "cuda" se tiver GPU
        )

    async def aretrieve_context(
        self, 
        query: str, 
        top_k_es: int = 40, 
        top_n_final: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Executa o pipeline completo da camada de dados: 
        1. Busca Híbrida Assíncrona Paralela com Reciprocal Rank Fusion (RRF)
        2. Deduplicação Parent-Child preservando metadados e maior score
        3. Reranking protegido por Semáforo devolvendo o contrato exato esperado pelo grafo
        """
        query = query.strip()
        if not query:
            logger.warning("⚠️ [ELASTIC MOTOR] Consulta vazia recebida.")
            return []

        logger.info(f"🔎 [ELASTIC MOTOR] Iniciando busca RRF e reranking para: '{query}'")
        
        try:
            # 1. Geração do vetor de forma assíncrona
            vector_query = await self.embeddings.aembed_query(query)

            # 2. Busca Híbrida Paralela + Fusão RRF
            hits = await self._hybrid_search_rrf(
                query=query, 
                vector=vector_query, 
                top_k=top_k_es
            )

            if not hits:
                logger.warning("❌ [ELASTIC MOTOR] Nenhum documento encontrado na busca híbrida.")
                return []

            # 3. Estratégia Parent-Child e Deduplicação Determinística
            docs_unicos = self._group_parent_documents(hits)

            if not docs_unicos:
                return []

            # 4. Reranking com controle de concorrência (Semáforo)
            if self.reranker and docs_unicos:
                resultados_finais = await self._rerank(
                    query=query,
                    documents=docs_unicos,
                    top_n=top_n_final
                )
                
                top_score = resultados_finais[0]["score"] if resultados_finais else -99.0
                logger.info(f"✅ [ELASTIC MOTOR] Reranking concluído. Top Score: {top_score:.2f}")
                return resultados_finais

            # Retorno sem Reranker (Fallback mantendo contrato da chave 'score')
            for doc in docs_unicos:
                doc["score"] = doc.get("rrf_score", 0.0)
                
            return docs_unicos[:top_n_final]

        except Exception as e:
            logger.error(f"❌ [ELASTIC MOTOR] Falha crítica na recuperação de contexto: {str(e)}", exc_info=True)
            raise e

    async def _hybrid_search_rrf(
        self, 
        query: str, 
        vector: List[float], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Executa BM25 e kNN em paralelo sem trafegar o array de vetores na rede."""
        window = max(top_k, self.rank_window_size)
        num_candidates = max(window, top_k * self.num_candidates_multiplier)
        
        # Economia drástica de I/O: não traz o campo 'vector' de volta do Elasticsearch
        source_fields = ["text", "metadata"]

        lexical_body = {
            "size": window,
            "_source": source_fields,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^1", "metadata.title^3"],
                    "type": "best_fields",
                }
            },
        }

        vector_body = {
            "size": window,
            "_source": source_fields,
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": window,
                "num_candidates": num_candidates,
            },
        }

        lexical_response, vector_response = await asyncio.gather(
            self.client.search(index=self.index_name, body=lexical_body),
            self.client.search(index=self.index_name, body=vector_body),
        )

        lexical_hits = lexical_response.get("hits", {}).get("hits", [])
        vector_hits = vector_response.get("hits", {}).get("hits", [])

        return self._rrf_fusion(lexical_hits, vector_hits, top_k=window)

    def _rrf_fusion(
        self, 
        lexical_hits: List[Dict[str, Any]], 
        vector_hits: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Combina os rankings baseando-se na posição (rank) para evitar distorção de escalas."""
        fused: Dict[str, Dict[str, Any]] = {}

        def process_ranking(hits: List[Dict[str, Any]], ranking_type: str):
            for rank, hit in enumerate(hits, start=1):
                doc_id = hit["_id"]
                if doc_id not in fused:
                    fused[doc_id] = {
                        **hit,
                        "_lexical_score": None,
                        "_vector_score": None,
                        "_rrf_score": 0.0,
                    }
                fused[doc_id][f"_{ranking_type}_score"] = hit.get("_score")
                fused[doc_id]["_rrf_score"] += 1.0 / (self.rank_constant + rank)

        process_ranking(lexical_hits, "lexical")
        process_ranking(vector_hits, "vector")

        results = list(fused.values())
        results.sort(key=lambda item: item["_rrf_score"], reverse=True)

        for result in results:
            result["_score"] = result["_rrf_score"]

        return results[:top_k]

    def _group_parent_documents(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Agrupa os chunks filhos nos documentos pais preservando páginas e o maior score RRF."""
        parent_documents: Dict[str, Dict[str, Any]] = {}

        for hit in hits:
            source = hit.get("_source", {})
            metadata = source.get("metadata", {}) or {}
            texto_filho = source.get("text", "") or ""

            parent_id = metadata.get("parent_id") or metadata.get("id") or hit.get("_id")
            parent_text = metadata.get("parent_content") or texto_filho

            if not parent_text.strip():
                continue

            # Hash determinístico SHA-256 caso não exista nenhum ID no banco
            if not parent_id:
                parent_id = hashlib.sha256(parent_text.encode('utf-8')).hexdigest()
            else:
                parent_id = str(parent_id)

            page = metadata.get("page", "N/A")
            pages_spanned = str(metadata.get("pages_spanned") or page)
            chunk_id = str(metadata.get("id") or hit.get("_id") or "")
            rrf_score = float(hit.get("_score") or 0.0)

            if parent_id not in parent_documents:
                parent_documents[parent_id] = {
                    "parent_id": parent_id,
                    "chunk_ids": [],
                    "source": metadata.get("source", "Desconhecido"),
                    "page": page,
                    "pages": [],
                    "pages_spanned": pages_spanned,
                    "title": metadata.get("title", ""),
                    "content": parent_text,
                    "rrf_score": rrf_score,
                }

            doc = parent_documents[parent_id]

            if chunk_id and chunk_id not in doc["chunk_ids"]:
                doc["chunk_ids"].append(chunk_id)

            if page not in doc["pages"]:
                doc["pages"].append(page)

            # Garante que o pai herde a maior pontuação RRF encontrada entre seus filhos
            doc["rrf_score"] = max(doc["rrf_score"], rrf_score)

        return list(parent_documents.values())

    async def _rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_n: int
    ) -> List[Dict[str, Any]]:
        """Executa o Cross-Encoder protegido por semáforo e injeta a chave 'score' esperada."""
        pares = [[query, doc["content"]] for doc in documents]

        async with self._reranker_semaphore:
            notas = await asyncio.to_thread(self.reranker.predict, pares)

        docs_com_notas = list(zip(documents, notas))
        docs_com_notas.sort(key=lambda x: x[1], reverse=True)

        resultados_finais = []
        for doc, score in docs_com_notas[:top_n]:
            # Contrato preservado: injeta 'score' exatamente como o retriever_node espera
            doc["score"] = float(score)
            resultados_finais.append(doc)

        return resultados_finais

    async def close(self):
        """Fecha a conexão assíncrona do Elasticsearch de forma idempotente."""
        if self._closed:
            return
        await self.client.close()
        self._closed = True