import os
import warnings
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch import ElasticsearchStore, BM25Strategy, DenseVectorStrategy
from .interfaces import DatabaseInterface

load_dotenv()

warnings.filterwarnings("ignore")


class ElasticMotor:
    def __init__(self, index_name="protocolos_enfermagem"):
        self.index_name = index_name
        self.url = os.getenv("ELASTIC_URL", "http://localhost:9200")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"}  # Altere para "cuda" se tiver GPU NVIDIA
        )

        self.store = ElasticsearchStore(
            embedding=self.embeddings,
            index_name=self.index_name,
            es_url=self.url
        )

    def get_retriever(self, top_k=30):
        """Retorna o retriever padrão do LangChain com pool expandido."""
        return self.store.as_retriever(search_kwargs={"k": top_k})

    def similarity_search(self, query: str, top_k=40) -> List[Document]:
        """
        Busca Híbrida Unificada (BM25 + Vetor Denso) em requisição única.
        Resgata chunks filhos com metadados do documento Pai.
        """
        cliente = self.store.client
        resultados_finais = []
        ids_vistos = set()

        try:
            # 1. Gera o vetor da pergunta usando o modelo de embedding
            vector_query = self.embeddings.embed_query(query)

            # 2. Query Híbrida Nativa do Elasticsearch (BM25 + KNN em 1 chamada)
            search_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            # BM25: Busca por palavras-chave no texto
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 0.4
                                    }
                                }
                            },
                            # BM25: Destaque para termos encontrados no título da página/seção
                            {
                                "match": {
                                    "metadata.title": {
                                        "query": query,
                                        "boost": 1.2
                                    }
                                }
                            }
                        ]
                    }
                },
                "knn": {
                    "field": "vector",
                    "query_vector": vector_query,
                    "k": top_k,
                    "num_candidates": top_k * 2,
                    "boost": 0.6
                }
            }

            resposta = cliente.search(index=self.index_name, body=search_body)
            hits = resposta.get("hits", {}).get("hits", [])

            for hit in hits:
                source = hit.get("_source", {})
                texto = source.get("text", "")
                metadata = source.get("metadata", {})

                # Chave de deduplicação usando o parent_id se existir, ou o hash do texto
                doc_id = metadata.get("parent_id") or hash(texto)

                if doc_id not in ids_vistos:
                    resultados_finais.append(
                        Document(page_content=texto, metadata=metadata)
                    )
                    ids_vistos.add(doc_id)

            return resultados_finais[:top_k]

        except Exception as e:
            print(f"[AVISO] Busca Híbrida nativa falhou: {e}. Executando fallback vetorial puro...")
            return self.store.similarity_search(query, k=top_k)

