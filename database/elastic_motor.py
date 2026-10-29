import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch import ElasticsearchStore, BM25Strategy, DenseVectorStrategy
from .interfaces import DatabaseInterface

load_dotenv()

class ElasticMotor:
    def __init__(self, index_name="protocolos_enfermagem"):
        self.index_name = index_name
        
        self.url = os.getenv("ELASTIC_URL", "http://localhost:9200")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"}  # Altere para "cuda" se tiver GPU NVIDIA
        )
        # Voltamos para a estratégia padrão que funciona em qualquer licença.
        # Nós faremos a lógica híbrida manualmente na função de busca!
        self.store = ElasticsearchStore(
            embedding=self.embeddings,
            index_name=self.index_name,
            es_url=self.url
        )
        
    def get_retriever(self, top_k=3):
        # Transforma o banco em um objeto retriever do Langchain
        return self.store.as_retriever(search_kwargs={"k": top_k})

    def similarity_search(self, query: str, top_k=8):
        cliente = self.store.client
        resultados_finais = []
        ids_vistos = set()
        from langchain_core.documents import Document

        def add_resultado(hit):
            texto = hit.get("_source", {}).get("text", "")
            metadata = hit.get("_source", {}).get("metadata", {})
            doc_id = hash(texto)
            if doc_id not in ids_vistos:
                resultados_finais.append(Document(page_content=texto, metadata=metadata))
                ids_vistos.add(doc_id)

        try:
            # -------------------------------------------------------------
            # BUSCA 1: A CAÇADORA DE TABELAS (Ignora a diluição do BM25)
            # -------------------------------------------------------------
            query_tabelas = {
                "size": 3, # Pega as 3 tabelas mais prováveis
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"metadata.tipo": "tabela"}}
                        ],
                        "should": [
                            # Procura a palavra no título (Peso gigante)
                            {"match": {"metadata.title": {"query": query, "boost": 5.0}}},
                            # Procura no conteúdo (Sem punição de tamanho longo)
                            {"match": {"text": query}}
                        ]
                    }
                }
            }
            res_tabelas = cliente.search(index=self.index_name, body=query_tabelas)
            for hit in res_tabelas.get("hits", {}).get("hits", []):
                add_resultado(hit)

            # -------------------------------------------------------------
            # BUSCA 2: TEXTO EXATO (Léxica para parágrafos curtos e CIDs)
            # -------------------------------------------------------------
            query_textos = {
                "size": 3,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"metadata.tipo": "texto"}}
                        ],
                        "should": [
                            {"match": {"text": {"query": query, "fuzziness": "AUTO"}}}
                        ]
                    }
                }
            }
            res_textos = cliente.search(index=self.index_name, body=query_textos)
            for hit in res_textos.get("hits", {}).get("hits", []):
                add_resultado(hit)

            # -------------------------------------------------------------
            # BUSCA 3: VETORIAL (Semântica para contexto geral)
            # -------------------------------------------------------------
            docs_vetoriais = self.store.similarity_search(query, k=3)
            for doc in docs_vetoriais:
                # Re-empacotando para manter o mesmo padrão da função interna
                fake_hit = {"_source": {"text": doc.page_content, "metadata": doc.metadata}}
                add_resultado(fake_hit)

            # Retorna os melhores resultados combinados (limitado ao top_k)
            return resultados_finais[:top_k]
            
        except Exception as e:
            print(f"[AVISO] Busca Tripla Híbrida Falhou: {e}")
            return self.store.similarity_search(query, k=top_k)