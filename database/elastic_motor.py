from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from .interfaces import DatabaseInterface

class ElasticMotor(DatabaseInterface):
    def __init__(self, index_name: str = "clinical_guidelines"):
        self.index_name = index_name
        
        print("[DATABASE] Inicializando Embeddings Locais...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print("[DATABASE] Conectando ao Elasticsearch (localhost:9200)...")
        self.store = ElasticsearchStore(
            es_url="http://localhost:9200", # Porta do seu container principal
            index_name=self.index_name,
            embedding=self.embeddings,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy()
        )

    def save_documents(self, documents: List[Document]) -> bool:
        try:
            print(f"[DATABASE] Salvando {len(documents)} chunks...")
            self.store.add_documents(documents)
            print("[DATABASE] Salvamento concluído com sucesso!")
            return True
        except Exception as e:
            print(f"[DATABASE ERRO] Falha ao salvar: {e}")
            return False

    def similarity_search(self, query: str, top_k: int = 3) -> List[Document]:
        try:
            return self.store.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"[DATABASE ERRO] Falha na busca: {e}")
            return []