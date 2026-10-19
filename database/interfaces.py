from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class DatabaseInterface(ABC):
    """
    Contrato base. O LangGraph só vai conversar com estes métodos,
    sem saber qual banco de dados está rodando por trás.
    """
    
    @abstractmethod
    def save_documents(self, documents: List[Document]) -> bool:
        """Recebe chunks de texto e salva no banco de dados."""
        pass
        
    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 3) -> List[Document]:
        """Busca os documentos mais relevantes para a pergunta."""
        pass