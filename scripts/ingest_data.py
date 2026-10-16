from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

print("Conectando ao Qdrant no Docker...")
client = QdrantClient(url="http://localhost:6333")

# Novo nome para não conflitar com os vetores antigos da OpenAI
collection_name = "clinical_guidelines_hf" 

# 1. Troca a OpenAI pelo modelo gratuito Multilíngue (baixa na 1ª vez que rodar)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. Atenção ao size=384, que é o tamanho exato deste modelo gratuito
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

docs = [
    Document(
        page_content="Paciente com suspeita de infarto agudo do miocárdio (IAM) deve receber AAS 300mg mastigado imediatamente para efeito antiplaquetário, salvo contraindicações.",
        metadata={"source": "Diretriz IAM 2024"}
    ),
    Document(
        page_content="No choque anafilático, a intervenção de primeira linha e salva-vidas é a injeção intramuscular de Epinefrina (Adrenalina) no vasto lateral da coxa. Não atrase a administração.",
        metadata={"source": "Protocolo Anafilaxia 2023"}
    )
]

print("Gerando vetores locais e salvando no Qdrant...")
vector_store.add_documents(docs)
print("✅ Dados inseridos com sucesso sem custo algum!")