import os
import sys

print(">>> [DEBUG 1] Script de Ingestão Iniciado <<<")

# Aponta para a raiz do projeto (para achar a pasta database)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database.elastic_motor import ElasticMotor

# ⚠️ COLOQUE O NOME DO SEU PDF QUE ESTÁ NA PASTA DOWNLOADS:
NOME_DO_PDF = "Manual-de-Procedimentos-Básicos-de-Enfermagem.pdf" 

# Descobre o caminho automático da sua pasta Downloads
PASTA_DOWNLOADS = os.path.join(os.path.expanduser('~'), 'Downloads')
ARQUIVO_PDF = os.path.join(PASTA_DOWNLOADS, NOME_DO_PDF)

def ingerir_documento():
    print(f"\n📖 [PASSO 1] Lendo o arquivo em:\n{ARQUIVO_PDF}")
    loader = PyMuPDFLoader(ARQUIVO_PDF)
    documentos_originais = loader.load()
    print(f"✅ O PDF tem {len(documentos_originais)} páginas.")

    print("\n✂️ [PASSO 2] Fatiando os documentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentos_fatiados = text_splitter.split_documents(documentos_originais)
    print(f"✅ Fatiado em {len(documentos_fatiados)} blocos de texto.")

    print("\n🚀 [PASSO 3] Conectando ao Elasticsearch...")
    db = ElasticMotor() 
    
    sucesso = db.save_documents(documentos_fatiados)
    if sucesso:
        print("\n🎉 Tudo salvo! Fazendo teste rápido de busca...")
        resultados = db.similarity_search("tratamento", top_k=1)
        if resultados:
            print(f"✅ SUCESSO! A IA vai ler isso: {resultados[0].page_content[:150]}...")
        else:
            print("⚠️ A busca não encontrou a palavra 'tratamento', mas o banco conectou.")

if __name__ == "__main__":
    print(f">>> [DEBUG 2] Buscando PDF na pasta Downloads... <<<")
    if not os.path.exists(ARQUIVO_PDF):
        print(f"❌ ERRO CRÍTICO: Não encontrei o PDF no caminho:\n{ARQUIVO_PDF}")
    else:
        ingerir_documento()