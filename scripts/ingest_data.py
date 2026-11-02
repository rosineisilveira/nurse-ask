import sys
import warnings
from pathlib import Path

# Adiciona a raiz do projeto (nurse-ask) ao sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from database.elastic_motor import ElasticMotor

warnings.filterwarnings("ignore")


def obter_caminho_pdf(nome_arquivo: str = "Manual-de-Procedimentos-Básicos-de-Enfermagem.pdf") -> Path:
    caminho_local = BASE_DIR / nome_arquivo
    if caminho_local.exists():
        return caminho_local

    caminho_downloads = Path.home() / "Downloads" / nome_arquivo
    if caminho_downloads.exists():
        return caminho_downloads

    encontrados = list((Path.home() / "Downloads").glob("*[Mm]anual*.pdf"))
    if encontrados:
        return encontrados[0]

    raise FileNotFoundError(f"❌ O PDF '{nome_arquivo}' não foi encontrado.")


def processar_pdf_direto(caminho_pdf: Path) -> list[Document]:
    print(f"📄 [EXTRAÇÃO INTEGRAL VIA PYMUPDF] Lendo arquivo: {caminho_pdf.name}...")

    doc = pymupdf.open(str(caminho_pdf))
    print(f"📖 Total de páginas no arquivo: {len(doc)}")

    # Chunk size expandido para 1500 chars com overlap de 300 para não quebrar a tabela
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )

    documentos_langchain = []

    for idx, page in enumerate(doc):
        num_pagina = idx + 1
        texto_pagina = page.get_text("text").strip()

        if not texto_pagina:
            print(f"⚠️ Página {num_pagina} sem camada de texto legível.")
            continue

        chunks = text_splitter.split_text(texto_pagina)
        for chunk_idx, chunk in enumerate(chunks):
            doc_obj = Document(
                page_content=f"--- PÁGINA {num_pagina} (Parte {chunk_idx + 1}) ---\n{chunk}",
                metadata={
                    "source": caminho_pdf.name,
                    "title": f"Página {num_pagina} (Parte {chunk_idx + 1})",
                    "tipo": "pymupdf_integral",
                    "page": num_pagina
                }
            )
            documentos_langchain.append(doc_obj)

    print(f"✅ Extração concluída: {len(documentos_langchain)} chunks gerados a partir de todas as páginas.")
    return documentos_langchain


if __name__ == "__main__":
    caminho_pdf = obter_caminho_pdf()
    
    # 1. Processamento direto e sem perda de blocos
    documentos = processar_pdf_direto(caminho_pdf)

    # 2. Gravação no Elasticsearch
    print("💾 Gravando no Elasticsearch via ElasticMotor.store...")
    motor = ElasticMotor(index_name="protocolos_enfermagem")
    motor.store.add_documents(documentos)
    print("🎉 Ingestão concluída com sucesso! Todos os dados da Página 30 e tabelas foram armazenados.")