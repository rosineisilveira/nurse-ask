import sys
import uuid
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pymupdf
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

    raise FileNotFoundError(f"❌ O PDF '{nome_arquivo}' não foi encontrado.")


def executar_ingestao_garantida(tamanho_lote: int = 30):
    caminho_pdf = obter_caminho_pdf()
    print(f"📄 [INGESTÃO DIRETA] Lendo PDF: {caminho_pdf.name}")

    doc = pymupdf.open(str(caminho_pdf))
    documentos = []
    ids_unicos = []

    for idx, page in enumerate(doc):
        num_pagina = idx + 1
        texto_pagina = page.get_text("text").strip()

        if not texto_pagina:
            continue

        # Incorpora tabelas no formato Markdown se existirem na página
        tabs = page.find_tables()
        if tabs.tables:
            for tab in tabs:
                texto_pagina += "\n\n" + tab.to_markdown()

        # Cada página gera um documento com ID exclusivo para evitar sobrescrita
        doc_obj = Document(
            page_content=f"--- PÁGINA {num_pagina} ---\n{texto_pagina}",
            metadata={
                "source": caminho_pdf.name,
                "title": f"Página {num_pagina} - {caminho_pdf.name}",
                "tipo": "pymupdf_integral",
                "page": num_pagina
            }
        )
        
        id_doc = f"doc_pag_{num_pagina}_{uuid.uuid4().hex[:6]}"
        
        documentos.append(doc_obj)
        ids_unicos.append(id_doc)

    total_docs = len(documentos)
    print(f"📦 Total de {total_docs} documentos preparados em memória.")

    # Conexão com o Elasticsearch
    motor = ElasticMotor(index_name="protocolos_enfermagem")
    store_target = motor.store if hasattr(motor, "store") else motor

    # Ingestão fracionada em lotes
    print(f"💾 Enviando para o Elasticsearch em lotes de {tamanho_lote}...")
    for i in range(0, total_docs, tamanho_lote):
        lote_docs = documentos[i : i + tamanho_lote]
        lote_ids = ids_unicos[i : i + tamanho_lote]
        
        inicio = i + 1
        fim = min(i + tamanho_lote, total_docs)
        print(f"⏳ Processando e indexando lote: {inicio} a {fim} de {total_docs}...")

        store_target.add_documents(documents=lote_docs, ids=lote_ids)

    print("🎉 Ingestão finalizada! Todos os documentos foram indexados com sucesso.")


if __name__ == "__main__":
    executar_ingestao_garantida(tamanho_lote=30)


        