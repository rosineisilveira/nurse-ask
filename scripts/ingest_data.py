import sys
import uuid
import re
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from elasticsearch.helpers import BulkIndexError
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


def limpar_caracteres_pua(texto: str) -> str:
    """Remove caracteres especiais de fontes Wingdings/Symbol (faixa PUA Unicode)."""
    return re.sub(r'[\uE000-\uF8FF]', '', texto)


def extrair_fluxo_continuo_pdf(doc: pymupdf.Document) -> str:
    """Extrai todas as páginas em uma única string contínua com tags de página."""
    texto_completo = []
    for idx, page in enumerate(doc):
        num_pagina = idx + 1
        texto_pagina = page.get_text("text").strip()
        if not texto_pagina:
            continue

        tabs = page.find_tables()
        if tabs.tables:
            for tab in tabs:
                texto_pagina += "\n\n" + tab.to_markdown()

        texto_limpo = limpar_caracteres_pua(texto_pagina)
        texto_completo.append(f"\n\n--- [INÍCIO PÁGINA {num_pagina}] ---\n{texto_limpo}\n--- [FIM PÁGINA {num_pagina}] ---")

    return "".join(texto_completo)


def executar_ingestao_parent_child_continuo(tamanho_lote: int = 50):
    caminho_pdf = obter_caminho_pdf()
    print(f"📄 [INGESTÃO CONTINUA CROSS-PAGE] Lendo PDF: {caminho_pdf.name}")

    doc = pymupdf.open(str(caminho_pdf))
    texto_integral = extrair_fluxo_continuo_pdf(doc)

    # 1. SPLITTER DO PAI: Janela ampla com sobreposição para não cortar tabelas entre páginas
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2800,
        chunk_overlap=700,
        separators=["\n--- [INÍCIO PÁGINA ", "\n\n", "\n", ". ", " "]
    )

    # 2. SPLITTER DO FILHO: Chunks curtos e precisos para a busca vetorial
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    parent_texts = parent_splitter.split_text(texto_integral)
    print(f"🧩 Gerados {len(parent_texts)} blocos Pais contínuos (Cross-Page).")

    documentos_filhos = []
    ids_unicos = []

    for idx_parent, parent_text in enumerate(parent_texts):
        parent_id = f"parent_block_{idx_parent + 1}_{uuid.uuid4().hex[:6]}"
        
        paginas_encontradas = [int(p) for p in re.findall(r'PÁGINA (\d+)', parent_text)]
        paginas_encontradas = sorted(list(set(paginas_encontradas)))
        pagina_referencia = paginas_encontradas[0] if paginas_encontradas else 0
        pages_str = ", ".join(map(str, paginas_encontradas))

        child_chunks = child_splitter.split_text(parent_text)

        for chunk_idx, child_text in enumerate(child_chunks):
            doc_filho = Document(
                page_content=child_text,
                metadata={
                    "source": caminho_pdf.name,
                    "title": f"Página(s) {pages_str} (Trecho {chunk_idx + 1})",
                    "tipo": "child_chunk_continuous",
                    "page": int(pagina_referencia),
                    "pages_spanned": pages_str,
                    "parent_id": parent_id,
                    "parent_content": parent_text
                }
            )

            child_id = f"child_b{idx_parent + 1}_c{chunk_idx + 1}_{uuid.uuid4().hex[:4]}"
            documentos_filhos.append(doc_filho)
            ids_unicos.append(child_id)

    total_chunks = len(documentos_filhos)
    print(f"📦 Total de {total_chunks} chunks filhos prontos para indexação.")

    motor = ElasticMotor(index_name="protocolos_enfermagem")
    store_target = motor.store if hasattr(motor, "store") else motor

    print(f"💾 Enviando para o Elasticsearch em lotes de {tamanho_lote}...")
    for i in range(0, total_chunks, tamanho_lote):
        lote_docs = documentos_filhos[i : i + tamanho_lote]
        lote_ids = ids_unicos[i : i + tamanho_lote]

        inicio = i + 1
        fim = min(i + tamanho_lote, total_chunks)
        print(f"⏳ Indexando lote: {inicio} a {fim} de {total_chunks}...")

        try:
            store_target.add_documents(documents=lote_docs, ids=lote_ids)
        except BulkIndexError as e:
            print(f"❌ Erro de indexação no Elasticsearch no lote {inicio}-{fim}:")
            for err_item in e.errors:
                print(err_item)
            raise e

    print("🎉 Ingestão Cross-Page concluída com sucesso!")


if __name__ == "__main__":
    executar_ingestao_parent_child_continuo(tamanho_lote=50)