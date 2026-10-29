import os
import sys
import warnings

# Silencia avisos do PyTorch / HuggingFace no terminal
warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

# Adiciona a raiz do projeto ao path para localizar a pasta database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.elastic_motor import ElasticMotor


def processar_pdf_para_elasticmotor(caminho_pdf: str):
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"❌ Arquivo não encontrado no caminho: {caminho_pdf}")

    print(f"📄 [DOCLING] Extraindo e analisando layout de: {os.path.basename(caminho_pdf)}...")

    # 1. Configura o Docling (OCR DESATIVADO para velocidade máxima)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # ⚡ OCR Desativado
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(caminho_pdf)
    docling_doc = result.document

    # 2. Tokenizer do MiniLM (mesmo modelo usado no ElasticMotor para cálculo exato dos 384 tokens)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    # 3. HybridChunker do Docling
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=800,
        merge_peers=True
    )

    print("✂️ [DOCLING] Fatiando e gerando chunks hierárquicos...")
    chunks_docling = list(chunker.chunk(docling_doc))
    documentos_langchain = []

    for chunk in chunks_docling:
        # A) Texto serializado com a hierarquia (breadcrumbs)
        texto_enriquecido = chunker.serialize(chunk)

        # B) Extrai os títulos para preencher o metadata.title
        titulos = [header for header in chunk.meta.headings] if hasattr(chunk.meta, 'headings') else []
        titulo_principal = titulos[0] if titulos else os.path.basename(caminho_pdf)
        hierarquia_str = " > ".join(titulos) if titulos else "Geral"

        # C) Descobre se este chunk é TABELA ou TEXTO (Alimenta os filtros do ElasticMotor)
        eh_tabela = any("table" in str(item.self_ref) for item in chunk.meta.doc_items)
        tipo_documento = "tabela" if eh_tabela else "texto"

        # D) Descobre a página principal
        paginas = set()
        for item in chunk.meta.doc_items:
            if hasattr(item, "prov") and item.prov:
                for p in item.prov:
                    paginas.add(p.page_no)
        pagina = min(paginas) if paginas else 1

        # E) Monta o objeto Document do LangChain para o ElasticMotor
        doc = Document(
            page_content=texto_enriquecido,
            metadata={
                "source": os.path.basename(caminho_pdf),
                "title": titulo_principal,    # Boost de 5.0 na Busca 1
                "tipo": tipo_documento,      # Filtro "tabela" ou "texto"
                "hierarquia": hierarquia_str,
                "page": pagina
            }
        )
        documentos_langchain.append(doc)

    print(f"✅ {len(documentos_langchain)} chunks gerados do PDF!")
    return documentos_langchain


if __name__ == "__main__":
    try:
       
        motor = ElasticMotor(index_name="protocolos_enfermagem")

        # 2. Localiza o PDF na pasta Downloads
        PASTA_DOWNLOADS = os.path.join(os.path.expanduser('~'), 'Downloads')
        NOME_DO_PDF = "Manual-de-Procedimentos-Básicos-de-Enfermagem.pdf"
        pdf_path = os.path.join(PASTA_DOWNLOADS, NOME_DO_PDF)

        # 3. Processa o PDF com Docling
        documentos = processar_pdf_para_elasticmotor(pdf_path)

        # 4. Salva no Elasticsearch via ElasticMotor.store
        print("💾 Gravando no Elasticsearch via ElasticMotor.store...")
        motor.store.add_documents(documentos)

        print("🎉 Ingestão concluída com sucesso! Pode rodar a API normalmente.")

    except Exception as e:
            print("\n❌ ERRO DETALHADO DO ELASTICSEARCH:")
            if hasattr(e, "errors") and e.errors:
            # Imprime o motivo exato da falha do primeiro documento
                print(e.errors[0])
            else:
                print(e)




        