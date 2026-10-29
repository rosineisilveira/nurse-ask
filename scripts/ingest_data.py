import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database.elastic_motor import ElasticMotor
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document 

print(">>> [DEBUG 1] Script de Ingestão Iniciado <<<")



# ⚠️ COLOQUE O NOME DO SEU PDF QUE ESTÁ NA PASTA DOWNLOADS:
NOME_DO_PDF = "Manual-de-Procedimentos-Básicos-de-Enfermagem.pdf" 

# Descobre o caminho automático da sua pasta Downloads
PASTA_DOWNLOADS = os.path.join(os.path.expanduser('~'), 'Downloads')
ARQUIVO_PDF = os.path.join(PASTA_DOWNLOADS, NOME_DO_PDF)

def ingerir_documento():
    
    print("\n📖 [PASSO 1] Docling Nativo: Extraindo estrutura completa...")
    converter = DocumentConverter()
    result = converter.convert(ARQUIVO_PDF)
    
    # O Docling gera um documento estruturado riquíssimo
    docling_doc = result.document
    
    documentos_preparados = []
    
    # Iteramos sobre os itens extraídos (textos, tabelas, imagens)
    print("⚙️ [PASSO 2] Mapeando Metadados (Títulos e Páginas)...")
    documentos_texto = []
    documentos_tabela = [] # Lista VIP
    
    # Uma sacada Sênior: O Docling lê de cima para baixo. Vamos guardar o "último título visto"
    # para que um parágrafo saiba a qual título ele pertence, mesmo que o título esteja na página anterior.
    ultimo_titulo_visto = "Introdução/Geral"
    
    for item, _ in docling_doc.iterate_items():
        if item.label == "section_header":
            ultimo_titulo_visto = item.text
            continue
            
        if item.label in ["text", "list_item", "table"]:
            pagina = str(item.prov[0].page_no) if hasattr(item, "prov") and item.prov else "Desconhecida"
            
            # SE FOR TABELA: Exporta o Markdown e NÃO FATIA
            if item.label == "table":
                conteudo = item.export_to_markdown()
                doc_langchain = Document(
                    page_content=conteudo,
                    metadata={"source": "protocolo.pdf", "page": pagina, "title": ultimo_titulo_visto, "tipo": "tabela"}
                )
                documentos_tabela.append(doc_langchain)
            
            # SE FOR TEXTO: Vai para a lista normal
            else:
                doc_langchain = Document(
                    page_content=item.text,
                    metadata={"source": "protocolo.pdf", "page": pagina, "title": ultimo_titulo_visto, "tipo": "texto"}
                )
                documentos_texto.append(doc_langchain)

    print("✂️ Fatiando apenas parágrafos (preservando tabelas)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks_texto = text_splitter.split_documents(documentos_texto)
    
    # Junta os textos fatiados com as tabelas INTACTAS
    chunks_finais = chunks_texto + documentos_tabela
    
    print(f"✅ Injetando {len(chunks_finais)} documentos no banco...")
    db = ElasticMotor()
    db.store.add_documents(chunks_finais)
    print("🎉 Ingestão Perfeita Finalizada!")

if __name__ == "__main__":
    ingerir_documento()