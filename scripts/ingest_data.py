
import asyncio
import os
import re
import sys
import uuid
import warnings
from pathlib import Path

import pymupdf
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import BulkIndexError, async_bulk
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database.elastic_motor import ElasticMotor

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
INDEX_NAME = "protocolos_enfermagem"
EMBEDDING_DIMS = 1024  # BAAI/bge-m3 output dimension


def get_pdf_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.pdf"))


def clean_pua_chars(text: str) -> str:
    """Removes special characters from Wingdings/Symbol fonts (Unicode PUA range)."""
    return re.sub(r"[\uE000-\uF8FF]", "", text)


def extract_continuous_pdf_flow(doc: pymupdf.Document) -> str:
    """Extracts all pages into a single continuous string with page markers."""
    full_text: list[str] = []

    for idx, page in enumerate(doc):
        page_num = idx + 1
        page_text = page.get_text("text").strip()
        if not page_text:
            continue

        tables = page.find_tables()
        if tables.tables:
            for table in tables:
                page_text += "\n\n" + table.to_markdown()

        cleaned_text = clean_pua_chars(page_text)
        full_text.append(
            f"\n\n--- [INÍCIO PÁGINA {page_num}] ---\n"
            f"{cleaned_text}\n"
            f"--- [FIM PÁGINA {page_num}] ---"
        )

    return "".join(full_text)


async def is_pdf_already_indexed(
    client: AsyncElasticsearch, index_name: str, filename: str
) -> bool:
    """Checks asynchronously if chunks from this PDF already exist in Elasticsearch."""
    try:
        if not await client.indices.exists(index=index_name):
            return False

        response = await client.count(
            index=index_name,
            query={
                "bool": {
                    "should": [
                        {"term": {"metadata.source": filename}},
                        {"term": {"metadata.source.keyword": filename}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        )
        return response.get("count", 0) > 0
    except Exception:
        return False


async def ensure_index_exists(client: AsyncElasticsearch, index_name: str) -> None:
    """Creates the Elasticsearch index with the exact schema expected by ElasticMotor."""
    if await client.indices.exists(index=index_name):
        return

    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMS,
                    "index": True,
                    "similarity": "cosine",
                },
                "metadata": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "title": {"type": "text"},
                        "tipo": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "pages_spanned": {"type": "keyword"},
                        "parent_id": {"type": "keyword"},
                        "parent_content": {"type": "text", "index": False},
                    }
                },
            }
        }
    }
    await client.indices.create(index=index_name, body=mapping)
    print(f"🆕 [ELASTICSEARCH] Índice '{index_name}' criado com sucesso.")


async def run_continuous_parent_child_ingestion(
    batch_size: int = 50, force: bool = False
) -> None:
    pdf_files = get_pdf_files()
    if not pdf_files:
        print(f"⚠️ [INGESTÃO] Nenhum arquivo PDF encontrado em '{DATA_DIR}/'.")
        return

    elastic_url = os.getenv("ELASTIC_URL", "http://localhost:9200")
    check_client = AsyncElasticsearch(elastic_url)
    pending_pdfs: list[Path] = []

    try:
        for pdf_path in pdf_files:
            if not force and await is_pdf_already_indexed(
                check_client, INDEX_NAME, pdf_path.name
            ):
                print(f"⏭️ [INGESTÃO] '{pdf_path.name}' já está indexado. Pulando...")
            else:
                pending_pdfs.append(pdf_path)
    finally:
        await check_client.close()

    if not pending_pdfs:
        print("✅ [INGESTÃO] Todos os PDFs já estão indexados no Elasticsearch.")
        return

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2800,
        chunk_overlap=700,
        separators=["\n--- [INÍCIO PÁGINA ", "\n\n", "\n", ". ", " "],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_documents: list[Document] = []
    unique_ids: list[str] = []

    for pdf_path in pending_pdfs:
        print(f"📄 [INGESTÃO CONTINUA CROSS-PAGE] Lendo PDF: {pdf_path.name}")

        with pymupdf.open(stream=pdf_path.read_bytes(), filetype="pdf") as doc:
            raw_content = extract_continuous_pdf_flow(doc)

        parent_texts = parent_splitter.split_text(raw_content)
        print(f"🧩 [{pdf_path.name}] Gerados {len(parent_texts)} blocos Pais contínuos.")

        for parent_idx, parent_text in enumerate(parent_texts):
            parent_hash = uuid.uuid5(
                uuid.NAMESPACE_DNS, f"{pdf_path.name}_parent_{parent_idx + 1}"
            ).hex[:6]
            parent_id = f"parent_block_{parent_idx + 1}_{parent_hash}"

            found_pages = sorted(
                {int(p) for p in re.findall(r"PÁGINA (\d+)", parent_text)}
            )
            reference_page = found_pages[0] if found_pages else 0
            pages_str = ", ".join(map(str, found_pages))

            child_chunks = child_splitter.split_text(parent_text)

            for chunk_idx, child_text in enumerate(child_chunks):
                child_doc = Document(
                    page_content=child_text,
                    metadata={
                        "source": pdf_path.name,
                        "title": f"Página(s) {pages_str} (Trecho {chunk_idx + 1})",
                        "tipo": "child_chunk_continuous",
                        "page": int(reference_page),
                        "pages_spanned": pages_str,
                        "parent_id": parent_id,
                        "parent_content": parent_text,
                    },
                )

                child_hash = uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{pdf_path.name}_b{parent_idx + 1}_c{chunk_idx + 1}",
                ).hex[:6]
                child_id = f"child_b{parent_idx + 1}_c{chunk_idx + 1}_{child_hash}"

                child_documents.append(child_doc)
                unique_ids.append(child_id)

    total_chunks = len(child_documents)
    if total_chunks == 0:
        print("⚠️ [INGESTÃO] Nenhum conteúdo extraído dos PDFs.")
        return

    print(f"📦 Total de {total_chunks} chunks filhos prontos para indexação.")

    engine = ElasticMotor(index_name=INDEX_NAME)
    try:
        await ensure_index_exists(engine.client, INDEX_NAME)
        print(f"💾 Gerando embeddings e enviando para o Elasticsearch em lotes de {batch_size}...")

        for i in range(0, total_chunks, batch_size):
            batch_docs = child_documents[i : i + batch_size]
            batch_ids = unique_ids[i : i + batch_size]

            start_idx = i + 1
            end_idx = min(i + batch_size, total_chunks)
            print(f"⏳ Indexando lote: {start_idx} a {end_idx} de {total_chunks}...")

            texts = [doc.page_content for doc in batch_docs]
            vectors = await engine.embeddings.aembed_documents(texts)

            actions = [
                {
                    "_index": INDEX_NAME,
                    "_id": doc_id,
                    "_source": {
                        "text": doc.page_content,
                        "vector": vector,
                        "metadata": doc.metadata,
                    },
                }
                for doc, doc_id, vector in zip(batch_docs, batch_ids, vectors)
            ]

            try:
                await async_bulk(engine.client, actions, refresh="wait_for")
            except BulkIndexError as err:
                print(f"❌ Erro de indexação no Elasticsearch no lote {start_idx}-{end_idx}:")
                for err_item in err.errors:
                    print(err_item)
                raise

        print("🎉 Ingestão Cross-Page concluída com sucesso!")
    finally:
        await engine.close()


if __name__ == "__main__":
    force_reingest = "--force" in sys.argv
    asyncio.run(run_continuous_parent_child_ingestion(batch_size=50, force=force_reingest))