import os
from elasticsearch import Elasticsearch
from pypdf import PdfReader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
pdf_path = BASE_DIR / "Manual-de-Procedimentos-Básicos-de-Enfermagem.pdf"

es = Elasticsearch(os.getenv("ELASTIC_URL", "http://localhost:9200"))

# Busca tudo o que foi salvo da Página 1
res = es.search(
    index="protocolos_enfermagem",
    body={
        "query": {
            "term": {"metadata.page": 31}
        }
    }
)

print(f"Total de chunks salvos da Pág 1: {res['hits']['total']['value']}\n")
for hit in res['hits']['hits']:
    print(f"--- CHUNK (Tipo: {hit['_source']['metadata'].get('tipo')}) ---")
    print(hit['_source']['text'][:300])
    print("\n" + "="*40)




    #python tests/test_page_metadata.py