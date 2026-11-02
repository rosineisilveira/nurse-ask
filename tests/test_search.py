import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database.elastic_motor import ElasticMotor

motor = ElasticMotor(index_name="protocolos_enfermagem")

PAGINA_ALVO = 31  # Altere para qualquer página que deseja inspecionar

# Consulta direta ao Elasticsearch por metadado exato (sem vetorização)
query_exata = {
    "query": {
        "term": {
            "metadata.page": PAGINA_ALVO
        }
    }
}

print(f"🔍 Consultando Elasticsearch para a Página {PAGINA_ALVO} (Filtro por Metadado Exato)...")

try:
    # Acessa o cliente Elasticsearch interno do motor
    if hasattr(motor, "client"):
        es_client = motor.client
    elif hasattr(motor.store, "client"):
        es_client = motor.store.client
    else:
        es_client = motor.store._client

    resultado = es_client.search(index="protocolos_enfermagem", body=query_exata)
    hits = resultado["hits"]["hits"]
    total = resultado["hits"]["total"]["value"]

    print(f"📊 Total de documentos da Página {PAGINA_ALVO} encontrados no banco: {total}\n")

    if total == 0:
        print(f"❌ A Página {PAGINA_ALVO} REALMENTE NÃO FOI INDEXADA no Elasticsearch!")
    else:
        for idx, hit in enumerate(hits):
            fonte = hit["_source"]
            texto = fonte.get("text") or fonte.get("page_content") or ""
            metadata = fonte.get("metadata", {})
            
            print(f"--- CHUNK {idx + 1} | ID: {hit['_id']} | Tipo: {metadata.get('tipo', 'N/A')} ---")
            print(texto[:400])
            print("\n" + "=" * 50 + "\n")

except Exception as e:
    print(f"⚠️ Erro ao executar busca nativa: {e}")