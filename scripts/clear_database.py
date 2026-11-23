import asyncio
import os
import sys
from pathlib import Path

# Adjusts sys.path to resolve project root imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from elasticsearch import AsyncElasticsearch

INDEX_NAME = "protocolos_enfermagem"


async def clear_database(index_name: str = INDEX_NAME) -> None:
    print("\n" + "=" * 50)
    print("🧹 INICIANDO LIMPEZA TOTAL DO ELASTICSEARCH")
    print("=" * 50)

    elastic_url = os.getenv("ELASTIC_URL", "http://localhost:9200")
    es_client = AsyncElasticsearch(elastic_url)

    try:
        if await es_client.indices.exists(index=index_name):
            print(f"🗑️ Deletando o índice: '{index_name}'...")
            await es_client.indices.delete(index=index_name)
            print("✅ [SUCESSO] Banco de dados limpo e zerado!")
            print("ℹ️ [INFO] Um novo índice vazio será criado na próxima ingestão.")
        else:
            print(f"ℹ️ [INFO] O índice '{index_name}' não existe ou já está vazio.")
    except Exception as err:
        print(f"❌ [ERRO] Não foi possível limpar o banco: {err}")
    finally:
        await es_client.close()

    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(clear_database())