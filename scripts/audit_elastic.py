import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.elastic_motor import ElasticMotor

def auditar_banco():
    db = ElasticMotor()
    cliente = db.store.client
    indice = db.index_name
    
    print("\n🔍 INICIANDO AUDITORIA DO ELASTICSEARCH")
    print("="*50)
    
    try:
        # Conta o total de documentos
        res_count = cliente.count(index=indice)
        total = res_count['count']
        print(f"📊 Total de trechos armazenados no índice '{indice}': {total}")
        
        if total == 0:
            print("❌ O banco está VAZIO! A Ingestão falhou completamente.")
            return

        print("\n🕵️ Buscando especificamente pelas TABELAS (tipo: tabela)...")
        
        # Busca direta no banco (sem usar vetores) filtrando apenas por tabelas
        query = {
            "size": 3,
            "query": {
                "match": {
                    "metadata.tipo": "tabela"
                }
            }
        }
        
        res = cliente.search(index=indice, body=query)
        hits = res.get("hits", {}).get("hits", [])
        
        print(f"✅ Encontradas {res['hits']['total']['value']} tabelas no banco.")
        
        print("\n" + "="*50)
        print(" AMOSTRA DE COMO AS TABELAS ESTÃO SALVAS NO BANCO ")
        print("="*50)
        
        for idx, hit in enumerate(hits, 1):
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            texto = source.get("text", "")
            
            print(f"\n--- Tabela {idx} ---")
            print(f"📌 Título (Metadata): {metadata.get('title', 'N/A')}")
            print(f"📄 Página (Metadata): {metadata.get('page', 'N/A')}")
            print("📝 Conteúdo Real (Primeiros 200 caracteres):")
            # Imprime a representação crua para vermos as quebras de linha Markdown
            print(repr(texto[:200]) + "...") 
            print("-" * 20)

    except Exception as e:
        print(f"[❌ ERRO FATAL] Não consegui ler o banco: {e}")

if __name__ == "__main__":
    auditar_banco()