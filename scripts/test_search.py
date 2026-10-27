import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database.elastic_motor import ElasticMotor
from pprint import pprint

def testar_leitura_tabela():
    print("\n🔎 TESTE FOCADO: LENDO A TABELA DE TUBOS IGNORANDO DILUIÇÃO")
    print("="*50)
    
    db = ElasticMotor()
    cliente = db.store.client
    indice = db.index_name
    
    pergunta = "classificaçao das diets hospitalares"
    
    try:
        # Busca Focada: Procura apenas em documentos que são "tabela" e 
        # não penaliza pelo tamanho longo do texto.
        query_body = {
            "size": 3,
            "query": {
                "bool": {
                    "must": [
                        # Filtro estrito: Só queremos olhar para as tabelas (ignorando textos)
                        {"match": {"metadata.tipo": "tabela"}},
                        # Busca pelas palavras-chave "tubos" ou "laboratório" no texto bruto
                        {"match": {"text": pergunta}}
                    ]
                }
            }
        }
        
        res = cliente.search(index=indice, body=query_body)
        hits = res.get("hits", {}).get("hits", [])
        
        print(f"✅ Encontrei {len(hits)} tabelas correspondentes.")
        
        if len(hits) == 0:
            print("❌ Não achei a tabela. Tente usar palavras-chave mais exatas do PDF.")
            return

        print("\n" + "="*50)
        print(" CONTEÚDO INTEGRAL DA MELHOR TABELA ENCONTRADA")
        print("="*50)
        
        # Pegamos o primeiro resultado (o mais relevante)
        melhor_tabela = hits[0].get("_source", {})
        texto = melhor_tabela.get("text", "")
        metadata = melhor_tabela.get("metadata", {})
        
        print(f"📌 Título: {metadata.get('title', 'N/A')} (Página {metadata.get('page', 'N/A')})")
        print("-" * 50)
        # Imprime o texto corrido (sem cortes) para vermos toda a estrutura
        print(texto)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Erro na busca: {e}")

if __name__ == "__main__":
    testar_leitura_tabela()