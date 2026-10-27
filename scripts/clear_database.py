import os
import sys

# Ajusta a rota para enxergar a raiz do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.elastic_motor import ElasticMotor

def limpar_banco():
    print("\n" + "="*50)
    print("🧹 INICIANDO LIMPEZA TOTAL DO ELASTICSEARCH")
    print("="*50)
    
    db = ElasticMotor()
    cliente_es = db.store.client
    
    # Busca a variável privada '_index_name'. Se não achar, usa o padrão 'langchain'
    nome_indice = getattr(db.store, "_index_name", "langchain")
    
    # 💡 DICA: Se no seu arquivo elastic_motor.py você digitou algo como 
    # index_name="protocolos", você pode apagar a linha acima e forçar o nome aqui:
    # nome_indice = "protocolos"
    
    try:
        print(f"🗑️ Deletando o índice: '{nome_indice}'...")
        
        # O parâmetro ignore_unavailable evita erros se o banco já estiver vazio
        cliente_es.indices.delete(index=nome_indice, ignore_unavailable=True)
        
        print(f"[✅ SUCESSO] Banco de dados limpo e zerado!")
        print("[INFO] Um novo índice vazio será criado na próxima ingestão.")
        
    except Exception as e:
        print(f"[❌ ERRO] Não foi possível limpar o banco: {e}")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    limpar_banco()