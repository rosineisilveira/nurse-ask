import os
import sys
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.elastic_motor import ElasticMotor

# Usamos o asyncio porque nosso motor é assíncrono por baixo dos panos (com o to_thread)
async def auditar_banco():
    print("\n" + "="*50)
    print("🕵️ INICIANDO AUDITORIA NO ELASTICSEARCH")
    print("="*50)
    
    db = ElasticMotor()
    
    # Faça uma pergunta genérica que force o Elastic a trazer algo
    # DICA: Se você souber de uma tabela no seu PDF (ex: "dosagem", "critérios"), busque por essa palavra.
    termo_busca = "enfermagem" # Troque por uma palavra que exista na sua tabela!
    
    print(f"\n[1] Buscando pela palavra: '{termo_busca}'...")
    documentos = await asyncio.to_thread(db.similarity_search, termo_busca, 1)
    
    if not documentos:
        print("\n❌ ERRO: O banco retornou vazio! O Docling pode ter falhado na ingestão.")
        return

    print("\n✅ Documento encontrado! Veja como o Elastic (e a IA) enxerga o texto:\n")
    print("-" * 50)
    
    # Pega o primeiro resultado
    doc = documentos[0]
    
    # Imprime o conteúdo exato (aqui você vai ver se a tabela virou Markdown ou se virou uma bagunça)
    print(doc.page_content)
    
    print("-" * 50)
    print("\n📌 METADADOS SALVOS:")
    for chave, valor in doc.metadata.items():
        print(f"  - {chave}: {valor}")

if __name__ == "__main__":
    asyncio.run(auditar_banco())