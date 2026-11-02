import asyncio
# Importe o seu grafo exatamente como ele é
from app.application.rag_workflow.graph import nurseask_engine

async def testar_saida_bruta():
    print("=== INICIANDO TESTE ISOLADO DO LANGGRAPH ===\n")
    initial_state = {"question": "O que é taquicardia?"} # Pergunta de teste
    
    try:
        # Vamos ouvir TUDO o que o LangGraph emite, evento por evento
        async for event in nurseask_engine.astream_events(initial_state, version="v2"):
            kind = event["event"]
            name = event.get("name", "desconhecido")
            
            # Se for um evento de nó (como retriever ou o seu jev/gerador)
            if kind == "on_chain_start" and not name.startswith("Runnable"):
                print(f"➡️ Entrando no nó: {name}")
            
            # SEGREDO: Aqui é onde o texto da Llama DEVE aparecer
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                print(f" [PEDAÇO] '{chunk}'")
            
            elif kind == "on_chat_model_end":
                print("\n✅ LLM terminou de gerar.")
                
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {e}")

if __name__ == "__main__":
    asyncio.run(testar_saida_bruta())