import time
from langchain_ollama import ChatOllama

print("1. Conectando ao Ollama Local...")

try:
    # Iniciamos o cronômetro
    inicio = time.time()
    
    llm = ChatOllama(model="llama3.1", temperature=0)
    
    print("2. Enviando a pergunta. O modelo está sendo carregado na memória (Pode levar até 2 minutos)...")
    resposta = llm.invoke("Responda apenas com a palavra OK")
    
    tempo_total = time.time() - inicio
    print(f"\n[✅ SUCESSO] Ollama respondeu: '{resposta.content}' em {tempo_total:.2f} segundos")
    
except Exception as e:
    print(f"\n[❌ ERRO] O Ollama falhou ou não está rodando. Motivo:")
    print(e)