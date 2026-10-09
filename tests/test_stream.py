import requests
import json

url = "http://127.0.0.1:8000/api/v1/chat/stream"
payload = {
    "session_id": "paciente_123",
    "question": "Qual é a conduta para paciente com sepse?"
}

print("Enviando requisição...\n")

# O parâmetro stream=True é o segredo para ler a resposta aos poucos
with requests.post(url, json=payload, stream=True) as response:
    # Lê a resposta linha por linha assim que ela chega do servidor
    for line in response.iter_lines():
        if line:
            # Decodifica os bytes para string (UTF-8)
            decoded_line = line.decode('utf-8')
            print(decoded_line)