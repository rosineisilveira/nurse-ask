import os

# Lista de pastas da Clean Architecture
pastas = [
    "app/domain/entities",
    "app/domain/interfaces",
    "app/application/rag_workflow",
    "app/application/ingestion",
    "app/infrastructure/database",
    "app/infrastructure/vector_store",
    "app/infrastructure/ai_clients",
    "app/presentation/api/v1",
    "app/presentation/middlewares",
    "app/core",
    "tests"
]

# Lista de arquivos base
arquivos = [
    "docker-compose.yml",
    "Dockerfile",
    "requirements.txt",
    ".env",
    ".gitignore",
    "main.py"
]

# Cria as pastas
for pasta in pastas:
    os.makedirs(pasta, exist_ok=True)
    print(f"Pasta criada: {pasta}")

# Cria os arquivos vazios
for arquivo in arquivos:
    with open(arquivo, 'a') as f:
        pass
    print(f"Arquivo criado: {arquivo}")

print("\n✅ Estrutura do NURSEASK criada com sucesso!")