import sys
import re
from pathlib import Path
import pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
pdf_path = Path.home() / "Downloads" / "Manual-de-Procedimentos-Básicos-de-Enfermagem.pdf"

if not pdf_path.exists():
    pdf_path = BASE_DIR / "Manual-de-Procedimentos-Básicos-de-Enfermagem.pdf"

doc = pymupdf.open(str(pdf_path))
page = doc[30] # Índice 29 = Página 30

texto_bruto = page.get_text("text")

print(f"📊 Tamanho total do texto bruto da Página 30: {len(texto_bruto)} caracteres.")

# Procura por caracteres nulos e de controle (exceto \n, \r, \t)
caracteres_invalidos = re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', texto_bruto)
print(f"⚠️ Caracteres de controle/nulos encontrados: {len(caracteres_invalidos)}")

if caracteres_invalidos:
    print(f"❌ Caracteres detectados: {set(caracteres_invalidos)}")
    # Mostra a posição exata do primeiro caractere nulo
    pos = re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', texto_bruto).start()
    print(f"📍 O texto corta exatamente no caractere {pos}:")
    print(f"--- ANTES DO CORTE ---")
    print(texto_bruto[max(0, pos-100):pos])
    print(f"--- DEPOIS DO CORTE (Trecho perdido) ---")
    print(texto_bruto[pos:pos+100])
else:
    print("✅ Nenhum caractere nulo detectado na extração direta do PyMuPDF.")
    print("\n--- ÚLTIMAS 300 LEIRAS EXTRAÍDAS DA PÁGINA 30 ---")
    print(texto_bruto[-300:])