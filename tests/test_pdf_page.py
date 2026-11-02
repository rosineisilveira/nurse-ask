from pathlib import Path
import pymupdf

# Localiza a pasta Downloads do usuário automaticamente
DOWNLOADS_DIR = Path.home() / "Downloads"

# Procura o arquivo na pasta Downloads
pdf_path = DOWNLOADS_DIR / "Manual-de-Procedimentos-Básicos-de-Enfermagem.pdf"

if not pdf_path.exists():
    # Fallback: procura qualquer arquivo com 'Manual' no nome dentro de Downloads
    encontrados = list(DOWNLOADS_DIR.glob("*[Mm]anual*.pdf"))
    if encontrados:
        pdf_path = encontrados[0]
    else:
        raise FileNotFoundError(f"❌ Nenhum PDF do manual foi encontrado em: {DOWNLOADS_DIR}")

print(f"📄 Arquivo localizado em: {pdf_path}")

doc = pymupdf.open(str(pdf_path))
print(f"📖 Total de páginas do PDF: {len(doc)}")

# Inspeção direcionada à Página 30 (índice 29)
page_index = 30 if len(doc) >= 31 else len(doc) - 1
page = doc[page_index]
text_blocks = page.get_text("blocks")

print(f"\n--- CONTEÚDO EXTRAÍDO DA PÁGINA {page_index + 1} ---")
print(f"📊 Total de blocos de texto encontrados: {len(text_blocks)}\n")

for b in text_blocks:
    texto = b[4].strip()
    if texto:
        print(texto)
        print("-" * 40)