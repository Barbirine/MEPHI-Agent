"""Create ONLY fictional, explicitly labelled demo PDFs. No API needed."""
from pathlib import Path
import fitz
from config import ROOT

DOCS = {
    'synthetic_a.pdf': [
        'SYNTHETIC EXAMPLE A. A lum is a fictional unit of stored light.',
        'SYNTHETIC EXAMPLE A. Stored light is Q = n * q. Here n is the number of cells and q is light per cell.',
        'SYNTHETIC EXAMPLE A. Equal-cell theorem: if every cell stores q lums, n cells store n*q lums. Proof: add q exactly n times.',
        'SYNTHETIC EXAMPLE A. The demonstration device has n = 3 cells.'],
    'synthetic_b.pdf': [
        'SYNTHETIC EXAMPLE B. In the demonstration device each cell stores q = 4 lums.'
    ]}

def create_demo(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCS.items():
        path = destination / name
        if path.exists():
            raise FileExistsError(f'Файл уже существует: {path}')
        with fitz.open() as pdf:
            for text in pages:
                page = pdf.new_page()
                page.insert_textbox(fitz.Rect(50,50,540,750), text, fontsize=12)
            pdf.save(path)

if __name__ == '__main__':
    create_demo(ROOT / 'data/pdfs')
