import logging
from pathlib import Path
import fitz
from rag.models import Page

log = logging.getLogger(__name__)

def parse_pdf(path: Path, document: str | None = None) -> list[Page]:
    """Preserve physical PDF page numbers (one-based), including empty pages."""
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with fitz.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError(f'PDF защищён паролем: {path.name}')
            pages = [Page(document or path.name, i + 1, page.get_text(sort=True).strip())
                     for i, page in enumerate(pdf)]
    except (RuntimeError, fitz.FileDataError) as exc:
        raise ValueError(f'Не удалось прочитать PDF: {path.name}') from exc
    empty = sum(not p.text for p in pages)
    log.info('PDF %s: %s страниц, %s без текста', path.name, len(pages), empty)
    if not any(p.text for p in pages):
        raise ValueError(f'В {path.name} нет текста. Для сканов требуется OCR.')
    return pages
