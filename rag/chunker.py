import hashlib
from rag.models import Chunk, Page

def chunk_pages(pages: list[Page], size: int = 1200, overlap: int = 200) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError('Требуется 0 <= overlap < size')
    chunks = []
    for page in pages:
        start = 0
        while start < len(page.text):
            end = min(start + size, len(page.text))
            text = page.text[start:end].strip()
            if text:
                identity = f'{page.document}\0{page.page}\0{start}\0{text}'
                chunks.append(Chunk(hashlib.sha256(identity.encode()).hexdigest(),
                                    page.document, page.page, text))
            if end == len(page.text):
                break
            start = end - overlap
    return chunks
