import logging
import re
from rag.gigachat_client import GigaChatClient
from config import Settings
from rag.chunker import chunk_pages
from rag.embeddings import LocalEmbedder
from rag.llm import Generator, GigaChatGenerator, REFUSAL
from rag.models import Answer
from rag.pdf_parser import parse_pdf
from rag.retriever import Retriever
from rag.vector_store import VectorStore

class Pipeline:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever, self.generator = retriever, generator

    def ask(self, question: str) -> Answer:
        hits = self.retriever.retrieve(question)
        if not hits:
            return Answer(REFUSAL, [], [])
        text = self.generator.generate(question, hits)
        ids = sorted({int(x) for x in re.findall(r'\[(\d+)\]', text)})
        if any(i < 1 or i > len(hits) for i in ids):
            raise ValueError('Ответ содержит несуществующую ссылку; повторите вопрос')
        if not ids and text.strip() != REFUSAL:
            raise ValueError('Модель не указала источники; повторите вопрос')
        return Answer(text, hits, ids)

def build_pipeline(settings: Settings) -> Pipeline:
    client = GigaChatClient(
        credentials=settings.credentials, scope=settings.scope,
        base_url=settings.base_url, auth_url=settings.auth_url,
        ca_bundle_file=settings.ca_bundle_file, timeout=settings.timeout)
    store = VectorStore(settings.index_path)
    embedder = LocalEmbedder()
    generator = GigaChatGenerator(client, settings.chat_model)
    return Pipeline(Retriever(store, embedder, settings.top_k, settings.min_similarity), generator)

def ingest(settings: Settings) -> int:
    paths = sorted(settings.pdf_dir.rglob('*')) if settings.pdf_dir.exists() else []
    paths = [p for p in paths if p.is_file() and p.suffix.lower() == '.pdf']
    if not paths:
        raise ValueError(f'Нет PDF в {settings.pdf_dir}')
    chunks = []
    for path in paths:
        chunks.extend(chunk_pages(parse_pdf(path, path.relative_to(settings.pdf_dir).as_posix()),
                                  settings.chunk_size, settings.chunk_overlap))
    store = VectorStore(settings.index_path)
    embedder = LocalEmbedder()
    store.replace(chunks, embedder.embed([c.text for c in chunks]), embedder.model)
    logging.getLogger(__name__).info('Индекс обновлён: %s фрагментов', len(chunks))
    return len(chunks)
